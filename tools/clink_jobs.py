"""Background job manager for long clink CLI runs.

Some MCP hosts enforce tool-call deadlines far shorter than a real CLI run
(observed: Claude Desktop's local-agent mode cancelling calls around a
minute while an agentic Copilot review legitimately needs several). Job
control is folded into the ``clink`` tool itself: every clink call waits a
bounded time and returns a ``job_id`` when the CLI is still running; calling
clink again with that ``job_id`` continues waiting, and ``cancel: true``
aborts. This module owns the job records and the background execution.

Design notes (from an adversarial design review):

- Every job runs on a **private** ``CLinkTool`` instance. The registry's
  cached singleton carries mutable per-request state; sharing it between a
  background job and foreground calls would race.
- The job runner disables the MCP progress heartbeat on its instance: by the
  time the job is still running, its originating request has completed, and
  progress against a completed request violates the protocol. The foreground
  waiting call emits heartbeats instead.
- State lives in-process. If the host restarts the server between calls,
  jobs are gone; clink says so explicitly instead of guessing.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from tools.models import ToolOutput
from tools.shared.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

#: How long a terminal job's result is retained for follow-up calls.
RESULT_TTL_SECONDS = 1800.0

#: Ceiling on simultaneously running jobs (each one is a paid CLI session).
MAX_CONCURRENT_JOBS = 3

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


@dataclass
class ClinkJob:
    """One background clink run and its lifecycle."""

    job_id: str
    cli_name: str
    state: str = "running"
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    expires_at: float | None = None
    idempotency_key: str | None = None
    task: asyncio.Task[None] | None = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    result_json: str | None = None
    error_json: str | None = None

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.created_at


class ClinkJobManager:
    """In-process registry of background clink jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, ClinkJob] = {}

    def start(self, arguments: dict[str, Any], idempotency_key: str | None = None) -> ClinkJob:
        """Spawn a background clink run and register its job record.

        ``arguments`` must already be stripped of job-control keys; they are
        handed to a private ``CLinkTool``'s blocking execution path.
        """
        self.collect_garbage()

        if idempotency_key:
            for job in self._jobs.values():
                if job.idempotency_key == idempotency_key and not job.terminal:
                    logger.info("clink: idempotency key matched running job %s", job.job_id)
                    return job

        running = sum(1 for job in self._jobs.values() if not job.terminal)
        if running >= MAX_CONCURRENT_JOBS:
            raise ToolExecutionError(
                ToolOutput(
                    status="error",
                    content=(
                        f"Too many concurrent clink jobs ({running} running, limit {MAX_CONCURRENT_JOBS}). "
                        "Continue (job_id) or cancel existing jobs before starting new ones."
                    ),
                    content_type="text",
                ).model_dump_json()
            )

        job = ClinkJob(
            job_id=uuid.uuid4().hex,
            cli_name=str(arguments.get("cli_name") or "auto"),
            idempotency_key=idempotency_key,
        )
        job.task = asyncio.create_task(self._run(job, dict(arguments)), name=f"clink-job-{job.job_id}")
        self._jobs[job.job_id] = job
        return job

    async def _run(self, job: ClinkJob, arguments: dict[str, Any]) -> None:
        from tools.clink import CLinkTool

        tool = CLinkTool()
        tool.disable_progress_heartbeat()
        try:
            result = await tool._execute_blocking(arguments)
            job.result_json = result[0].text if result else "{}"
            job.state = "completed"
        except ToolExecutionError as exc:
            job.error_json = exc.payload
            job.state = "failed"
        except asyncio.CancelledError:
            job.state = "cancelled"
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("clink job %s crashed", job.job_id)
            job.error_json = ToolOutput(
                status="error",
                content=f"clink job crashed unexpectedly: {exc}",
                content_type="text",
            ).model_dump_json()
            job.state = "failed"
        finally:
            job.finished_at = time.time()
            job.expires_at = job.finished_at + RESULT_TTL_SECONDS
            job.done_event.set()

    def get(self, job_id: str) -> ClinkJob | None:
        self.collect_garbage()
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> ClinkJob | None:
        job = self.get(job_id)
        if job is None:
            return None
        if not job.terminal and job.task is not None:
            job.task.cancel()
        return job

    def running_count(self) -> int:
        return sum(1 for job in self._jobs.values() if not job.terminal)

    def collect_garbage(self) -> None:
        """Drop terminal jobs whose retention window has passed."""
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.terminal and job.expires_at is not None and job.expires_at <= now
        ]
        for job_id in expired:
            del self._jobs[job_id]

    async def shutdown(self) -> None:
        """Cancel and await all running jobs (server teardown)."""
        tasks = [job.task for job in self._jobs.values() if job.task is not None and not job.task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown must not raise
                pass


_manager: ClinkJobManager | None = None


def get_clink_job_manager() -> ClinkJobManager:
    global _manager
    if _manager is None:
        _manager = ClinkJobManager()
    return _manager


def reset_clink_job_manager_for_tests() -> None:
    global _manager
    _manager = None


async def shutdown_clink_jobs() -> None:
    """Cancel all running clink jobs; safe to call when none were started."""
    if _manager is not None:
        await _manager.shutdown()
