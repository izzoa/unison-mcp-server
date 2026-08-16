"""Asynchronous job API for long clink CLI runs.

Some MCP hosts enforce tool-call deadlines far shorter than a real CLI run
(observed: Claude Desktop's local-agent mode cancelling calls around a
minute while an agentic Copilot review legitimately needs several). The job
API removes the single long-lived MCP request from the picture:

- ``clink_start`` accepts the same arguments as ``clink`` and returns within
  ~1 second. Runs that finish inside the grace window (including validation
  errors) return their result directly; anything slower returns a job id.
- ``clink_poll`` long-polls (bounded well under host deadlines) until the job
  reaches a terminal state, then returns the full clink result verbatim.
- ``clink_cancel`` cancels a running job; cancellation propagates through the
  agent's ``CancelledError`` path, which reaps the CLI process tree.

Design notes (from an adversarial design review):

- Every job runs on a **private** ``CLinkTool`` instance. The registry's
  cached singleton carries mutable per-request state; sharing it between a
  background job and foreground calls would race.
- The job runner disables the MCP progress heartbeat on its instance: the
  originating ``clink_start`` request has already completed, and progress
  against a completed request violates the protocol.
- State lives in-process. If the host restarts the server between calls,
  jobs are gone; ``clink_poll`` says so explicitly instead of guessing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, NoReturn

from mcp.types import TextContent

from tools.models import ToolModelCategory, ToolOutput
from tools.shared.base_models import ToolRequest
from tools.shared.base_tool import BaseTool
from tools.shared.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

#: Runs finishing inside this window return synchronously from clink_start —
#: fast CLI calls and validation errors never need a poll cycle.
GRACE_WINDOW_SECONDS = 1.0

#: How long a terminal job's result is retained for polling.
RESULT_TTL_SECONDS = 1800.0

#: Ceiling on simultaneously running jobs (each one is a paid CLI session).
MAX_CONCURRENT_JOBS = 3

#: Long-poll bounds: comfortably under any known host tool deadline.
DEFAULT_WAIT_MS = 20_000
MAX_WAIT_MS = 25_000

_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})

_POLL_CONTRACT = (
    "Job accepted and still running. You MUST call clink_poll with this job_id "
    "until it returns status completed, failed, or cancelled. Do not report the "
    "task as done before receiving the final result."
)


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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, arguments: dict[str, Any]) -> ClinkJob:
        """Spawn a background clink run and register its job record."""
        self.collect_garbage()

        idempotency_key = arguments.pop("idempotency_key", None)
        if idempotency_key:
            for job in self._jobs.values():
                if job.idempotency_key == idempotency_key and not job.terminal:
                    logger.info("clink_start: idempotency key matched running job %s", job.job_id)
                    return job

        running = sum(1 for job in self._jobs.values() if not job.terminal)
        if running >= MAX_CONCURRENT_JOBS:
            raise ToolExecutionError(
                ToolOutput(
                    status="error",
                    content=(
                        f"Too many concurrent clink jobs ({running} running, limit {MAX_CONCURRENT_JOBS}). "
                        "Poll or cancel existing jobs before starting new ones."
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
            result = await tool.execute(arguments)
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


# ---------------------------------------------------------------------------
# Shared plumbing for the three job tools
# ---------------------------------------------------------------------------


class _ClinkJobToolBase(BaseTool):
    """Utility base: no AI model, JSON payload responses."""

    def get_system_prompt(self) -> str:
        return ""

    def get_request_model(self):  # type: ignore[no-untyped-def]
        return ToolRequest

    def requires_model(self) -> bool:
        return False

    async def prepare_prompt(self, request: ToolRequest) -> str:
        return ""

    def format_response(self, response: str, request: ToolRequest, model_info: dict[str, Any] | None = None) -> str:
        return response

    def get_model_category(self) -> ToolModelCategory:
        return ToolModelCategory.FAST_RESPONSE

    def _respond(self, payload: dict[str, Any]) -> list[TextContent]:
        output = ToolOutput(
            status="success",
            content=json.dumps(payload, indent=2),
            content_type="json",
            metadata={"tool_name": self.get_name()},
        )
        return [TextContent(type="text", text=output.model_dump_json())]

    def _job_error(self, message: str) -> NoReturn:
        output = ToolOutput(status="error", content=message, content_type="text")
        raise ToolExecutionError(output.model_dump_json())

    @staticmethod
    def _job_summary(job: ClinkJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "status": job.state,
            "cli_name": job.cli_name,
            "elapsed_seconds": round(job.elapsed_seconds(), 1),
        }

    @staticmethod
    def _attach_terminal_payload(payload: dict[str, Any], job: ClinkJob) -> dict[str, Any]:
        if job.state == "completed" and job.result_json:
            try:
                payload["result"] = json.loads(job.result_json)
            except json.JSONDecodeError:
                payload["result"] = {"content": job.result_json}
        elif job.state == "failed" and job.error_json:
            try:
                payload["error"] = json.loads(job.error_json)
            except json.JSONDecodeError:
                payload["error"] = {"content": job.error_json}
        return payload


class ClinkStartTool(_ClinkJobToolBase):
    """Start a clink CLI run as a background job."""

    def get_name(self) -> str:
        return "clink_start"

    def get_description(self) -> str:
        return (
            "Start a clink CLI run as a background job that survives MCP host tool timeouts. "
            "Takes the same arguments as clink. Fast runs (and validation errors) return their "
            "result immediately; longer runs return a job_id you MUST follow up with clink_poll "
            "until a terminal status. Use this instead of clink when the run may exceed ~1 minute "
            "(large reviews, agentic tasks) or when a previous clink call timed out."
        )

    def get_annotations(self) -> dict[str, Any] | None:
        return {"readOnlyHint": False}

    def get_input_schema(self) -> dict[str, Any]:
        from tools.clink import CLinkTool

        schema = CLinkTool().get_input_schema()
        schema["properties"]["idempotency_key"] = {
            "type": "string",
            "description": (
                "Optional client-chosen key. Retrying clink_start with the same key while the "
                "original job is still running returns the existing job instead of launching "
                "duplicate paid CLI work."
            ),
        }
        return schema

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        manager = get_clink_job_manager()
        try:
            job = manager.start(dict(arguments))
        except ToolExecutionError:
            raise

        assert job.task is not None
        # Grace window: fast completions (and argument validation errors)
        # return synchronously so trivial calls never pay a poll round-trip.
        try:
            await asyncio.wait_for(asyncio.shield(job.task), timeout=GRACE_WINDOW_SECONDS)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            # Distinguish "the JOB got cancelled" (terminal — report it) from
            # "OUR start request got cancelled" (shield kept the job alive and
            # it stays pollable; propagate so the host sees its cancellation).
            if not job.terminal:
                raise
        except Exception:  # noqa: BLE001 - job errors are read from the record below
            pass

        payload = self._job_summary(job)
        if job.terminal:
            self._attach_terminal_payload(payload, job)
        else:
            payload["poll_after_ms"] = 5000
            payload["instructions"] = _POLL_CONTRACT
        return self._respond(payload)


class ClinkPollTool(_ClinkJobToolBase):
    """Poll a background clink job until it completes."""

    def get_name(self) -> str:
        return "clink_poll"

    def get_description(self) -> str:
        return (
            "Poll a clink background job started with clink_start. Long-polls up to wait_ms "
            "(default 20000, max 25000) and returns the job status; when the status is "
            "completed the full clink result is included under 'result' (failures under "
            "'error'). Keep calling until you receive a terminal status: completed, failed, "
            "or cancelled."
        )

    def get_annotations(self) -> dict[str, Any] | None:
        return {"readOnlyHint": True}

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job id returned by clink_start."},
                "wait_ms": {
                    "type": "integer",
                    "description": f"Long-poll duration in milliseconds (0-{MAX_WAIT_MS}, default {DEFAULT_WAIT_MS}).",
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        job_id = str(arguments.get("job_id") or "").strip()
        if not job_id:
            self._job_error("job_id is required")

        manager = get_clink_job_manager()
        job = manager.get(job_id)
        if job is None:
            self._job_error(
                f"Unknown clink job '{job_id}'. It may have expired ({int(RESULT_TTL_SECONDS / 60)} min retention) "
                "or the MCP server restarted since clink_start (jobs are in-memory). Start a new job."
            )

        raw_wait = arguments.get("wait_ms", DEFAULT_WAIT_MS)
        try:
            wait_ms = max(0, min(int(raw_wait), MAX_WAIT_MS))
        except (TypeError, ValueError):
            wait_ms = DEFAULT_WAIT_MS

        if not job.terminal and wait_ms > 0:
            # Waiting on the event is independent of the job task: a cancelled
            # poll request never cancels the job itself.
            try:
                await asyncio.wait_for(job.done_event.wait(), timeout=wait_ms / 1000.0)
            except asyncio.TimeoutError:
                pass

        payload = self._job_summary(job)
        if job.terminal:
            self._attach_terminal_payload(payload, job)
        else:
            payload["poll_after_ms"] = 5000
            payload["instructions"] = _POLL_CONTRACT
        return self._respond(payload)


class ClinkCancelTool(_ClinkJobToolBase):
    """Cancel a background clink job."""

    def get_name(self) -> str:
        return "clink_cancel"

    def get_description(self) -> str:
        return (
            "Cancel a clink background job started with clink_start. Cancellation kills the "
            "spawned CLI's entire process tree. Returns the job's state after the cancel request."
        )

    def get_annotations(self) -> dict[str, Any] | None:
        return {"readOnlyHint": False}

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job id returned by clink_start."},
            },
            "required": ["job_id"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        job_id = str(arguments.get("job_id") or "").strip()
        if not job_id:
            self._job_error("job_id is required")

        manager = get_clink_job_manager()
        job = manager.cancel(job_id)
        if job is None:
            self._job_error(f"Unknown clink job '{job_id}' (expired, or the server restarted).")

        # Give the cancellation a moment to propagate so the response usually
        # reflects the terminal state rather than 'running'.
        try:
            await asyncio.wait_for(job.done_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

        return self._respond(self._job_summary(job))
