"""Opencode-specific CLI agent hooks."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from clink.models import ResolvedCLIRole
from clink.parsers.base import ParsedCLIResponse, ParserError
from utils.env import get_env

from .base import AgentOutput, BaseCLIAgent, CLIAgentError

logger = logging.getLogger("clink.runner.opencode")

#: ``opencode --version`` prints ``opencode v2.2.0`` on v2 and ``1.18.35`` on v1.
_VERSION_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_PROBE_TIMEOUT_SECONDS = 5.0

#: ``standalone`` (default) runs opencode v2 on a private server owned by the
#: spawned process; ``service`` uses the user's shared background service.
OPENCODE_MODE_ENV = "CLINK_OPENCODE_MODE"

_SESSION_ID_PATTERN = re.compile(rb'"sessionID"\s*:\s*"([^"]+)"')

#: Probe results keyed by ``(path, mtime_ns, size)`` so an in-place upgrade is
#: re-probed. Failures are cached too: a hanging binary costs one probe, not one
#: per call.
_BinaryKey = tuple[str, int, int]
_VERSION_CACHE: dict[_BinaryKey, str | None] = {}
_INFLIGHT: dict[tuple[int, _BinaryKey], asyncio.Future[str | None]] = {}
_WARNED: set[str] = set()


def _probe_opencode_version(path: str) -> str | None:
    """Run ``<path> --version`` and return the first ``X.Y.Z`` it prints.

    Module-level so tests can replace it; returns ``None`` on any failure.
    """
    try:
        completed = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    match = _VERSION_PATTERN.search(f"{completed.stdout}\n{completed.stderr}")
    return match.group(0) if match else None


def _binary_key(path: str) -> _BinaryKey | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (path, stat.st_mtime_ns, stat.st_size)


def _major_version(version: str | None) -> int | None:
    return int(version.split(".", 1)[0]) if version else None


def reset_version_cache() -> None:
    """Forget every probe result and once-per-process warning (test hook)."""
    _VERSION_CACHE.clear()
    _INFLIGHT.clear()
    _WARNED.clear()


async def _probe_and_record(path: str, key: _BinaryKey) -> str | None:
    version = await asyncio.to_thread(_probe_opencode_version, path)
    _VERSION_CACHE[key] = version
    if version:
        logger.info("opencode at %s is v%s", path, version)
    else:
        logger.warning(
            "Could not determine the version of opencode at %s; v2 isolation (--standalone, --title) is not applied",
            path,
        )
    return version


async def _detect_version(path: str) -> str | None:
    """Version of the opencode binary at ``path``, probed once per binary off the event loop."""
    key = _binary_key(path)
    if key is None:
        return None
    if key in _VERSION_CACHE:
        return _VERSION_CACHE[key]
    flight = (id(asyncio.get_running_loop()), key)
    task = _INFLIGHT.get(flight)
    if task is None:
        task = asyncio.ensure_future(_probe_and_record(path, key))
        _INFLIGHT[flight] = task

        def _forget(_done: asyncio.Future[str | None]) -> None:
            _INFLIGHT.pop(flight, None)

        task.add_done_callback(_forget)
    # shield: one caller being cancelled must not cancel the probe the others await
    return await asyncio.shield(task)


def _warn_once(key: str, message: str, *args: object) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)


_SERVICE_MODE_WARNING = (
    "opencode v2 will use the shared background service (%s): clink's environment, including "
    "UNISON_CLINK_DEPTH and UNISON_CLINK_STRIP_SECRETS, does not reach its MCP servers or provider auth, so "
    "recursion-depth propagation and secret stripping are not guaranteed, and cancellation is only best-effort "
    "(the interrupt clink sends reaches the service, but nothing reclaims a session left by a force-kill). "
    "If no opencode background service is running, the call auto-starts a detached service that outlives clink and "
    "inherits clink's environment (including UNISON_CLINK_DEPTH and any unstripped secrets) for every later session "
    "that uses it, including your own."
)


def _opencode_mode() -> str:
    raw = (get_env(OPENCODE_MODE_ENV) or "").strip().lower()
    if raw in ("", "standalone"):
        return "standalone"
    if raw == "service":
        return "service"
    _warn_once(
        f"mode:{raw}",
        "Ignoring invalid %s=%r (expected 'standalone' or 'service'); using 'standalone'",
        OPENCODE_MODE_ENV,
        raw,
    )
    return "standalone"


class OpencodeAgent(BaseCLIAgent):
    """Opencode CLI agent with JSONL recovery support.

    Read-only enforcement for opencode is via layer-2 (prompt instruction) and
    layer-3 (filesystem snapshot diff) only — opencode has no CLI flag for
    read-only-while-still-executing semantics. ``--agent plan`` was tried in
    v11.8.0 but switches the agent persona (producing planning-language instead
    of executing the requested task), so it is not a true read-only sandbox.
    """

    model_flag_aliases: tuple[str, ...] = ("-m",)

    # opencode keeps the session outside this process. SIGINT makes its client
    # interrupt the session (releasing the execution claim); SIGKILL leaves the
    # claim behind and the user's opencode service resumes the session at its
    # next start, spending again with nobody listening.
    interrupt_signal = signal.SIGINT
    interrupt_grace_seconds = 5.0

    # Paths opencode v0.18/v1 create as bookkeeping on first-run bootstrap in any
    # working directory (captured against v0.18.x; see the regression fixture in
    # tests/test_clink_opencode_parser.py). opencode v2 writes no bookkeeping into
    # the project, so for v2 these are simply unused. The snapshot prunes ``.git``
    # and ``node_modules``, so ``.git/opencode`` and ``.opencode/node_modules/**``
    # only matter to callers that disable pruning. NOT a directory-wide glob over
    # .opencode/ because .opencode/skills/ and .opencode/commands/ are
    # user-extensions where model writes MUST classify as by_model, not
    # bookkeeping.
    fs_violation_ignore_patterns: tuple[str, ...] = (
        ".opencode/.gitignore",
        ".opencode/package.json",
        ".opencode/package-lock.json",
        ".opencode/node_modules/**",
        ".git/opencode",
    )

    def get_read_only_args(self) -> list[str]:
        """Opencode has no CLI flag for read-only-while-still-executing mode.

        ``--agent plan`` was tried in v11.8.0 but switches the agent persona to
        opencode's plan agent (which produces planning-language instead of
        executing the requested task), so it is not a true read-only sandbox.
        Layer-1 enforcement is intentionally absent for opencode; layer-2
        (prompt instruction) and layer-3 (filesystem snapshot diff) provide
        enforcement, both of which are CLI-agnostic.
        """
        return []

    def render_model_args(self, model: str) -> list[str]:
        return ["-m", model]

    # ------------------------------------------------------------------
    # opencode v2: version gate, private server, session title
    # ------------------------------------------------------------------

    def _is_opencode_launcher(self) -> bool:
        name = Path(self.client.executable[0]).name.lower()
        for suffix in (".exe", ".cmd", ".bat"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name in ("opencode", "opencode2")

    def _resolve_binary(self) -> str | None:
        # A wrapper (npx, bunx, node, env, docker) answers --version with its own
        # version, so its major says nothing about the opencode it launches: treat
        # it as an unknown version (pre-v2 command line) rather than guess.
        if not self._is_opencode_launcher():
            return None
        return shutil.which(self.client.executable[0])

    def _cached_major(self) -> int | None:
        """Major version already probed for the resolved binary, else ``None`` (never probes)."""
        binary = self._resolve_binary()
        key = _binary_key(binary) if binary else None
        return _major_version(_VERSION_CACHE.get(key)) if key else None

    async def run(
        self,
        *,
        role: ResolvedCLIRole,
        prompt: str,
        system_prompt: str | None = None,
        files: Sequence[str],
        images: Sequence[str],
        read_only: bool = False,
        model: str | None = None,
        working_dir: Path | None = None,
    ) -> AgentOutput:
        # Probe before building the command: the probe runs off the event loop,
        # and _build_command (synchronous) then only reads the cached result.
        binary = self._resolve_binary()
        if binary is not None:
            await _detect_version(binary)
        result = await super().run(
            role=role,
            prompt=prompt,
            system_prompt=system_prompt,
            files=files,
            images=images,
            read_only=read_only,
            model=model,
            working_dir=working_dir,
        )
        # opencode v1 exits 0 on session errors (upstream anomalyco/opencode#14551),
        # so an error-only stream is a failure whatever the exit status.
        if result.parsed.metadata.get("error_only"):
            self._raise_error_only(result.parsed, result.returncode, result.stdout, result.stderr)
        return result

    def _raise_error_only(self, parsed: ParsedCLIResponse, returncode: int, stdout: str, stderr: str) -> NoReturn:
        """Fail with opencode's own error type and message leading the text.

        stderr is empty on v2: the reason lives only in the JSON error event on stdout.
        """
        errors = parsed.metadata.get("errors") or [parsed.content]
        raise CLIAgentError(
            f"CLI '{self.client.name}' error: {errors[0]}",
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _build_command(
        self,
        *,
        role: ResolvedCLIRole,
        system_prompt: str | None,
        model: str | None = None,
    ) -> list[str]:
        command = super()._build_command(role=role, system_prompt=system_prompt, model=model)
        major = self._cached_major()
        if major is None or major < 2:
            return command
        return command + self._v2_arguments(command, role)

    def _v2_arguments(self, command: list[str], role: ResolvedCLIRole) -> list[str]:
        """Arguments only opencode v2 accepts, appended after the model flags."""
        extra: list[str] = []
        if any(token == "--server" or token.startswith("--server=") for token in command):
            _warn_once("manifest-server", _SERVICE_MODE_WARNING, "the manifest passes --server")
        elif "--standalone" not in command:
            if _opencode_mode() == "service":
                _warn_once("service-mode", _SERVICE_MODE_WARNING, f"{OPENCODE_MODE_ENV}=service")
            else:
                extra.append("--standalone")
        if "--title" not in command:
            # Without a title opencode makes an extra paid small-model request per run.
            extra.extend(["--title", f"clink:{role.name}"])
        return extra

    def _on_force_kill(self, captured_stdout: bytes, cwd: str | None) -> None:
        """Report the session a SIGKILLed opencode may have left suspended."""
        major = self._cached_major()
        # v1 runs in-process: its session is not resumed by a background service, and
        # its CLI rejects --standalone, so the v2 cleanup advice would be wrong there.
        v1 = major is not None and major < 2
        match = _SESSION_ID_PATTERN.search(captured_stdout)
        if match:
            session_id = match.group(1).decode("utf-8", errors="replace")
            if v1:
                logger.warning(
                    "opencode ignored the interrupt and was force-killed; its session %s may not have been closed "
                    "cleanly",
                    session_id,
                )
            else:
                logger.warning(
                    "opencode ignored the interrupt and was force-killed; its session %s may stay suspended and be "
                    "resumed by your opencode service. Delete it with: opencode session delete --standalone %s "
                    "(--standalone keeps the command from starting the background service)",
                    session_id,
                    session_id,
                )
            return
        if cwd:
            where = cwd
        else:
            try:
                where = os.getcwd()
            except OSError:  # the server's own working directory was deleted
                where = "<unknown: the server's working directory no longer exists>"
        if v1:
            logger.warning(
                "opencode ignored the interrupt and was force-killed before reporting a session ID; a session for "
                "working directory %s may not have been closed cleanly",
                where,
            )
        else:
            logger.warning(
                "opencode ignored the interrupt and was force-killed before reporting a session ID; a session for "
                "working directory %s may stay suspended and be resumed by your opencode service. List and delete "
                "it with: opencode session list --standalone / opencode session delete --standalone <id>",
                where,
            )

    def _recover_from_error(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        sanitized_command: list[str],
        duration_seconds: float,
        output_file_content: str | None,
    ) -> AgentOutput | None:
        try:
            parsed = self._parser.parse(stdout, stderr)
        except ParserError:
            return None

        if parsed.metadata.get("error_only"):
            self._raise_error_only(parsed, returncode, stdout, stderr)

        # Non-zero exit but assistant text came through (exit 130 after an
        # interrupt, a disconnect, an error after partial output): the content
        # may be incomplete, and the return code alone is the only other signal.
        parsed.metadata["partial"] = True

        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration_seconds,
            parser_name=self._parser.name,
            output_file_content=output_file_content,
        )
