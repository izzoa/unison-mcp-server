"""Execute configured CLI agents for the clink tool and parse output."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clink.constants import CLINK_DEPTH_ENV_VAR, DEFAULT_STREAM_LIMIT
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from clink.parsers import BaseParser, ParsedCLIResponse, ParserError, get_parser

logger = logging.getLogger("clink.agent")


def _noop_cleanup() -> None:
    """Cleanup function used for plans that don't need post-execution work."""


@dataclass
class AgentOutput:
    """Container returned by CLI agents after successful execution."""

    parsed: ParsedCLIResponse
    sanitized_command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    parser_name: str
    output_file_content: str | None = None


@dataclass(frozen=True)
class InvocationPlan:
    """Describes how to deliver the prompt + attachments to the spawned CLI.

    Constructed by :meth:`BaseCLIAgent.prepare_invocation` and consumed by
    :meth:`BaseCLIAgent.run` to dispatch transport-layer details that vary
    across CLIs. Supported kinds:

    - ``stdin`` (default): write the prompt to subprocess stdin (current
      behavior for Claude/Codex/Gemini/opencode).
    - ``argv``: append the prompt as a command-line argument, optionally
      preceded by ``flag`` (e.g. ``flag="--prompt"``).
    - ``message_file``: write the prompt to a temporary file and append
      ``flag <path>`` to the command (required for ``flag``). Used by Aider
      via ``--message-file``.
    - ``stream_json``: serialize prompt + images to a JSON message envelope
      and write to subprocess stdin. Used by Amp via ``--stream-json``.

    ``extra_payload`` carries kind-specific overrides — subclasses may stash
    structured fields here when the default serialization isn't enough (e.g.
    a custom Amp stream-json schema). The base materializer ignores keys it
    doesn't recognize.
    """

    kind: str
    flag: str | None = None
    extra_payload: dict[str, Any] = field(default_factory=dict)


class CLIAgentError(RuntimeError):
    """Raised when a CLI agent fails (non-zero exit, timeout, parse errors)."""

    def __init__(self, message: str, *, returncode: int | None = None, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class BaseCLIAgent:
    """Execute a configured CLI command and parse its output."""

    #: Flag tokens this CLI uses for model selection. Subclasses declare the
    #: forms (long, short) they emit from :meth:`render_model_args` so that
    #: :meth:`_strip_model_flags` can remove any pre-existing manifest-supplied
    #: model flag before the runtime model is appended. Empty tuple is a no-op.
    model_flag_aliases: tuple[str, ...] = ()

    #: Paths the CLI itself creates as bookkeeping (config caches, plugin
    #: installs, project-id files). Each pattern is either an exact relative
    #: path (string equality) OR a directory prefix ending in ``"/**"`` (matches
    #: the prefix or any descendant). ``fnmatch.fnmatch`` is intentionally NOT
    #: used because stdlib ``fnmatch`` does not implement bash-style globstar
    #: and produces incorrect results for path-shaped strings on every
    #: supported Python version. Empty tuple means "no bookkeeping declared";
    #: every detected change classifies as a model write.
    #:
    #: Consumed by :func:`utils.fs_snapshot.classify_changes` after the
    #: post-execution snapshot diff in :mod:`tools.clink`.
    fs_violation_ignore_patterns: tuple[str, ...] = ()

    def __init__(self, client: ResolvedCLIClient):
        self.client = client
        self._parser: BaseParser = get_parser(client.parser)
        self._logger = logging.getLogger(f"clink.runner.{client.name}")

    def get_read_only_args(self) -> list[str]:
        """Return CLI-specific flags for read-only mode.

        Subclasses override this to provide flags that restrict the CLI
        to read-only operations. The base implementation returns an empty
        list — enforcement relies on prompt injection and filesystem
        verification for unknown CLIs.
        """
        return []

    def prepare_invocation(
        self,
        prompt: str,
        files: Sequence[str],
        images: Sequence[str],
    ) -> InvocationPlan:
        """Return how to deliver ``prompt`` + ``files``/``images`` to the CLI.

        The default implementation returns ``InvocationPlan(kind="stdin")``
        which preserves the historical behavior of writing the prompt to
        subprocess stdin and ignoring ``files`` / ``images`` (the tool layer
        embeds those into ``prompt`` before calling ``run``).

        Subclasses override this when their CLI requires a different
        transport — e.g. Aider's ``--message-file`` needs ``kind="message_file"``,
        Amp's image input needs ``kind="stream_json"``.
        """
        _ = (prompt, files, images)
        return InvocationPlan(kind="stdin")

    def render_model_args(self, model: str) -> list[str]:
        """Return the argv fragment that conveys ``model`` to this CLI.

        Subclasses override to emit their CLI's flag form (e.g.
        ``["--model", model]`` or ``["-m", model]``). The base implementation
        returns ``[]`` so unknown CLIs that have no model selection silently
        ignore the runtime model.
        """
        return []

    def _strip_model_flags(self, command: list[str]) -> list[str]:
        """Remove every occurrence of ``model_flag_aliases`` from ``command``.

        Each matched alias is removed together with its immediately-following
        value. A bare alias that appears as the final token (no following
        value) is dropped and a WARNING is logged so misconfigured manifests
        are surfaced. Returns a new list; the input is not mutated.
        """
        if not self.model_flag_aliases:
            return list(command)

        aliases = set(self.model_flag_aliases)
        result: list[str] = []
        i = 0
        while i < len(command):
            token = command[i]
            if token in aliases:
                if i + 1 < len(command):
                    i += 2
                    continue
                self._logger.warning(
                    "Stripped trailing model flag alias '%s' with no value from CLI '%s' command",
                    token,
                    self.client.name,
                )
                i += 1
                continue
            result.append(token)
            i += 1
        return result

    def _apply_read_only(self, command: list[str]) -> list[str]:
        """Apply read-only restrictions to *command*.

        The default implementation appends :meth:`get_read_only_args`.
        Subclasses may override to strip conflicting flags first.
        """
        ro_args = self.get_read_only_args()
        if ro_args:
            command.extend(ro_args)
        return command

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
    ) -> AgentOutput:
        # The runner simply executes the configured CLI command for the selected role.
        command = self._build_command(role=role, system_prompt=system_prompt, model=model)
        env = self._build_environment()

        # Resolve executable path for cross-platform compatibility (especially Windows)
        executable_name = command[0]
        resolved_executable = shutil.which(executable_name)
        if resolved_executable is None:
            raise CLIAgentError(
                f"Executable '{executable_name}' not found in PATH for CLI '{self.client.name}'. "
                f"Ensure the command is installed and accessible."
            )
        command[0] = resolved_executable

        # Inject read-only sandbox flags when requested
        if read_only:
            command = self._apply_read_only(command)

        sanitized_command = list(command)

        cwd = str(self.client.working_dir) if self.client.working_dir else None
        limit = DEFAULT_STREAM_LIMIT

        stdout_text = ""
        stderr_text = ""
        output_file_content: str | None = None
        start_time = time.monotonic()

        output_file_path: Path | None = None
        command_with_output_flag = list(command)

        if self.client.output_to_file:
            fd, tmp_path = tempfile.mkstemp(prefix="clink-", suffix=".json")
            os.close(fd)
            output_file_path = Path(tmp_path)
            flag_template = self.client.output_to_file.flag_template
            try:
                rendered_flag = flag_template.format(path=str(output_file_path))
            except KeyError as exc:  # pragma: no cover - defensive
                raise CLIAgentError(f"Invalid output flag template '{flag_template}': missing placeholder {exc}")
            command_with_output_flag.extend(shlex.split(rendered_flag))
            sanitized_command = list(command_with_output_flag)

        # Materialize the invocation plan: stdin / argv / message_file / stream_json.
        # The default plan is "stdin" which preserves the historical behavior
        # (write the prompt to subprocess stdin). Subclasses override
        # prepare_invocation when their CLI needs a different transport.
        plan = self.prepare_invocation(prompt, files, images)
        extra_args, stdin_data, cleanup_plan = self._materialize_plan(plan, prompt, files, images)
        if extra_args:
            command_with_output_flag.extend(extra_args)
            sanitized_command = list(command_with_output_flag)

        self._logger.debug("Executing CLI command: %s", " ".join(sanitized_command))
        if cwd:
            self._logger.debug("Working directory: %s", cwd)

        try:
            process = await asyncio.create_subprocess_exec(
                *command_with_output_flag,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=limit,
                env=env,
            )
        except FileNotFoundError as exc:
            cleanup_plan()
            raise CLIAgentError(f"Executable not found for CLI '{self.client.name}': {exc}") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(stdin_data),
                timeout=self.client.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise CLIAgentError(
                f"CLI '{self.client.name}' timed out after {self.client.timeout_seconds} seconds",
                returncode=None,
            ) from exc
        finally:
            cleanup_plan()

        duration = time.monotonic() - start_time
        return_code = process.returncode
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        if output_file_path and output_file_path.exists():
            output_file_content = output_file_path.read_text(encoding="utf-8", errors="replace")
            if self.client.output_to_file and self.client.output_to_file.cleanup:
                try:
                    output_file_path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass

            if output_file_content and not stdout_text.strip():
                stdout_text = output_file_content

        if return_code != 0:
            recovered = self._recover_from_error(
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
                sanitized_command=sanitized_command,
                duration_seconds=duration,
                output_file_content=output_file_content,
            )
            if recovered is not None:
                return recovered

        if return_code != 0:
            raise CLIAgentError(
                f"CLI '{self.client.name}' exited with status {return_code}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            )

        try:
            parsed = self._parser.parse(stdout_text, stderr_text)
        except ParserError as exc:
            raise CLIAgentError(
                f"Failed to parse output from CLI '{self.client.name}': {exc}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            ) from exc

        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=return_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            parser_name=self._parser.name,
            output_file_content=output_file_content,
        )

    def _build_command(
        self,
        *,
        role: ResolvedCLIRole,
        system_prompt: str | None,
        model: str | None = None,
    ) -> list[str]:
        base = list(self.client.executable)
        base.extend(self.client.internal_args)
        base.extend(self.client.config_args)
        base.extend(role.role_args)

        if model:
            base = self._strip_model_flags(base)
            base.extend(self.render_model_args(model))

        return base

    def _build_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.client.env)
        # Propagate the clink recursion depth (incremented by one) so that any
        # CLI we spawn — if it itself re-invokes Unison via MCP — trips the
        # recursion guard in CLinkTool.execute() at the child process
        # boundary. See clink-multi-cli-infrastructure spec.
        current_depth = self._read_recursion_depth(env)
        env[CLINK_DEPTH_ENV_VAR] = str(current_depth + 1)
        return env

    @staticmethod
    def _read_recursion_depth(env: dict[str, str]) -> int:
        """Parse ``UNISON_CLINK_DEPTH`` from ``env``; default 0 if absent or invalid."""
        raw = env.get(CLINK_DEPTH_ENV_VAR, "")
        if not raw:
            return 0
        try:
            return int(raw)
        except ValueError:
            return 0

    def _materialize_plan(
        self,
        plan: InvocationPlan,
        prompt: str,
        files: Sequence[str],
        images: Sequence[str],
    ) -> tuple[list[str], bytes, Callable[[], None]]:
        """Convert an :class:`InvocationPlan` into ``(extra_args, stdin_bytes, cleanup)``.

        - ``stdin``: no extra args, prompt over stdin (current behavior).
        - ``argv``: prompt as a positional argument, optionally preceded by ``flag``.
        - ``message_file``: prompt written to a tempfile, ``flag <path>`` appended
          to the command, no stdin data. ``cleanup`` unlinks the tempfile.
        - ``stream_json``: prompt + images serialized into a default JSON envelope
          and written to stdin. Subclasses can override ``prepare_invocation`` to
          stash a fully custom payload in ``plan.extra_payload['serialized']``
          (bytes) when the default envelope doesn't match the CLI's schema.
        """
        if plan.kind == "stdin":
            return ([], prompt.encode("utf-8"), _noop_cleanup)

        if plan.kind == "argv":
            args: list[str] = []
            if plan.flag:
                args.append(plan.flag)
            args.append(prompt)
            return (args, b"", _noop_cleanup)

        if plan.kind == "message_file":
            if not plan.flag:
                raise CLIAgentError(
                    f"InvocationPlan kind='message_file' requires a 'flag' "
                    f"(CLI '{self.client.name}' returned a plan without one)"
                )
            fd, tmp_path = tempfile.mkstemp(prefix="clink-msg-", suffix=".txt")
            try:
                os.write(fd, prompt.encode("utf-8"))
            finally:
                os.close(fd)

            def _cleanup_tempfile() -> None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            return ([plan.flag, tmp_path], b"", _cleanup_tempfile)

        if plan.kind == "stream_json":
            # Subclasses may pre-serialize a CLI-specific payload and stash
            # the bytes in plan.extra_payload['serialized'] to override the
            # default envelope below.
            override = plan.extra_payload.get("serialized")
            if override is not None:
                if not isinstance(override, (bytes, bytearray)):
                    raise CLIAgentError(
                        f"InvocationPlan.extra_payload['serialized'] must be bytes; " f"got {type(override).__name__}"
                    )
                return ([], bytes(override), _noop_cleanup)
            payload: dict[str, Any] = {
                "messages": [{"role": "user", "content": prompt}],
            }
            if files:
                payload["files"] = list(files)
            if images:
                payload["images"] = list(images)
            return ([], json.dumps(payload).encode("utf-8"), _noop_cleanup)

        raise CLIAgentError(
            f"Unknown InvocationPlan kind '{plan.kind}' from CLI '{self.client.name}'. "
            f"Allowed: stdin, argv, message_file, stream_json."
        )

    # ------------------------------------------------------------------
    # Error recovery hooks
    # ------------------------------------------------------------------

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
        """Hook for subclasses to convert CLI errors into successful outputs.

        Return an AgentOutput to treat the failure as success, or None to signal
        that normal error handling should proceed.
        """

        return None
