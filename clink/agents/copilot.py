"""GitHub Copilot CLI agent for clink.

Copilot CLI (https://docs.github.com/copilot/how-tos/copilot-cli) is GitHub's
agentic coding CLI. clink invokes it non-interactively by piping the prompt to
stdin — no ``-p`` flag — which runs to completion and emits a terminal
``result`` event. Keeping the prompt off argv avoids the ``ARG_MAX`` ceiling
that clink prompts with embedded file contents can reach.

**Authentication:** Copilot reads ``COPILOT_GITHUB_TOKEN``, ``GH_TOKEN`` or
``GITHUB_TOKEN`` from the environment, or credentials stored by ``copilot
login``. Unison does not manage that state. Access is additionally gated by
GitHub organization policy; where the Copilot CLI policy is disabled, the BYOK
path (``COPILOT_PROVIDER_BASE_URL``) bypasses GitHub routing entirely and clink
parses the result identically.

**Read-only mode:** enforced fail-closed. ``--available-tools`` restricts the
model's tool schema to ``view``/``grep``/``glob``; an allowlist is used rather
than ``--excluded-tools`` because a denylist is fail-open — a tool added by a
future Copilot release would be permitted until someone noticed. Verified
against 1.0.78: the allowlist reduces the schema from 23 tools to exactly 3,
and it *does* reach MCP server tools (5 ``github-mcp-server-*`` tools were
removed along with the native ones), so no separate MCP handling is needed.
``--deny-tool`` for ``write`` and ``shell`` is retained as a second layer,
since Copilot documents that denial overrides all allow rules including
``--allow-all-tools``.

**Recursion guard:** Copilot is MCP-aware (``copilot mcp``,
``~/.copilot/mcp-config.json``). The cross-cutting guard in
``CLinkTool.execute()`` covers it; no Copilot-specific guard is implemented.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from typing import Any

from clink.models import ResolvedCLIClient
from clink.parsers.base import ParserError

from .base import AgentOutput, BaseCLIAgent, CLIAgentError, InvocationPlan

logger = logging.getLogger(__name__)

#: Tools the model may see in read-only mode. Everything else — including
#: `bash`, `create`, `edit`, `write_agent`, `task` (subagent delegation, whose
#: permission inheritance is not something we can verify) and `skill` — is
#: withheld from the schema entirely.
READ_ONLY_AVAILABLE_TOOLS = ("view", "grep", "glob")

#: Flags that broaden permissions and must not survive into a read-only
#: invocation. A user manifest override in ~/.unison/cli_clients could
#: otherwise reintroduce them. `--allow-all-tools` is deliberately NOT in this
#: list: Copilot requires it for non-interactive operation, and denial rules
#: outrank it.
_PERMISSION_BROADENING_FLAGS = frozenset(
    {
        "--yolo",
        "--allow-all",
        "--allow-all-paths",
        "--allow-all-urls",
    }
)

#: Attachment formats Copilot accepts.
_SUPPORTED_ATTACHMENT_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf", ".heic", ".heif"})


class CopilotAgent(BaseCLIAgent):
    """GitHub Copilot CLI agent — stdin transport, fail-closed read-only."""

    model_flag_aliases: tuple[str, ...] = ("--model",)

    # Copilot keeps session state under ~/.copilot (or $COPILOT_HOME), outside
    # the working directory. `--share` would write ./copilot-session-<id>.md
    # into the cwd, but clink never passes it.
    fs_violation_ignore_patterns: tuple[str, ...] = ()

    def __init__(self, client: ResolvedCLIClient) -> None:
        super().__init__(client)
        self._attachment_tempfiles: list[str] = []
        self._skipped_attachments: list[str] = []

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def render_model_args(self, model: str) -> list[str]:
        return ["--model", model]

    # ------------------------------------------------------------------
    # Read-only enforcement
    # ------------------------------------------------------------------

    def get_read_only_args(self) -> list[str]:
        """Restrict the model's tool schema to an explicit read-only allowlist."""
        return [
            "--available-tools",
            ",".join(READ_ONLY_AVAILABLE_TOOLS),
            "--deny-tool",
            "write",
            "--deny-tool",
            "shell",
        ]

    def _apply_read_only(self, command: list[str]) -> list[str]:
        """Strip permission-broadening flags before appending restrictions."""
        filtered = [arg for arg in command if arg not in _PERMISSION_BROADENING_FLAGS]
        return super()._apply_read_only(filtered)

    # ------------------------------------------------------------------
    # Attachments
    # ------------------------------------------------------------------

    def prepare_invocation(
        self,
        prompt: str,
        files: Sequence[str],
        images: Sequence[str],
    ) -> InvocationPlan:
        """Deliver the prompt on stdin, contributing ``--attachment`` argv pairs.

        Verified against 1.0.78 that a piped stdin prompt and ``--attachment``
        coexist in one invocation.
        """
        _ = files
        self._attachment_tempfiles = []
        self._skipped_attachments = []

        extra_args: list[str] = []
        for entry in images:
            path = self._materialize_attachment(entry)
            if path is None:
                continue
            extra_args.extend(["--attachment", path])

        return InvocationPlan(kind="stdin", extra_args=extra_args)

    def _materialize_attachment(self, entry: str) -> str | None:
        """Return a filesystem path for ``entry``, or None if unusable.

        clink's ``images`` field accepts a path or a base64 blob, but
        ``--attachment`` needs a path, so blobs are written to a temp file.
        """
        if not entry:
            return None

        if os.path.exists(entry):
            if os.path.splitext(entry)[1].lower() not in _SUPPORTED_ATTACHMENT_SUFFIXES:
                self._skipped_attachments.append(entry)
                logger.warning("Skipping attachment with unsupported format for Copilot: %s", entry)
                return None
            return entry

        payload, suffix = self._decode_blob(entry)
        if payload is None:
            self._skipped_attachments.append(entry[:64])
            logger.warning("Skipping unusable Copilot attachment entry (not a path or decodable blob)")
            return None

        fd, tmp_path = tempfile.mkstemp(prefix="clink-copilot-", suffix=suffix)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        self._attachment_tempfiles.append(tmp_path)
        return tmp_path

    @staticmethod
    def _decode_blob(entry: str) -> tuple[bytes | None, str]:
        """Decode a base64 blob, honouring an optional ``data:`` URI prefix."""
        suffix = ".png"
        payload = entry
        if entry.startswith("data:"):
            header, _, remainder = entry.partition(",")
            if not remainder:
                return None, suffix
            payload = remainder
            if "image/jpeg" in header:
                suffix = ".jpg"
            elif "application/pdf" in header:
                suffix = ".pdf"
            elif "image/webp" in header:
                suffix = ".webp"
            elif "image/gif" in header:
                suffix = ".gif"
        try:
            return base64.b64decode(payload, validate=True), suffix
        except (binascii.Error, ValueError):
            return None, suffix

    def cleanup_attachments(self) -> None:
        """Delete any temp files created for base64 attachments."""
        for path in self._attachment_tempfiles:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._attachment_tempfiles = []

    async def run(self, **kwargs: Any) -> AgentOutput:
        """Run the CLI, guaranteeing attachment temp files are cleaned up.

        ``_materialize_plan``'s cleanup callback only covers files the base
        materializer creates, so attachment temp files are released here — on
        the error and timeout paths as well as on success.
        """
        try:
            return await super().run(**kwargs)
        finally:
            self.cleanup_attachments()

    # ------------------------------------------------------------------
    # Error recovery
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
        """Salvage a response when Copilot exits non-zero but still answered.

        When it did not answer, re-raise with the diagnostic Copilot reported
        (an organization-policy denial, for instance) rather than letting the
        caller see a bare exit status or a parser complaint.
        """
        try:
            parsed = self._parser.parse(stdout, stderr)
        except ParserError:
            detail = self._extract_failure_detail(stdout) or (stderr or "").strip()
            if detail:
                raise CLIAgentError(
                    f"CLI '{self.client.name}' failed: {detail}",
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                ) from None
            return None

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

    @staticmethod
    def _extract_failure_detail(stdout: str) -> str | None:
        """Pull the most recent error message out of Copilot's JSONL stream."""
        detail: str | None = None
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, RecursionError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") not in ("session.error", "model.call_failure"):
                continue
            data = event.get("data")
            if not isinstance(data, dict):
                continue
            message = data.get("message") or data.get("errorMessage")
            if isinstance(message, str) and message.strip():
                detail = message.strip()
        return detail
