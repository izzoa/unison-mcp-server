"""Parser for GitHub Copilot CLI JSONL output from ``copilot --output-format json``.

Copilot emits one JSON object per line. Every event except the terminal one
shares an envelope::

    {"type": ..., "data": {...}, "ephemeral": bool?, "id": ..., "timestamp": ...,
     "parentId": ..., "agentId": ...?}

The terminal ``result`` event is **flat** — ``sessionId``, ``exitCode`` and
``usage`` sit at the top level with no ``data`` wrapper — so it needs its own
handling.

Response selection keys on ``type == "assistant.message"`` and reads
``data.content``. It deliberately does NOT key on ``ephemeral``: that flag means
"not persisted", not "not the answer", and coupling finality to a
session-storage concern would be fragile.

Subagent discrimination: Copilot can delegate through its ``task`` tool, and the
resulting subagent emits its own durable ``assistant.message`` events. Those
carry ``agentId`` on the envelope and ``parentToolCallId`` inside ``data``; root
messages carry neither. Both are checked, since either alone would be enough and
redundancy costs nothing.

Captured against GitHub Copilot CLI 1.0.78.
"""

from __future__ import annotations

import json
from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


def _is_subagent(event: dict[str, Any], data: dict[str, Any]) -> bool:
    """True when the event belongs to a delegated subagent rather than the root."""
    if event.get("agentId"):
        return True
    return bool(data.get("parentToolCallId"))


class CopilotJSONLParser(BaseParser):
    """Parse JSONL stdout from ``copilot --output-format json``."""

    name = "copilot_jsonl"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        root_messages: list[str] = []
        model_used: str | None = None
        session_id: str | None = None
        exit_code: int | None = None
        usage: dict[str, Any] | None = None
        diagnostics: list[str] = []

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

            event_type = event.get("type")

            # The terminal event is flat: no `data` wrapper.
            if event_type == "result":
                sid = event.get("sessionId")
                if isinstance(sid, str):
                    session_id = sid
                code = event.get("exitCode")
                if isinstance(code, int):
                    exit_code = code
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = event_usage
                continue

            data = event.get("data")
            if not isinstance(data, dict):
                data = {}

            if event_type == "assistant.message":
                if _is_subagent(event, data):
                    continue
                content = data.get("content")
                if isinstance(content, str) and content.strip():
                    root_messages.append(content.strip())
                    model = data.get("model")
                    if isinstance(model, str):
                        model_used = model

            elif event_type in ("session.error", "model.call_failure"):
                message = data.get("message") or data.get("errorMessage")
                if isinstance(message, str) and message.strip():
                    diagnostics.append(message.strip())

        if not root_messages:
            detail = diagnostics[-1] if diagnostics else (stderr or "").strip()
            if detail:
                raise ParserError(f"Copilot produced no assistant response. {detail}")
            raise ParserError(
                "Copilot produced no assistant response. stdout contained no "
                "'assistant.message' event with text content."
            )

        # Metadata is whitelisted rather than accumulated: Copilot's
        # `session.skills_loaded` event enumerates the user's personal skill
        # names and filesystem paths, which has no reason to reach any
        # response, log, or debug surface.
        metadata: dict[str, Any] = {}
        if model_used:
            metadata["model_used"] = model_used
        if session_id:
            metadata["session_id"] = session_id
        if exit_code is not None:
            metadata["exit_code"] = exit_code
        if usage:
            metadata["usage"] = usage
        if diagnostics:
            metadata["diagnostics"] = diagnostics
        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()

        return ParsedCLIResponse(content="\n\n".join(root_messages), metadata=metadata)
