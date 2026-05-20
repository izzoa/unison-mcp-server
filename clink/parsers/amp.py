"""Parser for Amp CLI JSONL output from ``amp --execute --stream-json``.

Amp emits a stream of typed JSON events on stdout, one per line:

- ``{"type":"system","subtype":"init","cwd":...,"session_id":...,"tools":[...],"mcp_servers":[...]}``
- ``{"type":"user","message":{"role":"user","content":[{"type":"text","text":...}]}, ...}``
- ``{"type":"assistant","message":{"type":"message","role":"assistant","content":[{"type":"text","text":...}], "usage":...}, ...}``
- ``{"type":"result","subtype":"success","result":"<final answer>","duration_ms":...,"is_error":false,...}``

The canonical response is the ``result`` field of the ``result`` event. We
also collect every assistant text block as a fallback (in case Amp's JSON
schema changes or the result event is absent).
"""

from __future__ import annotations

import json
from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


class AmpJSONLParser(BaseParser):
    """Parse JSONL stdout from ``amp --execute --stream-json``."""

    name = "amp_jsonl"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]

        events: list[dict[str, Any]] = []
        assistant_messages: list[str] = []
        result_text: str | None = None
        session_id: str | None = None
        usage: dict[str, Any] | None = None
        is_error: bool = False
        error_subtype: str | None = None

        for line in lines:
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            events.append(event)
            event_type = event.get("type")

            if event_type == "system":
                if event.get("subtype") == "init":
                    sid = event.get("session_id")
                    if isinstance(sid, str):
                        session_id = sid

            elif event_type == "assistant":
                message = event.get("message") or {}
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text.strip():
                                assistant_messages.append(text.strip())
                # Usage is on the latest assistant message
                msg_usage = message.get("usage")
                if isinstance(msg_usage, dict):
                    usage = msg_usage

            elif event_type == "result":
                subtype = event.get("subtype")
                if event.get("is_error") is True:
                    is_error = True
                    error_subtype = subtype if isinstance(subtype, str) else None
                # `result` is the canonical final text (success cases)
                res = event.get("result")
                if isinstance(res, str):
                    result_text = res

        # Prefer the canonical result; fall back to concatenated assistant messages.
        if result_text is not None and result_text.strip():
            content = result_text.strip()
        elif assistant_messages:
            content = "\n\n".join(assistant_messages).strip()
        else:
            # Nothing usable in stdout. If stderr has content (e.g. auth error
            # before any JSONL emitted), surface that as the parse error.
            stderr_stripped = (stderr or "").strip()
            if stderr_stripped:
                raise ParserError(f"Amp produced no parseable response events. stderr: {stderr_stripped}")
            raise ParserError(
                "Amp produced no parseable response events. stdout did not contain "
                "an 'assistant' or 'result' event with text content."
            )

        metadata: dict[str, Any] = {"events": events}
        if session_id:
            metadata["session_id"] = session_id
        if usage:
            metadata["usage"] = usage
        if is_error:
            metadata["is_error"] = True
            if error_subtype:
                metadata["error_subtype"] = error_subtype
        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()

        return ParsedCLIResponse(content=content, metadata=metadata)
