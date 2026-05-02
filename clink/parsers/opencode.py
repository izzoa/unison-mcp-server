"""Parser for opencode CLI JSONL output."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


class OpencodeJSONLParser(BaseParser):
    """Parse stdout emitted by ``opencode run --format json``.

    Opencode streams line-delimited JSON events. The parser collects assistant
    ``text`` events into the response content and retains other event types
    (``step_start``, ``tool_use``, summary events, …) under
    ``metadata['events']`` for observability. Token usage and the model
    identifier are extracted best-effort from any event that exposes them;
    missing fields produce no metadata key rather than raising.
    """

    name = "opencode_jsonl"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        events: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        usage: dict[str, Any] | None = None
        model_used: str | None = None
        errors: list[str] = []

        for line in lines:
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            events.append(event)
            event_type = event.get("type")

            if event_type == "text":
                text = self._extract_text(event)
                if text:
                    text_chunks.append(text)
            elif event_type == "error":
                message = event.get("message") or event.get("error")
                if isinstance(message, str) and message.strip():
                    errors.append(message.strip())

            extracted_usage = self._extract_usage(event)
            if extracted_usage is not None:
                usage = extracted_usage

            extracted_model = self._extract_model(event)
            if extracted_model:
                model_used = extracted_model

        if not events:
            raise ParserError("Opencode CLI JSONL output did not contain any parseable JSON events")

        if not text_chunks and errors:
            text_chunks.extend(errors)

        if not text_chunks:
            raise ParserError("Opencode CLI JSONL output did not include any assistant text events")

        content = "\n\n".join(text_chunks).strip()
        metadata: dict[str, Any] = {"events": events}
        if usage is not None:
            metadata["usage"] = usage
        if model_used:
            metadata["model_used"] = model_used
        if errors:
            metadata["errors"] = errors
        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()

        return ParsedCLIResponse(content=content, metadata=metadata)

    @staticmethod
    def _extract_text(event: dict[str, Any]) -> str | None:
        # Opencode wraps text content under event['part']['text']; older shapes
        # (and any future flattening) may put it directly on the event.
        candidates: list[Any] = [event.get("part"), event]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ("text", "content", "delta"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value
                if isinstance(value, dict):
                    nested = value.get("text") or value.get("content")
                    if isinstance(nested, str) and nested.strip():
                        return nested
        return None

    @staticmethod
    def _extract_usage(event: dict[str, Any]) -> dict[str, Any] | None:
        # Opencode emits tokens (and cost) on the nested 'part' of step_finish
        # events; check the wrapper first, then the event root.
        for source in (event.get("part"), event):
            if not isinstance(source, dict):
                continue
            for key in ("usage", "tokens", "token_usage"):
                value = source.get(key)
                if isinstance(value, dict) and value:
                    return value
        return None

    @staticmethod
    def _extract_model(event: dict[str, Any]) -> str | None:
        for source in (event.get("part"), event):
            if not isinstance(source, dict):
                continue
            for key in ("model_used", "model", "modelID"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    nested = value.get("id") or value.get("name")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
        return None
