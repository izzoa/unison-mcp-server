"""Parser for opencode CLI JSONL output."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


def _sum_usage(total: dict[str, Any] | None, addend: dict[str, Any]) -> dict[str, Any]:
    """Add ``addend`` into ``total`` key by key, recursing into nested counts.

    Numbers add (``cache: {read, write}`` is summed per key); for a
    non-numeric leaf the first value seen wins.
    """
    result: dict[str, Any] = dict(total or {})
    for key, value in addend.items():
        existing = result.get(key)
        if existing is None:
            result[key] = value
        elif isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _sum_usage(existing, value)
        elif _is_number(existing) and _is_number(value):
            result[key] = existing + value
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class OpencodeJSONLParser(BaseParser):
    """Parse stdout emitted by ``opencode run --format json``.

    Opencode streams line-delimited JSON events. The parser collects assistant
    ``text`` events into the response content and retains other event types
    (``step_start``, ``tool_use``, summary events, …) under
    ``metadata['events']`` for observability. Token usage is summed across
    every step summary; the model identifier and session ID are extracted
    best-effort from any event that exposes them, and missing fields produce no
    metadata key rather than raising.

    ``error`` events come in three shapes: a flat string (``message`` or
    ``error``), opencode v2's ``{"error": {"type", "message"}}``, and v1's
    ``{"error": {"name", "data": {"message"}}}``. The nested shapes are surfaced
    as ``"<type>: <message>"`` and a flat string verbatim, under
    ``metadata['errors']``.
    """

    name = "opencode_jsonl"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        events: list[dict[str, Any]] = []
        text_chunks: list[str] = []
        usage: dict[str, Any] | None = None
        model_used: str | None = None
        session_id: str | None = None
        step_starts = 0
        step_finishes = 0
        active_messages: set[str] = set()
        finished_messages: set[str] = set()
        errors: list[str] = []

        for line in lines:
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, RecursionError):
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
                error = self._extract_error(event)
                if error:
                    errors.append(error)
            elif event_type == "step_start":
                step_starts += 1
            elif event_type == "step_finish":
                step_finishes += 1

            message_id = self._extract_message_id(event)
            if message_id:
                (finished_messages if event_type == "step_finish" else active_messages).add(message_id)

            extracted_usage = self._extract_usage(event)
            if extracted_usage is not None:
                usage = _sum_usage(usage, extracted_usage)

            extracted_model = self._extract_model(event)
            if extracted_model:
                model_used = extracted_model

            if session_id is None:
                session_id = self._extract_session_id(event)

        if not events:
            raise ParserError("Opencode CLI JSONL output did not contain any parseable JSON events")

        error_only = not text_chunks and bool(errors)
        if error_only:
            text_chunks.extend(errors)

        if not text_chunks:
            raise ParserError("Opencode CLI JSONL output did not include any assistant text events")

        content = "\n\n".join(text_chunks).strip()
        metadata: dict[str, Any] = {"events": events}
        if usage is not None:
            metadata["usage"] = usage
        # opencode drops its final step summary (upstream #59652); in the captured
        # streams the dropped step has no step_start either, so counting starts
        # against finishes misses it. A message that produced events but never
        # got a summary is the reliable signal.
        metadata["usage_complete"] = (
            usage is not None and step_starts <= step_finishes and not (active_messages - finished_messages)
        )
        if model_used:
            metadata["model_used"] = model_used
        if session_id:
            metadata["session_id"] = session_id
        if errors:
            metadata["errors"] = errors
        if error_only:
            metadata["error_only"] = True
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
    def _extract_error(event: dict[str, Any]) -> str | None:
        raw = event.get("error")
        if isinstance(raw, dict):
            # v2: {type, message}. v1: {name, data: {message}}.
            error_type = raw.get("type") or raw.get("name")
            data = raw.get("data")
            message = raw.get("message") or (data.get("message") if isinstance(data, dict) else None)
            type_text = error_type.strip() if isinstance(error_type, str) and error_type.strip() else None
            message_text = message.strip() if isinstance(message, str) and message.strip() else None
            if type_text and message_text:
                return f"{type_text}: {message_text}"
            if message_text or type_text:
                return message_text or type_text
            raw = None  # an unusable error object: fall through to the top-level message
        flat = event.get("message") or raw
        if isinstance(flat, str) and flat.strip():
            return flat.strip()
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
    def _extract_message_id(event: dict[str, Any]) -> str | None:
        part = event.get("part")
        if isinstance(part, dict):
            value = part.get("messageID")
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _extract_session_id(event: dict[str, Any]) -> str | None:
        for source in (event, event.get("part")):
            if not isinstance(source, dict):
                continue
            value = source.get("sessionID")
            if isinstance(value, str) and value.strip():
                return value.strip()
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
