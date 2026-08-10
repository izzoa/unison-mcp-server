"""Structured JSON formatter for the MCP activity log.

Emits one single-line JSON object per record with a versioned schema. Every
string value — message, tool fields, ``extra`` values, exception text —
passes through the shared credential-redaction helper before serialization,
so field-level emission is not a bypass channel around the message-level
``RedactingFilter``. JSON logs are built for shipment to external aggregation
systems; an unredacted credential there is exfiltration, not a local artifact.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from utils.logging_setup import redact_text

SCHEMA_VERSION = "1.0"

#: Well-known tool-context fields lifted to top level when present as extras.
_TOOL_FIELDS = ("tool_name", "model", "thread_id", "tokens_in", "tokens_out", "latency_ms", "status")

#: LogRecord attributes that are internal plumbing, never "extras".
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


def _redact_value(value: Any) -> Any:
    """Recursively redact string content in JSON-serializable structures."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line, schema-versioned JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
            "schema_version": SCHEMA_VERSION,
        }

        extras: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in _RESERVED:
                continue
            if key in _TOOL_FIELDS:
                payload[key] = _redact_value(value)
            else:
                extras[key] = _redact_value(value)
        if extras:
            payload["extra"] = extras

        if record.exc_info and record.exc_info[0] is not None:
            exc_type, exc_value, _tb = record.exc_info
            payload["exception_type"] = f"{exc_type.__module__}.{exc_type.__qualname__}"
            payload["exception_message"] = redact_text(str(exc_value))

        return json.dumps(payload, ensure_ascii=False, default=str)
