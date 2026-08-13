"""
Logging configuration for the Unison MCP Server.

Encapsulates all logging setup: LocalTimeFormatter, rotating file handlers,
MCP activity logger, and log directory creation.
"""

import logging
import os
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

# (pattern, replacement) pairs used to scrub credential-shaped substrings from
# log records before they are written. Defense-in-depth: individual call sites
# should already avoid logging secrets, but LOG_LEVEL can be DEBUG and reviewed
# code/model output may contain credentials.
_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "***REDACTED***"),  # OpenAI-style
    (re.compile(r"xai-[A-Za-z0-9]{16,}"), "***REDACTED***"),  # xAI
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "***REDACTED***"),  # Google
    (re.compile(r"AKIA[0-9A-Z]{16}"), "***REDACTED***"),  # AWS access key id
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "***REDACTED***"),  # GitHub
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE), "Bearer ***REDACTED***"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "-----BEGIN PRIVATE KEY (REDACTED)-----"),
    (
        re.compile(r"((?:api[_-]?key|token|secret|password)\s*[=:]\s*)([^\s'\"]{8,})", re.IGNORECASE),
        r"\1***REDACTED***",
    ),
]


def redact_text(text: str) -> str:
    """Apply the credential-redaction patterns to *text*.

    The single public redaction surface shared by the logging filter, both
    log formatters (text-mode exception output leaks without it — the
    standard formatter appends ``exc_info`` after filters run), the JSON
    formatter's per-field pass, and telemetry export. One helper, one
    pattern set: pipelines cannot drift.
    """
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class RedactingFilter(logging.Filter):
    """Redact credential-shaped substrings from log messages (CWE-532)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact_text(message)
        if redacted != message:
            # Collapse args into the already-redacted message so re-formatting
            # cannot reintroduce the secret.
            record.msg = redacted
            record.args = ()
        return True


class LocalTimeFormatter(logging.Formatter):
    def formatException(self, ei) -> str:
        # The RedactingFilter mutates record.msg only; formatted exc_info is
        # appended AFTER filters run, so credential-shaped text inside an
        # exception would otherwise reach the log unredacted (in text mode
        # today, and in any future formatter that inherits this).
        return redact_text(super().formatException(ei))

    def formatTime(self, record, datefmt=None):
        """Override to use local timezone instead of UTC"""
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
            s = f"{t},{record.msecs:03.0f}"
        return s


def configure_logging(log_level: str = None) -> tuple[logging.Logger, logging.Logger]:
    """
    Configure all logging for the MCP server.

    Sets up stderr handler, rotating file handlers for server and activity logs,
    and returns logger references.

    Args:
        log_level: Log level string (DEBUG, INFO, WARNING, ERROR).
                   Defaults to LOG_LEVEL env var or INFO.

    Returns:
        Tuple of (server_logger, mcp_activity_logger)
    """
    from utils.env import get_env

    if log_level is None:
        log_level = (get_env("LOG_LEVEL", "INFO") or "INFO").upper()

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    redacting_filter = RedactingFilter()

    # Clear any existing handlers first
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create and configure stderr handler explicitly
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(getattr(logging, log_level, logging.INFO))
    stderr_handler.setFormatter(LocalTimeFormatter(log_format))
    stderr_handler.addFilter(redacting_filter)
    root_logger.addHandler(stderr_handler)

    # Set root logger level
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Add rotating file handler for local log monitoring
    try:
        # Create logs directory in project root
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)

        # Main server log with size-based rotation (20MB max per file)
        file_handler = RotatingFileHandler(
            log_dir / "mcp_server.log",
            maxBytes=20 * 1024 * 1024,  # 20MB max file size
            backupCount=5,  # Keep 5 rotated files (100MB total)
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, log_level, logging.INFO))
        file_handler.setFormatter(LocalTimeFormatter(log_format))
        file_handler.addFilter(redacting_filter)
        logging.getLogger().addHandler(file_handler)

        # Create a special logger for MCP activity tracking with size-based rotation
        mcp_activity_logger = logging.getLogger("mcp_activity")
        mcp_file_handler = RotatingFileHandler(
            log_dir / "mcp_activity.log",
            maxBytes=10 * 1024 * 1024,  # 10MB max file size
            backupCount=2,  # Keep 2 rotated files (20MB total)
            encoding="utf-8",
        )
        mcp_file_handler.setLevel(logging.INFO)
        # Dual-mode: UNISON_JSON_LOGS=true switches the ACTIVITY log to the
        # JSON formatter; the server log always stays text. The RedactingFilter
        # stays attached in both modes — swapping a formatter must never drop
        # or reorder handler filters.
        json_logs = os.getenv("UNISON_JSON_LOGS", "").strip().lower() == "true"
        if json_logs:
            from utils.json_log_formatter import JsonLogFormatter

            mcp_file_handler.setFormatter(JsonLogFormatter())
        else:
            mcp_file_handler.setFormatter(LocalTimeFormatter("%(asctime)s - %(message)s"))
        mcp_file_handler.addFilter(redacting_filter)
        mcp_activity_logger.addHandler(mcp_file_handler)
        mcp_activity_logger.setLevel(logging.INFO)
        # Ensure MCP activity also goes to stderr
        mcp_activity_logger.propagate = True

        # Log setup info
        logging.info(f"Logging to: {log_dir / 'mcp_server.log'}")
        logging.info(f"Process PID: {os.getpid()}")

    except Exception as e:
        print(f"Warning: Could not set up file logging: {e}", file=sys.stderr)
        mcp_activity_logger = logging.getLogger("mcp_activity")

    server_logger = logging.getLogger("server")
    return server_logger, mcp_activity_logger
