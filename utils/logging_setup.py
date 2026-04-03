"""
Logging configuration for the Unison MCP Server.

Encapsulates all logging setup: LocalTimeFormatter, rotating file handlers,
MCP activity logger, and log directory creation.
"""

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


class LocalTimeFormatter(logging.Formatter):
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
                   Defaults to LOG_LEVEL env var or DEBUG.

    Returns:
        Tuple of (server_logger, mcp_activity_logger)
    """
    from utils.env import get_env

    if log_level is None:
        log_level = (get_env("LOG_LEVEL", "DEBUG") or "DEBUG").upper()

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Clear any existing handlers first
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create and configure stderr handler explicitly
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(getattr(logging, log_level, logging.INFO))
    stderr_handler.setFormatter(LocalTimeFormatter(log_format))
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
        mcp_file_handler.setFormatter(LocalTimeFormatter("%(asctime)s - %(message)s"))
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
