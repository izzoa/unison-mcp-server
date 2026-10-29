"""Process-wide record of how the server was launched.

``server.run()`` switches into the server's own install directory when a host
launches it from a non-workspace directory (``/`` for Claude Desktop on macOS,
``System32`` on Windows). After that switch the process cwd says nothing about
the user's project, so consumers that derive a project directory from the cwd
(clink) must be able to tell that the cwd was substituted. This module holds
that one fact: pure and import-light, so any layer can read it.
"""

from __future__ import annotations

import threading
from pathlib import Path

_lock = threading.Lock()
_cwd_substitution: tuple[str, str] | None = None


def record_cwd_substitution(original: str | Path, substituted_to: str | Path) -> None:
    """Record that the server replaced its launch directory ``original`` with ``substituted_to``.

    Written once at startup, before any request is served. A later call
    replaces the earlier record.
    """
    global _cwd_substitution
    with _lock:
        _cwd_substitution = (str(original), str(substituted_to))


def cwd_substitution() -> tuple[str, str] | None:
    """Return ``(original, substituted_to)`` when the launch directory was replaced, else ``None``."""
    with _lock:
        return _cwd_substitution


def reset_launch_info_for_tests() -> None:
    """Forget any recorded substitution so state cannot leak between tests."""
    global _cwd_substitution
    with _lock:
        _cwd_substitution = None
