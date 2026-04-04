"""
SQLite storage backend for conversation threads.

Persistent alternative to InMemoryStorage that survives server restarts.
Uses Python's built-in sqlite3 module -- zero additional dependencies.

Enable via environment variable: STORAGE_BACKEND=sqlite

Configuration:
    STORAGE_BACKEND          -- "sqlite" to enable (default: "memory")
    STORAGE_SQLITE_PATH      -- database file path (default: .unison/conversations.db
                                relative to the working directory, giving per-project isolation)
    STORAGE_SWEEP_INTERVAL_SECONDS -- expired-row cleanup interval (default: 300)
"""

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from utils.env import get_env

logger = logging.getLogger(__name__)

# Default database path: .unison/conversations.db in the working directory.
# This gives per-project isolation — conversations for each project stay
# in that project's directory tree.
_DEFAULT_DB_PATH = os.path.join(os.getcwd(), ".unison", "conversations.db")

# Current schema version -- bump when adding migrations
_CURRENT_SCHEMA_VERSION = 1


def _glob_to_like(pattern: str) -> str:
    """Convert a Redis-style glob pattern to a SQL LIKE pattern.

    Supports ``*`` (any chars) and ``?`` (single char).  Literal ``%``
    and ``_`` in the input are escaped.
    """
    out = pattern.replace("%", r"\%").replace("_", r"\_")
    out = out.replace("*", "%").replace("?", "_")
    return out


class SQLiteStorageBackend:
    """Persistent storage backend using SQLite.

    Implements the ``StorageBackend`` protocol (``get``, ``set_with_ttl``,
    ``setex``, ``shutdown``) with identical semantics to ``InMemoryStorage``,
    but data persists across server restarts.

    Additional convenience methods (``delete``, ``keys``) are provided on the
    concrete class but are **not** part of the protocol contract.

    Concurrency model:
        * WAL journal mode -- allows concurrent readers at the database level.
        * A ``threading.Lock`` serialises **all** access to the shared
          ``sqlite3.Connection``.  The Python sqlite3 ``Connection`` object
          is not safe for concurrent use from multiple threads even when
          ``check_same_thread=False`` -- that flag only disables the
          thread-identity assertion, not internal state protection.
        * ``check_same_thread=False`` allows the background sweep thread
          and the main async-loop thread to share the connection.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or get_env("STORAGE_SQLITE_PATH") or _DEFAULT_DB_PATH

        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Open connection -- allow access from any thread
        self._connection = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )

        # WAL mode for concurrent reads; busy timeout for write contention
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")

        # Serialise all writes
        self._lock = threading.Lock()

        # Schema bootstrap + migrations
        self._init_schema()
        self._run_migrations()

        # Background sweep for expired rows
        self._shutdown_event = threading.Event()
        sweep_raw = get_env("STORAGE_SWEEP_INTERVAL_SECONDS", "300") or "300"
        try:
            self._sweep_interval = max(1, int(sweep_raw))
        except ValueError:
            self._sweep_interval = 300

        self._sweep_thread = threading.Thread(target=self._sweep_worker, daemon=True)
        self._sweep_thread.start()

        logger.info(
            "SQLite storage initialised at %s (sweep every %ds)",
            self._db_path,
            self._sweep_interval,
        )

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    # Migrations list: index 0 migrates from version 1 -> 2, etc.
    # Each callable receives a sqlite3.Connection (already inside a
    # transaction) and applies one schema upgrade.
    _MIGRATIONS: list = []

    def _init_schema(self) -> None:
        """Create initial tables if they don't exist."""
        with self._lock:
            with self._connection:
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kv_store (
                        key        TEXT PRIMARY KEY,
                        value      TEXT NOT NULL,
                        expires_at REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_version (
                        id      INTEGER PRIMARY KEY CHECK (id = 1),
                        version INTEGER NOT NULL
                    )
                    """
                )
                # Seed version row if absent
                cursor = self._connection.execute("SELECT version FROM schema_version WHERE id = 1")
                if cursor.fetchone() is None:
                    self._connection.execute(
                        "INSERT INTO schema_version (id, version) VALUES (1, ?)",
                        (_CURRENT_SCHEMA_VERSION,),
                    )

    def _run_migrations(self) -> None:
        """Apply any pending sequential migrations."""
        cursor = self._connection.execute("SELECT version FROM schema_version WHERE id = 1")
        row = cursor.fetchone()
        current_version = row[0] if row else 0

        for idx, migration_fn in enumerate(self._MIGRATIONS):
            target_version = idx + 2  # migrations start at version 2
            if target_version <= current_version:
                continue
            try:
                with self._lock:
                    with self._connection:
                        migration_fn(self._connection)
                        self._connection.execute(
                            "UPDATE schema_version SET version = ? WHERE id = 1",
                            (target_version,),
                        )
                logger.info("Applied storage migration to version %d", target_version)
            except Exception:
                logger.exception("Storage migration to version %d failed — stopping", target_version)
                break

    # ------------------------------------------------------------------
    # StorageBackend protocol methods
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        """Retrieve value if it exists and has not expired.

        Expired rows are deleted on read (lazy expiry) and ``None`` is
        returned.
        """
        now = time.time()
        with self._lock:
            cursor = self._connection.execute("SELECT value, expires_at FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row is None:
                return None

            value, expires_at = row
            if expires_at is not None and expires_at <= now:
                # Lazy expiry
                with self._connection:
                    self._connection.execute("DELETE FROM kv_store WHERE key = ?", (key,))
                logger.debug("Key %s expired and removed (lazy)", key)
                return None

        return value

    def set_with_ttl(self, key: str, ttl_seconds: int, value: str) -> None:
        """Store *value* under *key* with a TTL in seconds."""
        now = time.time()
        expires_at = now + ttl_seconds
        with self._lock:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO kv_store (key, value, expires_at, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value      = excluded.value,
                        expires_at = excluded.expires_at,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, expires_at, now, now),
                )
        logger.debug("Stored key %s with TTL %ds", key, ttl_seconds)

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        """Redis-compatible alias for :meth:`set_with_ttl`."""
        self.set_with_ttl(key, ttl_seconds, value)

    def shutdown(self) -> None:
        """Stop the sweep thread and close the database connection."""
        self._shutdown_event.set()
        if self._sweep_thread.is_alive():
            self._sweep_thread.join(timeout=2)
        try:
            self._connection.close()
        except Exception:
            pass
        logger.info("SQLite storage shut down")

    # ------------------------------------------------------------------
    # Extra convenience methods (not part of StorageBackend protocol)
    # ------------------------------------------------------------------

    def delete(self, key: str) -> None:
        """Remove *key* from the store. Silent no-op if key does not exist."""
        with self._lock:
            with self._connection:
                self._connection.execute("DELETE FROM kv_store WHERE key = ?", (key,))

    def keys(self, pattern: str) -> list[str]:
        """Return all non-expired keys matching a Redis-style glob *pattern*."""
        now = time.time()
        like_pattern = _glob_to_like(pattern)
        with self._lock:
            cursor = self._connection.execute(
                """
                SELECT key FROM kv_store
                WHERE key LIKE ? ESCAPE '\\'
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (like_pattern, now),
            )
            return [row[0] for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Background TTL sweep
    # ------------------------------------------------------------------

    def _sweep_expired(self) -> None:
        """Delete all rows whose TTL has elapsed."""
        now = time.time()
        with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    "DELETE FROM kv_store WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )
                deleted = cursor.rowcount
        if deleted:
            logger.debug("Swept %d expired storage entries", deleted)

    def _sweep_worker(self) -> None:
        """Daemon loop that calls :meth:`_sweep_expired` periodically."""
        while not self._shutdown_event.wait(timeout=self._sweep_interval):
            try:
                self._sweep_expired()
            except Exception:
                logger.exception("Error during storage sweep")
