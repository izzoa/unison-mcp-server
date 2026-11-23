"""Durable store for batch operations that outlive the tool call that started them.

A batch (see ``providers/shared/batch.py``) is processed by a provider over minutes to
hours, so the server must be able to find it again after the originating call returned and
even after the server process restarted. This module owns that record.

Why a store of its own (and not the conversation storage backend)
-----------------------------------------------------------------
``STORAGE_BACKEND`` defaults to ``memory``; gating batch on it would make the feature fail
out of the box. ``SQLiteStorageBackend`` resolves its path from ``STORAGE_SQLITE_PATH`` when
none is passed, which would silently place batch records in the conversation database. The
batch store therefore lives at its **own** path, ``<cwd>/.unison/batches.db`` (a sibling of,
and distinct from, ``conversations.db``), overridable only through ``BATCH_STORE_PATH``.
``STORAGE_BACKEND`` and ``STORAGE_SQLITE_PATH`` are never consulted. The default is
resolved against the working directory when the store is opened, not at import time, so a
server that changes directory at launch does not aim the store at ``/``.

What is persisted
-----------------
* one ``batches`` row per logical operation, keyed by a server-generated, unguessable
  ``batch_handle`` (the only identifier a caller ever sees): originating tool, continuation
  id, a resumable JSON payload, timestamps and an overall :class:`RecordState`;
* one ``batch_children`` row per provider batch beneath it: provider, provider batch id,
  the model spelling the caller requested, the model actually submitted, status, the
  custom_ids submitted (the adapter forgets them across a restart), the intent window used
  to reconcile an unrecorded submission, a claimed flag, and ``consumed`` outputs.

Outputs a provider still holds are **not** copied in: nothing here stores a fetch result
unless a caller explicitly hands it to :meth:`BatchStore.record_consumed` or folds it into
the payload via :meth:`BatchStore.update_payload` (outputs an operation has consumed into
its own state and could not fetch again in that form).

Retention and protection
------------------------
The store holds prompt and response content, so it is protected on the same terms as the
conversation store (parent directory ``0700``, database and ``-wal``/``-shm`` files
``0600``, a ``.gitignore`` of ``*``) and records are purged ``BATCH_RETENTION_DAYS`` (default
30, matching OpenRouter's result retention) after creation, terminal or not. A purge nulls
the payloads, drops the child rows and leaves a tombstone (handle, creation time and a non-null ``expired_at``) so an expired
handle is distinguishable from one never seen. The tombstone marker is ``expired_at IS NOT
NULL``, *not* the ``expired`` state, which a live record may legitimately carry when a
provider's own window lapsed unprocessed. Retention is enforced for the life of the process:
every read and write purges what has aged out first, so a long-running server cannot keep
content past retention until its next restart.
``secure_delete`` is on so purged text does not linger in free pages.
"""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from providers.shared.batch import BatchState
from utils.env import get_env

logger = logging.getLogger(__name__)

__all__ = [
    "BatchChild",
    "BatchRecord",
    "BatchStore",
    "BatchStoreError",
    "BatchStoreUnavailable",
    "ChildIntent",
    "ChildStateChanged",
    "ChildState",
    "HandleLookup",
    "LookupStatus",
    "ProviderBatchAlreadyClaimed",
    "RecordState",
    "batch_retention_days",
    "default_batch_store_path",
    "derive_record_state",
    "get_batch_store",
    "reset_batch_store",
]

HANDLE_PREFIX = "batch_"
_HANDLE_ENTROPY_BYTES = 32  # 256 bits, comfortably above the 128-bit floor
DEFAULT_RETENTION_DAYS = 30.0
_SCHEMA_VERSION = 1


class BatchStoreError(RuntimeError):
    """Base class for batch store failures."""


class BatchStoreUnavailable(BatchStoreError):
    """The durable batch store could not be opened (unwritable path, disk error)."""


class ProviderBatchAlreadyClaimed(BatchStoreError):
    """A provider batch id is already recorded under another child row."""


class ChildStateChanged(BatchStoreError):
    """A compare-and-set on a child's status lost: another writer resolved it first."""


class RecordState(str, Enum):
    """Overall state of one logical batch operation."""

    SUBMITTING = "submitting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    INDETERMINATE = "indeterminate"
    ORPHANED = "orphaned"

    @property
    def is_terminal(self) -> bool:
        """True when no further progress is expected (``orphaned``/``indeterminate`` are not)."""

        return self in (RecordState.COMPLETED, RecordState.FAILED, RecordState.CANCELLED, RecordState.EXPIRED)

    @property
    def needs_attention(self) -> bool:
        """True when an operator has to look at the record."""

        return self in (RecordState.INDETERMINATE, RecordState.ORPHANED)


class ChildState(str, Enum):
    """State of one provider batch beneath a record.

    ``SUBMITTING`` means intent is recorded but no provider id is (the unresolved case
    reconciliation works on). ``VALIDATING`` .. ``EXPIRED`` mirror the provider's own
    :class:`~providers.shared.batch.BatchState`. ``CANCELLED`` is also used for a local
    cancellation. ``INDETERMINATE`` and ``ORPHANED`` are operator-facing.
    """

    SUBMITTING = "submitting"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    INDETERMINATE = "indeterminate"
    ORPHANED = "orphaned"

    @property
    def is_terminal(self) -> bool:
        return self in (ChildState.COMPLETED, ChildState.FAILED, ChildState.EXPIRED, ChildState.CANCELLED)

    @property
    def is_unresolved(self) -> bool:
        """True while it is not known whether the provider holds a batch for this child."""

        return self in (ChildState.SUBMITTING, ChildState.INDETERMINATE)

    @property
    def is_running(self) -> bool:
        return self in (ChildState.VALIDATING, ChildState.IN_PROGRESS, ChildState.FINALIZING, ChildState.CANCELLING)

    @classmethod
    def from_batch_state(cls, state: BatchState) -> ChildState:
        return cls(state.value)


class LookupStatus(str, Enum):
    """Outcome of resolving a handle."""

    FOUND = "found"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChildIntent:
    """What a child submission is about to do; persisted before the provider is contacted.

    ``submitted_model`` is the identifier the adapter will put in the provider envelope
    (after alias and suffix mapping); ``requested_model`` is the caller's own spelling.
    """

    provider: str
    requested_model: str
    submitted_model: str
    endpoint: str
    request_total: int
    custom_ids: tuple[str, ...]
    #: The compact resolution the submission was made under (limits, parameters, the unclassified-newer note);
    #: ``None`` for a provider that resolves nothing. Written with the child, read back and never re-derived.
    resolution: dict[str, Any] | None = None


@dataclass(frozen=True)
class BatchChild:
    """One provider batch (or a not-yet-acknowledged submission) under a record."""

    handle: str
    index: int
    provider: str
    provider_batch_id: str | None
    requested_model: str
    submitted_model: str
    endpoint: str
    request_total: int
    status: ChildState
    custom_ids: tuple[str, ...]
    intent_window_start: float
    intent_window_end: float | None
    claimed: bool
    consumed: dict[str, Any] = field(default_factory=dict)
    detail: str | None = None
    #: What the submission resolved the model to (see :attr:`ChildIntent.resolution`); ``None`` when nothing was
    #: recorded (a row from an earlier build, or a provider with no resolution): "nothing recorded".
    resolution: dict[str, Any] | None = None


@dataclass(frozen=True)
class BatchRecord:
    """A logical batch operation and every provider batch beneath it."""

    handle: str
    tool_name: str
    continuation_id: str | None
    payload: Any
    state: RecordState
    state_detail: str | None
    unwound: bool
    created_at: float
    updated_at: float
    children: tuple[BatchChild, ...] = ()


@dataclass(frozen=True)
class HandleLookup:
    """Result of :meth:`BatchStore.lookup`.

    ``FOUND`` carries ``record``; ``EXPIRED`` (a tombstone survives, payloads purged)
    carries ``expired_at``; ``UNKNOWN`` means no row ever existed under that handle.
    """

    status: LookupStatus
    record: BatchRecord | None = None
    expired_at: float | None = None


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def default_batch_store_path() -> str:
    """Resolve the batch store location.

    ``BATCH_STORE_PATH`` when set, else ``<cwd>/.unison/batches.db``. Deliberately blind to
    ``STORAGE_BACKEND`` and ``STORAGE_SQLITE_PATH``.
    """

    override = (get_env("BATCH_STORE_PATH") or "").strip()
    if override:
        # ``~`` is not expanded by the process environment; without this it would create a literal "~" directory.
        return os.path.expanduser(override)
    return os.path.join(os.getcwd(), ".unison", "batches.db")


def batch_retention_days() -> float:
    """``BATCH_RETENTION_DAYS`` (default 30); invalid, non-positive or non-finite values use the default."""

    raw = (get_env("BATCH_RETENTION_DAYS") or "").strip()
    if not raw:
        return DEFAULT_RETENTION_DAYS
    try:
        value = float(raw)
    except ValueError:
        value = 0.0
    # ``nan`` and ``inf`` parse as floats but would silently disable retention (nothing is ever older than
    # an infinite or undefined cutoff): treat them like any other invalid value.
    if not math.isfinite(value) or value <= 0:
        logger.warning("Ignoring invalid BATCH_RETENTION_DAYS=%r; using %s days", raw, DEFAULT_RETENTION_DAYS)
        return DEFAULT_RETENTION_DAYS
    return value


def derive_record_state(children: Sequence[BatchChild], *, unwound: bool = False) -> RecordState:
    """Roll child states up into one record state.

    Precedence: an unacknowledged submission keeps the record ``submitting``; then
    ``indeterminate`` and ``orphaned`` (operator attention); an unwound (abandoned)
    operation is ``failed``; any running child makes it ``running``. When every child is
    terminal the record is ``completed`` if at least one child completed (callers inspect
    children for partial failure), ``cancelled`` if all were cancelled, ``expired`` if all
    expired unprocessed at the provider, else ``failed``. (``expired`` here is the
    provider-window outcome of a *live* record; a purged tombstone is distinguished by its
    ``expired_at`` stamp, never by this state.)
    """

    states = [child.status for child in children]
    if not states:
        return RecordState.FAILED
    if any(state is ChildState.SUBMITTING for state in states):
        return RecordState.SUBMITTING
    if any(state is ChildState.INDETERMINATE for state in states):
        return RecordState.INDETERMINATE
    if any(state is ChildState.ORPHANED for state in states):
        return RecordState.ORPHANED
    if unwound:
        return RecordState.FAILED
    if any(state.is_running for state in states):
        return RecordState.RUNNING
    if any(state is ChildState.COMPLETED for state in states):
        return RecordState.COMPLETED
    if all(state is ChildState.CANCELLED for state in states):
        return RecordState.CANCELLED
    if all(state is ChildState.EXPIRED for state in states):
        return RecordState.EXPIRED
    return RecordState.FAILED


# ----------------------------------------------------------------------
# The store
# ----------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    handle          TEXT PRIMARY KEY,
    tool_name       TEXT,
    continuation_id TEXT,
    payload         TEXT,
    state           TEXT NOT NULL,
    state_detail    TEXT,
    unwound         INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    expired_at      REAL
);
CREATE TABLE IF NOT EXISTS batch_children (
    handle              TEXT NOT NULL REFERENCES batches(handle) ON DELETE CASCADE,
    idx                 INTEGER NOT NULL,
    provider            TEXT NOT NULL,
    provider_batch_id   TEXT,
    requested_model     TEXT NOT NULL,
    submitted_model     TEXT NOT NULL,
    endpoint            TEXT NOT NULL,
    request_total       INTEGER NOT NULL,
    status              TEXT NOT NULL,
    custom_ids          TEXT NOT NULL,
    intent_window_start REAL NOT NULL,
    intent_window_end   REAL,
    claimed             INTEGER NOT NULL DEFAULT 0,
    consumed            TEXT,
    detail              TEXT,
    resolution          TEXT,
    PRIMARY KEY (handle, idx)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_batch_children_claim
    ON batch_children (provider, provider_batch_id) WHERE provider_batch_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_batches_state ON batches (state);
CREATE INDEX IF NOT EXISTS idx_batches_created ON batches (created_at);
"""

_CHILD_COLUMNS = (
    "handle, idx, provider, provider_batch_id, requested_model, submitted_model, endpoint, request_total, "
    "status, custom_ids, intent_window_start, intent_window_end, claimed, consumed, detail, resolution"
)
_RECORD_COLUMNS = (
    "handle, tool_name, continuation_id, payload, state, state_detail, unwound, created_at, updated_at, expired_at"
)


def _is_volatile_path(db_path: str) -> bool:
    """True for SQLite spellings that name a temporary or in-memory database."""

    cleaned = db_path.strip().lower()
    return cleaned in ("", ":memory:") or cleaned.startswith("file::memory:") or "mode=memory" in cleaned


def _refuse_foreign_database(db_path: str) -> None:
    """Refuse to adopt an existing file that is not a batch store.

    A mistaken ``BATCH_STORE_PATH`` pointing at some other SQLite database (or any other file) must
    not get batch tables added to it, nor its directory re-permissioned. An absent or empty file is
    fine (a new store), as is a database that already has the ``batches`` table.
    """

    try:
        if not os.path.isfile(db_path) or os.path.getsize(db_path) == 0:
            return
    except OSError:
        return
    try:
        probe = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
        try:
            tables = {row[0] for row in probe.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        finally:
            probe.close()
    except sqlite3.Error as exc:
        raise BatchStoreUnavailable(
            f"Refusing to use {db_path!r} as the batch store: it exists but is not a readable SQLite database ({exc}). "
            "Point BATCH_STORE_PATH at a new file."
        ) from exc
    if tables and "batches" not in tables:
        raise BatchStoreUnavailable(
            f"Refusing to use {db_path!r} as the batch store: it is an existing SQLite database that is not a batch "
            "store, and batch tables would be added to it. Point BATCH_STORE_PATH at a new file."
        )


def _protect_location(db_path: str) -> None:
    """Create the parent directory ``0700`` with a ``.gitignore`` of ``*``.

    Same terms as ``SQLiteStorageBackend``: the file holds plaintext prompts and responses
    and may land inside a project tree.
    """

    parent = Path(db_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except OSError:  # pragma: no cover - platform/permission dependent
        pass
    try:
        gitignore = parent / ".gitignore"
        # lexists: a dangling symlink here must not make us write through it to its target.
        if not os.path.lexists(gitignore):
            gitignore.write_text("*\n", encoding="utf-8")
    except OSError:  # pragma: no cover - best effort
        pass


def _protect_files(db_path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.chmod(f"{db_path}{suffix}", 0o600)
        except OSError:  # sidecars may not exist yet
            pass


class BatchStore:
    """SQLite-backed store of batch records. Thread-safe; one instance per database file.

    Every mutation runs inside a ``BEGIN IMMEDIATE`` transaction behind a process-local
    lock, so concurrent server processes sharing the file serialise on SQLite's write lock.
    """

    def __init__(
        self,
        db_path: str | None = None,
        *,
        retention_days: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db_path = db_path or default_batch_store_path()
        if _is_volatile_path(self._db_path):
            raise BatchStoreUnavailable(
                f"Refusing to open the batch store at {self._db_path!r}: an in-memory database does not survive "
                "a restart, so it can never be a durable batch store."
            )
        self._retention_days = retention_days if retention_days is not None else batch_retention_days()
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._file_backed = False
        self._checkpoint_pending = False

        try:
            _refuse_foreign_database(self._db_path)
            _protect_location(self._db_path)
            # isolation_level=None: autocommit; transactions are explicit (see _tx).
            self._connection = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA secure_delete=ON")
            # Ask SQLite itself what it opened rather than trusting the configured path.
            attached = self._connection.execute("PRAGMA database_list").fetchall()
            self._file_backed = any(row[1] == "main" and bool(row[2]) for row in attached)
            if not self._file_backed:
                self._connection.close()
                raise BatchStoreUnavailable(
                    f"The batch store at {self._db_path!r} opened as a volatile database; refusing to use it."
                )
            _protect_files(self._db_path)
            self._init_schema()
            _protect_files(self._db_path)
        except BatchStoreUnavailable:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise BatchStoreUnavailable(f"Could not open the batch store at {self._db_path}: {exc}") from exc

        try:
            purged = self.purge_expired()
            if purged:
                logger.info("Batch store purged %d record(s) past %s-day retention", purged, self._retention_days)
        except Exception:  # pragma: no cover - a failed purge must not block opening
            logger.exception("Batch store retention purge failed on open")

    # -- lifecycle ------------------------------------------------------

    @property
    def db_path(self) -> str:
        return self._db_path

    @property
    def retention_days(self) -> float:
        return self._retention_days

    @property
    def is_durable(self) -> bool:
        """True only while the connection is live *and* SQLite reports a real file behind it."""

        return self._file_backed and not self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._connection.close()
            except sqlite3.Error:  # pragma: no cover - defensive
                pass

    def _init_schema(self) -> None:
        with self._lock:
            self._connection.executescript(_SCHEMA)
            # A store file created before the ``resolution`` column existed keeps working: the column is added
            # (NULL for its rows, which read as "nothing recorded"). CREATE TABLE IF NOT EXISTS leaves it alone.
            columns = {row[1] for row in self._connection.execute("PRAGMA table_info(batch_children)").fetchall()}
            if "resolution" not in columns:
                try:
                    self._connection.execute("ALTER TABLE batch_children ADD COLUMN resolution TEXT")
                except sqlite3.OperationalError:
                    # Another process opening the same old file added it between our check and our ALTER (the
                    # check is not atomic with it). That is the outcome we wanted; anything else is a real error.
                    columns = {
                        row[1] for row in self._connection.execute("PRAGMA table_info(batch_children)").fetchall()
                    }
                    if "resolution" not in columns:
                        raise
            row = self._connection.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
            if row is None:
                self._connection.execute("INSERT INTO schema_version (id, version) VALUES (1, ?)", (_SCHEMA_VERSION,))

    @contextmanager
    def _tx(self, *, purge: bool = True) -> Iterator[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` transaction; ``sqlite3`` failures surface as :class:`BatchStoreError`.

        Records past retention are purged at the top of every transaction (``purge=True``),
        so no write ever builds on content that should already be gone. An
        ``IntegrityError`` passes through untouched for callers that map it themselves.
        """

        with self._lock:
            if self._closed:
                raise BatchStoreError("The batch store is closed")
            try:
                self._connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise BatchStoreError(f"The batch store could not start a transaction: {exc}") from exc
            try:
                if purge:
                    self._purge_in_tx(self._connection, self._clock())
                yield self._connection
            except BaseException as exc:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover - the transaction is already gone
                    pass
                if isinstance(exc, sqlite3.Error) and not isinstance(exc, sqlite3.IntegrityError):
                    raise BatchStoreError(f"The batch store failed: {exc}") from exc
                raise
            try:
                self._connection.execute("COMMIT")
            except sqlite3.Error as exc:
                raise BatchStoreError(f"The batch store could not commit: {exc}") from exc
            self._checkpoint_if_pending()

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        """Locked, retention-enforcing read access; ``sqlite3`` failures become :class:`BatchStoreError`."""

        with self._lock:
            if self._closed:
                raise BatchStoreError("The batch store is closed")
            try:
                self._purge_due()
                yield self._connection
            except sqlite3.Error as exc:
                raise BatchStoreError(f"The batch store failed: {exc}") from exc

    # -- row mapping ----------------------------------------------------

    @staticmethod
    def _decode_resolution(raw: Any) -> dict[str, Any] | None:
        """A stored resolution as a dict, or ``None`` for NULL or anything unreadable (nothing recorded)."""

        if not isinstance(raw, str) or not raw:
            return None
        try:
            value = json.loads(raw)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _child_from_row(row: Sequence[Any]) -> BatchChild:
        consumed_raw = row[13]
        return BatchChild(
            handle=row[0],
            index=row[1],
            provider=row[2],
            provider_batch_id=row[3],
            requested_model=row[4],
            submitted_model=row[5],
            endpoint=row[6],
            request_total=row[7],
            status=ChildState(row[8]),
            custom_ids=tuple(json.loads(row[9])),
            intent_window_start=row[10],
            intent_window_end=row[11],
            claimed=bool(row[12]),
            consumed=json.loads(consumed_raw) if consumed_raw else {},
            detail=row[14],
            resolution=BatchStore._decode_resolution(row[15]),
        )

    def _load_children(self, conn: sqlite3.Connection, handle: str) -> tuple[BatchChild, ...]:
        rows = conn.execute(
            f"SELECT {_CHILD_COLUMNS} FROM batch_children WHERE handle = ? ORDER BY idx", (handle,)
        ).fetchall()
        return tuple(self._child_from_row(row) for row in rows)

    def _record_from_row(self, conn: sqlite3.Connection, row: Sequence[Any]) -> BatchRecord:
        payload_raw = row[3]
        return BatchRecord(
            handle=row[0],
            tool_name=row[1] or "",
            continuation_id=row[2],
            payload=json.loads(payload_raw) if payload_raw is not None else None,
            state=RecordState(row[4]),
            state_detail=row[5],
            unwound=bool(row[6]),
            created_at=row[7],
            updated_at=row[8],
            children=self._load_children(conn, row[0]),
        )

    # -- creation -------------------------------------------------------

    def create_record(
        self,
        tool_name: str,
        children: Sequence[ChildIntent],
        *,
        continuation_id: str | None = None,
        payload: Any = None,
    ) -> str:
        """Persist a new record with one ``submitting`` child per intent; return its handle.

        The handle is generated here (``batch_`` + 256 random bits, URL-safe) and is the
        only identifier a caller ever receives, however many provider batches are beneath it.
        """

        if not children:
            raise ValueError("A batch record needs at least one child")
        now = self._clock()
        payload_json = json.dumps(payload) if payload is not None else None
        with self._tx() as conn:
            handle = ""
            for _ in range(5):
                candidate = HANDLE_PREFIX + secrets.token_urlsafe(_HANDLE_ENTROPY_BYTES)
                exists = conn.execute("SELECT 1 FROM batches WHERE handle = ?", (candidate,)).fetchone()
                if exists is None:
                    handle = candidate
                    break
            if not handle:  # pragma: no cover - 5 collisions of 256-bit values
                raise BatchStoreError("Could not generate a unique batch handle")
            conn.execute(
                "INSERT INTO batches (handle, tool_name, continuation_id, payload, state, unwound, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                (handle, tool_name, continuation_id, payload_json, RecordState.SUBMITTING.value, now, now),
            )
            for index, intent in enumerate(children):
                conn.execute(
                    "INSERT INTO batch_children (handle, idx, provider, provider_batch_id, requested_model, "
                    "submitted_model, endpoint, request_total, status, custom_ids, intent_window_start, claimed, "
                    "resolution) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (
                        handle,
                        index,
                        intent.provider,
                        intent.requested_model,
                        intent.submitted_model,
                        intent.endpoint,
                        intent.request_total,
                        ChildState.SUBMITTING.value,
                        json.dumps(list(intent.custom_ids)),
                        now,
                        json.dumps(intent.resolution, sort_keys=True) if intent.resolution is not None else None,
                    ),
                )
        return handle

    def delete_record(self, handle: str) -> bool:
        """Remove a record outright (no tombstone). For a submission that never produced a handle."""

        with self._tx() as conn:
            conn.execute("DELETE FROM batch_children WHERE handle = ?", (handle,))
            cursor = conn.execute("DELETE FROM batches WHERE handle = ?", (handle,))
            # The deleted prompts are still in the write-ahead log until it is checkpointed.
            self._checkpoint_pending = True
            return cursor.rowcount > 0

    # -- reads ----------------------------------------------------------

    def lookup(self, handle: str) -> HandleLookup:
        """Resolve a handle to FOUND (with the record), EXPIRED (tombstone) or UNKNOWN."""

        with self._read() as conn:
            row = conn.execute(f"SELECT {_RECORD_COLUMNS} FROM batches WHERE handle = ?", (handle,)).fetchone()
            if row is None:
                return HandleLookup(LookupStatus.UNKNOWN)
            if row[9] is not None:  # tombstone: purged for retention
                return HandleLookup(LookupStatus.EXPIRED, expired_at=row[9])
            return HandleLookup(LookupStatus.FOUND, record=self._record_from_row(conn, row))

    def get(self, handle: str) -> BatchRecord | None:
        """The live record, or ``None`` for an expired or unknown handle."""

        return self.lookup(handle).record

    def list_records(self, states: Sequence[RecordState] | None = None) -> list[BatchRecord]:
        """Live records (tombstones excluded), oldest first, optionally filtered by state."""

        with self._read() as conn:
            query = f"SELECT {_RECORD_COLUMNS} FROM batches WHERE expired_at IS NULL"
            params: list[Any] = []
            if states:
                query += f" AND state IN ({','.join('?' for _ in states)})"
                params.extend(state.value for state in states)
            query += " ORDER BY created_at, handle"
            rows = conn.execute(query, params).fetchall()
            return [self._record_from_row(conn, row) for row in rows]

    def needs_attention(self) -> list[BatchRecord]:
        """Records an operator has to inspect: ``indeterminate`` or ``orphaned``.

        Includes any record with an indeterminate or orphaned child, whatever the record's
        own rolled-up state. Never discards anything.
        """

        flagged = []
        for record in self.list_records():
            if record.state.needs_attention or any(
                child.status in (ChildState.INDETERMINATE, ChildState.ORPHANED) for child in record.children
            ):
                flagged.append(record)
        return flagged

    def unresolved_intents(self, *, min_age_seconds: float = 0.0) -> list[BatchRecord]:
        """Records with a child still ``submitting`` (no provider id recorded) older than the cutoff.

        ``min_age_seconds`` keeps reconciliation away from a submission another process is
        legitimately performing right now.
        """

        cutoff = self._clock() - min_age_seconds
        out = []
        for record in self.list_records():
            if any(
                child.status is ChildState.SUBMITTING and child.intent_window_start <= cutoff
                for child in record.children
            ):
                out.append(record)
        return out

    def find_claim(self, provider: str, provider_batch_id: str) -> tuple[str, int] | None:
        """``(handle, child index)`` already holding this provider batch id, if any."""

        with self._read() as conn:
            row = conn.execute(
                "SELECT handle, idx FROM batch_children WHERE provider = ? AND provider_batch_id = ?",
                (provider, provider_batch_id),
            ).fetchone()
        return (row[0], row[1]) if row else None

    # -- child mutations ------------------------------------------------

    def record_provider_batch(
        self,
        handle: str,
        index: int,
        provider_batch_id: str,
        *,
        status: ChildState = ChildState.VALIDATING,
        intent_window_end: float | None = None,
        detail: str | None = None,
        expect_status: ChildState | None = None,
    ) -> None:
        """Record the provider's id for a child and mark it claimed.

        The record's rolled-up state is refreshed in the same transaction, so a crash can
        never leave a record ``submitting`` beside a child that already has its id.

        Args:
            expect_status: compare-and-set. Only write if the child still has this status
                (a reconciler passes ``SUBMITTING`` so it cannot clobber a concurrent
                resolution).

        Raises:
            ProviderBatchAlreadyClaimed: another child already holds this provider batch id.
            ChildStateChanged: ``expect_status`` no longer holds.
            BatchStoreError: no such child.
        """

        now = self._clock()
        sql = (
            "UPDATE batch_children SET provider_batch_id = ?, claimed = 1, status = ?, "
            "intent_window_end = COALESCE(?, intent_window_end, ?), detail = ? WHERE handle = ? AND idx = ?"
        )
        params: list[Any] = [provider_batch_id, status.value, intent_window_end, now, detail, handle, index]
        if expect_status is not None:
            sql += " AND status = ?"
            params.append(expect_status.value)
        try:
            with self._tx() as conn:
                cursor = conn.execute(sql, params)
                if cursor.rowcount == 0:
                    self._raise_missing_or_changed(conn, handle, index, expect_status)
                conn.execute("UPDATE batches SET updated_at = ? WHERE handle = ?", (now, handle))
                self._recompute_in_tx(conn, handle, now)
        except sqlite3.IntegrityError as exc:
            raise ProviderBatchAlreadyClaimed(
                f"Provider batch {provider_batch_id} is already recorded under another child"
            ) from exc

    @staticmethod
    def _raise_missing_or_changed(
        conn: sqlite3.Connection, handle: str, index: int, expect_status: ChildState | None
    ) -> None:
        exists = conn.execute("SELECT 1 FROM batch_children WHERE handle = ? AND idx = ?", (handle, index)).fetchone()
        if exists is not None and expect_status is not None:
            raise ChildStateChanged(f"Child {index} under handle {handle} is no longer {expect_status.value}")
        raise BatchStoreError(f"No child {index} under handle {handle}")

    def set_child_status(
        self,
        handle: str,
        index: int,
        status: ChildState,
        *,
        detail: str | None = None,
        expect_status: ChildState | None = None,
    ) -> None:
        """Set a child's status (and optionally its detail text, which is replaced).

        Refreshes the record's rolled-up state in the same transaction. ``expect_status``
        makes it a compare-and-set (raises :class:`ChildStateChanged` when it no longer holds).
        """

        now = self._clock()
        sql = "UPDATE batch_children SET status = ?, detail = COALESCE(?, detail) WHERE handle = ? AND idx = ?"
        params: list[Any] = [status.value, detail, handle, index]
        if expect_status is not None:
            sql += " AND status = ?"
            params.append(expect_status.value)
        with self._tx() as conn:
            cursor = conn.execute(sql, params)
            if cursor.rowcount == 0:
                self._raise_missing_or_changed(conn, handle, index, expect_status)
            conn.execute("UPDATE batches SET updated_at = ? WHERE handle = ?", (now, handle))
            self._recompute_in_tx(conn, handle, now)

    def set_child_window_end(self, handle: str, index: int, window_end: float) -> None:
        """Record when the provider call for this child resolved (success or ambiguous)."""

        with self._tx() as conn:
            conn.execute(
                "UPDATE batch_children SET intent_window_end = ? WHERE handle = ? AND idx = ?",
                (window_end, handle, index),
            )

    def record_consumed(self, handle: str, index: int, key: str, value: Any, *, refetchable: bool = False) -> bool:
        """Persist an output the operation has consumed into its own state, under ``key``.

        Returns ``True`` when stored. An output the provider still holds (``refetchable``)
        is **not** stored: the record is not a mirror of provider results, and a refetch is
        free. Pass ``refetchable=False`` (the default) only for outputs that could not be
        fetched again in that form.
        """

        if refetchable:
            logger.debug("Not persisting refetchable output %r for %s child %d", key, handle, index)
            return False
        encoded = json.dumps(value)
        now = self._clock()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT consumed FROM batch_children WHERE handle = ? AND idx = ?", (handle, index)
            ).fetchone()
            if row is None:
                raise BatchStoreError(f"No child {index} under handle {handle}")
            consumed = json.loads(row[0]) if row[0] else {}
            consumed[key] = json.loads(encoded)
            conn.execute(
                "UPDATE batch_children SET consumed = ? WHERE handle = ? AND idx = ?",
                (json.dumps(consumed), handle, index),
            )
            conn.execute("UPDATE batches SET updated_at = ? WHERE handle = ?", (now, handle))
        return True

    # -- record mutations -----------------------------------------------

    def update_payload(self, handle: str, payload: Any) -> None:
        """Replace the record's resumable payload (the operation's accumulating state)."""

        now = self._clock()
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE batches SET payload = ?, updated_at = ? WHERE handle = ? AND expired_at IS NULL",
                (json.dumps(payload) if payload is not None else None, now, handle),
            )
            if cursor.rowcount == 0:
                raise BatchStoreError(f"No live record under handle {handle}")
            # The replaced payload (plaintext prompts) lingers in the write-ahead log until a checkpoint.
            self._checkpoint_pending = True

    def set_state(self, handle: str, state: RecordState, *, detail: str | None = None) -> None:
        """Set a live record's state. Any :class:`RecordState` is allowed, ``expired`` included.

        A purged tombstone cannot be modified, and a live record can never *become* a
        tombstone this way: the tombstone marker is ``expired_at``, written only by retention.
        """

        now = self._clock()
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE batches SET state = ?, state_detail = ?, updated_at = ? WHERE handle = ? "
                "AND expired_at IS NULL",
                (state.value, detail, now, handle),
            )
            if cursor.rowcount == 0:
                raise BatchStoreError(f"No live record under handle {handle}")

    def mark_unwound(self, handle: str) -> None:
        """Flag the operation as abandoned (a saga unwind); later adoption yields ``orphaned``."""

        with self._tx() as conn:
            conn.execute("UPDATE batches SET unwound = 1 WHERE handle = ? AND expired_at IS NULL", (handle,))

    def recompute_state(self, handle: str) -> RecordState | None:
        """Re-derive and persist the record state from its children; ``None`` if not live.

        A record already ``cancelled`` (locally) keeps that state.
        """

        with self._tx() as conn:
            return self._recompute_in_tx(conn, handle, self._clock())

    def _recompute_in_tx(self, conn: sqlite3.Connection, handle: str, now: float) -> RecordState | None:
        row = conn.execute(f"SELECT {_RECORD_COLUMNS} FROM batches WHERE handle = ?", (handle,)).fetchone()
        if row is None or row[9] is not None:
            return None
        if row[4] == RecordState.CANCELLED.value:
            return RecordState.CANCELLED
        children = self._load_children(conn, handle)
        state = derive_record_state(children, unwound=bool(row[6]))
        if state.value != row[4]:
            conn.execute("UPDATE batches SET state = ?, updated_at = ? WHERE handle = ?", (state.value, now, handle))
        return state

    # -- retention ------------------------------------------------------

    def _cutoff(self, now: float) -> float:
        return now - self._retention_days * 86400.0

    def _purge_due(self) -> None:
        """Purge aged-out records if any exist; a cheap read when there are none."""

        cutoff = self._cutoff(self._clock())
        due = self._connection.execute(
            "SELECT 1 FROM batches WHERE expired_at IS NULL AND created_at < ? LIMIT 1", (cutoff,)
        ).fetchone()
        if due is not None:
            self.purge_expired()

    def purge_expired(self, now: float | None = None) -> int:
        """Purge records older than the retention period; return how many were purged.

        Applies to every state, terminal or not, measured from creation. Payloads and
        consumed outputs are nulled, child rows dropped, and a tombstone (handle,
        ``created_at``, non-null ``expired_at``) remains. Runs automatically ahead of every
        read and write, so retention holds for the life of the process, not just at open.
        """

        current = self._clock() if now is None else now
        with self._tx(purge=False) as conn:
            purged = self._purge_in_tx(conn, current)
        return purged

    def _purge_in_tx(self, conn: sqlite3.Connection, current: float) -> int:
        cutoff = self._cutoff(current)
        handles = [
            row[0]
            for row in conn.execute(
                "SELECT handle FROM batches WHERE expired_at IS NULL AND created_at < ?", (cutoff,)
            ).fetchall()
        ]
        for handle in handles:
            # Retention outranks operator visibility, but never silently: an unresolved or untracked provider
            # batch may still be billed, and its id is about to be deleted. Say so in the server log.
            for child_row in conn.execute(
                "SELECT provider, provider_batch_id, status FROM batch_children WHERE handle = ? AND status IN (?, ?, ?)",
                (handle, ChildState.ORPHANED.value, ChildState.INDETERMINATE.value, ChildState.SUBMITTING.value),
            ).fetchall():
                logger.warning(
                    "Retention is purging batch record %s while its %s child is still %s (provider batch %s): "
                    "that provider batch may still be billed and is no longer tracked here",
                    handle,
                    child_row[0],
                    child_row[2],
                    child_row[1] or "<unknown>",
                )
            conn.execute("DELETE FROM batch_children WHERE handle = ?", (handle,))
            conn.execute(
                "UPDATE batches SET tool_name = NULL, continuation_id = NULL, payload = NULL, state = ?, "
                "state_detail = NULL, unwound = 0, updated_at = ?, expired_at = ? WHERE handle = ?",
                (RecordState.EXPIRED.value, current, current, handle),
            )
        if handles:
            self._checkpoint_pending = True
        return len(handles)

    def _checkpoint_if_pending(self) -> None:
        if not self._checkpoint_pending:
            return
        self._checkpoint_pending = False
        try:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:  # pragma: no cover - another connection may hold the WAL
            pass


# ----------------------------------------------------------------------
# Process-wide instance
# ----------------------------------------------------------------------

_store_instance: BatchStore | None = None
_store_lock = threading.Lock()


def get_batch_store(store: BatchStore | None = None) -> BatchStore:
    """The process-wide batch store, opened lazily at :func:`default_batch_store_path`.

    Args:
        store: install this instance as the singleton (tests, dependency injection).

    Raises:
        BatchStoreUnavailable: the store could not be opened. Batch submission must then
            refuse to proceed; it never falls back to a volatile store.
    """

    global _store_instance
    with _store_lock:
        if store is not None:
            _store_instance = store
        elif _store_instance is None or not _store_instance.is_durable:
            _store_instance = BatchStore()
        return _store_instance


def reset_batch_store() -> None:
    """Close and forget the process-wide store (test teardown)."""

    global _store_instance
    with _store_lock:
        if _store_instance is not None:
            _store_instance.close()
        _store_instance = None
