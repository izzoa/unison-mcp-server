"""Tests for SQLiteStorageBackend — persistent conversation storage."""

import os
import threading
import time

import pytest

from utils.sqlite_storage import SQLiteStorageBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path):
    """Return a temporary database file path."""
    return str(tmp_path / "test_conversations.db")


@pytest.fixture()
def storage(db_path):
    """Create a SQLiteStorageBackend with a short sweep interval for testing."""
    os.environ.pop("STORAGE_SQLITE_PATH", None)
    os.environ.pop("STORAGE_SWEEP_INTERVAL_SECONDS", None)
    backend = SQLiteStorageBackend(db_path=db_path)
    yield backend
    backend.shutdown()


# ---------------------------------------------------------------------------
# 9.1 — CRUD tests
# ---------------------------------------------------------------------------


class TestCRUD:
    """Basic set/get/delete/keys operations."""

    def test_set_get_round_trip(self, storage):
        storage.set_with_ttl("key1", 60, "value1")
        assert storage.get("key1") == "value1"

    def test_setex_get_round_trip(self, storage):
        storage.setex("key2", 60, "value2")
        assert storage.get("key2") == "value2"

    def test_get_nonexistent_returns_none(self, storage):
        assert storage.get("nonexistent") is None

    def test_delete_removes_key(self, storage):
        storage.setex("key3", 60, "value3")
        storage.delete("key3")
        assert storage.get("key3") is None

    def test_delete_nonexistent_is_silent(self, storage):
        # Should not raise
        storage.delete("never_existed")

    def test_set_overwrites_existing(self, storage):
        storage.setex("key4", 60, "value_a")
        storage.setex("key4", 60, "value_b")
        assert storage.get("key4") == "value_b"

    def test_keys_pattern_matching(self, storage):
        storage.setex("thread:abc", 60, "v1")
        storage.setex("thread:def", 60, "v2")
        storage.setex("session:ghi", 60, "v3")

        matched = sorted(storage.keys("thread:*"))
        assert matched == ["thread:abc", "thread:def"]

    def test_keys_no_match_returns_empty(self, storage):
        storage.setex("thread:abc", 60, "v1")
        assert storage.keys("nonexistent:*") == []


# ---------------------------------------------------------------------------
# 9.2 — TTL tests
# ---------------------------------------------------------------------------


class TestTTL:
    """TTL enforcement: lazy expiry on read."""

    def test_value_accessible_before_ttl(self, storage):
        storage.setex("ttl_key", 60, "alive")
        assert storage.get("ttl_key") == "alive"

    def test_value_expired_after_ttl(self, storage):
        storage.setex("ttl_key", 1, "short_lived")
        time.sleep(1.1)
        assert storage.get("ttl_key") is None

    def test_expired_row_deleted_on_read(self, storage):
        storage.setex("expire_me", 1, "gone")
        time.sleep(1.1)
        # First get triggers lazy delete
        assert storage.get("expire_me") is None
        # Verify it's actually deleted from the DB
        cursor = storage._connection.execute("SELECT COUNT(*) FROM kv_store WHERE key = ?", ("expire_me",))
        assert cursor.fetchone()[0] == 0

    def test_setex_overwrites_with_new_ttl(self, storage):
        storage.setex("refresh", 1, "old")
        storage.setex("refresh", 60, "new")
        time.sleep(1.1)
        # Should still be alive because second setex extended TTL
        assert storage.get("refresh") == "new"

    def test_expired_keys_excluded_from_keys(self, storage):
        storage.setex("alive_key", 60, "yes")
        storage.setex("dead_key", 1, "no")
        time.sleep(1.1)
        assert storage.keys("*_key") == ["alive_key"]


# ---------------------------------------------------------------------------
# 9.3 — Sweep tests
# ---------------------------------------------------------------------------


class TestSweep:
    """Background sweep for expired rows."""

    def test_sweep_deletes_expired_rows(self, storage):
        storage.setex("sweep1", 1, "a")
        storage.setex("sweep2", 1, "b")
        storage.setex("sweep3", 60, "c")
        time.sleep(1.1)

        storage._sweep_expired()

        cursor = storage._connection.execute("SELECT COUNT(*) FROM kv_store")
        assert cursor.fetchone()[0] == 1  # only sweep3 remains

    def test_sweep_no_op_when_nothing_expired(self, storage):
        storage.setex("fresh", 60, "value")
        storage._sweep_expired()  # should not raise or delete
        assert storage.get("fresh") == "value"


# ---------------------------------------------------------------------------
# 9.4 — Concurrent access tests
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Thread-safe writes via multiple threads."""

    def test_concurrent_writes(self, storage):
        errors: list[Exception] = []

        def writer(prefix: str, count: int):
            try:
                for i in range(count):
                    storage.setex(f"{prefix}:{i}", 60, f"val_{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"t{n}", 50)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent write errors: {errors}"

        # Verify all writes landed
        cursor = storage._connection.execute("SELECT COUNT(*) FROM kv_store")
        assert cursor.fetchone()[0] == 250  # 5 threads * 50 keys

    def test_concurrent_read_during_write(self, storage):
        """Reads should not block while writes are in progress (WAL mode)."""
        storage.setex("existing", 60, "data")
        read_results: list[str | None] = []
        errors: list[Exception] = []

        def slow_writer():
            try:
                for i in range(20):
                    storage.setex(f"write:{i}", 60, f"v{i}")
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(20):
                    read_results.append(storage.get("existing"))
            except Exception as exc:
                errors.append(exc)

        t_write = threading.Thread(target=slow_writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert errors == [], f"Concurrent access errors: {errors}"
        # All reads should succeed (no None due to blocking)
        assert all(r == "data" for r in read_results)


# ---------------------------------------------------------------------------
# 9.5 — Migration tests
# ---------------------------------------------------------------------------


class TestMigrations:
    """Schema migration support."""

    def test_initial_schema_version_is_1(self, storage):
        cursor = storage._connection.execute("SELECT version FROM schema_version WHERE id = 1")
        assert cursor.fetchone()[0] == 1

    def test_migration_runs_on_version_mismatch(self, db_path):
        """Create DB at v1, add a migration, re-open — migration should run."""
        # First: create a v1 database
        backend1 = SQLiteStorageBackend(db_path=db_path)
        backend1.shutdown()

        # Define a migration that adds an index
        migration_applied = []

        def add_index(conn):
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON kv_store (expires_at)")
            migration_applied.append(True)

        # Patch the class-level migrations list
        original = SQLiteStorageBackend._MIGRATIONS
        try:
            SQLiteStorageBackend._MIGRATIONS = [add_index]
            backend2 = SQLiteStorageBackend(db_path=db_path)
            assert migration_applied == [True]
            cursor = backend2._connection.execute("SELECT version FROM schema_version WHERE id = 1")
            assert cursor.fetchone()[0] == 2
            backend2.shutdown()
        finally:
            SQLiteStorageBackend._MIGRATIONS = original

    def test_no_migration_when_up_to_date(self, db_path):
        """Re-opening a database at the current version should not run migrations."""
        migration_calls = []

        def should_not_run(conn):
            migration_calls.append(True)

        original = SQLiteStorageBackend._MIGRATIONS
        try:
            # First open (no migrations defined, version stays at 1)
            b1 = SQLiteStorageBackend(db_path=db_path)
            b1.shutdown()

            # Second open with empty migrations list
            SQLiteStorageBackend._MIGRATIONS = []
            b2 = SQLiteStorageBackend(db_path=db_path)
            b2.shutdown()
            assert migration_calls == []
        finally:
            SQLiteStorageBackend._MIGRATIONS = original


# ---------------------------------------------------------------------------
# 9.6 — Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    """create_storage_backend() factory function."""

    def test_default_returns_in_memory(self, monkeypatch):
        monkeypatch.delenv("STORAGE_BACKEND", raising=False)
        # Re-import to pick up env change
        from utils.storage_backend import InMemoryStorage, create_storage_backend

        backend = create_storage_backend()
        assert isinstance(backend, InMemoryStorage)
        backend.shutdown()

    def test_memory_returns_in_memory(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "memory")
        from utils.storage_backend import InMemoryStorage, create_storage_backend

        backend = create_storage_backend()
        assert isinstance(backend, InMemoryStorage)
        backend.shutdown()

    def test_sqlite_returns_sqlite_backend(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("STORAGE_SQLITE_PATH", str(tmp_path / "factory_test.db"))
        from utils.storage_backend import create_storage_backend

        backend = create_storage_backend()
        assert isinstance(backend, SQLiteStorageBackend)
        backend.shutdown()

    def test_unknown_falls_back_to_in_memory(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "redis")
        from utils.storage_backend import InMemoryStorage, create_storage_backend

        backend = create_storage_backend()
        assert isinstance(backend, InMemoryStorage)
        backend.shutdown()


# ---------------------------------------------------------------------------
# 9.7 — Persistence integration test
# ---------------------------------------------------------------------------


class TestPersistenceIntegration:
    """End-to-end: data survives backend recreation (simulates restart)."""

    def test_data_persists_across_restarts(self, db_path):
        # Write data
        b1 = SQLiteStorageBackend(db_path=db_path)
        b1.setex("thread:restart-test", 3600, '{"thread_id": "abc"}')
        b1.shutdown()

        # Re-open — data should be there
        b2 = SQLiteStorageBackend(db_path=db_path)
        assert b2.get("thread:restart-test") == '{"thread_id": "abc"}'
        b2.shutdown()

    def test_ttl_continues_across_restarts(self, db_path):
        b1 = SQLiteStorageBackend(db_path=db_path)
        b1.setex("thread:ttl-test", 60, '{"alive": true}')
        b1.shutdown()

        # Re-open within TTL
        b2 = SQLiteStorageBackend(db_path=db_path)
        assert b2.get("thread:ttl-test") is not None
        b2.shutdown()

    def test_ttl_expires_across_restarts(self, db_path):
        b1 = SQLiteStorageBackend(db_path=db_path)
        b1.setex("thread:expire-across", 1, '{"short": true}')
        b1.shutdown()

        time.sleep(1.1)

        b2 = SQLiteStorageBackend(db_path=db_path)
        assert b2.get("thread:expire-across") is None
        b2.shutdown()

    def test_conversation_store_round_trip(self, monkeypatch, db_path):
        """Full round-trip through the conversation_store API with SQLite."""
        monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("STORAGE_SQLITE_PATH", db_path)

        # Reset the singleton so it picks up the new env vars
        from utils.storage_backend import reset_storage_backend

        reset_storage_backend()

        try:
            from utils.conversation_store import add_turn, create_thread, get_thread

            tid = create_thread("chat", {"prompt": "hello"})
            add_turn(tid, "user", "hello world", tool_name="chat")
            add_turn(tid, "assistant", "hi there", tool_name="chat")

            ctx = get_thread(tid)
            assert ctx is not None
            assert len(ctx.turns) == 2
            assert ctx.turns[0].content == "hello world"
            assert ctx.turns[1].content == "hi there"
        finally:
            reset_storage_backend()
