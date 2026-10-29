"""
Tests for the working directory stored with a conversation thread.

Covers the storage half of the clink-working-dir-derivation change (task 4.1):
- ``ThreadContext.working_dir`` / ``working_dir_source`` default to ``None`` so a thread persisted
  before they existed still loads on both storage backends
- ``set_thread_working_dir`` stores both fields atomically without disturbing the turns
- the generic initial-context merge in ``reconstruct_thread_context`` no longer replays ``working_dir``

``utils.conversation_store`` is reloaded for every test by an autouse fixture in ``conftest.py``,
so the module is always reached through its attribute (``store.X``) rather than imported by name.
"""

import json
import threading
import time
import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

import utils.conversation_store as store
from utils.sqlite_storage import SQLiteStorageBackend
from utils.storage_backend import InMemoryStorage, get_storage_backend, reset_storage_backend

# One in-memory backend for the module: each instance starts a (daemon) cleanup thread that
# sleeps for minutes, so tests clear it instead of creating a new one.
_MEMORY_BACKEND = InMemoryStorage()

BACKENDS = ["memory", "sqlite"]


def _legacy_thread_json(thread_id: str) -> str:
    """A thread exactly as persisted before working_dir / working_dir_source existed."""
    return json.dumps(
        {
            "thread_id": thread_id,
            "parent_thread_id": None,
            "created_at": "2026-01-01T00:00:00+00:00",
            "last_updated_at": "2026-01-01T00:00:00+00:00",
            "tool_name": "clink",
            "turns": [
                {
                    "role": "user",
                    "content": "hello",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "files": ["/proj/a.py"],
                    "images": None,
                    "tool_name": "clink",
                    "model_provider": None,
                    "model_name": None,
                    "model_metadata": None,
                }
            ],
            "initial_context": {"prompt": "hello"},
        }
    )


def _expires_at(backend, key: str) -> float:
    """Absolute expiry time of a stored key, read straight from the backend."""
    if isinstance(backend, SQLiteStorageBackend):
        row = backend._connection.execute("SELECT expires_at FROM kv_store WHERE key = ?", (key,)).fetchone()
        return float(row[0])
    return float(backend._store[key][1])


@pytest.fixture(params=BACKENDS)
def backend(request, tmp_path):
    """Install a fresh storage backend as the process singleton for the duration of one test."""
    if request.param == "sqlite":
        instance = SQLiteStorageBackend(db_path=str(tmp_path / "threads.db"))
    else:
        with _MEMORY_BACKEND._lock:
            _MEMORY_BACKEND._store.clear()
        instance = _MEMORY_BACKEND

    get_storage_backend(instance)
    try:
        yield instance
    finally:
        reset_storage_backend()
        if isinstance(instance, SQLiteStorageBackend):
            instance.shutdown()


@pytest.fixture()
def sqlite_backend(tmp_path):
    instance = SQLiteStorageBackend(db_path=str(tmp_path / "threads.db"))
    get_storage_backend(instance)
    try:
        yield instance
    finally:
        reset_storage_backend()
        instance.shutdown()


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibleLoad:
    def test_model_defaults_to_none(self):
        context = store.ThreadContext.model_validate_json(_legacy_thread_json(str(uuid.uuid4())))

        assert context.working_dir is None
        assert context.working_dir_source is None
        assert len(context.turns) == 1

    def test_legacy_json_loads_on_both_backends(self, backend):
        thread_id = str(uuid.uuid4())
        raw = _legacy_thread_json(thread_id)
        assert "working_dir" not in raw
        backend.setex(f"thread:{thread_id}", 3600, raw)

        context = store.get_thread(thread_id)

        assert context is not None
        assert context.working_dir is None
        assert context.working_dir_source is None
        assert [t.content for t in context.turns] == ["hello"]

    def test_legacy_thread_still_takes_turns_and_gets_a_working_dir(self, backend):
        thread_id = str(uuid.uuid4())
        backend.setex(f"thread:{thread_id}", 3600, _legacy_thread_json(thread_id))

        assert store.add_turn(thread_id, "assistant", "hi there", tool_name="clink") is True
        assert store.set_thread_working_dir(thread_id, "/proj", "files") is True

        context = store.get_thread(thread_id)
        assert context is not None
        assert [t.content for t in context.turns] == ["hello", "hi there"]
        assert context.working_dir == "/proj"
        assert context.working_dir_source == "files"

    def test_fresh_thread_has_no_working_dir(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})

        context = store.get_thread(thread_id)

        assert context is not None
        assert context.working_dir is None
        assert context.working_dir_source is None


# ---------------------------------------------------------------------------
# set_thread_working_dir
# ---------------------------------------------------------------------------


class TestSetThreadWorkingDir:
    def test_round_trip(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})

        assert store.set_thread_working_dir(thread_id, "/proj/repo", "files") is True

        context = store.get_thread(thread_id)
        assert context is not None
        assert context.working_dir == "/proj/repo"
        assert context.working_dir_source == "files"

    def test_round_trip_survives_the_serialised_form(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})
        store.set_thread_working_dir(thread_id, "/proj/repo", "explicit")

        stored = json.loads(backend.get(f"thread:{thread_id}"))

        assert stored["working_dir"] == "/proj/repo"
        assert stored["working_dir_source"] == "explicit"

    def test_a_later_call_replaces_both_fields(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})
        store.set_thread_working_dir(thread_id, "/proj/one", "files")

        assert store.set_thread_working_dir(thread_id, "/proj/two", "continuation") is True

        context = store.get_thread(thread_id)
        assert context is not None
        assert context.working_dir == "/proj/two"
        assert context.working_dir_source == "continuation"

    def test_turns_and_file_order_are_untouched(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})
        store.add_turn(thread_id, "user", "q1", files=["/p/b.py", "/p/a.py"], tool_name="clink")
        store.add_turn(thread_id, "assistant", "a1", tool_name="clink", model_name="m1")
        before = store.get_thread(thread_id)

        assert store.set_thread_working_dir(thread_id, "/p", "files") is True

        after = store.get_thread(thread_id)
        assert before is not None and after is not None
        assert after.turns == before.turns
        assert after.turns[0].files == ["/p/b.py", "/p/a.py"]
        assert after.initial_context == before.initial_context
        assert after.created_at == before.created_at

    def test_a_turn_added_afterwards_keeps_the_working_dir(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})
        store.set_thread_working_dir(thread_id, "/proj", "files")

        assert store.add_turn(thread_id, "user", "next") is True

        context = store.get_thread(thread_id)
        assert context is not None
        assert context.working_dir == "/proj"
        assert context.working_dir_source == "files"
        assert len(context.turns) == 1

    def test_unknown_thread_returns_false_and_stores_nothing(self, backend):
        thread_id = str(uuid.uuid4())

        assert store.set_thread_working_dir(thread_id, "/proj", "files") is False

        assert backend.get(f"thread:{thread_id}") is None
        assert store.get_thread(thread_id) is None

    @pytest.mark.parametrize("bad_id", ["", "not-a-uuid", "thread:../../etc/passwd"])
    def test_invalid_thread_id_returns_false(self, backend, bad_id):
        assert store.set_thread_working_dir(bad_id, "/proj", "files") is False

    def test_non_string_thread_id_returns_false(self, backend):
        assert store.set_thread_working_dir(None, "/proj", "files") is False  # type: ignore[arg-type]
        assert store.set_thread_working_dir(123, "/proj", "files") is False  # type: ignore[arg-type]

    def test_expired_thread_returns_false(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})
        backend.setex(f"thread:{thread_id}", 1, backend.get(f"thread:{thread_id}"))
        time.sleep(1.2)

        assert store.set_thread_working_dir(thread_id, "/proj", "files") is False

    def test_refreshes_the_ttl(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})
        key = f"thread:{thread_id}"
        backend.setex(key, 60, backend.get(key))  # nearly expired compared with the configured timeout
        short_expiry = _expires_at(backend, key)

        assert store.set_thread_working_dir(thread_id, "/proj", "files") is True

        assert _expires_at(backend, key) > short_expiry + 60

    def test_write_failure_returns_false_instead_of_raising(self):
        thread_id = str(uuid.uuid4())
        context = store.ThreadContext(
            thread_id=thread_id,
            created_at="2026-01-01T00:00:00+00:00",
            last_updated_at="2026-01-01T00:00:00+00:00",
            tool_name="clink",
            turns=[],
            initial_context={},
        )
        mock_storage = Mock()
        mock_storage.get.return_value = context.model_dump_json()
        mock_storage.setex.side_effect = RuntimeError("disk full")

        with patch("utils.conversation_store.get_storage", return_value=mock_storage):
            assert store.set_thread_working_dir(thread_id, "/proj", "files") is False

    def test_storage_lookup_failure_returns_false_instead_of_raising(self):
        with patch("utils.conversation_store.get_storage", side_effect=RuntimeError("no storage")):
            assert store.set_thread_working_dir(str(uuid.uuid4()), "/proj", "files") is False

    def test_sqlite_atomic_failure_returns_false_instead_of_raising(self, sqlite_backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})

        with patch.object(sqlite_backend, "atomic_update", side_effect=RuntimeError("database is locked")):
            assert store.set_thread_working_dir(thread_id, "/proj", "files") is False

    def test_corrupt_stored_json_returns_false_on_sqlite(self, sqlite_backend):
        thread_id = str(uuid.uuid4())
        sqlite_backend.setex(f"thread:{thread_id}", 3600, "{not json")

        assert store.set_thread_working_dir(thread_id, "/proj", "files") is False

    def test_uses_the_atomic_path_on_sqlite(self, sqlite_backend):
        thread_id = store.create_thread("clink", {"prompt": "hello"})

        with patch.object(sqlite_backend, "atomic_update", wraps=sqlite_backend.atomic_update) as atomic:
            assert store.set_thread_working_dir(thread_id, "/proj", "files") is True

        atomic.assert_called_once()
        assert atomic.call_args[0][0] == f"thread:{thread_id}"
        assert atomic.call_args[0][1] == store.CONVERSATION_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def _run_together(*workers):
    """Start every worker at the same instant and collect their return values."""
    barrier = threading.Barrier(len(workers))
    results = [None] * len(workers)
    errors: list[BaseException] = []

    def runner(index, worker):
        try:
            barrier.wait(timeout=10)
            results[index] = worker()
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(i, w)) for i, w in enumerate(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    return results


class TestConcurrentSets:
    def test_two_threads_each_persist_their_own_directory(self, backend):
        first = store.create_thread("clink", {"prompt": "one"})
        second = store.create_thread("clink", {"prompt": "two"})
        store.add_turn(first, "user", "turn in first")
        store.add_turn(second, "user", "turn in second")

        results = _run_together(
            lambda: store.set_thread_working_dir(first, "/proj/first", "files"),
            lambda: store.set_thread_working_dir(second, "/proj/second", "explicit"),
        )

        assert results == [True, True]
        ctx_first, ctx_second = store.get_thread(first), store.get_thread(second)
        assert ctx_first is not None and ctx_second is not None
        assert (ctx_first.working_dir, ctx_first.working_dir_source) == ("/proj/first", "files")
        assert (ctx_second.working_dir, ctx_second.working_dir_source) == ("/proj/second", "explicit")
        assert [t.content for t in ctx_first.turns] == ["turn in first"]
        assert [t.content for t in ctx_second.turns] == ["turn in second"]

    def test_two_sets_on_one_thread_never_mix_directory_and_source(self, backend):
        for _ in range(25):
            thread_id = store.create_thread("clink", {"prompt": "hello"})
            store.add_turn(thread_id, "user", "q")
            store.add_turn(thread_id, "assistant", "a")

            results = _run_together(
                lambda tid=thread_id: store.set_thread_working_dir(tid, "/proj/one", "files"),
                lambda tid=thread_id: store.set_thread_working_dir(tid, "/proj/two", "explicit"),
            )

            assert results == [True, True]
            context = store.get_thread(thread_id)
            assert context is not None
            assert (context.working_dir, context.working_dir_source) in {
                ("/proj/one", "files"),
                ("/proj/two", "explicit"),
            }
            assert [t.content for t in context.turns] == ["q", "a"]

    def test_set_racing_add_turn_loses_neither_on_sqlite(self, sqlite_backend):
        for round_number in range(25):
            thread_id = store.create_thread("clink", {"prompt": "hello"})

            results = _run_together(
                lambda tid=thread_id: store.set_thread_working_dir(tid, "/proj", "files"),
                lambda tid=thread_id, n=round_number: store.add_turn(tid, "user", f"turn {n}"),
            )

            assert results == [True, True]
            context = store.get_thread(thread_id)
            assert context is not None
            assert context.working_dir == "/proj"
            assert context.working_dir_source == "files"
            assert [t.content for t in context.turns] == [f"turn {round_number}"]

    def test_set_racing_add_turn_across_connections_loses_neither(self, tmp_path, monkeypatch):
        """Two backends on one database file (as two server processes would) must not drop an update."""
        db_path = str(tmp_path / "shared.db")
        backend_a, backend_b = SQLiteStorageBackend(db_path=db_path), SQLiteStorageBackend(db_path=db_path)
        per_thread = threading.local()
        monkeypatch.setattr(store, "get_storage", lambda: per_thread.backend)

        def as_backend(chosen, func):
            def run():
                per_thread.backend = chosen
                return func()

            return run

        try:
            for round_number in range(25):
                per_thread.backend = backend_a
                thread_id = store.create_thread("clink", {"prompt": "hello"})

                results = _run_together(
                    as_backend(backend_a, lambda tid=thread_id: store.set_thread_working_dir(tid, "/proj", "files")),
                    as_backend(backend_b, lambda tid=thread_id, n=round_number: store.add_turn(tid, "user", f"t{n}")),
                )

                assert results == [True, True]
                per_thread.backend = backend_a
                context = store.get_thread(thread_id)
                assert context is not None
                assert context.working_dir == "/proj"
                assert [t.content for t in context.turns] == [f"t{round_number}"]
        finally:
            backend_a.shutdown()
            backend_b.shutdown()


# ---------------------------------------------------------------------------
# Initial-context merge in reconstruct_thread_context
# ---------------------------------------------------------------------------


class TestInitialContextMerge:
    """A continuation must not inherit turn one's working_dir as if the caller had passed it."""

    @staticmethod
    async def _reconstruct(arguments):
        from server import reconstruct_thread_context

        with patch("utils.model_context.ModelContext.calculate_token_allocation") as mock_calc:
            mock_calc.return_value = MagicMock(
                total_tokens=200000,
                content_tokens=160000,
                response_tokens=40000,
                file_tokens=64000,
                history_tokens=64000,
            )
            with patch("utils.conversation_memory.build_conversation_history") as mock_build:
                mock_build.return_value = ("=== CONVERSATION HISTORY ===\n", 1000)
                return await reconstruct_thread_context(arguments)

    @pytest.mark.asyncio
    async def test_working_dir_is_not_copied_into_the_continuation(self, backend):
        thread_id = store.create_thread(
            "clink",
            {
                "prompt": "first",
                "cli_name": "gemini",
                "working_dir": "/proj/explicit",
                "absolute_file_paths": ["/proj/explicit/a.py"],
            },
        )
        store.add_turn(thread_id, "assistant", "reply", tool_name="clink", model_name="gemini-2.5-flash")

        enhanced = await self._reconstruct({"continuation_id": thread_id, "prompt": "second"})

        assert "working_dir" not in enhanced
        # Other initial_context keys still merge, so the exclusion is specific to working_dir.
        assert enhanced["cli_name"] == "gemini"
        assert enhanced["absolute_file_paths"] == ["/proj/explicit/a.py"]

    @pytest.mark.asyncio
    async def test_the_stored_resolved_directory_is_not_replayed_either(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "first", "cli_name": "gemini"})
        store.add_turn(thread_id, "assistant", "reply", tool_name="clink", model_name="gemini-2.5-flash")
        assert store.set_thread_working_dir(thread_id, "/proj/derived", "files") is True

        enhanced = await self._reconstruct({"continuation_id": thread_id, "prompt": "second"})

        assert "working_dir" not in enhanced
        assert "working_dir_source" not in enhanced
        assert enhanced["cli_name"] == "gemini"

    @pytest.mark.asyncio
    async def test_a_working_dir_passed_on_the_continuation_is_kept(self, backend):
        thread_id = store.create_thread("clink", {"prompt": "first", "working_dir": "/proj/old"})
        store.add_turn(thread_id, "assistant", "reply", tool_name="clink", model_name="gemini-2.5-flash")

        enhanced = await self._reconstruct(
            {"continuation_id": thread_id, "prompt": "second", "working_dir": "/proj/new"}
        )

        assert enhanced["working_dir"] == "/proj/new"
