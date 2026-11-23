"""Tests for the durable batch store (change add-batch-model-support, section 4).

No network, no API keys. ``is_durable`` (4.1) is covered here too, next to the store that
depends on the same distinction.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import stat
import time

import pytest

from utils import batch_store as bs
from utils.batch_store import (
    BatchStore,
    BatchStoreUnavailable,
    ChildIntent,
    ChildState,
    LookupStatus,
    ProviderBatchAlreadyClaimed,
    RecordState,
)
from utils.sqlite_storage import SQLiteStorageBackend
from utils.storage_backend import InMemoryStorage, create_storage_backend


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("BATCH_STORE_PATH", "BATCH_RETENTION_DAYS", "STORAGE_BACKEND", "STORAGE_SQLITE_PATH"):
        monkeypatch.delenv(name, raising=False)
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()


@pytest.fixture
def store(tmp_path):
    instance = BatchStore(str(tmp_path / "state" / "batches.db"))
    yield instance
    instance.close()


def _intent(model: str = "anthropic/claude-opus-4.5", ids: tuple[str, ...] = ("a", "b")) -> ChildIntent:
    return ChildIntent(
        provider="openrouter",
        requested_model=model + ":batch",
        submitted_model=model,
        endpoint="/v1/chat/completions",
        request_total=len(ids),
        custom_ids=ids,
    )


class _Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# ----------------------------------------------------------------------
# 4.1 is_durable
# ----------------------------------------------------------------------


class TestIsDurable:
    def test_in_memory_backend_is_not_durable(self):
        backend = InMemoryStorage()
        try:
            assert backend.is_durable is False
        finally:
            backend.shutdown()

    def test_live_sqlite_backend_is_durable_and_not_after_shutdown(self, tmp_path):
        backend = SQLiteStorageBackend(db_path=str(tmp_path / "c.db"))
        assert backend.is_durable is True
        backend.shutdown()
        assert backend.is_durable is False

    def test_forced_sqlite_init_failure_reports_not_durable_despite_env(self, monkeypatch):
        monkeypatch.setenv("STORAGE_BACKEND", "sqlite")

        def boom(self, *args, **kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(SQLiteStorageBackend, "__init__", boom)

        backend = create_storage_backend()
        try:
            assert isinstance(backend, InMemoryStorage)
            assert backend.is_durable is False
        finally:
            backend.shutdown()

    def test_working_sqlite_selection_reports_durable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
        monkeypatch.setenv("STORAGE_SQLITE_PATH", str(tmp_path / "conv.db"))
        backend = create_storage_backend()
        try:
            assert backend.is_durable is True
        finally:
            backend.shutdown()


# ----------------------------------------------------------------------
# 4.2 dedicated, independently-located store
# ----------------------------------------------------------------------


class TestDedicatedLocation:
    def test_works_with_no_storage_configuration(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        opened = bs.get_batch_store()
        assert opened.db_path == os.path.join(str(tmp_path), ".unison", "batches.db")
        assert opened.is_durable is True
        handle = opened.create_record("consensus", [_intent()])
        assert opened.get(handle) is not None
        assert os.path.exists(opened.db_path)

    def test_default_is_a_sibling_distinct_from_the_conversation_default(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from utils.sqlite_storage import _DEFAULT_DB_PATH

        batch_path = bs.default_batch_store_path()
        assert os.path.basename(batch_path) != os.path.basename(_DEFAULT_DB_PATH)
        assert os.path.basename(os.path.dirname(batch_path)) == os.path.basename(os.path.dirname(_DEFAULT_DB_PATH))

    @pytest.mark.parametrize("backend", ["sqlite", "memory", "bogus"])
    def test_conversation_storage_settings_do_not_redirect_it(self, monkeypatch, tmp_path, backend):
        monkeypatch.chdir(tmp_path)
        elsewhere = tmp_path / "conversations" / "conv.db"
        monkeypatch.setenv("STORAGE_BACKEND", backend)
        monkeypatch.setenv("STORAGE_SQLITE_PATH", str(elsewhere))

        opened = bs.get_batch_store()
        opened.create_record("chat", [_intent()])

        assert opened.db_path == os.path.join(str(tmp_path), ".unison", "batches.db")
        assert not elsewhere.exists()

    def test_batch_store_path_override_is_honoured(self, monkeypatch, tmp_path):
        target = tmp_path / "custom" / "my-batches.db"
        monkeypatch.setenv("BATCH_STORE_PATH", str(target))
        opened = bs.get_batch_store()
        assert opened.db_path == str(target)
        assert target.exists()

    def test_unopenable_path_raises_unavailable_instead_of_falling_back(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(BatchStoreUnavailable):
            BatchStore(str(blocker / "sub" / "batches.db"))


# ----------------------------------------------------------------------
# 4.3 / 4.4 record shape and handles
# ----------------------------------------------------------------------


class TestRecord:
    def test_three_child_record_round_trips_under_one_handle(self, store):
        intents = [
            _intent("anthropic/claude-opus-4.5", ("a", "b")),
            _intent("google/gemini-2.5-pro", ("c",)),
            _intent("openai/gpt-5", ("d", "e", "f")),
        ]
        payload = {"original_proposal": "Ship it?", "models": [{"model": "opus", "stance": "for"}], "n": 3}
        handle = store.create_record("consensus", intents, continuation_id="thread-1", payload=payload)
        for index, batch_id in enumerate(("b_one", "b_two", "b_three")):
            store.record_provider_batch(handle, index, batch_id)

        record = store.get(handle)
        assert record is not None
        assert record.handle == handle
        assert record.tool_name == "consensus"
        assert record.continuation_id == "thread-1"
        assert record.payload == payload
        assert record.created_at <= record.updated_at
        assert [c.provider_batch_id for c in record.children] == ["b_one", "b_two", "b_three"]
        assert [c.submitted_model for c in record.children] == [
            "anthropic/claude-opus-4.5",
            "google/gemini-2.5-pro",
            "openai/gpt-5",
        ]
        assert [c.requested_model for c in record.children][0].endswith(":batch")
        assert [c.custom_ids for c in record.children] == [("a", "b"), ("c",), ("d", "e", "f")]
        assert [c.request_total for c in record.children] == [2, 1, 3]
        assert all(
            c.claimed and c.status is ChildState.VALIDATING and c.provider == "openrouter" for c in record.children
        )
        assert len(store.list_records()) == 1

    def test_state_is_derived_from_children(self, store):
        handle = store.create_record("consensus", [_intent(), _intent("x/y")])
        assert store.recompute_state(handle) is RecordState.SUBMITTING
        store.record_provider_batch(handle, 0, "p0")
        store.record_provider_batch(handle, 1, "p1")
        assert store.recompute_state(handle) is RecordState.RUNNING
        store.set_child_status(handle, 0, ChildState.COMPLETED)
        assert store.recompute_state(handle) is RecordState.RUNNING  # one child still processing
        store.set_child_status(handle, 1, ChildState.FAILED)
        assert store.recompute_state(handle) is RecordState.COMPLETED  # partial: children carry the failure

    def test_handles_are_prefixed_unique_and_high_entropy(self, store):
        handles = {store.create_record("chat", [_intent()]) for _ in range(200)}
        assert len(handles) == 200
        for handle in handles:
            assert handle.startswith(bs.HANDLE_PREFIX)
            assert len(handle) - len(bs.HANDLE_PREFIX) >= 43  # >= 256 bits as unpadded base64url
        # Not derived from a counter, time, or each other: no shared long prefix beyond the marker.
        bodies = sorted(h[len(bs.HANDLE_PREFIX) :] for h in handles)
        assert len({b[:6] for b in bodies}) > 150

    def test_three_batch_submission_has_exactly_one_handle(self, store):
        handle = store.create_record("consensus", [_intent(), _intent("x/y"), _intent("z/w")])
        assert len(store.list_records()) == 1
        assert len(store.get(handle).children) == 3

    def test_duplicate_provider_batch_id_cannot_be_claimed_twice(self, store):
        first = store.create_record("chat", [_intent()])
        second = store.create_record("chat", [_intent()])
        store.record_provider_batch(first, 0, "same")
        with pytest.raises(ProviderBatchAlreadyClaimed):
            store.record_provider_batch(second, 0, "same")
        assert store.find_claim("openrouter", "same") == (first, 0)


# ----------------------------------------------------------------------
# 4.5 retention, tombstone, protection
# ----------------------------------------------------------------------


class TestRetention:
    def test_retention_defaults_to_thirty_days_and_is_configurable(self, monkeypatch):
        assert bs.batch_retention_days() == 30
        monkeypatch.setenv("BATCH_RETENTION_DAYS", "7")
        assert bs.batch_retention_days() == 7
        monkeypatch.setenv("BATCH_RETENTION_DAYS", "nonsense")
        assert bs.batch_retention_days() == 30
        monkeypatch.setenv("BATCH_RETENTION_DAYS", "-3")
        assert bs.batch_retention_days() == 30

    def test_purge_nulls_payloads_and_leaves_a_tombstone(self, tmp_path):
        clock = _Clock()
        path = str(tmp_path / "s" / "batches.db")
        store = BatchStore(path, retention_days=30, clock=clock)
        marker = "TOP-SECRET-PROMPT-MARKER"
        handle = store.create_record("consensus", [_intent()], continuation_id="c1", payload={"p": marker})
        store.record_provider_batch(handle, 0, "prov-1")
        store.record_consumed(handle, 0, "a", {"verdict": marker})

        clock.now += 31 * 86400
        assert store.purge_expired() == 1

        raw = sqlite3.connect(path)
        try:
            row = raw.execute(
                "SELECT tool_name, continuation_id, payload, state, expired_at FROM batches WHERE handle = ?",
                (handle,),
            ).fetchone()
            assert row[0] is None and row[1] is None and row[2] is None
            assert row[3] == "expired"
            assert row[4] == pytest.approx(clock.now)
            assert raw.execute("SELECT COUNT(*) FROM batch_children").fetchone()[0] == 0
        finally:
            raw.close()
        store.close()
        for suffix in ("", "-wal"):
            candidate = path + suffix
            if os.path.exists(candidate):
                with open(candidate, "rb") as handle_file:
                    assert marker.encode() not in handle_file.read()

    def test_terminal_and_cancelled_records_are_purged_too(self, tmp_path):
        clock = _Clock()
        store = BatchStore(str(tmp_path / "b.db"), retention_days=30, clock=clock)
        done = store.create_record("chat", [_intent()])
        store.record_provider_batch(done, 0, "p")
        store.set_child_status(done, 0, ChildState.COMPLETED)
        store.recompute_state(done)
        cancelled = store.create_record("chat", [_intent()])
        store.set_state(cancelled, RecordState.CANCELLED)
        fresh_age = clock.now
        clock.now += 29 * 86400
        young = store.create_record("chat", [_intent()])
        clock.now = fresh_age + 31 * 86400

        assert store.purge_expired() == 2
        assert store.lookup(done).status is LookupStatus.EXPIRED
        assert store.lookup(cancelled).status is LookupStatus.EXPIRED
        assert store.lookup(young).status is LookupStatus.FOUND
        store.close()

    def test_purge_runs_when_the_store_is_opened(self, tmp_path):
        clock = _Clock()
        path = str(tmp_path / "b.db")
        first = BatchStore(path, retention_days=1, clock=clock)
        handle = first.create_record("chat", [_intent()], payload={"x": 1})
        first.close()

        clock.now += 2 * 86400
        reopened = BatchStore(path, retention_days=1, clock=clock)
        assert reopened.lookup(handle).status is LookupStatus.EXPIRED
        reopened.close()

    def test_location_is_protected_like_the_conversation_store(self, tmp_path):
        conv = SQLiteStorageBackend(db_path=str(tmp_path / "conv" / "conversations.db"))
        conv.set_with_ttl("k", 60, "v")
        batch = BatchStore(str(tmp_path / "batch" / "batches.db"))
        batch.create_record("chat", [_intent()])

        def mode(path):
            return stat.S_IMODE(os.stat(path).st_mode)

        assert mode(tmp_path / "batch") == mode(tmp_path / "conv") == 0o700
        assert mode(tmp_path / "batch" / "batches.db") == mode(tmp_path / "conv" / "conversations.db") == 0o600
        assert (tmp_path / "batch" / ".gitignore").read_text() == (tmp_path / "conv" / ".gitignore").read_text()
        for suffix in ("-wal", "-shm"):
            sidecar = tmp_path / "batch" / f"batches.db{suffix}"
            if sidecar.exists():
                assert mode(sidecar) == 0o600
        conv.shutdown()
        batch.close()


# ----------------------------------------------------------------------
# 4.6 consumed outputs vs provider-retrievable results
# ----------------------------------------------------------------------


class TestConsumedOutputs:
    def test_consumed_outputs_are_persisted_and_survive_reopen(self, tmp_path):
        path = str(tmp_path / "b.db")
        first = BatchStore(path)
        handle = first.create_record("consensus", [_intent()], payload={"accumulated_responses": []})
        first.record_provider_batch(handle, 0, "p")
        first.record_consumed(handle, 0, "stance-for", {"verdict": "yes", "model": "opus"})
        first.record_consumed(handle, 0, "stance-against", {"verdict": "no"})
        first.update_payload(handle, {"accumulated_responses": [{"model": "opus", "verdict": "yes"}]})
        first.close()

        reopened = BatchStore(path)
        record = reopened.get(handle)
        assert record.children[0].consumed == {
            "stance-for": {"verdict": "yes", "model": "opus"},
            "stance-against": {"verdict": "no"},
        }
        assert record.payload == {"accumulated_responses": [{"model": "opus", "verdict": "yes"}]}
        reopened.close()

    def test_provider_retrievable_results_are_not_copied_in(self, tmp_path):
        path = str(tmp_path / "b.db")
        store = BatchStore(path)
        handle = store.create_record("chat", [_intent()], payload={"prompt": "hello"})
        store.record_provider_batch(handle, 0, "p")

        # An output the provider still holds is declined; one that could not be fetched again is kept.
        assert (
            store.record_consumed(handle, 0, "refetchable", {"text": "RESULT-STILL-AT-PROVIDER"}, refetchable=True)
            is False
        )
        assert store.record_consumed(handle, 0, "folded", {"text": "ONLY-COPY"}) is True
        store.set_child_status(handle, 0, ChildState.COMPLETED)

        record = store.get(handle)
        assert record.state is RecordState.COMPLETED
        assert record.children[0].consumed == {"folded": {"text": "ONLY-COPY"}}
        assert record.payload == {"prompt": "hello"}
        store.close()
        with open(path, "rb") as raw:
            assert b"RESULT-STILL-AT-PROVIDER" not in raw.read()
        wal = path + "-wal"
        if os.path.exists(wal):
            with open(wal, "rb") as raw:
                assert b"RESULT-STILL-AT-PROVIDER" not in raw.read()


class TestFetchedResultsAreNotMirrored:
    """The real fetch path (executor -> provider.fetch_batch) must not copy re-fetchable outputs in."""

    def test_a_completed_batch_fetched_by_the_executor_leaves_no_result_text_in_the_store(self, tmp_path):
        from tests.batch_test_helpers import FakeClock, FakeProvider, FakeRemote, make_executor, make_plan
        from utils.batch_execution import BatchRunStatus

        path = str(tmp_path / "fetch" / "batches.db")
        store = BatchStore(path)
        try:
            remote = FakeRemote()
            provider = FakeProvider(remote)
            executor = make_executor(store, {"openrouter": provider}, FakeClock())
            started = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0)
            remote.finish("prov_batch_1", results={"chat": "RESULT-STILL-AT-PROVIDER-9f3"})

            done = executor.resume(started.handle)

            # The caller really did receive the text (so the fetch ran) ...
            assert done.status is BatchRunStatus.COMPLETED
            assert done.outcomes_by_custom_id()["chat"].response.content == "RESULT-STILL-AT-PROVIDER-9f3"
            assert "fetch_batch" in provider.calls
            # ... yet nothing in the record, its children or the files on disk holds it.
            record = store.get(started.handle)
            assert record.children[0].consumed == {}
            assert "RESULT-STILL-AT-PROVIDER" not in repr(record)
        finally:
            store.close()
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(path + suffix):
                with open(path + suffix, "rb") as raw:
                    assert b"RESULT-STILL-AT-PROVIDER" not in raw.read(), suffix


# ----------------------------------------------------------------------
# 4.7 unknown vs expired
# ----------------------------------------------------------------------


class TestLookup:
    def test_purged_handle_is_expired_and_never_seen_handle_is_unknown(self, tmp_path):
        clock = _Clock()
        store = BatchStore(str(tmp_path / "b.db"), retention_days=30, clock=clock)
        handle = store.create_record("chat", [_intent()], payload={"p": 1})
        clock.now += 40 * 86400
        store.purge_expired()

        expired = store.lookup(handle)
        assert expired.status is LookupStatus.EXPIRED
        assert expired.record is None
        assert expired.expired_at == pytest.approx(clock.now)

        unknown = store.lookup("batch_never_issued")
        assert unknown.status is LookupStatus.UNKNOWN
        assert unknown.record is None and unknown.expired_at is None

        assert store.get(handle) is None
        assert store.list_records() == []
        store.close()

    def test_found_handle_returns_the_record(self, store):
        handle = store.create_record("chat", [_intent()])
        found = store.lookup(handle)
        assert found.status is LookupStatus.FOUND
        assert found.record.handle == handle

    def test_expired_record_cannot_be_modified(self, tmp_path):
        clock = _Clock()
        store = BatchStore(str(tmp_path / "b.db"), retention_days=1, clock=clock)
        handle = store.create_record("chat", [_intent()])
        clock.now += 2 * 86400
        store.purge_expired()
        with pytest.raises(bs.BatchStoreError):
            store.update_payload(handle, {"x": 1})
        store.close()


# ----------------------------------------------------------------------
# Retention is enforced for the life of the process, not only at open
# ----------------------------------------------------------------------


class TestRetentionInARunningProcess:
    def _store(self, tmp_path, clock, days=30):
        return BatchStore(str(tmp_path / "b.db"), retention_days=days, clock=clock)

    def test_lookup_after_retention_lapses_reports_expired_with_no_payload_and_no_reopen(self, tmp_path):
        clock = _Clock()
        store = self._store(tmp_path, clock)
        handle = store.create_record("chat", [_intent()], payload={"p": "secret"})
        assert store.lookup(handle).record.payload == {"p": "secret"}

        clock.now += 60 * 86400  # same process, same connection: no reopen, no manual purge
        looked = store.lookup(handle)
        assert looked.status is LookupStatus.EXPIRED and looked.record is None
        assert looked.expired_at == pytest.approx(clock.now)
        row = store._connection.execute("SELECT payload, tool_name FROM batches WHERE handle = ?", (handle,)).fetchone()
        assert row == (None, None)
        assert store.get(handle) is None
        store.close()

    @pytest.mark.parametrize("reader", ["get", "list_records", "needs_attention", "unresolved_intents", "find_claim"])
    def test_every_read_purges_first(self, tmp_path, reader):
        clock = _Clock()
        store = self._store(tmp_path, clock)
        handle = store.create_record("chat", [_intent()], payload={"p": "secret"})
        store.record_provider_batch(handle, 0, "p1")
        clock.now += 31 * 86400

        call = {
            "get": lambda: store.get(handle),
            "list_records": store.list_records,
            "needs_attention": store.needs_attention,
            "unresolved_intents": store.unresolved_intents,
            "find_claim": lambda: store.find_claim("openrouter", "p1"),
        }[reader]
        result = call()
        assert result in (None, [])
        assert store._connection.execute("SELECT payload FROM batches WHERE handle = ?", (handle,)).fetchone() == (
            None,
        )
        store.close()

    def test_writes_cannot_resurrect_or_extend_an_aged_record(self, tmp_path):
        clock = _Clock()
        store = self._store(tmp_path, clock, days=1)
        handle = store.create_record("chat", [_intent()], payload={"p": 1})
        clock.now += 2 * 86400
        for write in (
            lambda: store.update_payload(handle, {"p": 2}),
            lambda: store.set_state(handle, RecordState.RUNNING),
            lambda: store.set_child_status(handle, 0, ChildState.IN_PROGRESS),
            lambda: store.record_consumed(handle, 0, "k", 1),
        ):
            with pytest.raises(bs.BatchStoreError):
                write()
        assert store.lookup(handle).status is LookupStatus.EXPIRED
        store.close()

    def test_cancelled_and_completed_records_age_out_in_process_too(self, tmp_path):
        clock = _Clock()
        store = self._store(tmp_path, clock)
        handle = store.create_record("chat", [_intent()], payload={"p": 1})
        store.set_state(handle, RecordState.CANCELLED)
        clock.now += 31 * 86400
        assert store.lookup(handle).status is LookupStatus.EXPIRED
        store.close()


# ----------------------------------------------------------------------
# The provider-window "expired" state is not the retention tombstone
# ----------------------------------------------------------------------


class TestExpiredStateVersusTombstone:
    def test_a_live_record_can_carry_the_expired_state_with_its_payload_readable(self, tmp_path):
        clock = _Clock()
        store = BatchStore(str(tmp_path / "b.db"), retention_days=30, clock=clock)
        handle = store.create_record("chat", [_intent()], payload={"p": "keep"})
        store.set_state(handle, RecordState.EXPIRED, detail="provider window lapsed")

        found = store.lookup(handle)
        assert found.status is LookupStatus.FOUND
        assert found.record.state is RecordState.EXPIRED and found.record.payload == {"p": "keep"}
        assert found.record.children  # still resumable / reportable
        assert [r.handle for r in store.list_records()] == [handle]
        store.update_payload(handle, {"p": "still-writable"})
        store.close()

    def test_retention_still_purges_a_record_in_the_expired_state(self, tmp_path):
        clock = _Clock()
        path = str(tmp_path / "b.db")
        store = BatchStore(path, retention_days=30, clock=clock)
        handle = store.create_record("chat", [_intent()], payload={"p": "PURGE-ME-MARKER"})
        store.set_state(handle, RecordState.EXPIRED)
        clock.now += 31 * 86400

        assert store.purge_expired() == 1
        looked = store.lookup(handle)
        assert looked.status is LookupStatus.EXPIRED and looked.expired_at == pytest.approx(clock.now)
        assert store._connection.execute("SELECT payload FROM batches WHERE handle = ?", (handle,)).fetchone() == (
            None,
        )
        assert store.purge_expired() == 0  # the tombstone is never purged twice
        store.close()
        with open(path, "rb") as raw:
            assert b"PURGE-ME-MARKER" not in raw.read()

    def test_all_children_expired_derives_the_expired_record_state(self, store):
        handle = store.create_record("chat", [_intent(), _intent("x/y")])
        for index, batch_id in enumerate(("p0", "p1")):
            store.record_provider_batch(handle, index, batch_id, status=ChildState.EXPIRED)
        assert store.get(handle).state is RecordState.EXPIRED
        assert store.lookup(handle).status is LookupStatus.FOUND

        mixed = store.create_record("chat", [_intent(), _intent("x/y")])
        store.record_provider_batch(mixed, 0, "m0", status=ChildState.EXPIRED)
        store.record_provider_batch(mixed, 1, "m1", status=ChildState.FAILED)
        assert store.get(mixed).state is RecordState.FAILED


# ----------------------------------------------------------------------
# Durability of the file itself, typed failures, atomic state
# ----------------------------------------------------------------------


class TestVolatileAndFailureModes:
    @pytest.mark.parametrize("path", [":memory:", "file::memory:?cache=shared", "file:x?mode=memory"])
    def test_in_memory_databases_are_refused_and_nothing_is_written(self, monkeypatch, tmp_path, path):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(BatchStoreUnavailable, match="in-memory"):
            BatchStore(path)
        assert list(tmp_path.iterdir()) == []

    def test_a_real_store_reports_durable_until_closed(self, tmp_path):
        store = BatchStore(str(tmp_path / "b.db"))
        assert store.is_durable is True
        store.close()
        assert store.is_durable is False

    def test_sqlite_failures_surface_as_batch_store_errors(self, tmp_path):
        store = BatchStore(str(tmp_path / "b.db"))
        handle = store.create_record("chat", [_intent()])
        store._connection.execute("PRAGMA busy_timeout=0")
        blocker = sqlite3.connect(store.db_path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(bs.BatchStoreError):
                store.create_record("chat", [_intent()])
            with pytest.raises(bs.BatchStoreError):
                store.set_child_status(handle, 0, ChildState.IN_PROGRESS)
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        store.close()

    def test_a_sqlite_error_inside_a_transaction_is_typed_and_rolls_the_whole_write_back(self, store):
        """An error after the first statement (not just at BEGIN) must not escape as sqlite3.Error."""

        real = store._connection

        class Failing:
            def __init__(self) -> None:
                self.armed = True

            def execute(self, sql: str, *args):
                if self.armed and sql.startswith("INSERT INTO batch_children"):
                    raise sqlite3.OperationalError("disk I/O error (forced)")
                return real.execute(sql, *args)

            def __getattr__(self, name: str):
                return getattr(real, name)

        failing = Failing()
        store._connection = failing  # type: ignore[assignment]
        try:
            with pytest.raises(bs.BatchStoreError) as excinfo:
                store.create_record("chat", [_intent()], payload={"p": 1})
            assert not isinstance(excinfo.value, sqlite3.Error)
            assert isinstance(excinfo.value.__cause__, sqlite3.OperationalError)
        finally:
            failing.armed = False
            store._connection = real
        # The batches row inserted before the failing statement was rolled back with it.
        assert store.list_records() == []
        assert real.execute("SELECT COUNT(*) FROM batches").fetchone() == (0,)
        assert store.create_record("chat", [_intent()])  # and the store is still usable

    def test_record_state_follows_the_child_write_in_the_same_transaction(self, store):
        handle = store.create_record("chat", [_intent()])
        assert store.get(handle).state is RecordState.SUBMITTING
        store.record_provider_batch(handle, 0, "p")  # no recompute_state call
        assert store.get(handle).state is RecordState.RUNNING
        store.set_child_status(handle, 0, ChildState.COMPLETED)
        assert store.get(handle).state is RecordState.COMPLETED

    def test_compare_and_set_refuses_a_child_that_already_moved(self, store):
        handle = store.create_record("chat", [_intent()])
        store.record_provider_batch(handle, 0, "p")
        with pytest.raises(bs.ChildStateChanged):
            store.set_child_status(handle, 0, ChildState.INDETERMINATE, expect_status=ChildState.SUBMITTING)
        with pytest.raises(bs.ChildStateChanged):
            store.record_provider_batch(handle, 0, "other", expect_status=ChildState.SUBMITTING)
        assert store.get(handle).children[0].provider_batch_id == "p"
        with pytest.raises(bs.BatchStoreError) as missing:
            store.set_child_status(handle, 9, ChildState.FAILED, expect_status=ChildState.SUBMITTING)
        assert not isinstance(missing.value, bs.ChildStateChanged)


# ----------------------------------------------------------------------
# Hardening found in review
# ----------------------------------------------------------------------


class TestRetentionSettingMustBeFinite:
    @pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity", "1e999"])
    def test_non_finite_values_fall_back_to_the_default_with_a_warning(self, monkeypatch, caplog, raw):
        """``nan`` and ``inf`` parse as floats but would silently switch retention off."""

        monkeypatch.setenv("BATCH_RETENTION_DAYS", raw)
        with caplog.at_level(logging.WARNING, logger="utils.batch_store"):
            assert bs.batch_retention_days() == 30
        assert "BATCH_RETENTION_DAYS" in caplog.text

    def test_retention_is_still_enforced_when_the_setting_was_nan(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BATCH_RETENTION_DAYS", "nan")
        clock = _Clock()
        store = BatchStore(str(tmp_path / "b.db"), clock=clock)
        handle = store.create_record("chat", [_intent()], payload={"p": 1})
        clock.now += 40 * 86400
        assert store.lookup(handle).status is LookupStatus.EXPIRED
        store.close()


class TestNoPlaintextResidueInTheWriteAheadLog:
    """``secure_delete`` zeroes main-database pages only; the WAL keeps earlier plaintext until a checkpoint."""

    @staticmethod
    def _wal_bytes(path: str) -> bytes:
        wal = path + "-wal"
        return open(wal, "rb").read() if os.path.exists(wal) else b""

    def test_a_deleted_record_does_not_linger_in_the_wal(self, tmp_path):
        path = str(tmp_path / "b.db")
        store = BatchStore(path)
        handle = store.create_record("chat", [_intent()], payload={"p": "PAYLOAD_MARK_DELETE"})
        store.delete_record(handle)
        assert b"PAYLOAD_MARK_DELETE" not in self._wal_bytes(path)
        store.close()

    def test_a_replaced_payload_does_not_linger_in_the_wal(self, tmp_path):
        path = str(tmp_path / "b.db")
        store = BatchStore(path)
        handle = store.create_record("chat", [_intent()], payload={"p": "PAYLOAD_MARK_OLD"})
        store.update_payload(handle, {"p": "PAYLOAD_MARK_NEW"})
        assert b"PAYLOAD_MARK_OLD" not in self._wal_bytes(path)
        store.close()


class TestPurgeNeverDropsAnUntrackedBatchSilently:
    @pytest.mark.parametrize("status", [ChildState.ORPHANED, ChildState.INDETERMINATE])
    def test_a_purged_attention_record_is_named_in_a_warning(self, tmp_path, caplog, status):
        clock = _Clock()
        store = BatchStore(str(tmp_path / "b.db"), retention_days=30, clock=clock)
        handle = store.create_record("chat", [_intent()])
        store.record_provider_batch(handle, 0, "PROV_STILL_BILLED", status=status)
        clock.now += 31 * 86400
        with caplog.at_level(logging.WARNING, logger="utils.batch_store"):
            assert store.purge_expired() == 1
        assert handle in caplog.text and "PROV_STILL_BILLED" in caplog.text and status.value in caplog.text
        store.close()

    def test_a_purged_ordinary_record_is_not_warned_about(self, tmp_path, caplog):
        clock = _Clock()
        store = BatchStore(str(tmp_path / "b.db"), retention_days=30, clock=clock)
        handle = store.create_record("chat", [_intent()])
        store.record_provider_batch(handle, 0, "p")
        store.set_child_status(handle, 0, ChildState.COMPLETED)
        clock.now += 31 * 86400
        with caplog.at_level(logging.WARNING, logger="utils.batch_store"):
            store.purge_expired()
        assert "Retention is purging" not in caplog.text
        store.close()


class TestStorePathHandling:
    def test_a_tilde_in_the_configured_path_is_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("BATCH_STORE_PATH", "~/dedicated/b.db")
        assert bs.default_batch_store_path() == str(tmp_path / "dedicated" / "b.db")

    def test_a_dangling_gitignore_symlink_is_not_written_through(self, tmp_path):
        directory = tmp_path / "dedicated"
        directory.mkdir()
        (tmp_path / "elsewhere").mkdir()
        elsewhere = tmp_path / "elsewhere" / "created.txt"
        (directory / ".gitignore").symlink_to(elsewhere)
        BatchStore(str(directory / "b.db")).close()
        assert not elsewhere.exists()

    def test_an_existing_unrelated_sqlite_database_is_refused_untouched(self, tmp_path):
        directory = tmp_path / "shared"
        directory.mkdir()
        os.chmod(directory, 0o755)
        foreign = directory / "mystuff.db"
        connection = sqlite3.connect(foreign)
        connection.execute("CREATE TABLE notes (id INTEGER)")
        connection.commit()
        connection.close()
        size = foreign.stat().st_size

        with pytest.raises(BatchStoreUnavailable, match="not a batch store"):
            BatchStore(str(foreign))

        assert foreign.stat().st_size == size
        assert stat.S_IMODE(os.stat(directory).st_mode) == 0o755  # the directory was not re-permissioned
        assert not (directory / ".gitignore").exists()
        check = sqlite3.connect(foreign)
        try:
            assert [r[0] for r in check.execute("SELECT name FROM sqlite_master WHERE type='table'")] == ["notes"]
        finally:
            check.close()

    def test_a_file_that_is_not_sqlite_is_refused(self, tmp_path):
        garbage = tmp_path / "notes.txt"
        garbage.write_text("this is just a text file, long enough to not look like an empty database " * 4)
        with pytest.raises(BatchStoreUnavailable, match="not a readable SQLite database"):
            BatchStore(str(garbage))
        assert "just a text file" in garbage.read_text()

    def test_an_existing_batch_store_and_an_empty_file_are_still_accepted(self, tmp_path):
        path = str(tmp_path / "b.db")
        first = BatchStore(path)
        handle = first.create_record("chat", [_intent()])
        first.close()
        again = BatchStore(path)
        assert again.lookup(handle).status is LookupStatus.FOUND
        again.close()

        empty = tmp_path / "empty.db"
        empty.touch()
        BatchStore(str(empty)).close()


class TestResolutionColumn:
    """Section 15.7: what a submission resolved the model to is stored with its child and read back as recorded."""

    def test_a_child_record_stores_and_returns_its_resolution(self, tmp_path):
        store = BatchStore(str(tmp_path / "s.db"))
        resolved = {"v": 1, "slug": "a/b", "limits": {"context_window": 10, "max_output_tokens": 5}}
        handle = store.create_record(
            "chat",
            [
                ChildIntent("openrouter", "opus", "a/b", "/v1/chat/completions", 1, ("chat",), resolution=resolved),
                ChildIntent("openrouter", "m", "m", "/v1/chat/completions", 1, ("x",)),
            ],
        )

        record = store.get(handle)

        assert record is not None
        assert record.children[0].resolution == resolved and record.children[1].resolution is None
        store.close()

    def test_a_store_file_from_before_the_column_existed_is_upgraded_in_place(self, tmp_path):
        path = tmp_path / "old.db"
        connection = sqlite3.connect(path)
        now = str(time.time())
        connection.executescript("""
            CREATE TABLE schema_version (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (1, 1);
            CREATE TABLE batches (handle TEXT PRIMARY KEY, tool_name TEXT, continuation_id TEXT, payload TEXT,
                state TEXT NOT NULL, state_detail TEXT, unwound INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, expired_at REAL);
            CREATE TABLE batch_children (handle TEXT NOT NULL, idx INTEGER NOT NULL, provider TEXT NOT NULL,
                provider_batch_id TEXT, requested_model TEXT NOT NULL, submitted_model TEXT NOT NULL,
                endpoint TEXT NOT NULL, request_total INTEGER NOT NULL, status TEXT NOT NULL, custom_ids TEXT NOT NULL,
                intent_window_start REAL NOT NULL, intent_window_end REAL, claimed INTEGER NOT NULL DEFAULT 0,
                consumed TEXT, detail TEXT, PRIMARY KEY (handle, idx));
            INSERT INTO batches VALUES ('batch_old', 'chat', NULL, NULL, 'running', NULL, 0, {now}, {now}, NULL);
            INSERT INTO batch_children VALUES ('batch_old', 0, 'openrouter', 'pb', 'opus', 'a/b', '/e', 1, 'validating',
                '["chat"]', {now}, NULL, 0, NULL, NULL);
            """.replace("{now}", now))
        connection.commit()
        connection.close()

        store = BatchStore(str(path))
        old = store.get("batch_old")
        handle = store.create_record(
            "chat", [ChildIntent("openrouter", "m", "m", "/e", 1, ("c",), resolution={"v": 1})]
        )

        assert old is not None and old.children[0].resolution is None  # an earlier row reads as "nothing recorded"
        fresh = store.get(handle)
        assert fresh is not None and fresh.children[0].resolution == {"v": 1}
        store.close()

    def test_a_provider_that_resolves_nothing_stores_null(self, store):
        handle = store.create_record("chat", [_intent()])

        record = store.get(handle)

        assert record is not None and record.children[0].resolution is None

    def test_an_unreadable_stored_value_reads_as_nothing_recorded(self, tmp_path):
        path = tmp_path / "s.db"
        first = BatchStore(str(path))
        handle = first.create_record("chat", [_intent()])
        first.close()
        raw = sqlite3.connect(path)
        raw.execute("UPDATE batch_children SET resolution = ?", ("{not json",))
        raw.commit()
        raw.close()

        second = BatchStore(str(path))
        record = second.get(handle)

        assert record is not None and record.children[0].resolution is None
        second.close()
