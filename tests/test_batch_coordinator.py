"""Tests for the submission saga and reconciliation (add-batch-model-support, section 5).

No network, no API keys: providers are in-process fakes, plus the real OpenRouter adapter
with its transport seam stubbed where its cancel semantics matter.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import utils.model_restrictions as model_restrictions
from providers.openrouter import OpenRouterProvider
from providers.shared import (
    BatchListEntry,
    BatchListing,
    BatchProviderError,
    BatchRequest,
    BatchState,
    CancelOutcome,
    CancelState,
    UnsupportedBatchOperation,
)
from utils import batch_store as bs
from utils.batch_coordinator import (
    BatchCoordinator,
    BatchSubmissionError,
    ChildPlan,
    ReconcileAction,
    SubmissionOutcomeUnknown,
    reconcile_at_startup,
)
from utils.batch_store import BatchStore, BatchStoreError, ChildState, LookupStatus, RecordState

UNKNOWN = "submission_outcome_unknown"
MODEL = "anthropic/claude-opus-4.5"


class _Clock:
    def __init__(self, now: float = 2_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeProvider:
    """Duck-typed batch provider that records every call, in order."""

    def __init__(
        self,
        name: str = "openrouter",
        *,
        submits: list[Any] | None = None,
        cancel: Any = None,
        listing: Any = None,
    ) -> None:
        self.name = name
        self.calls: list[str] = []
        self.submit_args: list[tuple[str, list[BatchRequest]]] = []
        self.submit_resolutions: list[Any] = []
        self.cancelled: list[str] = []
        self._submits = list(submits or [])
        self._cancel = cancel if cancel is not None else CancelOutcome(CancelState.CANCELLED)
        self._listing = listing
        self.on_submit: Any = None
        self._counter = 0

    def get_provider_type(self) -> Any:
        return SimpleNamespace(value=self.name)

    def batch_submission_model(self, model_name: str) -> str:
        return model_name.removesuffix(":batch")

    def submit_batch(self, model_name: str, requests: list[BatchRequest], resolution: Any = None) -> str:
        self.calls.append("submit_batch")
        self.submit_args.append((model_name, list(requests)))
        self.submit_resolutions.append(resolution)
        if self.on_submit is not None:
            self.on_submit(model_name, requests)
        if self._submits:
            item = self._submits.pop(0)
            if isinstance(item, BaseException):
                raise item
            return str(item)
        self._counter += 1
        return f"prov_{self.name}_{self._counter}"

    def cancel_batch(self, batch_id: str) -> CancelOutcome:
        self.calls.append("cancel_batch")
        self.cancelled.append(batch_id)
        if isinstance(self._cancel, BaseException):
            raise self._cancel
        return self._cancel

    def poll_batch(self, batch_id: str) -> Any:  # pragma: no cover - must never be reached
        self.calls.append("poll_batch")
        raise AssertionError("poll must not happen during reconciliation")

    def list_batches(self, limit: int = 100) -> BatchListing:
        self.calls.append("list_batches")
        if isinstance(self._listing, BaseException):
            raise self._listing
        if self._listing is None:
            raise UnsupportedBatchOperation(self.name, "list_batches")
        return self._listing


def _reqs(*ids: str) -> list[BatchRequest]:
    return [BatchRequest(custom_id=i, prompt=f"prompt {i}") for i in ids]


def _plan(provider: Any, model: str = MODEL, ids: tuple[str, ...] = ("a", "b")) -> ChildPlan:
    return ChildPlan(provider=provider, model=model, requests=_reqs(*ids))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("BATCH_STORE_PATH", "BATCH_RETENTION_DAYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENROUTER_ALLOWED_MODELS", raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def store(tmp_path, clock):
    instance = BatchStore(str(tmp_path / "s" / "batches.db"), clock=clock)
    yield instance
    instance.close()


def _coordinator(store, providers: dict[str, Any], clock) -> BatchCoordinator:
    return BatchCoordinator(store, provider_resolver=lambda name: providers.get(name), clock=clock)


def _fail_first_write(store: BatchStore, monkeypatch, *, always: bool = False) -> None:
    """Make recording a provider id fail; later (orphan) writes succeed unless ``always``."""

    real = store.record_provider_batch

    def flaky(handle, index, provider_batch_id, **kwargs):
        if always or kwargs.get("status", ChildState.VALIDATING) is ChildState.VALIDATING:
            raise BatchStoreError("disk I/O error (forced)")
        return real(handle, index, provider_batch_id, **kwargs)

    monkeypatch.setattr(store, "record_provider_batch", flaky)


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


class TestSubmitSuccess:
    def test_three_models_yield_one_handle_and_three_recorded_children(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        handle = coordinator.submit(
            "consensus",
            [_plan(provider, "m/one:batch"), _plan(provider, "m/two", ("c",)), _plan(provider, "m/three", ("d", "e"))],
            continuation_id="thread",
            payload={"original_proposal": "x"},
        )
        record = store.get(handle)
        assert record.state is RecordState.RUNNING
        assert [c.provider_batch_id for c in record.children] == [f"prov_openrouter_{i}" for i in (1, 2, 3)]
        assert [c.status for c in record.children] == [ChildState.VALIDATING] * 3
        assert record.children[0].requested_model == "m/one:batch"  # caller's spelling kept
        assert record.children[0].submitted_model == "m/one"
        assert len(store.list_records()) == 1
        assert provider.submit_args[0][0] == "m/one:batch"  # passed to the adapter untouched

    def test_duplicate_custom_ids_fail_before_anything_is_persisted(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(ValueError, match="Duplicate"):
            coordinator.submit("chat", [_plan(provider, ids=("a", "a"))])
        assert store.list_records() == []
        assert provider.calls == []


# ----------------------------------------------------------------------
# Section 15.5/15.7: the resolution admission made is the one that is recorded and submitted
# ----------------------------------------------------------------------


def _resolution(slug: str, *, window: int = 1_000_000, note: tuple[str, ...] = ()) -> Any:
    from providers.shared import ModelCapabilities, ProviderType
    from providers.shared.batch_resolution import BatchModelResolution

    capabilities = ModelCapabilities(
        provider=ProviderType.OPENROUTER,
        model_name=slug,
        friendly_name="OpenRouter",
        context_window=window,
        max_output_tokens=window // 8,
    )
    return BatchModelResolution(
        requested="alias", slug=slug, capabilities=capabilities, floated=True, derived=True, unclassified=note
    )


class TestResolutionIsCarriedAndRecorded:
    def test_a_plan_with_a_resolution_records_it_and_hands_it_to_the_provider(self, store, clock):
        provider = FakeProvider()
        resolution = _resolution("vendor/model-5", window=500_000, note=("vendor/model-6-x",))
        plan = ChildPlan(
            provider=provider, model="alias", requests=[BatchRequest(custom_id="a", prompt="p")], resolution=resolution
        )

        handle = _coordinator(store, {"openrouter": provider}, clock).submit("chat", [plan])

        record = store.get(handle)
        assert provider.submit_resolutions == [resolution]  # exactly the admission's value, not a new resolution
        child = record.children[0]
        assert child.requested_model == "alias" and child.submitted_model == "vendor/model-5"
        assert child.resolution is not None and child.resolution["slug"] == "vendor/model-5"
        assert child.resolution["limits"] == {"context_window": 500_000, "max_output_tokens": 62_500}
        assert child.resolution["unclassified_newer_batch_models"] == ["vendor/model-6-x"]

    def test_a_plan_without_one_submits_as_before_and_records_null(self, store, clock):
        provider = FakeProvider()

        handle = _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider, "m/one")])

        assert provider.submit_resolutions == [None]
        record = store.get(handle)
        assert record.children[0].resolution is None and record.children[0].submitted_model == "m/one"

    def test_each_child_of_a_multi_batch_operation_keeps_its_own_resolution(self, store, clock):
        provider = FakeProvider()
        plans = [
            ChildPlan(
                provider=provider,
                model="a",
                requests=[BatchRequest(custom_id="x", prompt="p")],
                resolution=_resolution("v/a-1", window=100_000),
            ),
            ChildPlan(
                provider=provider,
                model="b",
                requests=[BatchRequest(custom_id="y", prompt="p")],
                resolution=_resolution("v/b-2", window=900_000, note=("v/b-3-z",)),
            ),
            ChildPlan(provider=provider, model="c", requests=[BatchRequest(custom_id="z", prompt="p")]),
        ]

        handle = _coordinator(store, {"openrouter": provider}, clock).submit("consensus", plans)

        children = store.get(handle).children
        assert [c.resolution["slug"] if c.resolution else None for c in children] == ["v/a-1", "v/b-2", None]
        assert children[0].resolution["limits"]["context_window"] == 100_000
        assert children[1].resolution["limits"]["context_window"] == 900_000
        assert "unclassified_newer_batch_models" not in children[0].resolution

    def test_the_recorded_resolution_survives_a_restart_of_the_store(self, tmp_path, clock):
        path = str(tmp_path / "state" / "batches.db")
        first = BatchStore(path)
        provider = FakeProvider()
        plan = ChildPlan(
            provider=provider,
            model="alias",
            requests=[BatchRequest(custom_id="a", prompt="p")],
            resolution=_resolution("vendor/model-5"),
        )
        handle = _coordinator(first, {"openrouter": provider}, clock).submit("chat", [plan])
        first.close()

        second = BatchStore(path)
        record = second.get(handle)
        second.close()

        assert record is not None and record.children[0].resolution["slug"] == "vendor/model-5"


# ----------------------------------------------------------------------
# 5.1 intent before the provider call
# ----------------------------------------------------------------------


class TestIntentFirst:
    def test_intent_record_exists_before_provider_is_called(self, store, clock):
        provider = FakeProvider()
        observed: list[Any] = []

        def inspect(model_name, requests):
            records = store.list_records()
            observed.append((len(records), records[0].state, records[0].children[0]))

        provider.on_submit = inspect
        _coordinator(store, {"openrouter": provider}, clock).submit(
            "chat", [_plan(provider, "m/one:batch", ("x", "y", "z"))], payload={"prompt": "hello"}
        )

        count, state, child = observed[0]
        assert count == 1
        assert state is RecordState.SUBMITTING
        assert child.status is ChildState.SUBMITTING
        assert child.provider_batch_id is None and child.claimed is False
        assert child.provider == "openrouter"
        assert child.requested_model == "m/one:batch" and child.submitted_model == "m/one"
        assert child.endpoint == "/v1/chat/completions"
        assert child.request_total == 3
        assert child.custom_ids == ("x", "y", "z")
        assert child.intent_window_start == clock.now

    def test_unavailable_store_blocks_submission_before_any_provider_call(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        store.close()
        with pytest.raises(BatchSubmissionError, match="nothing was submitted"):
            coordinator.submit("chat", [_plan(provider)])
        assert provider.calls == []

    def test_intent_write_failure_means_no_provider_call(self, store, clock, monkeypatch):
        provider = FakeProvider()
        monkeypatch.setattr(store, "create_record", lambda *a, **k: (_ for _ in ()).throw(BatchStoreError("full")))
        with pytest.raises(BatchSubmissionError, match="nothing was submitted"):
            _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider)])
        assert provider.calls == []


# ----------------------------------------------------------------------
# 5.2 accepted but unrecorded
# ----------------------------------------------------------------------


class TestUnrecordedAcceptance:
    def test_cancel_supported_is_cancelled_and_no_handle_returned(self, store, clock, monkeypatch):
        provider = FakeProvider("anthropic", cancel=CancelOutcome(CancelState.CANCELLED))
        _fail_first_write(store, monkeypatch)
        with pytest.raises(BatchSubmissionError) as excinfo:
            _coordinator(store, {"anthropic": provider}, clock).submit("chat", [_plan(provider)])

        error = excinfo.value
        assert error.handle is None and error.orphans == ()
        assert provider.cancelled == ["prov_anthropic_1"]
        record = store.list_records()[0]
        assert record.children[0].status is ChildState.CANCELLED
        assert record.children[0].provider_batch_id == "prov_anthropic_1"
        assert record.state is RecordState.CANCELLED
        assert store.needs_attention() == []

    def test_cancel_unsupported_orphans_the_batch_and_names_it(self, store, clock, monkeypatch):
        provider = FakeProvider(
            "openrouter", cancel=CancelOutcome(CancelState.UNSUPPORTED, "OpenRouter has no cancel endpoint")
        )
        _fail_first_write(store, monkeypatch)
        with pytest.raises(BatchSubmissionError) as excinfo:
            _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider)])

        error = excinfo.value
        assert error.handle is None
        assert [(o.provider, o.batch_id) for o in error.orphans] == [("openrouter", "prov_openrouter_1")]
        text = str(error)
        assert "openrouter" in text and "prov_openrouter_1" in text
        assert "continues" in text and "billed" in text
        record = store.list_records()[0]
        assert record.state is RecordState.ORPHANED
        assert record.children[0].status is ChildState.ORPHANED
        assert record.children[0].provider_batch_id == "prov_openrouter_1"
        assert [r.handle for r in store.needs_attention()] == [record.handle]

    def test_cancel_failure_orphans_the_batch_and_names_it(self, store, clock, monkeypatch):
        provider = FakeProvider("openai", cancel=RuntimeError("cancel endpoint down"))
        _fail_first_write(store, monkeypatch)
        with pytest.raises(BatchSubmissionError) as excinfo:
            _coordinator(store, {"openai": provider}, clock).submit("chat", [_plan(provider)])

        error = excinfo.value
        assert error.handle is None
        assert [(o.provider, o.batch_id) for o in error.orphans] == [("openai", "prov_openai_1")]
        assert "cancel endpoint down" in str(error) and "billed" in str(error)
        assert store.list_records()[0].children[0].status is ChildState.ORPHANED

    def test_error_names_the_batch_even_when_the_store_cannot_record_anything(self, store, clock, monkeypatch):
        provider = FakeProvider("openrouter", cancel=CancelOutcome(CancelState.UNSUPPORTED))
        _fail_first_write(store, monkeypatch, always=True)
        with pytest.raises(BatchSubmissionError) as excinfo:
            _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider)])
        assert "prov_openrouter_1" in str(excinfo.value) and "openrouter" in str(excinfo.value)
        assert excinfo.value.handle is None
        # Nothing could be written, so the child is still an unresolved intent, not a silent success.
        assert store.list_records()[0].children[0].status is ChildState.SUBMITTING

    def test_real_openrouter_adapter_never_issues_a_delete(self, store, clock, monkeypatch):
        provider = OpenRouterProvider(api_key="test-key")
        calls: list[tuple[str, str]] = []

        def send(method: str, path: str, content: str | None = None) -> httpx.Response:
            calls.append((method, path))
            if method == "POST":
                return httpx.Response(202, json={"id": "or_batch_1", "status": "validating"})
            return httpx.Response(200, json={"id": "or_batch_1", "status": "in_progress", "request_counts": {}})

        monkeypatch.setattr(provider, "_batch_send", send)
        _fail_first_write(store, monkeypatch)
        with pytest.raises(BatchSubmissionError) as excinfo:
            _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider)])

        assert {method for method, _ in calls} <= {"POST", "GET"}
        assert "DELETE" not in {method for method, _ in calls}
        assert [(o.provider, o.batch_id) for o in excinfo.value.orphans] == [("openrouter", "or_batch_1")]
        assert "billed" in str(excinfo.value)


# ----------------------------------------------------------------------
# 5.3 multi-batch unwind
# ----------------------------------------------------------------------


class TestPartialMultiBatch:
    def test_first_batch_is_cancelled_on_a_cancel_capable_provider(self, store, clock):
        provider = FakeProvider("anthropic", submits=["acc_1", BatchProviderError("quota exceeded", status_code=429)])
        with pytest.raises(BatchSubmissionError) as excinfo:
            _coordinator(store, {"anthropic": provider}, clock).submit(
                "consensus", [_plan(provider, "m/one"), _plan(provider, "m/two", ("c",))]
            )
        error = excinfo.value
        assert error.handle is None and error.orphans == ()
        assert provider.cancelled == ["acc_1"]
        assert provider.calls.count("submit_batch") == 2
        record = store.list_records()[0]
        assert record.children[0].status is ChildState.CANCELLED
        assert record.children[1].status is ChildState.FAILED
        assert record.state is RecordState.FAILED

    def test_first_batch_is_named_as_left_running_on_openrouter(self, store, clock):
        provider = FakeProvider(
            "openrouter",
            submits=["or_1", BatchProviderError("HTTP 400", status_code=400)],
            cancel=CancelOutcome(CancelState.UNSUPPORTED),
        )
        with pytest.raises(BatchSubmissionError) as excinfo:
            _coordinator(store, {"openrouter": provider}, clock).submit(
                "consensus", [_plan(provider, "m/one"), _plan(provider, "m/two", ("c",))]
            )
        error = excinfo.value
        assert error.handle is None
        assert [(o.provider, o.batch_id) for o in error.orphans] == [("openrouter", "or_1")]
        assert "or_1" in str(error) and "billed" in str(error)
        record = store.list_records()[0]
        assert record.state is RecordState.ORPHANED
        assert record.children[0].status is ChildState.ORPHANED
        assert record.children[1].status is ChildState.FAILED

    def test_later_children_are_not_submitted_after_a_failure(self, store, clock):
        provider = FakeProvider(submits=[ValueError("restricted model")])
        with pytest.raises(BatchSubmissionError):
            _coordinator(store, {"openrouter": provider}, clock).submit(
                "consensus", [_plan(provider, "m/one"), _plan(provider, "m/two"), _plan(provider, "m/three")]
            )
        assert provider.calls == ["submit_batch"]

    def test_clean_failure_retains_nothing(self, store, clock):
        provider = FakeProvider(submits=[ValueError("restricted model")])
        with pytest.raises(BatchSubmissionError) as excinfo:
            _coordinator(store, {"openrouter": provider}, clock).submit(
                "chat", [_plan(provider)], payload={"prompt": "secret"}
            )
        assert excinfo.value.handle is None
        assert store.list_records() == []


# ----------------------------------------------------------------------
# Ambiguous submission outcome
# ----------------------------------------------------------------------


class TestOutcomeUnknown:
    def _submit_unknown(self, store, clock, provider=None, plans=1):
        provider = provider or FakeProvider(
            submits=[BatchProviderError("connection reset", provider="openrouter", code=UNKNOWN)]
        )
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider, ids=(str(i),)) for i in range(plans)])
        return provider, coordinator, excinfo.value

    def test_intent_stays_error_names_handle_and_nothing_is_retried(self, store, clock):
        provider, _, error = self._submit_unknown(store, clock)
        assert provider.calls == ["submit_batch"]
        assert error.handle is not None
        text = str(error)
        assert "UNKNOWN" in text and error.handle in text and "NOT retried" in text
        record = store.get(error.handle)
        assert record.state is RecordState.SUBMITTING
        assert record.children[0].status is ChildState.SUBMITTING
        assert record.children[0].intent_window_end is not None

    def test_untyped_transport_style_failure_is_also_treated_as_ambiguous(self, store, clock):
        provider = FakeProvider(submits=[TimeoutError("timed out")])
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])
        assert provider.calls == ["submit_batch"]
        assert store.get(excinfo.value.handle).children[0].status is ChildState.SUBMITTING

    @pytest.mark.parametrize("status", [408, 425, 409, 418, 499, 500, 502, 504, 301, 200])
    def test_a_provider_error_whose_status_does_not_prove_non_acceptance_is_ambiguous(self, store, clock, status):
        """The coordinator's own classification, for adapters that do not tag the error themselves."""

        provider = FakeProvider(submits=[BatchProviderError("no answer", provider="openrouter", status_code=status)])
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])

        assert provider.calls == ["submit_batch"]  # never retried
        record = store.get(excinfo.value.handle)
        assert record is not None and record.children[0].status is ChildState.SUBMITTING

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 405, 413, 422, 429])
    def test_a_provider_error_whose_status_proves_non_acceptance_is_a_clean_refusal(self, store, clock, status):
        provider = FakeProvider(submits=[BatchProviderError("refused", provider="openrouter", status_code=status)])
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(BatchSubmissionError) as excinfo:
            coordinator.submit("chat", [_plan(provider)])

        assert not isinstance(excinfo.value, SubmissionOutcomeUnknown)
        assert store.list_records() == []  # the intent was deleted: nothing can exist at the provider

    def test_multi_batch_ambiguity_unwinds_earlier_children_and_abandons_the_operation(self, store, clock):
        provider = FakeProvider(
            "openrouter",
            submits=["or_1", BatchProviderError("reset", code=UNKNOWN)],
            cancel=CancelOutcome(CancelState.UNSUPPORTED),
        )
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("consensus", [_plan(provider, "m/one"), _plan(provider, "m/two", ("c",))])
        error = excinfo.value
        assert [(o.provider, o.batch_id) for o in error.orphans] == [("openrouter", "or_1")]
        # The operation was abandoned: no resumable handle goes to the caller, and the message says so.
        assert error.handle is None
        assert error.retained_handle is not None and error.retained_handle not in str(error)
        assert "no handle is returned" in str(error)
        record = store.get(error.retained_handle)
        assert record.unwound is True
        assert record.children[0].status is ChildState.ORPHANED
        assert record.children[1].status is ChildState.SUBMITTING

    def test_single_batch_ambiguity_still_names_its_resumable_handle(self, store, clock):
        _, _, error = self._submit_unknown(store, clock)
        assert error.handle is not None and error.retained_handle == error.handle


# ----------------------------------------------------------------------
# 5.4 adapter list operation (the correlation mechanism recorded in design.md)
# ----------------------------------------------------------------------


class TestOpenRouterListOperation:
    def test_list_batches_is_a_metadata_get_with_limit(self, monkeypatch):
        provider = OpenRouterProvider(api_key="test-key")
        seen: list[tuple[str, str]] = []

        def send(method: str, path: str, content: str | None = None) -> httpx.Response:
            seen.append((method, path))
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "b1",
                            "status": "in_progress",
                            "model": MODEL,
                            "endpoint": "/v1/chat/completions",
                            "created_at": 1_700_000_000,
                            "request_counts": {"total": 4, "completed": 0, "failed": 0},
                            "results": None,
                        },
                        {"id": "b2", "status": "completed", "created_at": "2026-11-21T10:00:00Z"},
                        {"status": "completed"},
                    ]
                },
            )

        monkeypatch.setattr(provider, "_batch_send", send)
        listing = provider.list_batches(limit=50)

        assert seen == [("GET", "/batches?limit=50")]
        first, second = listing.entries
        assert (first.batch_id, first.state, first.model, first.endpoint, first.total) == (
            "b1",
            BatchState.IN_PROGRESS,
            MODEL,
            "/v1/chat/completions",
            4,
        )
        assert first.created_at == 1_700_000_000
        assert second.batch_id == "b2" and second.model is None and second.total is None
        assert second.created_at == pytest.approx(1_795_000_000, rel=0.05)
        assert listing.may_be_truncated is False

    def test_full_page_is_flagged_possibly_truncated(self, monkeypatch):
        provider = OpenRouterProvider(api_key="test-key")
        rows = [{"id": f"b{i}", "status": "completed"} for i in range(3)]
        monkeypatch.setattr(provider, "_batch_send", lambda *a, **k: httpx.Response(200, json=rows))
        assert provider.list_batches(limit=3).may_be_truncated is True

    def test_documented_list_envelope_is_parsed_and_has_more_decides_truncation(self, monkeypatch):
        """Shape per https://openrouter.ai/docs/api/api-reference/batch/list-batches (checked 2026-11-21):

        ``{"object": "list", "data": [...], "first_id", "last_id", "has_more"}``, newest first, items are
        metadata-only batch objects whose ``created_at`` is a Unix timestamp and ``results`` is null.
        """

        provider = OpenRouterProvider(api_key="test-key")

        def page(has_more: bool, rows: int) -> dict[str, Any]:
            data = [
                {
                    "object": "batch",
                    "id": f"batch_{i}",
                    "status": "validating",
                    "model": MODEL,
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "created_at": 1_790_000_000 - i,
                    "expires_at": 1_790_086_400,
                    "finalized_at": None,
                    "provider": "Anthropic",
                    "request_counts": {"total": 2, "completed": 0, "failed": 0},
                    "results": None,
                }
                for i in range(rows)
            ]
            return {
                "object": "list",
                "data": data,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None,
                "has_more": has_more,
            }

        # Fewer rows than the limit but has_more=True: still truncated (len(rows) alone would say no).
        monkeypatch.setattr(provider, "_batch_send", lambda *a, **k: httpx.Response(200, json=page(True, 3)))
        more = provider.list_batches(limit=100)
        assert more.may_be_truncated is True
        assert [e.batch_id for e in more.entries] == ["batch_0", "batch_1", "batch_2"]
        assert more.entries[0].created_at == 1_790_000_000 and more.entries[0].total == 2
        assert more.entries[0].state is BatchState.VALIDATING

        # A full page with has_more=False is the whole list.
        monkeypatch.setattr(provider, "_batch_send", lambda *a, **k: httpx.Response(200, json=page(False, 3)))
        assert provider.list_batches(limit=3).may_be_truncated is False

        # The documented empty list.
        empty = {"data": [], "first_id": None, "has_more": False, "last_id": None, "object": "list"}
        monkeypatch.setattr(provider, "_batch_send", lambda *a, **k: httpx.Response(200, json=empty))
        listing = provider.list_batches()
        assert listing.entries == () and listing.may_be_truncated is False

    def test_limit_is_clamped_to_the_documented_maximum_of_100(self, monkeypatch):
        provider = OpenRouterProvider(api_key="test-key")
        seen: list[str] = []

        def send(method: str, path: str, content: str | None = None) -> httpx.Response:
            seen.append(path)
            return httpx.Response(200, json={"object": "list", "data": [], "has_more": False})

        monkeypatch.setattr(provider, "_batch_send", send)
        provider.list_batches(limit=1000)
        provider.list_batches(limit=0)
        assert seen == ["/batches?limit=100", "/batches?limit=1"]  # a larger value is a 400 at OpenRouter

    def test_provider_without_the_operation_raises_unsupported(self):
        from providers.gemini import GeminiModelProvider

        with pytest.raises(UnsupportedBatchOperation, match="list_batches"):
            GeminiModelProvider(api_key="test-key").list_batches()

    def test_submission_model_reports_the_slug_the_envelope_would_carry(self):
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.batch_submission_model(MODEL + ":batch") == MODEL
        assert provider.batch_submission_model(MODEL) == MODEL


# ----------------------------------------------------------------------
# 5.5 reconciliation
# ----------------------------------------------------------------------


def _entry(
    batch_id: str,
    *,
    model: str | None = MODEL,
    endpoint: str | None = "/v1/chat/completions",
    total: int | None = 2,
    created_at: float | None,
    state: BatchState | None = BatchState.IN_PROGRESS,
) -> BatchListEntry:
    return BatchListEntry(
        batch_id=batch_id, state=state, model=model, endpoint=endpoint, created_at=created_at, total=total
    )


class TestReconciliation:
    def _unresolved(self, store, clock, provider, *, ids=("a", "b"), model=MODEL):
        """Persist an ambiguous submission, then age it past the startup grace period."""

        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        provider._submits.append(BatchProviderError("reset", provider="openrouter", code=UNKNOWN))
        start = clock.now
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider, model, ids)])
        clock.now += 600
        return coordinator, excinfo.value.handle, start

    def test_adopts_exactly_one_unclaimed_matching_batch(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(
            entries=(
                _entry("theirs", model="other/model", created_at=start),
                _entry("ours", created_at=start + 1),
                _entry("old", created_at=start - 3600),
            )
        )
        results = coordinator.reconcile_pending()

        assert [(r.handle, r.action, r.provider_batch_id) for r in results] == [
            (handle, ReconcileAction.ADOPTED, "ours")
        ]
        record = store.get(handle)
        assert record.children[0].provider_batch_id == "ours"
        assert record.children[0].claimed is True
        assert record.children[0].status is ChildState.IN_PROGRESS
        assert record.state is RecordState.RUNNING
        assert provider.calls.count("submit_batch") == 1  # the original attempt only

    def test_closes_out_intent_with_no_corresponding_batch(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(
            entries=(
                _entry("wrong-size", total=9, created_at=start + 1),
                _entry("wrong-time", created_at=start + 7200),
                _entry("wrong-endpoint", endpoint="/v1/embeddings", created_at=start + 1),
            )
        )
        results = coordinator.reconcile_pending()

        assert [r.action for r in results] == [ReconcileAction.CLOSED_OUT]
        record = store.get(handle)
        assert record.children[0].status is ChildState.FAILED
        assert record.state is RecordState.FAILED
        assert "no provider batch" in record.children[0].detail
        assert provider.calls.count("submit_batch") == 1

    def test_several_matches_are_indeterminate(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("m1", created_at=start), _entry("m2", created_at=start + 2)))
        results = coordinator.reconcile_pending()

        assert [r.action for r in results] == [ReconcileAction.INDETERMINATE]
        assert store.get(handle).state is RecordState.INDETERMINATE
        assert store.get(handle).children[0].provider_batch_id is None

    def test_match_already_claimed_by_another_record_is_indeterminate(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        other = store.create_record("chat", [bs.ChildIntent("openrouter", "x", "x", "/e", 1, ("z",))])
        store.record_provider_batch(other, 0, "taken")
        provider._listing = BatchListing(entries=(_entry("taken", created_at=start + 1),))

        results = coordinator.reconcile_pending()
        assert [r.action for r in results] == [ReconcileAction.INDETERMINATE]
        assert "already claimed" in store.get(handle).children[0].detail
        assert store.get(handle).children[0].provider_batch_id is None

    def test_provider_without_a_list_operation_is_indeterminate_and_not_resubmitted(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, _ = self._unresolved(store, clock, provider)
        provider._listing = None  # list_batches raises UnsupportedBatchOperation
        results = coordinator.reconcile_pending()

        assert [r.action for r in results] == [ReconcileAction.INDETERMINATE]
        assert "no way to enumerate" in store.get(handle).children[0].detail
        assert provider.calls.count("submit_batch") == 1

    def test_listing_that_cannot_rule_a_batch_out_is_indeterminate_not_closed_out(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)

        # Truncated page that does not reach back to the submission time.
        provider._listing = BatchListing(
            entries=(_entry("n1", model="other", created_at=start + 100),), may_be_truncated=True
        )
        assert [r.action for r in coordinator.reconcile_pending()] == [ReconcileAction.INDETERMINATE]

    def test_unverifiable_listing_entry_blocks_close_out(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("opaque", model=None, created_at=start + 1),))
        assert [r.action for r in coordinator.reconcile_pending()] == [ReconcileAction.INDETERMINATE]

    def test_a_listed_batch_that_differs_only_in_model_spelling_blocks_close_out(self, store, clock):
        """The provider may report the model differently from the slug sent; closing out is irreversible."""

        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(
            entries=(_entry("maybe-ours", model=MODEL + "-20251101", created_at=start + 1),)
        )
        results = coordinator.reconcile_pending()

        assert [r.action for r in results] == [ReconcileAction.INDETERMINATE]
        assert (
            "maybe-ours" not in results[0].detail
        )  # ids stay server-side (see TestReconciliationKeepsIdentifiersInternal)
        assert store.get(handle).children[0].status is ChildState.INDETERMINATE  # retained, not FAILED
        assert store.get(handle).children[0].provider_batch_id is None

    def test_a_match_created_outside_the_window_is_closed_out_but_flagged_as_possible_clock_skew(
        self, store, clock, caplog
    ):
        """A skewed clock makes a real, billed batch look off-window: the close-out must not read as certain."""

        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("skewed-ours", created_at=start + 60),))
        with caplog.at_level(logging.WARNING, logger="utils.batch_coordinator"):
            results = coordinator.reconcile_pending()

        assert [r.action for r in results] == [ReconcileAction.CLOSED_OUT]
        detail = store.get(handle).children[0].detail
        assert "clock is skewed" in detail and "skewed-ours" not in detail  # said plainly, no provider id to a caller
        assert "skewed-ours" in caplog.text  # the operator gets the candidate id in the log

    def test_a_close_out_with_no_off_window_candidate_carries_no_skew_note(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("unrelated", model="other/model", created_at=start + 60),))
        assert [r.action for r in coordinator.reconcile_pending()] == [ReconcileAction.CLOSED_OUT]
        assert "skewed" not in store.get(handle).children[0].detail

    def test_truncated_listing_that_reaches_the_window_still_allows_close_out(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(
            entries=(_entry("n1", model="other", created_at=start - 500),), may_be_truncated=True
        )
        assert [r.action for r in coordinator.reconcile_pending()] == [ReconcileAction.CLOSED_OUT]

    def test_competing_unresolved_intents_are_not_guessed_between(self, store, clock):
        provider = FakeProvider()
        coordinator, first, start = self._unresolved(store, clock, provider)
        clock.now = start  # second intent in the same window
        provider._submits.append(BatchProviderError("reset", code=UNKNOWN))
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])
        second = excinfo.value.handle
        clock.now = start + 600
        provider._listing = BatchListing(entries=(_entry("only-one", created_at=start + 1),))

        results = coordinator.reconcile_pending()
        assert {(r.handle, r.action) for r in results} == {
            (first, ReconcileAction.INDETERMINATE),
            (second, ReconcileAction.INDETERMINATE),
        }
        assert store.find_claim("openrouter", "only-one") is None

    def test_unreadable_listing_or_missing_provider_leaves_intent_unresolved(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, _ = self._unresolved(store, clock, provider)
        provider._listing = RuntimeError("503")
        assert [r.action for r in coordinator.reconcile_pending()] == [ReconcileAction.SKIPPED]
        assert store.get(handle).children[0].status is ChildState.SUBMITTING

        absent = _coordinator(store, {}, clock)
        assert [r.action for r in absent.reconcile_pending()] == [ReconcileAction.SKIPPED]
        assert store.get(handle).children[0].status is ChildState.SUBMITTING

    def test_a_young_intent_is_adopted_at_once_so_a_restart_right_after_a_crash_still_finds_it(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        clock.now = start + 5  # the server restarted seconds after the crash
        provider._listing = BatchListing(entries=(_entry("ours", created_at=start + 1),))

        results = coordinator.reconcile_pending()  # default (startup) parameters: no age guard
        assert [(r.action, r.provider_batch_id) for r in results] == [(ReconcileAction.ADOPTED, "ours")]
        assert store.get(handle).children[0].provider_batch_id == "ours"

    def test_a_young_intent_with_no_listed_batch_is_not_closed_out_yet(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        clock.now = start + 5
        provider._listing = BatchListing(entries=())

        assert [r.action for r in coordinator.reconcile_pending()] == [ReconcileAction.SKIPPED]
        assert store.get(handle).children[0].status is ChildState.SUBMITTING

    def test_a_submission_still_running_in_this_process_is_never_reconciled(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        seen: list[Any] = []

        def during_submit(model_name, requests):
            provider._listing = BatchListing(entries=())
            seen.append(coordinator.reconcile_pending())
            seen.append(coordinator.reconcile_handle(store.list_records()[0].handle))

        provider.on_submit = during_submit
        coordinator.submit("chat", [_plan(provider)])
        assert seen == [[], []]
        assert "list_batches" not in provider.calls

    def test_early_close_out_cannot_orphan_a_batch_the_provider_lists_late(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        provider._submits.append(BatchProviderError("timed out", code=UNKNOWN))
        start = clock.now
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])
        handle = excinfo.value.handle

        # Resume three seconds later: the list does not show the batch yet.
        clock.now = start + 3
        provider._listing = BatchListing(entries=())
        assert coordinator.open_for_resume(handle).record.children[0].status is ChildState.SUBMITTING

        # Twenty seconds on the provider's list catches up; the batch is still adoptable.
        clock.now = start + 23
        provider._listing = BatchListing(entries=(_entry("late-listed", created_at=start + 1),))
        results = coordinator.reconcile_pending()
        assert [(r.action, r.provider_batch_id) for r in results] == [(ReconcileAction.ADOPTED, "late-listed")]
        assert store.get(handle).state is RecordState.RUNNING

    def test_two_concurrent_reconcilers_never_turn_an_adopted_batch_indeterminate(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("ours", created_at=start + 1),))
        other = BatchCoordinator(store, provider_resolver=lambda name: provider, clock=clock)

        for trial in range(25):
            store.set_child_status(handle, 0, ChildState.SUBMITTING)
            store._connection.execute("UPDATE batch_children SET provider_batch_id = NULL, claimed = 0")
            barrier = threading.Barrier(2)
            outcomes: list[Any] = []

            def run(c: BatchCoordinator, gate: threading.Barrier = barrier, sink: list[Any] = outcomes) -> None:
                gate.wait()
                sink.append(c.reconcile_handle(handle))

            threads = [threading.Thread(target=run, args=(c,)) for c in (coordinator, other)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10)
            child = store.get(handle).children[0]
            assert child.status is ChildState.IN_PROGRESS and child.provider_batch_id == "ours", trial
            assert store.get(handle).state is RecordState.RUNNING

    def test_a_stale_reconciler_loses_the_compare_and_set_instead_of_clobbering(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("ours", created_at=start + 1),))

        real_list = provider.list_batches

        def list_then_lose_the_race(limit: int = 100) -> BatchListing:
            listing = real_list(limit)
            # Another process adopts the very same batch after we read the child and the list.
            store.record_provider_batch(handle, 0, "ours", status=ChildState.IN_PROGRESS)
            return listing

        provider.list_batches = list_then_lose_the_race  # type: ignore[method-assign]
        results = coordinator.reconcile_handle(handle)

        # The list is read before the reconcile lock is taken, so the other process's adoption lands before the
        # locked re-read: the child is seen as already resolved and left alone (nothing to report). Either way
        # nothing is adopted, closed out or marked indeterminate over it. The compare-and-set itself, after the
        # checks, is pinned by the next test.
        assert all(r.action is ReconcileAction.SKIPPED for r in results)
        child = store.get(handle).children[0]
        assert child.status is ChildState.IN_PROGRESS and child.provider_batch_id == "ours"

    def test_the_reconcile_lock_is_not_held_while_the_providers_batch_list_is_being_read(self, store, clock):
        """A reconciler stuck in a slow list must not make every other resume wait on the module-wide lock."""

        from utils import batch_coordinator

        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("ours", created_at=start + 1),))
        in_list = threading.Event()
        release = threading.Event()
        real_list = provider.list_batches
        lock_held_during_list: list[bool] = []

        def slow_list(limit: int = 100) -> BatchListing:
            lock_held_during_list.append(batch_coordinator._RECONCILE_LOCK.locked())
            in_list.set()
            release.wait(10)
            return real_list(limit)

        provider.list_batches = slow_list  # type: ignore[method-assign]
        results: list[Any] = []
        worker = threading.Thread(target=lambda: results.extend(coordinator.reconcile_handle(handle)))
        worker.start()
        try:
            assert in_list.wait(5)
            assert lock_held_during_list == [False]
            # While it is stuck, another reconciler can take the lock at once instead of queueing behind the list.
            acquired = batch_coordinator._RECONCILE_LOCK.acquire(timeout=0.5)
            assert acquired, "the reconcile lock was held across the provider's list call"
            batch_coordinator._RECONCILE_LOCK.release()
        finally:
            release.set()
            worker.join(10)

        assert [(r.action, r.provider_batch_id) for r in results] == [(ReconcileAction.ADOPTED, "ours")]

    def test_losing_the_race_after_the_checks_cannot_overwrite_an_adopted_child_with_the_adoption(
        self, store, clock, monkeypatch
    ):
        """The compare-and-set itself, not the earlier claim check: the child moves between check and write."""

        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("ours", created_at=start + 1),))
        real = coordinator._competing_intent

        def other_process_finishes_first(child, lo, hi):
            store.record_provider_batch(handle, 0, "ours", status=ChildState.COMPLETED)
            return real(child, lo, hi)

        monkeypatch.setattr(coordinator, "_competing_intent", other_process_finishes_first)
        results = coordinator.reconcile_handle(handle)

        assert [r.action for r in results] == [ReconcileAction.SKIPPED]
        assert store.get(handle).children[0].status is ChildState.COMPLETED  # not reset to in_progress

    def test_losing_the_race_cannot_overwrite_an_adopted_child_with_indeterminate(self, store, clock, monkeypatch):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("m1", created_at=start), _entry("m2", created_at=start + 2)))
        real = coordinator._classify
        raced: list[bool] = []

        def adopted_meanwhile(child, entry, lo, hi):
            if not raced:
                raced.append(True)
                store.record_provider_batch(handle, 0, "m1", status=ChildState.IN_PROGRESS)
            return real(child, entry, lo, hi)

        monkeypatch.setattr(coordinator, "_classify", adopted_meanwhile)
        results = coordinator.reconcile_handle(handle)

        assert [r.action for r in results] == [ReconcileAction.SKIPPED]
        child = store.get(handle).children[0]
        assert child.status is ChildState.IN_PROGRESS and child.provider_batch_id == "m1"
        assert store.get(handle).state is RecordState.RUNNING

    def test_reconciliation_never_submits_or_cancels(self, store, clock):
        provider = FakeProvider()
        coordinator, handle, start = self._unresolved(store, clock, provider)
        before = provider.calls.count("submit_batch")
        for listing in (
            BatchListing(entries=(_entry("m", created_at=start),)),
            BatchListing(entries=()),
            None,
        ):
            provider._listing = listing
            coordinator.reconcile_pending()
            coordinator.reconcile_handle(handle)
        assert provider.calls.count("submit_batch") == before
        assert "cancel_batch" not in provider.calls

    def test_adopting_into_an_abandoned_operation_unwinds_it(self, store, clock):
        provider = FakeProvider(
            "openrouter",
            submits=["or_1", BatchProviderError("reset", code=UNKNOWN)],
            cancel=CancelOutcome(CancelState.UNSUPPORTED),
        )
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        start = clock.now
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("consensus", [_plan(provider, MODEL), _plan(provider, MODEL, ("c", "d"))])
        assert excinfo.value.handle is None
        handle = excinfo.value.retained_handle
        clock.now += 600
        provider._listing = BatchListing(entries=(_entry("late", total=2, created_at=start + 1),))

        results = coordinator.reconcile_pending()
        assert [r.action for r in results] == [ReconcileAction.ADOPTED]
        record = store.get(handle)
        assert record.children[1].provider_batch_id == "late"
        assert record.children[1].status is ChildState.ORPHANED  # billed work of an abandoned operation
        assert record.state is RecordState.ORPHANED


# ----------------------------------------------------------------------
# 5.6 operator-inspectable indeterminate intent
# ----------------------------------------------------------------------


class TestReconciliationKeepsIdentifiersInternal:
    """What a reconciler stores as ``detail`` is shown to any caller who resumes the handle.

    It must name neither provider batch ids (they may belong to other batches on the account) nor
    another operation's handle (a bearer capability). The server log carries them for the operator.
    """

    @staticmethod
    def _unresolved_pair(store, clock, provider):
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        handles = []
        start = clock.now
        for _ in range(2):
            clock.now = start
            provider._submits.append(BatchProviderError("reset", provider="openrouter", code=UNKNOWN))
            with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
                coordinator.submit("chat", [_plan(provider)])
            handles.append(excinfo.value.handle)
        clock.now = start + 600
        return coordinator, handles, start

    @staticmethod
    def _rendered(store, handle):
        """Everything a resuming caller is shown for ``handle``: the status payload and the message."""

        from utils.batch_execution import (
            BatchRunResult,
            BatchRunStatus,
            ChildResult,
            build_status_payload,
            result_message,
        )

        record = store.get(handle)
        result = BatchRunResult(
            BatchRunStatus.INDETERMINATE, handle, record=record, children=tuple(ChildResult(c) for c in record.children)
        )
        return json.dumps(build_status_payload(result, tool_name="chat")) + result_message(result, tool_name="chat")

    def test_a_competing_intent_neither_leaks_its_handle_nor_the_listed_batch_id(self, store, clock, caplog):
        provider = FakeProvider()
        coordinator, (first, second), start = self._unresolved_pair(store, clock, provider)
        provider._listing = BatchListing(entries=(_entry("PROVIDER_ID_OF_SOMEONE_ELSE", created_at=start + 1),))
        with caplog.at_level(logging.WARNING, logger="utils.batch_coordinator"):
            results = coordinator.reconcile_pending()

        assert {r.action for r in results} == {ReconcileAction.INDETERMINATE}
        for handle, other in ((first, second), (second, first)):
            shown = self._rendered(store, handle) + store.get(handle).children[0].detail
            assert "PROVIDER_ID_OF_SOMEONE_ELSE" not in shown
            assert other not in shown
        assert "PROVIDER_ID_OF_SOMEONE_ELSE" in caplog.text  # still recoverable by the operator
        assert first in caplog.text and second in caplog.text

    def test_a_claimed_match_does_not_leak_the_claiming_handle_or_id(self, store, clock, caplog):
        provider = FakeProvider()
        coordinator, handle, start = TestReconciliation()._unresolved(store, clock, provider)
        other = store.create_record("chat", [bs.ChildIntent("openrouter", "x", "x", "/e", 1, ("z",))])
        store.record_provider_batch(other, 0, "taken-id")
        provider._listing = BatchListing(entries=(_entry("taken-id", created_at=start + 1),))
        with caplog.at_level(logging.WARNING, logger="utils.batch_coordinator"):
            coordinator.reconcile_pending()

        shown = self._rendered(store, handle)
        assert other not in shown and "taken-id" not in shown
        assert other in caplog.text and "taken-id" in caplog.text

    def test_several_matches_do_not_leak_any_of_their_ids(self, store, clock, caplog):
        provider = FakeProvider()
        coordinator, handle, start = TestReconciliation()._unresolved(store, clock, provider)
        provider._listing = BatchListing(
            entries=(_entry("cand-one", created_at=start + 1), _entry("cand-two", created_at=start + 2))
        )
        with caplog.at_level(logging.WARNING, logger="utils.batch_coordinator"):
            coordinator.reconcile_pending()

        shown = self._rendered(store, handle)
        assert "cand-one" not in shown and "cand-two" not in shown
        assert "2 listed batches match" in shown
        assert "cand-one" in caplog.text and "cand-two" in caplog.text


class TestIndeterminateIsRetained:
    def test_survives_reconciliation_and_is_reportable(self, store, clock, caplog):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        provider._submits.append(BatchProviderError("reset", code=UNKNOWN))
        start = clock.now
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])
        handle = excinfo.value.handle
        clock.now += 600
        provider._listing = BatchListing(entries=(_entry("a", created_at=start), _entry("b", created_at=start + 1)))

        with caplog.at_level(logging.WARNING):
            coordinator.reconcile_pending()
        assert any(handle in rec.getMessage() and "openrouter" in rec.getMessage() for rec in caplog.records)

        # A second pass neither discards nor re-decides it.
        provider._listing = BatchListing(entries=())
        assert coordinator.reconcile_pending() == []
        flagged = store.needs_attention()
        assert [r.handle for r in flagged] == [handle]
        assert flagged[0].children[0].status is ChildState.INDETERMINATE
        assert flagged[0].children[0].detail

        # Survives a restart of the store as well.
        path = store.db_path
        store.close()
        reopened = BatchStore(path, clock=clock)
        try:
            assert [r.handle for r in reopened.needs_attention()] == [handle]
        finally:
            reopened.close()

    def test_startup_warning_names_handle_and_provider_for_orphans_and_indeterminate(
        self, store, clock, caplog, monkeypatch
    ):
        provider = FakeProvider("openrouter", submits=["or_1"], cancel=CancelOutcome(CancelState.UNSUPPORTED))
        _fail_first_write(store, monkeypatch)
        with pytest.raises(BatchSubmissionError):
            _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider)])
        handle = store.list_records()[0].handle

        with caplog.at_level(logging.WARNING):
            reconcile_at_startup(store=store, provider_resolver=lambda name: provider)
        messages = [rec.getMessage() for rec in caplog.records]
        assert any(handle in m and "openrouter" in m and "or_1" in m for m in messages)


# ----------------------------------------------------------------------
# 5.7 resume reconciles before polling
# ----------------------------------------------------------------------


class TestResume:
    def test_resume_reconciles_unresolved_intent_before_anything_is_polled(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        provider._submits.append(BatchProviderError("reset", code=UNKNOWN))
        start = clock.now
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])
        handle = excinfo.value.handle
        provider._listing = BatchListing(entries=(_entry("adopted-on-resume", created_at=start + 1),))

        # No time has passed: adoption does not wait on any grace period.
        lookup = coordinator.open_for_resume(handle)

        assert provider.calls == ["submit_batch", "list_batches"]  # reconciled; nothing polled
        assert lookup.status is LookupStatus.FOUND
        child = lookup.record.children[0]
        assert child.provider_batch_id == "adopted-on-resume" and child.status is ChildState.IN_PROGRESS
        assert lookup.record.state is RecordState.RUNNING

    def test_resume_reports_a_closed_out_intent_as_failed(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        provider._submits.append(BatchProviderError("reset", code=UNKNOWN))
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])
        provider._listing = BatchListing(entries=())
        clock.now += 600  # long after the submission window, so the empty list is conclusive

        lookup = coordinator.open_for_resume(excinfo.value.handle)
        assert lookup.record.state is RecordState.FAILED

    def test_resume_heals_a_stale_record_state_even_when_no_child_is_submitting(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        handle = coordinator.submit("chat", [_plan(provider)])
        # A crash (or an older build) left the roll-up behind the child that already has its id.
        store.set_state(handle, RecordState.SUBMITTING)
        assert store.get(handle).state is RecordState.SUBMITTING
        assert store.get(handle).children[0].status is ChildState.VALIDATING
        provider.calls.clear()

        lookup = coordinator.open_for_resume(handle)

        assert lookup.record.state is RecordState.RUNNING
        assert store.get(handle).state is RecordState.RUNNING  # persisted, not just reported
        assert provider.calls == []  # nothing to reconcile, so the provider is not touched

    def test_resume_keeps_a_locally_cancelled_record_cancelled(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        handle = coordinator.submit("chat", [_plan(provider)])
        store.set_state(handle, RecordState.CANCELLED)
        assert coordinator.open_for_resume(handle).record.state is RecordState.CANCELLED

    def test_resume_of_a_resolved_unknown_or_expired_handle_never_touches_the_provider(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        handle = coordinator.submit("chat", [_plan(provider)])
        provider.calls.clear()

        assert coordinator.open_for_resume(handle).status is LookupStatus.FOUND
        assert coordinator.open_for_resume("batch_unknown").status is LookupStatus.UNKNOWN
        assert provider.calls == []


# ----------------------------------------------------------------------
# Startup wiring
# ----------------------------------------------------------------------


class TestStartup:
    def test_nothing_is_created_when_no_batch_store_exists(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert reconcile_at_startup() == []
        assert not os.path.exists(tmp_path / ".unison")

    def test_startup_reconciles_an_existing_default_store(self, monkeypatch, tmp_path, clock):
        monkeypatch.chdir(tmp_path)
        opened = bs.get_batch_store()
        provider = FakeProvider()
        coordinator = BatchCoordinator(opened, provider_resolver=lambda n: provider)
        provider._submits.append(BatchProviderError("reset", code=UNKNOWN))
        with pytest.raises(SubmissionOutcomeUnknown):
            coordinator.submit("chat", [_plan(provider)])
        provider._listing = BatchListing(entries=())

        results = reconcile_at_startup(provider_resolver=lambda n: provider, clock=lambda: time.time() + 600)
        assert [r.action for r in results] == [ReconcileAction.CLOSED_OUT]

    def test_startup_never_raises(self, monkeypatch, tmp_path):
        class Broken:
            def unresolved_intents(self, **kwargs):
                raise RuntimeError("db on fire")

        assert reconcile_at_startup(store=Broken()) == []  # type: ignore[arg-type]

    def test_server_starts_reconciliation_in_the_background_and_survives_its_failure(self, monkeypatch):
        import server
        import utils.batch_coordinator as coordinator_module

        done = threading.Event()
        monkeypatch.setattr(coordinator_module, "reconcile_at_startup", lambda: done.set())
        server._start_batch_reconciliation()
        assert done.wait(5)

        failed = threading.Event()

        def boom() -> None:
            failed.set()
            raise RuntimeError("provider exploded")

        monkeypatch.setattr(coordinator_module, "reconcile_at_startup", boom)
        server._start_batch_reconciliation()  # must not raise
        assert failed.wait(5)


# ----------------------------------------------------------------------
# The real OpenRouter adapter's HTTP answers, driven through the saga
# ----------------------------------------------------------------------


class TestRealAdapterThroughTheSaga:
    """A provider that may hold a billable batch must never be treated as a clean refusal."""

    def _provider(self, monkeypatch, response: httpx.Response | Exception):
        provider = OpenRouterProvider(api_key="test-key")
        calls: list[str] = []

        def send(method: str, path: str, content: str | None = None) -> httpx.Response:
            calls.append(method)
            if isinstance(response, Exception):
                raise response
            return response

        monkeypatch.setattr(provider, "_batch_send", send)
        return provider, calls

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(202, json={}),
            httpx.Response(202, json={"status": "validating"}),
            httpx.Response(202, content=b"<html>accepted</html>"),
            httpx.Response(500, json={"error": {"message": "boom"}}),
            httpx.Response(502, content=b"Bad Gateway"),
            httpx.Response(504, content=b"Gateway Timeout"),
            httpx.Response(408, content=b"Request Timeout"),
            httpx.Response(408, json={"error": {"message": "upstream timed out"}}),
            httpx.Response(425, content=b"Too Early"),
            httpx.Response(409, json={"error": {"message": "conflict"}}),
            httpx.Response(499, content=b"client closed request"),
            httpx.Response(307, headers={"location": "/elsewhere"}),
            httpx.ReadTimeout("slow"),
            httpx.ConnectError("reset"),
        ],
        ids=[
            "202-empty",
            "202-no-id",
            "202-html",
            "500",
            "502",
            "504",
            "408-proxy",
            "408-json",
            "425",
            "409",
            "499",
            "3xx",
            "read-timeout",
            "connect-error",
        ],
    )
    def test_possibly_accepted_submissions_keep_their_intent_and_are_never_retried(
        self, store, clock, monkeypatch, response
    ):
        provider, calls = self._provider(monkeypatch, response)
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])

        assert calls == ["POST"]
        error = excinfo.value
        assert error.handle is not None and "NOT retried" in str(error)
        record = store.get(error.handle)
        assert record is not None and record.children[0].status is ChildState.SUBMITTING

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 422, 429])
    def test_definite_refusals_are_clean_and_retain_nothing(self, store, clock, monkeypatch, status):
        provider, calls = self._provider(monkeypatch, httpx.Response(status, json={"error": {"message": "no"}}))
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(BatchSubmissionError) as excinfo:
            coordinator.submit("chat", [_plan(provider)])

        assert not isinstance(excinfo.value, SubmissionOutcomeUnknown)
        assert excinfo.value.handle is None and excinfo.value.orphans == ()
        assert store.list_records() == []

    @pytest.mark.parametrize("status", [408, 425])
    def test_a_408_or_425_on_create_is_typed_ambiguous_by_the_adapter_itself(self, monkeypatch, status):
        """The adapter, not just the coordinator, must not call a timeout a refusal (a retry could bill twice)."""

        provider, _ = self._provider(monkeypatch, httpx.Response(status, content=b"proxy says no"))
        with pytest.raises(BatchProviderError) as excinfo:
            provider.submit_batch(MODEL, _reqs("a", "b"))

        assert excinfo.value.code == UNKNOWN and excinfo.value.status_code == status
        assert excinfo.value.retryable is False

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 422])
    def test_a_definite_refusal_is_not_typed_ambiguous_by_the_adapter(self, monkeypatch, status):
        provider, _ = self._provider(monkeypatch, httpx.Response(status, json={"error": {"message": "no"}}))
        with pytest.raises(BatchProviderError) as excinfo:
            provider.submit_batch(MODEL, _reqs("a", "b"))

        assert excinfo.value.code != UNKNOWN and excinfo.value.status_code == status

    def test_a_408_on_a_multi_batch_submission_keeps_the_intent_and_unwinds_the_accepted_batch(
        self, store, clock, monkeypatch
    ):
        provider = OpenRouterProvider(api_key="test-key")
        responses = [httpx.Response(202, json={"id": "or_1", "status": "validating"}), httpx.Response(408)]
        sent: list[str] = []

        def send(method: str, path: str, content: str | None = None) -> httpx.Response:
            sent.append(f"{method} {path}")
            return (
                responses.pop(0)
                if method == "POST"
                else httpx.Response(200, json={"id": "or_1", "status": "in_progress"})
            )

        monkeypatch.setattr(provider, "_batch_send", send)
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("consensus", [_plan(provider, MODEL, ("a",)), _plan(provider, "openai/gpt-5", ("b",))])

        assert sent.count("POST /batches") == 2  # the 408 was not retried
        assert excinfo.value.handle is None and excinfo.value.retained_handle is not None
        record = store.get(excinfo.value.retained_handle)
        assert record is not None and record.children[1].status is ChildState.SUBMITTING  # intent retained

    def test_an_accepted_202_whose_batch_is_listed_later_is_adopted(self, store, clock, monkeypatch):
        provider, _ = self._provider(monkeypatch, httpx.Response(202, json={}))
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        start = clock.now
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])
        handle = excinfo.value.handle

        def listing(method: str, path: str, content: str | None = None) -> httpx.Response:
            assert (method, path.split("?")[0]) == ("GET", "/batches")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "or_real_1",
                            "status": "in_progress",
                            "model": MODEL,
                            "endpoint": "/v1/chat/completions",
                            "created_at": start + 1,
                            "request_counts": {"total": 2},
                        }
                    ]
                },
            )

        monkeypatch.setattr(provider, "_batch_send", listing)
        results = coordinator.reconcile_handle(handle)
        assert [(r.action, r.provider_batch_id) for r in results] == [(ReconcileAction.ADOPTED, "or_real_1")]


# ----------------------------------------------------------------------
# Store failures reach the caller as the typed submission error
# ----------------------------------------------------------------------


class TestStoreFailuresAreTyped:
    def test_a_locked_database_is_a_submission_error_and_nothing_is_submitted(self, store, clock):
        import sqlite3

        provider = FakeProvider()
        store._connection.execute("PRAGMA busy_timeout=0")
        blocker = sqlite3.connect(store.db_path, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(BatchSubmissionError, match="nothing was submitted"):
                _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider)])
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        assert provider.calls == []


# ----------------------------------------------------------------------
# Default configuration and production wiring
# ----------------------------------------------------------------------


class TestDefaultConfiguration:
    def test_submission_succeeds_with_no_storage_configuration_and_resumes_after_a_restart(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        for name in ("STORAGE_BACKEND", "STORAGE_SQLITE_PATH"):
            monkeypatch.delenv(name, raising=False)
        provider = FakeProvider()
        first = bs.get_batch_store()
        handle = BatchCoordinator(first, provider_resolver=lambda n: provider).submit(
            "chat", [_plan(provider)], payload={"prompt": "hi"}
        )
        assert first.db_path == os.path.join(str(tmp_path), ".unison", "batches.db")

        bs.reset_batch_store()  # the process "restarts": nothing in memory survives
        second = bs.get_batch_store()
        assert second is not first
        lookup = BatchCoordinator(second, provider_resolver=lambda n: provider).open_for_resume(handle)
        assert lookup.status is LookupStatus.FOUND
        assert lookup.record.payload == {"prompt": "hi"}
        assert lookup.record.children[0].provider_batch_id == "prov_openrouter_1"

    def test_submission_consults_the_stores_own_durability_property(self, store, clock, monkeypatch):
        provider = FakeProvider()
        monkeypatch.setattr(type(store), "is_durable", property(lambda self: False))
        with pytest.raises(BatchSubmissionError, match="durable"):
            _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider)])
        assert provider.calls == []

    def test_a_volatile_store_path_cannot_reach_a_submission(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BATCH_STORE_PATH", ":memory:")
        with pytest.raises(bs.BatchStoreUnavailable, match="in-memory"):
            bs.get_batch_store()
        assert list(tmp_path.iterdir()) == []  # nothing was chmod-ed or written into the working directory

    def test_registry_resolver_reaches_the_configured_provider_and_its_list_operation(self, monkeypatch):
        from providers.registry import ModelProviderRegistry, get_default_registry, set_default_registry
        from providers.shared import ProviderType
        from utils.batch_coordinator import registry_provider_resolver

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        previous = get_default_registry()
        registry = ModelProviderRegistry(config={"OPENROUTER_API_KEY": "test-key"})
        registry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        set_default_registry(registry)
        try:
            resolved = registry_provider_resolver("openrouter")
            assert isinstance(resolved, OpenRouterProvider)
            monkeypatch.setattr(
                resolved,
                "_batch_send",
                lambda *a, **k: httpx.Response(200, json={"data": [{"id": "x1", "status": "completed"}]}),
            )
            assert [e.batch_id for e in resolved.list_batches().entries] == ["x1"]
            assert registry_provider_resolver("anthropic") is None  # not registered
            assert registry_provider_resolver("no-such-provider") is None
        finally:
            set_default_registry(previous)

    def test_startup_reconciliation_resolves_providers_through_the_default_registry(self, monkeypatch, tmp_path):
        from providers.registry import ModelProviderRegistry, get_default_registry, set_default_registry
        from providers.shared import ProviderType

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        previous = get_default_registry()
        registry = ModelProviderRegistry(config={"OPENROUTER_API_KEY": "test-key"})
        registry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        set_default_registry(registry)
        try:
            opened = bs.get_batch_store()
            provider = registry.get_provider(ProviderType.OPENROUTER)
            sent: list[str] = []

            def send(method: str, path: str, content: str | None = None) -> httpx.Response:
                sent.append(f"{method} {path.split('?')[0]}")
                if method == "POST":
                    raise httpx.ReadTimeout("slow")
                return httpx.Response(200, json={"data": []})

            monkeypatch.setattr(provider, "_batch_send", send)
            coordinator = BatchCoordinator(opened)  # production resolver: the default registry
            with pytest.raises(SubmissionOutcomeUnknown):
                coordinator.submit("chat", [_plan(provider)])

            results = reconcile_at_startup(clock=lambda: time.time() + 600)  # also the production resolver
            assert [r.action for r in results] == [ReconcileAction.CLOSED_OUT]
            assert sent == ["POST /batches", "GET /batches"]  # one submission attempt, then only a list
        finally:
            set_default_registry(previous)


# ----------------------------------------------------------------------
# Bounded provider calls (a hung submit or list cannot hold a call open)
# ----------------------------------------------------------------------


class TestBoundedProviderCalls:
    def test_call_within_returns_the_value_and_reraises_the_error(self):
        from utils.batch_coordinator import call_within

        assert call_within(lambda: 7, 1.0) == 7
        with pytest.raises(KeyError):
            call_within(lambda: {}["missing"], 1.0)

    def test_a_call_that_overruns_is_abandoned_and_its_late_value_reaches_the_hook(self):
        from utils.batch_coordinator import CallTimedOut, call_within

        release = threading.Event()
        late: list[Any] = []
        arrived = threading.Event()

        def hook(value: Any) -> None:
            late.append(value)
            arrived.set()

        started = time.monotonic()
        with pytest.raises(CallTimedOut):
            call_within(lambda: (release.wait(30), "late")[1], 0.1, on_late_result=hook)
        assert time.monotonic() - started < 3.0
        assert late == []
        release.set()
        assert arrived.wait(5) and late == ["late"]

    def test_a_call_that_finishes_in_time_never_reaches_the_late_hook(self):
        from utils.batch_coordinator import call_within

        late: list[Any] = []
        assert call_within(lambda: "ok", 1.0, on_late_result=late.append) == "ok"
        time.sleep(0.05)
        assert late == []

    def test_no_time_left_is_a_timeout_without_calling_the_provider(self):
        from utils.batch_coordinator import CallTimedOut, call_within

        called: list[int] = []
        with pytest.raises(CallTimedOut):
            call_within(lambda: called.append(1), 0)
        assert called == []

    def test_a_submit_that_never_answers_is_ambiguous_not_a_refusal(self, store, clock):
        provider = FakeProvider()
        release = threading.Event()
        provider.on_submit = lambda _model, _requests: release.wait(30)
        coordinator = BatchCoordinator(
            store, provider_resolver=lambda name: provider, clock=clock, call_timeout_seconds=0.1
        )
        try:
            started = time.monotonic()
            with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
                coordinator.submit("chat", [_plan(provider)])
            assert time.monotonic() - started < 3.0
        finally:
            release.set()
        assert excinfo.value.handle is not None and "did not answer" in str(excinfo.value)
        assert store.get(excinfo.value.handle).children[0].status is ChildState.SUBMITTING

    def test_a_late_acceptance_of_an_abandoned_multi_batch_operation_is_recorded_orphaned(self, store, clock):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        handle = store.create_record(
            "consensus",
            [
                bs.ChildIntent("openrouter", MODEL, MODEL, "/v1/chat/completions", 1, ("a",)),
                bs.ChildIntent("openrouter", "m/two", "m/two", "/v1/chat/completions", 1, ("b",)),
            ],
        )
        store.mark_unwound(handle)

        coordinator._record_late_acceptance(handle, 0, _plan(provider, MODEL, ("a",)), "late_batch")

        child = store.get(handle).children[0]
        assert child.provider_batch_id == "late_batch" and child.status is ChildState.ORPHANED
        assert "abandoned" in (child.detail or "")

    def test_a_late_acceptance_that_cannot_be_recorded_is_logged_loudly_not_raised(self, store, clock, caplog):
        provider = FakeProvider()
        coordinator = _coordinator(store, {"openrouter": provider}, clock)
        with caplog.at_level(logging.ERROR):
            coordinator._record_late_acceptance("batch_missing", 0, _plan(provider), "late_batch")

        assert any("late_batch" in record.getMessage() and "billed" in record.getMessage() for record in caplog.records)

    def test_an_unreadable_batch_list_leaves_the_intent_unresolved(self, store, clock):
        provider = FakeProvider(submits=[BatchProviderError("reset", provider="openrouter", code=UNKNOWN)])
        release = threading.Event()
        provider.list_batches = lambda limit=100: (release.wait(30), None)[1]
        coordinator = BatchCoordinator(
            store, provider_resolver=lambda name: provider, clock=clock, call_timeout_seconds=0.1
        )
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            coordinator.submit("chat", [_plan(provider)])
        try:
            started = time.monotonic()
            results = coordinator.reconcile_handle(excinfo.value.handle)
            assert time.monotonic() - started < 3.0
        finally:
            release.set()

        assert [r.action for r in results] == [ReconcileAction.SKIPPED]
        assert store.get(excinfo.value.handle).children[0].status is ChildState.SUBMITTING
