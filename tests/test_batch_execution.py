"""Tests for the bounded wait, resume and cancel of batches (add-batch-model-support, section 6).

No network, no API keys: providers are in-process fakes over a shared ``FakeRemote``, the clock
and sleep are injected, and the durable store lives in a temporary directory.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx
import pytest

import utils.model_restrictions as model_restrictions
from providers.openrouter import OpenRouterProvider
from providers.shared import (
    BatchError,
    BatchFailed,
    BatchIncompleteResults,
    BatchListEntry,
    BatchListing,
    BatchNotFound,
    BatchNotReady,
    BatchProviderError,
    BatchRateLimited,
    BatchResultsExpired,
    BatchState,
    CancelOutcome,
    CancelState,
)
from tests.batch_test_helpers import MODEL, FakeClock, FakeProvider, FakeRemote, RemoteBatch, make_executor, make_plan
from utils import batch_store as bs
from utils.batch_coordinator import BatchCoordinator, BatchSubmissionError, SubmissionOutcomeUnknown
from utils.batch_execution import (
    BatchExecutor,
    BatchHandleToolMismatch,
    BatchRunStatus,
    WaitPolicy,
    build_status_payload,
    describe_submission_error,
    result_message,
)
from utils.batch_store import BatchStore, ChildIntent, ChildState, RecordState

UNKNOWN = "submission_outcome_unknown"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("BATCH_STORE_PATH", "BATCH_RETENTION_DAYS"):
        monkeypatch.delenv(name, raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


@pytest.fixture
def store(tmp_path):
    instance = BatchStore(str(tmp_path / "s" / "batches.db"))
    yield instance
    instance.close()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def remote():
    return FakeRemote()


@pytest.fixture
def provider(remote):
    return FakeProvider(remote)


@pytest.fixture
def executor(store, provider, clock):
    return make_executor(store, {"openrouter": provider}, clock)


# ----------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------


class TestWaitPolicy:
    def test_default_and_clamp_stay_inside_host_deadlines(self):
        policy = WaitPolicy()
        assert policy.budget(None) == 45.0
        assert policy.budget(10) == 10.0
        assert policy.budget(10_000) == 50.0  # clamped, never rejected
        assert policy.budget(0) == 0.0
        assert policy.budget(-5) == 0.0
        assert policy.max_seconds < 60.0  # Claude Desktop's local-agent mode cancels near a minute

    @pytest.mark.parametrize("junk", ["soon", float("nan"), True, object()])
    def test_unusable_values_fall_back_to_the_default(self, junk):
        assert WaitPolicy().budget(junk) == 45.0

    def test_delays_back_off_then_repeat_the_last(self):
        policy = WaitPolicy()
        assert [policy.delay(i) for i in range(8)] == [1.0, 2.0, 4.0, 8.0, 15.0, 15.0, 15.0, 15.0]


# ----------------------------------------------------------------------
# 6.5 / 6.6 submit and bounded wait
# ----------------------------------------------------------------------


class TestSubmitAndWait:
    @staticmethod
    def _finish_on_submit(provider):
        """Make the fake provider finish every batch the moment it is submitted."""

        original = provider.submit_batch

        def submit_and_finish(model, requests):
            batch_id = original(model, requests)
            provider.remote.finish(batch_id)
            return batch_id

        provider.submit_batch = submit_and_finish

    def test_fast_batch_returns_its_result_inline(self, executor, provider, clock):
        self._finish_on_submit(provider)
        result = executor.submit_and_wait("chat", [make_plan(provider)], payload={"prompt": "hi"})

        assert result.status is BatchRunStatus.COMPLETED
        assert result.is_terminal
        outcome = result.outcomes_by_custom_id()["chat"]
        assert outcome.success and outcome.response.content == "answer:chat"
        # The response model is the one recorded at submission, not the adapter's own guess.
        assert outcome.response.model_name == MODEL
        assert provider.cancelled == []

    def test_subsecond_completion_is_detected_at_the_first_check_not_at_the_end(self, executor, provider, clock):
        self._finish_on_submit(provider)
        result = executor.submit_and_wait("chat", [make_plan(provider)])

        assert result.status is BatchRunStatus.COMPLETED
        assert result.polls == 1
        assert clock.sleeps == [1.0]  # one prompt check
        assert result.waited_seconds == pytest.approx(1.0)
        assert result.waited_seconds < result.wait_budget_seconds

    def test_slow_batch_returns_a_handle_and_never_cancels(self, executor, provider, clock):
        result = executor.submit_and_wait("chat", [make_plan(provider)], payload={"prompt": "hi"})

        assert result.status is BatchRunStatus.RUNNING
        assert not result.is_terminal
        assert result.handle.startswith("batch_")
        assert provider.cancelled == []  # a slow batch is never cancelled
        assert result.waited_seconds == pytest.approx(45.0)
        message = result_message(result, tool_name="chat")
        assert "NOT finished" in message and f"batch_handle='{result.handle}'" in message
        assert "still running" in message
        payload = build_status_payload(result, tool_name="chat")
        assert payload["status"] == "running" and payload["resume_with"] == {"batch_handle": result.handle}
        # Provider identifiers stay internal.
        assert "prov_batch_1" not in str(payload)

    def test_wait_seconds_zero_returns_the_handle_without_polling(self, executor, provider, clock):
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0)
        assert result.status is BatchRunStatus.RUNNING
        assert result.polls == 0 and clock.sleeps == []

    def test_caller_wait_is_honoured_and_clamped(self, executor, provider, clock):
        short = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=5)
        assert short.waited_seconds == pytest.approx(5.0)
        clock.sleeps.clear()
        long_ = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=10_000)
        assert long_.wait_budget_seconds == 50.0
        assert long_.waited_seconds <= 50.0 + 1e-6

    def test_full_length_wait_polls_a_bounded_number_of_times_with_backoff(self, executor, provider, clock):
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=10_000)

        assert result.status is BatchRunStatus.RUNNING
        assert result.polls == provider.calls.count("poll_batch")
        assert result.polls <= WaitPolicy().max_polls
        assert clock.sleeps[:5] == [1.0, 2.0, 4.0, 8.0, 15.0]  # increases from its initial value
        assert sum(clock.sleeps) <= 50.0 + 1e-6

    def test_poll_rounds_are_hard_capped_whatever_the_budget(self, store, provider, clock):
        capped = make_executor(
            store, {"openrouter": provider}, clock, policy=WaitPolicy(poll_delays=(0.1,), max_polls=4)
        )
        result = capped.submit_and_wait("chat", [make_plan(provider)], wait_seconds=50)
        assert result.polls == 4
        assert result.status is BatchRunStatus.RUNNING

    def test_rate_limited_poll_is_absorbed_and_retry_after_is_honoured(self, executor, provider, clock):
        original = provider.submit_batch

        def submit(model, requests):
            batch_id = original(model, requests)
            provider.remote.batches[batch_id].script = [BatchState.IN_PROGRESS, BatchState.COMPLETED]
            provider.remote.batches[batch_id].results = {"chat": "done"}
            return batch_id

        provider.submit_batch = submit
        provider.poll_errors = [BatchRateLimited("429", retry_after=5.0)]
        result = executor.submit_and_wait("chat", [make_plan(provider)])

        assert result.status is BatchRunStatus.COMPLETED  # the call did not fail
        assert clock.sleeps[0] == 1.0
        assert clock.sleeps[1] == 5.0  # retry_after lengthened the next delay

    def test_retry_after_that_cannot_fit_ends_the_wait_with_a_handle(self, executor, provider, clock):
        provider.poll_errors = [BatchRateLimited("429", retry_after=60.0)]
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=3)

        assert result.status is BatchRunStatus.RUNNING
        assert result.polls == 1  # no point polling again inside the window
        assert result.handle

    def test_retryable_provider_errors_are_absorbed(self, executor, provider, clock):
        provider.poll_errors = [
            BatchProviderError("bad gateway", status_code=502, retryable=True),
            BatchProviderError("transport", retryable=True),
        ]
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=20)
        assert result.status is BatchRunStatus.RUNNING
        assert result.polls >= 3  # kept polling after both failures

    def test_a_non_retryable_poll_error_stops_polling_that_child_and_is_reported(self, executor, provider, clock):
        provider.poll_errors = [BatchProviderError("invalid api key", status_code=401)]
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=30)

        assert result.status is BatchRunStatus.RUNNING
        assert result.polls == 1
        assert "invalid api key" in (result.children[0].poll_problem or "")
        assert "invalid api key" in result_message(result, tool_name="chat")

    def test_provider_forgetting_the_batch_marks_the_child_failed(self, executor, provider, clock):
        provider.poll_errors = [BatchNotFound("no such batch", status_code=404)]
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=30)
        assert result.status is BatchRunStatus.FAILED
        assert "no longer knows" in result_message(result, tool_name="chat")

    def test_provider_status_cancelling_is_followed_until_it_becomes_cancelled(self, executor, provider, remote):
        """``cancelling`` (documented by OpenRouter) is in flight, not an unknown status: polling goes on."""

        handle = _start(executor, provider, make_plan(provider))
        remote.batches["prov_batch_1"].script = [BatchState.CANCELLING, BatchState.CANCELLING, BatchState.CANCELLED]

        first = executor.resume(handle, wait_seconds=0)
        assert first.status is BatchRunStatus.RUNNING
        assert first.children[0].child.status is ChildState.CANCELLING
        assert first.children[0].poll_problem is None
        assert "unrecognised" not in result_message(first, tool_name="chat")

        executor.resume(handle, wait_seconds=0)
        last = executor.resume(handle, wait_seconds=0)
        assert last.status is BatchRunStatus.CANCELLED
        assert store_child_status(executor, handle) is ChildState.CANCELLED

    def test_a_poll_answering_410_is_reported_as_results_expired_not_running_forever(self, executor, provider, remote):
        """HTTP 410 on a poll means the provider deleted the batch with its results (30-day retention)."""

        handle = _start(executor, provider, make_plan(provider))
        provider.poll_errors = [BatchResultsExpired("gone", status_code=410)]
        remote.batches["prov_batch_1"].fetch_error = BatchResultsExpired("gone", status_code=410)

        result = executor.resume(handle, wait_seconds=0)

        assert result.status is BatchRunStatus.RESULTS_EXPIRED
        assert result.children[0].results_expired
        assert "still running" not in result_message(result, tool_name="chat")
        assert "expired at the provider" in result_message(result, tool_name="chat")
        assert remote.submissions == 1  # nothing resubmitted
        again = executor.resume(handle, wait_seconds=0)
        assert again.status is BatchRunStatus.RESULTS_EXPIRED  # and it stays that way

    def test_submission_errors_propagate_and_name_the_handle_only_when_resumable(self, executor, provider):
        provider.submit_error = BatchProviderError("reset", provider="openrouter", code=UNKNOWN)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait("chat", [make_plan(provider)])
        text = describe_submission_error(excinfo.value)
        assert excinfo.value.handle in text and "Do not resubmit" in text

        provider.submit_error = BatchProviderError("quota", status_code=429, provider="openrouter")
        with pytest.raises(BatchSubmissionError) as clean:
            executor.submit_and_wait("chat", [make_plan(provider)])
        assert clean.value.handle is None
        assert "batch_handle" not in describe_submission_error(clean.value)


# ----------------------------------------------------------------------
# 6.7 resume
# ----------------------------------------------------------------------


def store_child_status(executor, handle, index=0):
    return executor.store.get(handle).children[index].status


def _start(executor, provider, *plans, wait=0, **kwargs):
    result = executor.submit_and_wait(
        "consensus" if len(plans) > 1 else "chat", list(plans), wait_seconds=wait, **kwargs
    )
    return result.handle


class TestResume:
    def test_terminal_handle_returns_results_and_repeating_it_submits_nothing(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider, ids=("a", "b")))
        remote.finish("prov_batch_1")

        first = executor.resume(handle)
        second = executor.resume(handle)
        third = executor.resume(handle, wait_seconds=0)

        assert first.status is BatchRunStatus.COMPLETED
        contents = {cid: o.response.content for cid, o in first.outcomes_by_custom_id().items()}
        assert contents == {"a": "answer:a", "b": "answer:b"}  # correlated by id though returned reversed
        for repeat in (second, third):
            assert {c: o.response.content for c, o in repeat.outcomes_by_custom_id().items()} == contents
        assert remote.submissions == 1
        assert provider.calls.count("submit_batch") == 1  # nothing was billed twice
        assert second.polls == 0  # the stored terminal state is not polled again

    def test_running_handle_waits_again_and_returns_the_same_handle(self, executor, provider, clock):
        handle = _start(executor, provider, make_plan(provider))
        before = provider.calls.count("poll_batch")

        again = executor.resume(handle, wait_seconds=10)

        assert again.status is BatchRunStatus.RUNNING
        assert again.handle == handle
        assert provider.calls.count("poll_batch") > before
        assert again.waited_seconds == pytest.approx(10.0)

    def test_resume_checks_immediately_before_any_delay(self, executor, provider, remote, clock):
        handle = _start(executor, provider, make_plan(provider))
        remote.finish("prov_batch_1")
        clock.sleeps.clear()

        result = executor.resume(handle)

        assert result.status is BatchRunStatus.COMPLETED
        assert clock.sleeps == []  # a long-finished batch costs no waiting at all

    def test_fetch_receives_the_stored_custom_ids_and_the_recorded_model(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider, model="m/x:batch", ids=("a", "b")))
        remote.finish("prov_batch_1")

        result = executor.resume(handle)

        assert provider.fetch_expected_ids[-1] == ("a", "b")
        response = result.outcomes_by_custom_id()["a"].response
        assert response.model_name == "m/x"  # the slug recorded at submission
        assert response.metadata["requested_model"] == "m/x:batch"  # the caller's own spelling
        assert result.child_for_custom_id("b").requested_model == "m/x:batch"

    def test_unknown_and_expired_handles_are_reported_distinctly(self, tmp_path, clock, provider):
        wall = [10_000_000.0]
        aged = BatchStore(str(tmp_path / "aged" / "batches.db"), retention_days=1, clock=lambda: wall[0])
        executor = make_executor(aged, {"openrouter": provider}, clock)
        handle = _start(executor, provider, make_plan(provider))
        wall[0] += 2 * 86400

        expired = executor.resume(handle)
        unknown = executor.resume("batch_never_issued")

        assert expired.status is BatchRunStatus.HANDLE_EXPIRED and expired.expired_at is not None
        assert unknown.status is BatchRunStatus.UNKNOWN_HANDLE
        expired_text = result_message(expired, tool_name="chat")
        unknown_text = result_message(unknown, tool_name="chat")
        assert "expired" in expired_text and "Unknown" not in expired_text
        assert "Unknown batch handle" in unknown_text and "expired" not in unknown_text.replace("not an expired", "")
        aged.close()

    def test_results_past_the_providers_retention_are_reported_expired(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider))
        remote.finish("prov_batch_1")
        remote.batches["prov_batch_1"].fetch_error = BatchResultsExpired("gone", status_code=410)

        result = executor.resume(handle)

        assert result.status is BatchRunStatus.RESULTS_EXPIRED
        assert result.children[0].results_expired
        assert "expired at the provider" in result_message(result, tool_name="chat")
        assert remote.submissions == 1

    def test_a_batch_that_expired_unprocessed_is_reported_and_not_resubmitted(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider))
        remote.batches["prov_batch_1"].script = [BatchState.EXPIRED]
        remote.batches["prov_batch_1"].fetch_error = BatchFailed("window lapsed", state=BatchState.EXPIRED)

        result = executor.resume(handle)

        assert result.status is BatchRunStatus.EXPIRED
        assert "not resubmitted" in result_message(result, tool_name="chat")
        assert remote.submissions == 1

    def test_failed_child_is_reported_alongside_a_successful_one(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider, "m/one", ("a",)), make_plan(provider, "m/two", ("b",)))
        remote.finish("prov_batch_1")
        remote.batches["prov_batch_2"].script = [BatchState.FAILED]
        remote.batches["prov_batch_2"].fetch_error = BatchFailed("model overloaded", state=BatchState.FAILED)

        result = executor.resume(handle)

        assert result.status is BatchRunStatus.COMPLETED
        assert set(result.outcomes_by_custom_id()) == {"a"}  # the successful results are still returned
        failed = [c for c in result.children if c.error]
        assert len(failed) == 1 and "model overloaded" in failed[0].error

    def test_operation_is_running_while_any_child_batch_is_non_terminal(self, executor, provider, remote):
        handle = _start(
            executor,
            provider,
            make_plan(provider, "m/one", ("a",)),
            make_plan(provider, "m/two", ("b",)),
            make_plan(provider, "m/three", ("c",)),
        )
        remote.finish("prov_batch_1")
        remote.finish("prov_batch_2")  # the third is still processing

        result = executor.resume(handle, wait_seconds=0)

        assert result.status is BatchRunStatus.RUNNING
        assert provider.calls.count("fetch_batch") == 0  # nothing to collect until all are terminal
        assert [c.child.status for c in result.children][:2] == [ChildState.COMPLETED, ChildState.COMPLETED]

    def test_results_that_cannot_be_fetched_yet_keep_the_handle_resumable(self, executor, provider, remote, clock):
        handle = _start(executor, provider, make_plan(provider))
        remote.finish("prov_batch_1")
        remote.batches["prov_batch_1"].fetch_error = BatchRateLimited("slow down")

        deferred = executor.resume(handle)
        assert deferred.status is BatchRunStatus.RUNNING
        assert "could not be fetched" in result_message(deferred, tool_name="chat")

        remote.batches["prov_batch_1"].fetch_error = None
        assert executor.resume(handle).status is BatchRunStatus.COMPLETED

    def test_a_handle_belongs_to_the_tool_that_issued_it(self, executor, provider):
        handle = _start(executor, provider, make_plan(provider))
        with pytest.raises(BatchHandleToolMismatch, match="issued by the 'chat' tool"):
            executor.resume(handle, tool_name="consensus")
        with pytest.raises(BatchHandleToolMismatch):
            executor.cancel(handle, tool_name="consensus")
        assert provider.cancelled == []


class TestResumeReconcilesFirst:
    """5.7: unresolved intent is reconciled on resume, before anything is polled."""

    def test_unresolved_intent_triggers_reconciliation_before_polling(self, store, provider, remote, clock):
        executor = make_executor(store, {"openrouter": provider}, clock)
        provider.submit_error = BatchProviderError("connection reset", provider="openrouter", code=UNKNOWN)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait("chat", [make_plan(provider)])
        handle = excinfo.value.handle
        child = store.get(handle).children[0]
        assert child.status is ChildState.SUBMITTING and child.provider_batch_id is None

        # The provider did create a batch the server never recorded; it shows up in the list.
        remote.batches["or_late"] = RemoteBatch(model=MODEL, custom_ids=["chat"], script=[BatchState.COMPLETED])
        remote.batches["or_late"].results = {"chat": "late answer"}
        provider.listing = BatchListing(
            entries=(
                BatchListEntry(
                    batch_id="or_late",
                    state=BatchState.IN_PROGRESS,
                    model=child.submitted_model,
                    endpoint=child.endpoint,
                    created_at=child.intent_window_start + 1,
                    total=1,
                ),
            )
        )
        provider.submit_error = None
        provider.calls.clear()

        result = executor.resume(handle)

        assert provider.calls.index("list_batches") < provider.calls.index("poll_batch")
        assert "submit_batch" not in provider.calls  # reconciliation never resubmits
        assert result.status is BatchRunStatus.COMPLETED
        assert result.outcomes_by_custom_id()["chat"].response.content == "late answer"

    def test_an_unconfirmed_submission_is_not_reported_as_running_at_the_provider(self, store, provider, clock):
        """With the provider's list unreadable nothing is known: do not claim a batch is running."""

        executor = make_executor(store, {"openrouter": provider}, clock)
        provider.submit_error = BatchProviderError("connection reset", provider="openrouter", code=UNKNOWN)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait("chat", [make_plan(provider)])
        handle = excinfo.value.handle

        def unreadable():
            raise BatchProviderError("unauthorised", status_code=401)

        provider.list_batches = unreadable  # type: ignore[method-assign]
        provider.submit_error = None
        result = executor.resume(handle, wait_seconds=0)

        assert result.status is BatchRunStatus.RUNNING
        assert result.children[0].child.status is ChildState.SUBMITTING
        message = result_message(result, tool_name="chat")
        assert "still running at the provider" not in message
        assert "NOT confirmed" in message and "may or may not hold a batch" in message
        assert "Do not resubmit" in message and handle in message
        payload = build_status_payload(result, tool_name="chat")
        assert "not confirmed yet" in payload["children"][0]["note"]
        assert "submit_batch" not in provider.calls[1:]  # never resubmitted

    def test_undecidable_intent_is_reported_not_polled(self, store, provider, clock):
        executor = make_executor(store, {"openrouter": provider}, clock)
        provider.submit_error = BatchProviderError("connection reset", provider="openrouter", code=UNKNOWN)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait("chat", [make_plan(provider)])
        provider.calls.clear()  # provider has no list operation: the intent stays unresolved

        result = executor.resume(excinfo.value.handle)

        assert result.status is BatchRunStatus.INDETERMINATE
        assert store.get(excinfo.value.handle).state is RecordState.INDETERMINATE
        assert "poll_batch" not in provider.calls
        assert "submit_batch" not in provider.calls


# ----------------------------------------------------------------------
# 6.8 cancel
# ----------------------------------------------------------------------


class TestCancel:
    def test_running_batch_is_cancelled_where_the_provider_supports_it(self, store, remote, clock):
        provider = FakeProvider(remote, cancel=CancelOutcome(CancelState.CANCELLED))
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = _start(executor, provider, make_plan(provider))

        result = executor.cancel(handle, tool_name="chat")

        assert provider.cancelled == ["prov_batch_1"]
        assert result.status is BatchRunStatus.CANCELLED
        assert store.get(handle).state is RecordState.CANCELLED
        assert result.cancel_reports[0].action == "cancelled" and not result.cancel_reports[0].batch_continues
        assert "continues" not in result_message(result, tool_name="chat")

    def test_cancel_without_a_cancel_mechanism_is_local_and_says_the_batch_continues(self, executor, provider, store):
        handle = _start(executor, provider, make_plan(provider))

        result = executor.cancel(handle)

        assert result.status is BatchRunStatus.CANCELLED
        record = store.get(handle)
        assert record.state is RecordState.CANCELLED and record.children[0].status is ChildState.CANCELLED
        report = result.cancel_reports[0]
        assert report.action == "continues" and report.batch_continues
        message = result_message(result, tool_name="chat")
        assert "continues and will still be billed" in message
        assert "NOT stopped" in message
        for claim in ("was stopped", "has been stopped", "cancelled at provider"):
            assert claim not in message  # never claims the provider batch was stopped
        assert build_status_payload(result, tool_name="chat")["provider_batches_continue"] is True

    def test_cancelled_record_stops_polling_and_ignores_results(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider))
        executor.cancel(handle)
        remote.finish("prov_batch_1")
        provider.calls.clear()

        result = executor.resume(handle)

        assert result.status is BatchRunStatus.CANCELLED
        assert provider.calls == []  # no poll, no fetch
        assert result.outcomes_by_custom_id() == {}
        assert "continues" in result_message(result, tool_name="chat")  # the stored detail is repeated

    def test_multi_batch_cancel_reaches_every_child(self, store, remote, clock):
        provider = FakeProvider(remote, cancel=CancelOutcome(CancelState.CANCELLED))
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = _start(
            executor,
            provider,
            make_plan(provider, "m/one", ("a",)),
            make_plan(provider, "m/two", ("b",)),
            make_plan(provider, "m/three", ("c",)),
        )

        result = executor.cancel(handle)

        assert provider.cancelled == ["prov_batch_1", "prov_batch_2", "prov_batch_3"]
        assert result.status is BatchRunStatus.CANCELLED
        assert [r.state_after for r in result.cancel_reports] == [ChildState.CANCELLED] * 3
        assert store.get(handle).state is RecordState.CANCELLED

    def test_cancelling_a_finished_batch_reports_its_state_without_raising(self, executor, provider, remote, store):
        handle = _start(executor, provider, make_plan(provider))
        remote.finish("prov_batch_1")
        executor.resume(handle)  # records the terminal state
        provider.calls.clear()

        result = executor.cancel(handle)

        assert result.status is BatchRunStatus.COMPLETED
        assert provider.cancelled == [] and result.cancel_reports == ()
        assert store.get(handle).state is RecordState.COMPLETED

    def test_cancelling_twice_reports_the_existing_state(self, executor, provider):
        handle = _start(executor, provider, make_plan(provider))
        executor.cancel(handle)
        provider.calls.clear()

        again = executor.cancel(handle)

        assert again.status is BatchRunStatus.CANCELLED
        assert provider.calls == []

    def test_a_cancel_the_provider_reports_as_already_finished_is_not_claimed_as_cancelled(self, store, remote, clock):
        provider = FakeProvider(remote, cancel=CancelOutcome(CancelState.ALREADY_TERMINAL))
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = _start(executor, provider, make_plan(provider))
        remote.finish("prov_batch_1")  # finished just as the caller cancelled

        result = executor.cancel(handle)

        assert result.cancel_reports[0].action == "already_terminal"
        assert store.get(handle).children[0].status is ChildState.COMPLETED
        assert result.status is BatchRunStatus.COMPLETED

    def test_a_failing_cancel_request_still_marks_it_local_and_warns(self, store, remote, clock):
        provider = FakeProvider(remote, cancel=BatchProviderError("boom", status_code=500))
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = _start(executor, provider, make_plan(provider))

        result = executor.cancel(handle)

        assert result.status is BatchRunStatus.CANCELLED
        assert result.cancel_reports[0].batch_continues
        assert "could not be requested" in result.cancel_reports[0].detail
        assert "still be billed" in result.cancel_reports[0].detail

    def test_a_child_with_an_unconfirmed_submission_is_flagged_not_assumed_stopped(self, store, remote, clock):
        provider = FakeProvider(remote, cancel=CancelOutcome(CancelState.CANCELLED))
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = store.create_record(
            "consensus",
            [
                ChildIntent("openrouter", "m/one", "m/one", "/v1/chat/completions", 1, ("a",)),
                ChildIntent("openrouter", "m/two", "m/two", "/v1/chat/completions", 1, ("b",)),
            ],
        )
        store.record_provider_batch(handle, 0, "prov_batch_x", status=ChildState.IN_PROGRESS)
        remote.batches["prov_batch_x"] = RemoteBatch(model="m/one", custom_ids=["a"])

        result = executor.cancel(handle)

        by_index = {r.index: r for r in result.cancel_reports}
        assert by_index[0].action == "cancelled"
        assert by_index[1].action == "unconfirmed" and by_index[1].batch_continues
        assert provider.cancelled == ["prov_batch_x"]  # nothing was asked of a batch with no id
        # Not erased by the cancel: still unresolved, and on the operator list if reconciliation could not decide.
        assert store.get(handle).children[1].status.is_unresolved
        assert handle in [r.handle for r in store.needs_attention()]
        assert store.get(handle).state is RecordState.CANCELLED

    def test_cancel_of_an_unknown_handle_is_reported_as_unknown(self, executor):
        assert executor.cancel("batch_nope").status is BatchRunStatus.UNKNOWN_HANDLE

    def test_real_openrouter_adapter_cancel_never_issues_a_delete(self, store, clock, monkeypatch):
        adapter = OpenRouterProvider(api_key="test-key")
        calls: list[str] = []

        def send(method: str, path: str, content: str | None = None) -> httpx.Response:
            calls.append(method)
            if method == "POST":
                return httpx.Response(202, json={"id": "or_batch_1", "status": "validating"})
            return httpx.Response(200, json={"id": "or_batch_1", "status": "in_progress", "request_counts": {}})

        monkeypatch.setattr(adapter, "_batch_send", send)
        executor = make_executor(store, {"openrouter": adapter}, clock)
        handle = executor.submit_and_wait("chat", [make_plan(adapter)], wait_seconds=0).handle

        result = executor.cancel(handle)

        assert "DELETE" not in calls and set(calls) <= {"POST", "GET"}
        assert result.cancel_reports[0].batch_continues
        assert "still be billed" in result_message(result, tool_name="chat")


# ----------------------------------------------------------------------
# Review fixes: latency budget, multi-child bound, id hygiene, incomplete results, races
# ----------------------------------------------------------------------


class TestWaitBudgetCountsProviderLatency:
    def test_slow_provider_calls_do_not_push_a_wait_past_its_budget(self, executor, provider, clock):
        """A 20 s poll against a 50 s wait used to run to 66 s, beyond a host's ~60 s cancel point."""

        provider.on_poll = lambda _batch_id: setattr(clock, "now", clock.now + 20.0)
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=50)

        assert result.status is BatchRunStatus.RUNNING
        assert result.waited_seconds <= 50.0
        assert result.polls == 2  # a third round could not have fitted
        assert provider.cancelled == []

    def test_the_sleep_before_a_round_is_shortened_so_that_a_slow_poll_still_ends_inside_the_budget(
        self, store, provider, clock
    ):
        """Delays 1 s then 30 s; each poll takes 20 s. Sleeping the full 30 s would start round two at 31 s
        and end it at 51 s; shortened to what leaves room for the poll it ends exactly at the 50 s budget."""

        executor = make_executor(store, {"openrouter": provider}, clock, policy=WaitPolicy(poll_delays=(1.0, 30.0)))
        provider.on_poll = lambda _batch_id: setattr(clock, "now", clock.now + 20.0)

        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=50)

        assert result.polls == 2
        assert result.waited_seconds <= 50.0

    def test_a_slow_failing_fetch_is_not_retried_when_the_retry_cannot_fit(self, executor, provider, remote, clock):
        handle = _start(executor, provider, make_plan(provider))
        remote.finish("prov_batch_1")
        attempts = []

        def slow_failure(batch_id, expected_custom_ids=None):
            attempts.append(clock.now)
            clock.now += 20.0
            raise BatchProviderError("upstream hiccup", retryable=True, status_code=503)

        provider.fetch_batch = slow_failure

        started = clock.now
        result = executor.resume(handle, wait_seconds=30)

        assert len(attempts) == 1  # 10 s left after a 20 s attempt: a second one could not have completed
        assert clock.now - started <= 30.0
        assert result.status is BatchRunStatus.RUNNING and result.children[0].pending_reason

    def test_fast_provider_behaviour_is_unchanged_by_the_latency_accounting(self, executor, provider, clock):
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=50)
        assert clock.sleeps[:5] == [1.0, 2.0, 4.0, 8.0, 15.0]
        assert result.waited_seconds == pytest.approx(50.0)


class TestMultiChildPollBound:
    def test_a_full_length_wait_over_several_children_issues_a_bounded_number_of_polls(self, executor, provider, clock):
        plans = [make_plan(provider, f"m/model{i}", (f"c{i}",)) for i in range(3)]
        result = executor.submit_and_wait("consensus", plans, wait_seconds=10_000)

        assert result.status is BatchRunStatus.RUNNING
        assert result.polls == provider.calls.count("poll_batch")
        assert result.polls <= 3 * WaitPolicy().max_polls  # rounds x children
        assert sum(clock.sleeps) <= 50.0 + 1e-6

    def test_rate_limit_on_every_child_is_absorbed_and_still_bounded(self, executor, provider, clock):
        plans = [make_plan(provider, f"m/model{i}", (f"c{i}",)) for i in range(3)]
        provider.poll_errors = [BatchRateLimited("429", retry_after=2.0) for _ in range(100)]
        result = executor.submit_and_wait("consensus", plans, wait_seconds=50)

        assert result.status is BatchRunStatus.RUNNING
        assert result.polls <= 3 * WaitPolicy().max_polls


class TestProviderIdsStayInternal:
    def test_a_failed_batch_error_does_not_carry_the_provider_batch_id(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider))
        remote.batches["prov_batch_1"].script = [BatchState.FAILED]
        remote.batches["prov_batch_1"].fetch_error = BatchFailed(
            "OpenRouter batch prov_batch_1 failed: overloaded", state=BatchState.FAILED
        )

        result = executor.resume(handle)

        assert result.status is BatchRunStatus.FAILED
        text = result.children[0].error + result_message(result, tool_name="chat")
        assert "prov_batch_1" not in text and "overloaded" in text

    def test_a_per_request_error_does_not_carry_the_provider_batch_id(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider, ids=("a", "b")))
        remote.finish("prov_batch_1")
        remote.batches["prov_batch_1"].results = {
            "a": "fine",
            "b": BatchError("no result (batch prov_batch_1 ended as 'completed')", code="missing_result"),
        }

        result = executor.resume(handle)

        failed = result.outcomes_by_custom_id()["b"]
        assert not failed.success and "prov_batch_1" not in failed.error.message
        assert "ended as 'completed'" in failed.error.message

    def test_a_non_retryable_poll_error_does_not_carry_the_provider_batch_id(self, executor, provider):
        provider.poll_errors = [BatchProviderError("batch prov_batch_1 rejected the key", status_code=401)]
        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=30)
        assert "prov_batch_1" not in (result.children[0].poll_problem or "")
        assert "rejected the key" in result.children[0].poll_problem


class TestIncompleteResults:
    def test_fewer_results_than_submitted_are_kept_and_flagged_not_dropped(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider, ids=("a", "b")))
        remote.finish("prov_batch_1")
        original = provider.fetch_batch

        def partial(batch_id, expected_custom_ids=None):
            outcomes = original(batch_id, expected_custom_ids=expected_custom_ids)
            raise BatchIncompleteResults(
                "1 of 2 requests returned a result", outcomes=[o for o in outcomes if o.custom_id == "a"]
            )

        provider.fetch_batch = partial

        result = executor.resume(handle)

        assert result.status is BatchRunStatus.COMPLETED
        assert set(result.outcomes_by_custom_id()) == {"a"}
        assert result.children[0].error and "fewer results" in result.children[0].error


class TestConcurrentCancelAndResume:
    def test_a_cancel_that_lands_while_a_poll_is_in_flight_wins(self, executor, provider, remote):
        handle = _start(executor, provider, make_plan(provider))
        remote.finish("prov_batch_1")  # the in-flight poll will report COMPLETED
        provider.on_poll = lambda _batch_id: (
            setattr(provider, "on_poll", None),
            executor.cancel(handle, tool_name="chat"),
        )

        result = executor.resume(handle, tool_name="chat")

        record = executor.store.get(handle)
        assert record.state is RecordState.CANCELLED
        assert record.children[0].status is ChildState.CANCELLED  # not overwritten by the late poll result
        assert result.status is BatchRunStatus.CANCELLED


class TestHungProviderCallsAreAbandoned:
    """A provider call that never returns must not hold the tool call past budget + grace (real clock)."""

    @staticmethod
    def _real_time_executor(store, provider):
        coordinator = BatchCoordinator(store, provider_resolver=lambda name: {"openrouter": provider}.get(name))
        return BatchExecutor(coordinator, policy=WaitPolicy(call_grace_seconds=0.3))  # real clock and sleep

    def test_a_hung_poll_is_abandoned_at_the_hard_ceiling_and_the_handle_survives(self, store, provider, remote):
        executor = self._real_time_executor(store, provider)
        handle = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0).handle
        release = threading.Event()
        provider.on_poll = lambda _batch_id: release.wait(30)
        try:
            started = time.monotonic()
            result = executor.resume(handle, wait_seconds=0.2)
            elapsed = time.monotonic() - started

            assert elapsed < 3.0  # budget 0.2 s + grace 0.3 s, never the 30 s the provider would have taken
            assert result.status is BatchRunStatus.RUNNING and result.handle == handle
            assert provider.cancelled == [] and "submit_batch" not in provider.calls[1:]
        finally:
            release.set()
        provider.on_poll = None
        remote.finish("prov_batch_1")
        assert executor.resume(handle).status is BatchRunStatus.COMPLETED

    def test_a_hung_fetch_leaves_the_handle_resumable_instead_of_blocking(self, store, provider, remote):
        executor = self._real_time_executor(store, provider)
        handle = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0).handle
        remote.finish("prov_batch_1")
        release = threading.Event()
        original = provider.fetch_batch
        provider.fetch_batch = lambda *args, **kwargs: (release.wait(30), original(*args, **kwargs))[1]
        try:
            started = time.monotonic()
            result = executor.resume(handle, wait_seconds=0.2)
            elapsed = time.monotonic() - started

            assert elapsed < 3.0
            assert result.status is BatchRunStatus.RUNNING  # not terminal: the results are not in hand
            assert "did not fit" in result.children[0].pending_reason
        finally:
            release.set()
        provider.fetch_batch = original
        done = executor.resume(handle)
        assert done.status is BatchRunStatus.COMPLETED
        assert done.outcomes_by_custom_id()["chat"].response.content == "answer:chat"

    def test_provider_errors_still_propagate_through_the_bounded_call(self, store, provider):
        executor = self._real_time_executor(store, provider)
        handle = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0).handle
        provider.poll_errors = [BatchProviderError("batch rejected the key", status_code=401)]

        result = executor.resume(handle, wait_seconds=0.2)

        assert "rejected the key" in result.children[0].poll_problem


class TestFetchPathCompareAndSet:
    def test_a_child_resolved_while_its_fetch_was_in_flight_is_not_overwritten_by_a_late_not_ready(
        self, executor, provider, remote, store
    ):
        handle = _start(executor, provider, make_plan(provider))
        remote.finish("prov_batch_1")  # the poll sees COMPLETED, then the fetch goes out

        def not_ready_after_a_concurrent_cancel(batch_id, expected_custom_ids=None):
            store.set_child_status(handle, 0, ChildState.CANCELLED, detail="cancelled concurrently")
            raise BatchNotReady("still finalizing", state=BatchState.IN_PROGRESS)

        provider.fetch_batch = not_ready_after_a_concurrent_cancel

        executor.resume(handle)

        child = store.get(handle).children[0]
        assert child.status is ChildState.CANCELLED  # the observation 'in progress' did not clobber it
        assert child.detail == "cancelled concurrently"


# ----------------------------------------------------------------------
# Carry-over (a): submission and reconciliation latency count against the wait budget
# ----------------------------------------------------------------------


class TestSubmissionLatencyCountsAgainstTheBudget:
    """``submit_and_wait`` used to start its clock after ``coordinator.submit``: a 6 s submit with wait_seconds=4 took 10 s."""

    def test_a_submit_that_used_the_whole_budget_returns_the_handle_without_waiting_any_longer(
        self, executor, provider, clock
    ):
        provider.on_submit = lambda: setattr(clock, "now", clock.now + 6.0)
        started = clock.now

        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=4)

        assert result.status is BatchRunStatus.RUNNING and result.handle
        assert clock.now - started == 6.0  # the 6 s of submission, and not 4 s of waiting on top of it
        assert clock.sleeps == []
        assert provider.calls == ["submit_batch"]  # nothing left to poll with
        assert result.waited_seconds == 6.0  # reported from the start of the call, submission included
        assert provider.cancelled == []

    def test_the_wait_that_follows_a_slow_submit_is_only_what_is_left_of_the_budget(self, executor, provider, clock):
        provider.on_submit = lambda: setattr(clock, "now", clock.now + 2.0)
        started = clock.now

        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=4)

        assert result.status is BatchRunStatus.RUNNING
        assert clock.now - started <= 4.0  # 2 s submit + at most 2 s of waiting (it was 6 s before)
        assert provider.calls.count("poll_batch") >= 1  # it did use the time that was left

    def test_a_fast_submit_still_gets_its_full_wait(self, executor, provider, clock):
        started = clock.now

        executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=4)

        assert 3.0 <= clock.now - started <= 4.0

    def test_reconciling_on_resume_is_charged_to_the_wait_too(self, store, provider, remote, clock):
        executor = make_executor(store, {"openrouter": provider}, clock)
        provider.submit_error = BatchProviderError("connection reset", provider="openrouter", code=UNKNOWN)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait("chat", [make_plan(provider)])
        handle = excinfo.value.handle
        child = store.get(handle).children[0]
        remote.batches["or_late"] = RemoteBatch(model=MODEL, custom_ids=["chat"])
        listing = BatchListing(
            entries=(
                BatchListEntry(
                    batch_id="or_late",
                    state=BatchState.IN_PROGRESS,
                    model=child.submitted_model,
                    endpoint=child.endpoint,
                    created_at=child.intent_window_start + 1,
                    total=1,
                ),
            )
        )

        def slow_list(limit: int = 100) -> BatchListing:
            provider.calls.append("list_batches")
            clock.now += 3.0
            return listing

        provider.list_batches = slow_list
        provider.submit_error = None
        started = clock.now

        result = executor.resume(handle, wait_seconds=4)

        assert result.status is BatchRunStatus.RUNNING
        assert clock.now - started <= 4.0  # the 3 s listing left 1 s, not a fresh 4
        assert result.waited_seconds <= 4.0

    def test_the_ceiling_for_the_next_provider_call_shrinks_with_the_wait_but_never_below_the_grace(
        self, store, provider, clock
    ):
        executor = make_executor(store, {"openrouter": provider}, clock, policy=WaitPolicy(call_grace_seconds=5.0))
        started = clock.now
        ceiling = executor._call_timeout(started, 4.0)

        assert ceiling() == 9.0  # budget 4 + grace 5
        clock.now += 6.0
        assert ceiling() == 5.0  # 3 s left to the hard deadline: floored at the grace
        clock.now = started + 100.0
        assert ceiling() == 5.0  # a later child of a multi-batch submit is not cut off after a slow first one

    def test_a_multi_batch_submission_is_not_abandoned_because_its_first_batch_was_slow(
        self, executor, provider, clock
    ):
        provider.on_submit = lambda: setattr(clock, "now", clock.now + 6.0)

        result = executor.submit_and_wait(
            "consensus", [make_plan(provider, "m/one", ("a",)), make_plan(provider, "m/two", ("b",))], wait_seconds=0
        )

        assert provider.calls.count("submit_batch") == 2  # both submitted: abandoning one would strand paid work
        assert result.status is BatchRunStatus.RUNNING


class TestHungSubmitAndListAreBounded:
    """Real clock: a provider that never answers a submit or a batch list cannot hold the call open."""

    @staticmethod
    def _real_time_executor(store, provider):
        coordinator = BatchCoordinator(store, provider_resolver=lambda name: {"openrouter": provider}.get(name))
        return BatchExecutor(coordinator, policy=WaitPolicy(max_seconds=0.5, call_grace_seconds=0.3))

    def test_a_hung_submit_is_abandoned_as_an_unknown_outcome_and_keeps_its_intent(self, store, provider, remote):
        executor = self._real_time_executor(store, provider)
        release = threading.Event()
        provider.on_submit = lambda: release.wait(30)
        try:
            started = time.monotonic()
            with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
                executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0.2)
            elapsed = time.monotonic() - started

            assert 0.7 <= elapsed < 3.0  # longest permitted wait 0.5 s + grace 0.3 s, never the 30 s it would take
            assert "NOT retried" in str(excinfo.value)
            handle = excinfo.value.handle
            assert handle is not None
            child = store.get(handle).children[0]
            assert child.status is ChildState.SUBMITTING and child.provider_batch_id is None
            assert provider.calls == ["submit_batch"]
        finally:
            release.set()

        # The abandoned call is accepted after all: its batch is adopted under the same handle, not lost.
        deadline = time.monotonic() + 5
        while store.get(handle).children[0].provider_batch_id is None and time.monotonic() < deadline:
            time.sleep(0.02)
        late = store.get(handle).children[0]
        assert late.provider_batch_id == "prov_batch_1" and late.status is ChildState.VALIDATING
        remote.finish("prov_batch_1")
        assert executor.resume(handle).status is BatchRunStatus.COMPLETED
        assert provider.calls.count("submit_batch") == 1  # never resubmitted

    def test_a_hung_batch_list_on_resume_is_abandoned_and_the_intent_stays_unresolved(self, store, provider):
        executor = self._real_time_executor(store, provider)
        provider.submit_error = BatchProviderError("connection reset", provider="openrouter", code=UNKNOWN)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0)
        handle = excinfo.value.handle
        provider.submit_error = None
        release = threading.Event()
        provider.list_batches = lambda limit=100: (release.wait(30), None)[1]
        try:
            started = time.monotonic()
            result = executor.resume(handle, wait_seconds=0.2)
            elapsed = time.monotonic() - started

            assert elapsed < 3.0
            assert result.status is BatchRunStatus.RUNNING and result.handle == handle
            child = store.get(handle).children[0]
            assert child.status is ChildState.SUBMITTING  # neither adopted nor closed out nor marked indeterminate
        finally:
            release.set()
        assert provider.calls.count("submit_batch") == 1


class TestTheSubmitCeilingDoesNotDependOnTheCallersWait:
    """``wait_seconds`` says how long to wait for the *result*; it must not decide whether a create is abandoned.

    The submit used to share the wait's ceiling (remaining wait + grace, floored at the grace), so
    ``wait_seconds=0`` turned any create slower than the grace into an abandoned, possibly billed batch with an
    unknown outcome, and for a multi-batch submission into an unwound operation with no handle.
    """

    @staticmethod
    def _real_time_executor(store, provider):
        coordinator = BatchCoordinator(store, provider_resolver=lambda name: {"openrouter": provider}.get(name))
        # Longest wait 0.5 s, grace 0.3 s: a create that takes 0.6 s is slower than the grace, inside the ceiling.
        return BatchExecutor(coordinator, policy=WaitPolicy(max_seconds=0.5, call_grace_seconds=0.3))

    def test_the_submit_timeout_is_the_longest_permitted_wait_plus_grace_whatever_the_wait(
        self, store, provider, clock
    ):
        executor = make_executor(
            store, {"openrouter": provider}, clock, policy=WaitPolicy(max_seconds=50.0, call_grace_seconds=5.0)
        )
        started = clock.now
        ceiling = executor._submit_timeout(started)

        assert ceiling() == 55.0  # the same for wait_seconds 0 and for wait_seconds 50: it never sees the wait
        clock.now += 20.0
        assert ceiling() == 35.0  # shrinks with the time the call has spent
        clock.now = started + 500.0
        assert ceiling() == 5.0  # a later child of a multi-batch submit still gets the grace

    def test_a_slow_but_successful_single_batch_create_is_not_abandoned_at_wait_zero(self, store, provider, remote):
        executor = self._real_time_executor(store, provider)
        provider.on_submit = lambda: time.sleep(0.6)  # longer than the 0.3 s grace the old ceiling floored at

        result = executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0)

        assert result.status is BatchRunStatus.RUNNING and result.handle  # a handle, not "outcome unknown"
        child = store.get(result.handle).children[0]
        assert child.provider_batch_id == "prov_batch_1" and child.status is ChildState.VALIDATING
        assert provider.calls == ["submit_batch"]

    def test_a_slow_first_batch_does_not_unwind_a_multi_batch_submission_at_wait_zero(self, store, provider, remote):
        executor = self._real_time_executor(store, provider)
        slow_first = iter([0.6, 0.0])
        provider.on_submit = lambda: time.sleep(next(slow_first))

        result = executor.submit_and_wait(
            "consensus", [make_plan(provider, "m/one", ("a",)), make_plan(provider, "m/two", ("b",))], wait_seconds=0
        )

        assert result.status is BatchRunStatus.RUNNING and result.handle  # not abandoned, so not stranded
        assert provider.calls.count("submit_batch") == 2
        children = store.get(result.handle).children
        assert [c.provider_batch_id for c in children] == ["prov_batch_1", "prov_batch_2"]
        assert not store.get(result.handle).unwound
        assert store.needs_attention() == []  # nothing orphaned


class TestALateAcceptedSubmitIsNeverLost:
    """An abandoned submit that the provider accepts after all must leave its batch id on the record, whatever
    a reconciler did to the child in the meantime."""

    @staticmethod
    def _real_time_executor(store, provider):
        coordinator = BatchCoordinator(store, provider_resolver=lambda name: {"openrouter": provider}.get(name))
        return BatchExecutor(coordinator, policy=WaitPolicy(max_seconds=0.2, call_grace_seconds=0.1))

    @staticmethod
    def _wait_for(predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.02)
        return predicate()

    def _abandoned_single_batch(self, store, provider):
        """A single-batch submit whose call hangs past its ceiling; returns (executor, handle, release event)."""

        executor = self._real_time_executor(store, provider)
        release = threading.Event()
        provider.on_submit = lambda: release.wait(30)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait("chat", [make_plan(provider)], wait_seconds=0)
        provider.on_submit = None
        return executor, excinfo.value.handle, release

    def test_a_late_batch_is_attached_even_after_a_resume_marked_the_child_indeterminate(
        self, store, provider, remote, caplog
    ):
        executor, handle, release = self._abandoned_single_batch(store, provider)
        # The provider's list holds one entry nobody can verify, so the resume cannot adopt or close out: indeterminate.
        provider.listing = BatchListing(entries=(BatchListEntry(batch_id="someone_elses"),))
        assert executor.resume(handle, wait_seconds=0).status is BatchRunStatus.INDETERMINATE
        assert store.get(handle).children[0].status is ChildState.INDETERMINATE
        try:
            assert store.get(handle).children[0].provider_batch_id is None  # the call is still in flight
        finally:
            release.set()  # ... and now the provider answers: it did create a batch

        assert self._wait_for(lambda: store.get(handle).children[0].provider_batch_id is not None)
        child = store.get(handle).children[0]
        assert child.provider_batch_id == "prov_batch_1"  # not lost: it is on the record, not only in a log line
        assert child.status is ChildState.VALIDATING  # the id resolves the indeterminacy, so the handle polls it
        assert store.get(handle).state is not RecordState.INDETERMINATE
        remote.finish("prov_batch_1")
        assert executor.resume(handle).status is BatchRunStatus.COMPLETED
        assert provider.calls.count("submit_batch") == 1  # never resubmitted

    def test_a_late_batch_is_recorded_orphaned_when_the_child_was_already_closed_out_as_failed(
        self, store, provider, remote
    ):
        executor, handle, release = self._abandoned_single_batch(store, provider)
        store.set_child_status(handle, 0, ChildState.FAILED, detail="closed out by reconciliation")
        release.set()

        assert self._wait_for(lambda: store.get(handle).children[0].provider_batch_id is not None)
        child = store.get(handle).children[0]
        assert child.provider_batch_id == "prov_batch_1"
        assert child.status is ChildState.ORPHANED  # a billed batch the record had called failed: for an operator
        assert "already been resolved as failed" in child.detail
        assert [r.handle for r in store.needs_attention()] == [handle]

    def test_a_late_batch_of_an_abandoned_multi_batch_operation_is_recorded_orphaned(self, store, provider, remote):
        executor = self._real_time_executor(store, provider)
        release = threading.Event()
        calls = iter([lambda: release.wait(30), lambda: None])
        provider.on_submit = lambda: next(calls)()
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait(
                "consensus",
                [make_plan(provider, "m/one", ("a",)), make_plan(provider, "m/two", ("b",))],
                wait_seconds=0,
            )
        handle = excinfo.value.retained_handle
        assert store.get(handle).unwound
        release.set()

        assert self._wait_for(lambda: store.get(handle).children[0].provider_batch_id is not None)
        child = store.get(handle).children[0]
        assert child.status is ChildState.ORPHANED and child.provider_batch_id == "prov_batch_1"

    def test_a_child_that_changes_state_between_the_read_and_the_write_is_re_read_not_lost(
        self, store, provider, remote
    ):
        """A reconciler can resolve the child in the instant between our read and our compare-and-set write."""

        executor = make_executor(store, {"openrouter": provider}, FakeClock())
        provider.submit_error = BatchProviderError("reset", provider="openrouter", code=UNKNOWN)
        with pytest.raises(SubmissionOutcomeUnknown) as excinfo:
            executor.submit_and_wait("chat", [make_plan(provider)])
        handle = excinfo.value.handle
        assert store.get(handle).children[0].status is ChildState.SUBMITTING

        real_write = store.record_provider_batch
        raced: list[bool] = []

        def lose_the_race_once(*args, **kwargs):
            if not raced:
                raced.append(True)
                store.set_child_status(handle, 0, ChildState.INDETERMINATE, detail="resolved by a reconciler")
            return real_write(*args, **kwargs)

        store.record_provider_batch = lose_the_race_once  # type: ignore[method-assign]
        try:
            outcome = executor.coordinator._attach_late_batch(handle, 0, "late_batch")
        finally:
            store.record_provider_batch = real_write  # type: ignore[method-assign]

        assert raced == [True] and outcome == "adopted"
        child = store.get(handle).children[0]
        assert child.provider_batch_id == "late_batch" and child.status is ChildState.VALIDATING

    def test_a_late_batch_that_the_child_already_holds_is_not_an_error(self, store, provider, remote, caplog):
        executor, handle, release = self._abandoned_single_batch(store, provider)
        # A reconciler attributed this very batch to the child while the call was still in flight.
        store.record_provider_batch(handle, 0, "prov_batch_1", status=ChildState.IN_PROGRESS)
        with caplog.at_level("ERROR"):
            release.set()
            time.sleep(0.3)

        child = store.get(handle).children[0]
        assert child.provider_batch_id == "prov_batch_1" and child.status is ChildState.IN_PROGRESS  # untouched
        assert not [r for r in caplog.records if "could not be recorded" in r.getMessage()]

    def test_a_late_batch_that_cannot_be_attached_is_logged_with_provider_and_id(self, store, provider, remote, caplog):
        executor, handle, release = self._abandoned_single_batch(store, provider)
        store.record_provider_batch(handle, 0, "a_different_batch", status=ChildState.IN_PROGRESS)
        with caplog.at_level("ERROR"):
            release.set()
            assert self._wait_for(lambda: any("could not be recorded" in r.getMessage() for r in caplog.records))

        message = next(r.getMessage() for r in caplog.records if "could not be recorded" in r.getMessage())
        assert "prov_batch_1" in message and "openrouter" in message and handle in message
        assert store.get(handle).children[0].provider_batch_id == "a_different_batch"  # never overwritten


# ----------------------------------------------------------------------
# Carry-over (c): child-level already-terminal handling inside a mixed multi-child record
# ----------------------------------------------------------------------


class TestCancelOfAMixedRecordKeepsFinishedChildrenFinished:
    def test_a_completed_child_is_reported_already_terminal_and_never_as_still_billing(self, store, remote, clock):
        provider = FakeProvider(remote)  # OpenRouter-like: no cancel mechanism
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = _start(
            executor,
            provider,
            make_plan(provider, "m/one", ("a",)),
            make_plan(provider, "m/two", ("b",)),
            make_plan(provider, "m/three", ("c",)),
        )
        # Child 0 finished at the provider and was observed by an earlier resume; 1 and 2 are still running.
        remote.finish("prov_batch_1")
        remote.batches["prov_batch_2"].script = [BatchState.IN_PROGRESS]
        remote.batches["prov_batch_3"].script = [BatchState.IN_PROGRESS]
        executor.resume(handle, wait_seconds=0)
        assert [c.status for c in store.get(handle).children] == [
            ChildState.COMPLETED,
            ChildState.IN_PROGRESS,
            ChildState.IN_PROGRESS,
        ]
        provider.cancelled.clear()

        result = executor.cancel(handle)

        by_index = {report.index: report for report in result.cancel_reports}
        done = by_index[0]
        assert done.action == "already_terminal"
        assert done.batch_continues is False  # it must not be counted among the batches that "continue and bill"
        assert done.state_after is ChildState.COMPLETED
        assert "nothing to cancel" in done.detail and "billed" not in done.detail
        assert provider.cancelled == ["prov_batch_2", "prov_batch_3"]  # no cancel was even requested for it
        assert by_index[1].action == by_index[2].action == "continues"
        assert by_index[1].batch_continues and by_index[2].batch_continues

        children = store.get(handle).children
        assert children[0].status is ChildState.COMPLETED  # not overwritten with cancelled
        assert children[1].status is children[2].status is ChildState.CANCELLED
        assert store.get(handle).state is not RecordState.CANCELLED  # a record holding finished work is not "cancelled"
        payload = build_status_payload(result, tool_name="consensus")
        assert payload["provider_batches_continue"] is True  # because of children 1 and 2 only
        assert (
            "m/one: the provider reports" not in payload["message"] and "m/one: already completed" in payload["message"]
        )

    def test_cancelling_a_run_with_one_finished_child_reports_cancelled_and_how_to_get_the_results(
        self, store, remote, clock
    ):
        provider = FakeProvider(remote)
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = _start(executor, provider, make_plan(provider, "m/one", ("a",)), make_plan(provider, "m/two", ("b",)))
        remote.finish("prov_batch_1")
        remote.batches["prov_batch_2"].script = [BatchState.IN_PROGRESS]
        executor.resume(handle, wait_seconds=0)

        result = executor.cancel(handle, tool_name="consensus")

        assert result.status is BatchRunStatus.CANCELLED  # this call did cancel a child; not "completed"
        assert store.get(handle).state is not RecordState.CANCELLED  # the finished child's results stay retrievable
        payload = build_status_payload(result, tool_name="consensus")
        assert payload["status"] == "cancelled"
        assert "ONLY the batch_handle" in payload["message"]
        assert "is completed" not in payload["message"]
        assert payload["provider_batches_continue"] is True

    def test_cancelling_names_the_provider_batch_that_keeps_running_in_the_server_log_only(
        self, executor, provider, caplog
    ):
        handle = _start(executor, provider, make_plan(provider))
        with caplog.at_level(logging.WARNING, logger="utils.batch_execution"):
            result = executor.cancel(handle)

        assert "prov_batch_1" in caplog.text and handle in caplog.text and "NOT stopped" in caplog.text
        shown = result_message(result, tool_name="chat") + str(build_status_payload(result, tool_name="chat"))
        assert "prov_batch_1" not in shown  # the caller still sees only the handle

    def test_when_every_child_already_finished_nothing_is_reported_as_continuing(self, store, remote, clock):
        provider = FakeProvider(remote)
        executor = make_executor(store, {"openrouter": provider}, clock)
        handle = _start(executor, provider, make_plan(provider, "m/one", ("a",)), make_plan(provider, "m/two", ("b",)))
        remote.finish("prov_batch_1")
        remote.finish("prov_batch_2")
        executor.resume(handle, wait_seconds=0)
        provider.cancelled.clear()

        result = executor.cancel(handle)

        assert provider.cancelled == []
        assert result.cancel_reports == ()  # the record was terminal: reported as it stands
        assert result.status is BatchRunStatus.COMPLETED
