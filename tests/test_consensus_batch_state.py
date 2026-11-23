"""The pure state helpers of a batched consensus run (``utils/consensus_batch.py``), plus the executor and
context additions section 8 and 9 rely on. No network, no API keys."""

from __future__ import annotations

import json
from typing import Any

import pytest

from providers.shared import BatchError, BatchOutcome, BatchState, ModelResponse, ProviderType
from tests.batch_test_helpers import FakeClock, FakeProvider, FakeRemote, make_executor, make_plan
from tools.consensus import ConsensusTool
from tools.shared.exceptions import ToolExecutionError
from utils.batch_execution import ChildResult
from utils.batch_store import BatchChild, BatchStore, ChildState
from utils.consensus_batch import (
    EXECUTION_BATCH,
    PAYLOAD_KEYS,
    ConsensusPayloadError,
    build_payload,
    custom_id_for,
    entry_index,
    interrupted_sync_responses,
    merge_responses,
    ordered_responses,
    parse_payload,
    persisted_roster,
    responses_for_child,
    unanswered_indexes,
)
from utils.model_context import ModelContext


def _roster() -> list[dict[str, Any]]:
    return persisted_roster(
        [
            {"model": "a", "stance": "for"},
            {"model": "b", "stance": "against", "stance_prompt": "Be blunt."},
            {"model": "a"},
        ],
        ["batch", None, "batch"],
    )


def _child(*ids: str, status: ChildState = ChildState.COMPLETED, detail: str | None = None) -> BatchChild:
    return BatchChild(
        handle="batch_x",
        index=0,
        provider="openrouter",
        provider_batch_id="p1",
        requested_model="a",
        submitted_model="vendor/a",
        endpoint="/v1/chat/completions",
        request_total=len(ids),
        status=status,
        custom_ids=tuple(ids),
        intent_window_start=0.0,
        intent_window_end=None,
        claimed=True,
        detail=detail,
    )


def _ok(custom_id: str, text: str) -> BatchOutcome:
    return BatchOutcome.succeeded(
        custom_id,
        ModelResponse(content=text, model_name="vendor/a", provider=ProviderType.OPENROUTER),
    )


class TestRosterAndPayload:
    def test_the_roster_records_stance_stance_prompt_and_only_the_batched_mode(self):
        roster = _roster()

        assert roster == [
            {"model": "a", "stance": "for", "execution_mode": "batch"},
            {"model": "b", "stance": "against", "stance_prompt": "Be blunt."},
            {"model": "a", "stance": "neutral", "execution_mode": "batch"},
        ]

    def test_the_payload_is_exactly_the_four_documented_fields_in_order(self):
        payload = build_payload("proposal", "findings", _roster(), [])

        assert (
            tuple(payload)
            == PAYLOAD_KEYS
            == (
                "original_proposal",
                "findings",
                "models",
                "accumulated_responses",
            )
        )
        assert "work_history" not in payload

    def test_a_payload_round_trips_through_json_and_parses_back(self):
        payload = build_payload("p — 日本語", "f", _roster(), [{"model": "b", "stance": "against"}])

        state = parse_payload(json.loads(json.dumps(payload)))

        assert state.original_proposal == "p — 日本語" and state.findings == "f"
        assert state.batched_indexes() == [0, 2]
        assert unanswered_indexes(state) == [0, 2]  # b (index 1) already has its response

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "text",
            {},
            {"original_proposal": "p"},
            {"original_proposal": 5, "findings": None, "models": [], "accumulated_responses": []},
            {"original_proposal": "p", "findings": None, "models": "x", "accumulated_responses": []},
            {"original_proposal": "p", "findings": None, "models": [], "accumulated_responses": [1]},
        ],
    )
    def test_a_payload_that_is_not_a_consensus_payload_is_refused_by_name(self, raw):
        with pytest.raises(ConsensusPayloadError):
            parse_payload(raw)

    def test_custom_ids_name_the_roster_position(self):
        assert custom_id_for(3) == "consensus-3" and entry_index("consensus-3") == 3
        assert entry_index("chat") is None and entry_index("consensus-x") is None


class TestResponses:
    def test_a_successful_outcome_becomes_a_synchronous_shaped_response_in_the_requested_spelling(self):
        state = parse_payload(build_payload("p", "f", _roster(), []))
        result = ChildResult(
            _child("consensus-0", "consensus-2"), outcomes=(_ok("consensus-0", "yes"), _ok("consensus-2", "meh"))
        )

        responses = responses_for_child(state, result)

        assert responses is not None
        first = responses[0]
        assert (first["model"], first["stance"], first["status"], first["verdict"]) == ("a", "for", "success", "yes")
        assert first["metadata"]["provider"] == "openrouter" and first["metadata"]["model_name"] == "a"
        assert first["metadata"]["model_requested"] == "a" and first["metadata"]["model_resolved"] == "vendor/a"
        assert first["metadata"]["execution_mode"] == EXECUTION_BATCH

    def test_one_failed_request_fails_only_its_own_entry(self):
        state = parse_payload(build_payload("p", "f", _roster(), []))
        outcomes = (
            _ok("consensus-0", "yes"),
            BatchOutcome.failed("consensus-2", BatchError("refused", retryable=False)),
        )

        responses = responses_for_child(state, ChildResult(_child("consensus-0", "consensus-2"), outcomes=outcomes))

        assert responses is not None
        assert [r["status"] for r in responses] == ["success", "error"] and responses[1]["error"] == "refused"

    def test_a_batch_that_failed_as_a_whole_fails_every_entry_it_owned(self):
        state = parse_payload(build_payload("p", "f", _roster(), []))
        result = ChildResult(_child("consensus-0", "consensus-2", status=ChildState.FAILED), error="provider gave up")

        responses = responses_for_child(state, result)

        assert responses is not None and [r["status"] for r in responses] == ["error", "error"]
        assert {r["error"] for r in responses} == {"provider gave up"}

    def test_a_result_the_provider_never_returned_is_an_error_for_that_entry_only(self):
        state = parse_payload(build_payload("p", "f", _roster(), []))
        result = ChildResult(_child("consensus-0", "consensus-2"), outcomes=(_ok("consensus-0", "yes"),))

        responses = responses_for_child(state, result)

        assert responses is not None
        assert responses[0]["status"] == "success"
        assert responses[1]["status"] == "error" and "no result" in responses[1]["error"]

    def test_an_empty_completion_is_an_error_not_a_blank_verdict(self):
        state = parse_payload(build_payload("p", "f", _roster(), []))
        blank = BatchOutcome.succeeded(
            "consensus-0", ModelResponse(content="", model_name="x", provider=ProviderType.OPENROUTER)
        )

        responses = responses_for_child(state, ChildResult(_child("consensus-0"), outcomes=(blank,)))

        assert responses is not None and responses[0]["status"] == "error"

    def test_a_cancelled_child_with_no_results_names_how_it_ended(self):
        state = parse_payload(build_payload("p", "f", _roster(), []))

        responses = responses_for_child(
            state, ChildResult(_child("consensus-0", status=ChildState.CANCELLED, detail="stopped by the provider"))
        )

        assert (
            responses is not None
            and "cancelled" in responses[0]["error"]
            and "stopped by the provider" in responses[0]["error"]
        )

    def test_a_child_whose_results_cannot_be_fetched_yet_records_nothing(self):
        state = parse_payload(build_payload("p", "f", _roster(), []))

        assert responses_for_child(state, ChildResult(_child("consensus-0"), pending_reason="try later")) is None

    def test_a_recorded_response_is_final_and_a_repeated_merge_adds_nothing(self):
        first = {"model": "a", "stance": "for", "status": "success", "verdict": "first"}
        late = {"model": "a", "stance": "for", "status": "error", "error": "later"}
        other = {"model": "b", "stance": "against", "status": "success", "verdict": "b"}

        once = merge_responses([first], [late, other])

        assert [r.get("verdict") or r.get("error") for r in once] == ["first", "b"]
        assert merge_responses(once, [late, other]) == once

    def test_responses_are_reported_in_roster_order(self):
        roster = _roster()
        scrambled = [
            {"model": "a", "stance": "neutral"},
            {"model": "b", "stance": "against"},
            {"model": "a", "stance": "for"},
        ]

        assert [(r["model"], r["stance"]) for r in ordered_responses(roster, scrambled)] == [
            ("a", "for"),
            ("b", "against"),
            ("a", "neutral"),
        ]

    def test_only_synchronous_entries_without_a_result_are_reported_interrupted(self):
        state = parse_payload(build_payload("p", "f", _roster(), []))

        interrupted = interrupted_sync_responses(state)

        assert [(r["model"], r["status"]) for r in interrupted] == [("b", "error")]
        assert "not re-run" in interrupted[0]["error"]


class TestExecutorAdditions:
    def test_fetch_false_reports_a_finished_record_by_state_and_fetches_nothing(self, tmp_path):
        remote, store, provider = FakeRemote(), BatchStore(str(tmp_path / "b" / "batches.db")), None
        provider = FakeProvider(remote)
        executor = make_executor(store, {"openrouter": provider}, FakeClock())
        handle = executor.submit_and_wait("consensus", [make_plan(provider)], wait_seconds=0, fetch=False).handle
        remote.finish("prov_batch_1")

        result = executor.resume(handle, tool_name="consensus", fetch=False)

        assert result.status.value == "completed" and all(not c.outcomes for c in result.children)
        assert "fetch_batch" not in provider.calls
        fetched = executor.resume(handle, tool_name="consensus")  # the default still fetches, as before
        assert [o.custom_id for c in fetched.children for o in c.outcomes] == ["chat"]
        store.close()

    def test_fetch_child_returns_that_childs_results_mapped_onto_the_recorded_model(self, tmp_path):
        remote, store = FakeRemote(), BatchStore(str(tmp_path / "b" / "batches.db"))
        provider = FakeProvider(remote)
        executor = make_executor(store, {"openrouter": provider}, FakeClock())
        handle = executor.submit_and_wait(
            "consensus", [make_plan(provider, "m/one", ("x", "y"))], wait_seconds=0
        ).handle
        remote.finish("prov_batch_1", results={"x": "ex", "y": "why"})
        executor.resume(handle, fetch=False)  # polls: the child is now recorded as completed
        child = store.get(handle).children[0]

        result = executor.fetch_child(child)

        assert {o.custom_id: o.response.content for o in result.outcomes} == {"x": "ex", "y": "why"}
        assert {o.response.model_name for o in result.outcomes} == {"m/one"}  # the recorded model, not the adapter's
        assert provider.fetch_expected_ids[-1] == ("x", "y")
        store.close()

    def test_fetch_child_reports_a_failed_batch_as_an_error_not_an_exception(self, tmp_path):
        remote, store = FakeRemote(), BatchStore(str(tmp_path / "b" / "batches.db"))
        provider = FakeProvider(remote)
        executor = make_executor(store, {"openrouter": provider}, FakeClock())
        handle = executor.submit_and_wait("consensus", [make_plan(provider)], wait_seconds=0).handle
        remote.batches["prov_batch_1"].script = [BatchState.FAILED]
        executor.resume(handle, fetch=False)

        result = executor.fetch_child(store.get(handle).children[0])

        assert result.error and not result.outcomes
        store.close()


class TestModelContextProvider:
    def test_a_pinned_provider_is_used_instead_of_the_registrys_choice(self):
        sentinel = object()

        context = ModelContext("some-model", provider=sentinel)

        assert context.provider is sentinel

    def test_without_a_provider_the_context_still_resolves_lazily(self):
        assert ModelContext("some-model")._provider is None


class TestConsensusRecordsThatAreNotConsensusRecords:
    @pytest.mark.asyncio
    async def test_resuming_a_consensus_handle_with_no_consensus_payload_is_an_explicit_error(self, tmp_path):
        remote, store = FakeRemote(), BatchStore(str(tmp_path / "b" / "batches.db"))
        provider = FakeProvider(remote)
        executor = make_executor(store, {"openrouter": provider}, FakeClock())
        handle = executor.submit_and_wait("consensus", [make_plan(provider)], wait_seconds=0, fetch=False).handle
        tool = ConsensusTool()
        tool._batch_executor = executor

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute({"batch_handle": handle})

        payload = json.loads(excinfo.value.payload)
        assert (
            payload["metadata"]["batch_error"] == "payload_unreadable" and payload["metadata"]["batch_handle"] == handle
        )
        store.close()
