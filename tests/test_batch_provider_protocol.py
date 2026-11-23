"""Tests for the provider-neutral batch types and the default ModelProvider batch surface."""

import dataclasses

import pytest

from providers.base import ModelProvider
from providers.shared import (
    BatchError,
    BatchOutcome,
    BatchRequest,
    BatchState,
    BatchStatus,
    CancelOutcome,
    CancelState,
    ModelResponse,
    ProviderType,
    UnsupportedBatchOperation,
    validate_batch_requests,
)


class _NoBatchProvider(ModelProvider):
    """Minimal concrete provider that does not implement batch."""

    def get_provider_type(self) -> ProviderType:
        return ProviderType.XAI

    def generate_content(self, prompt, model_name, system_prompt=None, temperature=0.3, max_output_tokens=None, **kw):
        return ModelResponse(content="sync")

    def get_all_model_capabilities(self):
        return {}

    def list_models(self, **kwargs):
        return []

    def validate_model_name(self, model_name):
        return True


def _req(custom_id: str, prompt: str = "hi") -> BatchRequest:
    return BatchRequest(custom_id=custom_id, prompt=prompt)


class TestDefaultBatchSurface:
    @pytest.fixture
    def provider(self) -> _NoBatchProvider:
        return _NoBatchProvider(api_key="k")

    def test_submit_raises_naming_provider_and_operation(self, provider):
        with pytest.raises(UnsupportedBatchOperation) as exc:
            provider.submit_batch("some-model", [_req("a")])
        assert "xai" in str(exc.value)
        assert "submit_batch" in str(exc.value)

    @pytest.mark.parametrize(
        "operation, call",
        [
            ("poll_batch", lambda p: p.poll_batch("batch_1")),
            ("fetch_batch", lambda p: p.fetch_batch("batch_1")),
            ("cancel_batch", lambda p: p.cancel_batch("batch_1")),
        ],
    )
    def test_other_operations_raise_naming_provider_and_operation(self, provider, operation, call):
        with pytest.raises(UnsupportedBatchOperation) as exc:
            call(provider)
        assert "xai" in str(exc.value)
        assert operation in str(exc.value)
        assert exc.value.provider == "xai"
        assert exc.value.operation == operation

    def test_synchronous_surface_is_untouched(self, provider):
        assert provider.generate_content("p", "m").content == "sync"

    def test_no_silent_synchronous_substitution(self, provider):
        # A rejected batch submission must never fall back to generate_content.
        with pytest.raises(UnsupportedBatchOperation):
            provider.submit_batch("m", [_req("a")])


class TestBatchState:
    @pytest.mark.parametrize(
        "state", [BatchState.COMPLETED, BatchState.FAILED, BatchState.EXPIRED, BatchState.CANCELLED]
    )
    def test_terminal_states(self, state):
        assert state.is_terminal

    @pytest.mark.parametrize("state", [BatchState.VALIDATING, BatchState.IN_PROGRESS, BatchState.FINALIZING])
    def test_non_terminal_states(self, state):
        assert not state.is_terminal

    def test_status_reflects_state_terminality(self):
        assert not BatchStatus(state=BatchState.VALIDATING, provider="openrouter", total=3).is_terminal
        assert BatchStatus(state=BatchState.FAILED, error_message="bad input").is_terminal


class TestBatchOutcome:
    def test_success_outcome_carries_response(self):
        response = ModelResponse(content="answer", usage={"total_tokens": 5})
        outcome = BatchOutcome.succeeded("req-1", response)
        assert outcome.success is True
        assert outcome.custom_id == "req-1"
        assert outcome.response is response
        assert outcome.error is None

    def test_retryable_failure(self):
        outcome = BatchOutcome.failed("req-2", BatchError("rate limited", code="429", retryable=True))
        assert outcome.success is False
        assert outcome.response is None
        assert outcome.error is not None and outcome.error.retryable is True

    def test_non_retryable_failure(self):
        outcome = BatchOutcome.failed("req-3", BatchError("malformed request", code="invalid_request"))
        assert outcome.success is False
        assert outcome.error is not None and outcome.error.retryable is False

    def test_mixed_outcomes_keep_sibling_results(self):
        outcomes = [
            BatchOutcome.succeeded("a", ModelResponse(content="ok")),
            BatchOutcome.failed("b", BatchError("nope")),
        ]
        assert [o.custom_id for o in outcomes if o.success] == ["a"]
        assert [o.custom_id for o in outcomes if not o.success] == ["b"]


class TestCancelOutcome:
    def test_states_cover_the_spec(self):
        assert {s.value for s in CancelState} == {"cancelled", "cancelling", "already_terminal", "unsupported"}

    def test_unsupported_reports_batch_continues(self):
        outcome = CancelOutcome(CancelState.UNSUPPORTED, "no cancel endpoint; batch continues and is billed")
        assert outcome.batch_continues is True
        assert "continues" in outcome.detail

    def test_already_terminal_and_cancelled_do_not_continue(self):
        assert CancelOutcome(CancelState.ALREADY_TERMINAL).batch_continues is False
        assert CancelOutcome(CancelState.CANCELLED).batch_continues is False


class TestValidateBatchRequests:
    def test_accepts_unique_ids(self):
        validate_batch_requests([_req("a"), _req("b")])

    def test_duplicate_id_rejected_and_named(self):
        with pytest.raises(ValueError) as exc:
            validate_batch_requests([_req("a"), _req("dup"), _req("dup")])
        assert "dup" in str(exc.value)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            validate_batch_requests([])


class TestBatchRequest:
    def test_request_is_immutable_and_carries_images(self):
        request = BatchRequest(custom_id="a", prompt="p", images=("/tmp/x.png",))
        assert request.has_non_text_content is True
        assert BatchRequest(custom_id="b", prompt="p").has_non_text_content is False
        with pytest.raises(dataclasses.FrozenInstanceError):
            request.custom_id = "z"  # type: ignore[misc]
