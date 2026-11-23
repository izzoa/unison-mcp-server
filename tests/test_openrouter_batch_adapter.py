"""Tests for the OpenRouter batch adapter (change add-batch-model-support, section 3).

No network and no API key: HTTP goes through ``OpenRouterProvider._batch_send`` (stubbed)
or through an ``httpx.MockTransport`` injected as ``_test_transport``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import utils.model_restrictions as model_restrictions
from providers.openrouter import OpenRouterProvider
from providers.shared import (
    BatchFailed,
    BatchIncompleteResults,
    BatchNotFound,
    BatchNotReady,
    BatchProviderError,
    BatchRateLimited,
    BatchRequest,
    BatchResultsExpired,
    BatchState,
    CancelState,
    ProviderType,
)

OPUS = "anthropic/claude-opus-4.5"  # batch-capable, has aliases
OPUS_LATEST = "anthropic/claude-opus-5.5"  # what the family alias `opus` means under batch (fixture catalog)
FLASH = "google/gemini-2.5-flash"  # no batch listing


def _accepted(batch_id: str = "batch_1", **overrides: Any) -> dict[str, Any]:
    payload = {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "model": OPUS,
        "completion_window": "24h",
        "status": "validating",
        "created_at": 1,
        "finalized_at": None,
        "provider": None,
        "expires_at": "2026-11-23T00:00:00Z",
        "request_counts": {"total": 2, "completed": 0, "failed": 0},
        "usage": None,
        "results": None,
        "error": None,
    }
    payload.update(overrides)
    return payload


def _ok(custom_id: str, text: str, usage: dict | None = None) -> dict[str, Any]:
    return {
        "id": f"res_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"req_{custom_id}",
            "body": {
                "id": f"gen_{custom_id}",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": (
                    usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                ),
            },
        },
        "error": None,
    }


def _err(custom_id: str, error_type: str, message: str = "boom") -> dict[str, Any]:
    return {
        "id": f"res_{custom_id}",
        "custom_id": custom_id,
        "response": None,
        "error": {"type": error_type, "message": message, "param": None, "error_type": error_type},
    }


class _Recorder:
    """Stub for ``_batch_send`` that records calls and replays canned responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, method: str, path: str, content: str | None = None) -> httpx.Response:
        self.calls.append((method, path, content))
        if not self.responses:
            raise AssertionError(f"unexpected network call: {method} {path}")
        return self.responses.pop(0)


def _json(status: int, payload: Any, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers)


@pytest.fixture(autouse=True)
def _reset_restrictions(monkeypatch):
    monkeypatch.delenv("OPENROUTER_ALLOWED_MODELS", raising=False)
    model_restrictions._restriction_service = None
    yield
    model_restrictions._restriction_service = None


@pytest.fixture
def provider() -> OpenRouterProvider:
    return OpenRouterProvider(api_key="test-key")


def _stub(provider: OpenRouterProvider, monkeypatch, *responses: httpx.Response) -> _Recorder:
    recorder = _Recorder(*responses)
    monkeypatch.setattr(provider, "_batch_send", recorder)
    return recorder


def _two() -> list[BatchRequest]:
    return [
        BatchRequest(custom_id="a", prompt="alpha", system_prompt="be for"),
        BatchRequest(custom_id="b", prompt="beta", system_prompt="be against"),
    ]


# ----------------------------------------------------------------------
# 3.1 endpoint routing
# ----------------------------------------------------------------------


class TestEndpointRouting:
    def test_batch_submission_targets_batches_path_and_sync_still_targets_chat_completions(self, monkeypatch):
        seen: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.method, str(request.url)))
            if request.url.path.endswith("/batches"):
                return _json(202, _accepted())
            return _json(
                200,
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": OPUS,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        provider = OpenRouterProvider(api_key="test-key")
        provider._test_transport = httpx.MockTransport(handler)

        batch_id = provider.submit_batch(OPUS, _two())
        sync = provider.generate_content("hi", OPUS)

        assert batch_id == "batch_1"
        assert sync.content == "hello"
        assert seen[0] == ("POST", "https://openrouter.ai/api/v1/batches")
        assert seen[1] == ("POST", "https://openrouter.ai/api/v1/chat/completions")
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_batch_call_carries_auth_and_openrouter_headers(self):
        captured: dict[str, httpx.Request] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return _json(202, _accepted())

        provider = OpenRouterProvider(api_key="sk-test")
        provider._test_transport = httpx.MockTransport(handler)
        provider.submit_batch(OPUS, _two())

        headers = captured["request"].headers
        assert headers["authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"
        assert "x-title" in headers and "http-referer" in headers

    def test_poll_and_fetch_hit_the_batch_resource_by_id(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(200, _accepted("batch_9")), _json(200, _accepted("batch_9")))
        provider.poll_batch("batch_9")
        with pytest.raises(BatchNotReady):
            provider.fetch_batch("batch_9")
        assert [(m, p) for m, p, _ in rec.calls] == [("GET", "/batches/batch_9"), ("GET", "/batches/batch_9")]


# ----------------------------------------------------------------------
# 3.2 envelope
# ----------------------------------------------------------------------


class TestEnvelope:
    def test_endpoint_and_model_serialize_before_requests_with_inline_requests(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(202, _accepted()))
        provider.submit_batch(OPUS, _two())

        assert len(rec.calls) == 1  # one POST: no file-upload step
        method, path, content = rec.calls[0]
        assert (method, path) == ("POST", "/batches")
        assert content is not None
        assert content.index('"endpoint"') < content.index('"model"') < content.index('"requests"')
        assert list(json.loads(content).keys()) == ["endpoint", "model", "requests"]

        body = json.loads(content)
        assert body["endpoint"] == "/v1/chat/completions"
        assert body["model"] == OPUS
        assert [r["custom_id"] for r in body["requests"]] == ["a", "b"]
        first = body["requests"][0]["body"]
        assert first["messages"] == [
            {"role": "system", "content": "be for"},
            {"role": "user", "content": "alpha"},
        ]
        assert "model" not in first  # batch-level model applies
        assert all(p == "/batches" for _, p, _ in rec.calls)  # nothing like /files

    def test_sampling_parameters_follow_the_models_batch_listing(self, provider, monkeypatch):
        # Changed by D9 (was ``..._follow_the_synchronous_rules``): a curated slug sends the parameters its ``:batch``
        # listing names. opus-4.5's lists temperature, stop and max_tokens (no top_p), sonnet-4.5's also top_p.
        rec = _stub(provider, monkeypatch, _json(202, _accepted()))
        provider.submit_batch(
            OPUS, [BatchRequest("a", "p", temperature=0.2, max_output_tokens=99, extra={"top_p": 0.9, "stop": ["x"]})]
        )
        body = json.loads(rec.calls[0][2])["requests"][0]["body"]
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 99
        assert body["stop"] == ["x"]
        assert "top_p" not in body  # the listing does not name it

        rec = _stub(provider, monkeypatch, _json(202, _accepted()))
        provider.submit_batch(
            "anthropic/claude-sonnet-4.5",
            [BatchRequest("a", "p", temperature=0.2, max_output_tokens=99, extra={"top_p": 0.9})],
        )
        assert json.loads(rec.calls[0][2])["requests"][0]["body"]["top_p"] == 0.9

    def test_model_without_temperature_support_sends_no_temperature_but_its_listed_token_cap_and_seed(
        self, provider, monkeypatch
    ):
        # Changed by D9 (was ``..._omits_sampling_params``): o3 takes no temperature, but its ``:batch`` listing names
        # max_tokens and seed, which used to be dropped with it.
        rec = _stub(provider, monkeypatch, _json(202, _accepted(model="openai/o3")))
        provider.submit_batch(
            "openai/o3", [BatchRequest("a", "p", temperature=0.2, max_output_tokens=99, extra={"seed": 5, "top_p": 1})]
        )
        body = json.loads(rec.calls[0][2])["requests"][0]["body"]
        assert "temperature" not in body and "top_p" not in body
        assert body["max_tokens"] == 99 and body["seed"] == 5

    def test_duplicate_custom_ids_rejected_before_network(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch)
        with pytest.raises(ValueError, match="Duplicate"):
            provider.submit_batch(OPUS, [BatchRequest("a", "x"), BatchRequest("a", "y")])
        assert rec.calls == []

    def test_empty_submission_rejected_before_network(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch)
        with pytest.raises(ValueError):
            provider.submit_batch(OPUS, [])
        assert rec.calls == []


# ----------------------------------------------------------------------
# 3.3 one model per batch
# ----------------------------------------------------------------------


class TestOneModel:
    def test_two_model_submission_rejected_before_any_provider_call(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch)
        reqs = [
            BatchRequest("a", "x", extra={"model": OPUS}),
            BatchRequest("b", "y", extra={"model": "anthropic/claude-sonnet-4.5"}),
        ]
        with pytest.raises(ValueError, match="one model"):
            provider.submit_batch(OPUS, reqs)
        assert rec.calls == []

    def test_same_model_multi_stance_submission_accepted_and_bodies_omit_model(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(202, _accepted()))
        stances = [
            BatchRequest("for", "q", system_prompt="argue for", extra={"model": "opus"}),
            BatchRequest("against", "q", system_prompt="argue against", extra={"model": "opus:batch"}),
            BatchRequest("neutral", "q", system_prompt="be neutral"),
        ]
        # 14.13: the batch is named by the alias; a full-slug batch with `opus` requests would now be a mix
        # of two models (4.5 vs the latest), see tests/test_batch_dynamic_resolution.py.
        assert provider.submit_batch("opus", stances) == "batch_1"
        body = json.loads(rec.calls[0][2])
        assert len(body["requests"]) == 3
        assert all("model" not in r["body"] for r in body["requests"])


# ----------------------------------------------------------------------
# 3.4 text-only
# ----------------------------------------------------------------------


class TestTextOnly:
    def test_image_request_rejected_naming_it_with_no_provider_call(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch)
        reqs = [BatchRequest("ok", "text"), BatchRequest("pic-1", "look", images=("/tmp/x.png",))]
        with pytest.raises(ValueError) as exc:
            provider.submit_batch(OPUS, reqs)
        message = str(exc.value)
        assert "pic-1" in message
        assert "text-only" in message
        assert "synchronous" in message
        assert rec.calls == []

    @pytest.mark.parametrize("key", ["audio", "video", "files", "attachments"])
    def test_other_attachment_carriers_rejected(self, provider, monkeypatch, key):
        rec = _stub(provider, monkeypatch)
        with pytest.raises(ValueError, match="req-x"):
            provider.submit_batch(OPUS, [BatchRequest("req-x", "p", extra={key: ["blob"]})])
        assert rec.calls == []

    def test_files_ingested_into_the_prompt_as_text_are_accepted(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(202, _accepted()))
        prompt = "Review this.\n\n=== FILE: app.py ===\nprint('hi')\n=== END FILE ===\n"
        assert provider.submit_batch(OPUS, [BatchRequest("r1", prompt)]) == "batch_1"
        sent = json.loads(rec.calls[0][2])["requests"][0]["body"]["messages"][-1]["content"]
        assert "=== FILE: app.py ===" in sent


# ----------------------------------------------------------------------
# 3.5 acceptance is not completion
# ----------------------------------------------------------------------


class TestAcceptanceIsNotCompletion:
    def test_accepted_submission_reports_no_completed_requests(self, provider, monkeypatch):
        accepted = _accepted("batch_q")
        _stub(provider, monkeypatch, _json(202, accepted), _json(200, accepted))

        batch_id = provider.submit_batch(OPUS, _two())
        status = provider.poll_batch(batch_id)

        assert status.state is BatchState.VALIDATING
        assert not status.is_terminal
        assert status.completed == 0 and status.failed == 0
        assert status.total == 2

    @pytest.mark.parametrize("state", ["in_progress", "finalizing"])
    def test_in_flight_states_poll_without_raising(self, provider, monkeypatch, state):
        _stub(provider, monkeypatch, _json(200, _accepted(status=state)))
        assert not provider.poll_batch("batch_1").is_terminal

    def test_cancelling_is_a_documented_non_terminal_status(self, provider, monkeypatch):
        """OpenRouter documents ``cancelling`` between a cancel and ``cancelled``; it must not be an unknown status."""

        _stub(provider, monkeypatch, _json(200, _accepted(status="cancelling")))
        status = provider.poll_batch("batch_1")
        assert status.state is BatchState.CANCELLING
        assert not status.is_terminal

    def test_every_documented_status_is_recognised(self, provider, monkeypatch):
        documented = (
            "validating",
            "in_progress",
            "finalizing",
            "completed",
            "failed",
            "expired",
            "cancelling",
            "cancelled",
        )
        for name in documented:
            _stub(provider, monkeypatch, _json(200, _accepted(status=name)))
            assert provider.poll_batch("batch_1").state.value == name

    @pytest.mark.parametrize("state", ["completed", "failed", "expired", "cancelled"])
    def test_terminal_states(self, provider, monkeypatch, state):
        _stub(provider, monkeypatch, _json(200, _accepted(status=state, error={"message": "why"})))
        status = provider.poll_batch("batch_1")
        assert status.is_terminal and status.state.value == state
        assert status.error_message == "why"


# ----------------------------------------------------------------------
# 3.6 poll / fetch / cancel
# ----------------------------------------------------------------------


def _completed(results: list[dict], **overrides: Any) -> dict[str, Any]:
    return _accepted(
        status="completed",
        results=results,
        request_counts={"total": len(results), "completed": len(results), "failed": 0},
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.01, "is_byok": False},
        **overrides,
    )


class TestFetch:
    def test_outcomes_correlate_by_custom_id_regardless_of_order(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(200, _completed([_ok("c", "third"), _ok("a", "first"), _ok("b", "second")])))
        outcomes = {o.custom_id: o for o in provider.fetch_batch("batch_1")}
        assert set(outcomes) == {"a", "b", "c"}
        assert outcomes["a"].response.content == "first"
        assert outcomes["b"].response.content == "second"
        assert outcomes["c"].response.content == "third"

    def test_success_maps_to_standard_response_with_usage_and_batch_marker(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(200, _completed([_ok("a", "answer")])))
        (outcome,) = provider.fetch_batch("batch_1")

        assert outcome.success and outcome.error is None
        response = outcome.response
        assert response.content == "answer"
        assert response.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        assert response.provider is ProviderType.OPENROUTER
        assert response.model_name == OPUS
        assert response.friendly_name == "OpenRouter"
        assert response.metadata["execution_mode"] == "batch"
        assert response.metadata["batch_id"] == "batch_1"
        assert response.metadata["submitted_model"] == OPUS

    def test_per_request_errors_map_with_retryability(self, provider, monkeypatch):
        results = [
            _ok("good", "fine"),
            _err("limited", "rate_limit_error", "slow down"),
            _err("down", "server_error"),
            _err("bad", "invalid_request_error", "malformed"),
        ]
        _stub(provider, monkeypatch, _json(200, _completed(results)))
        by_id = {o.custom_id: o for o in provider.fetch_batch("batch_1")}

        assert by_id["good"].success
        assert not by_id["limited"].success and by_id["limited"].error.retryable is True
        assert by_id["down"].error.retryable is True
        assert by_id["bad"].error.retryable is False
        assert by_id["bad"].error.message == "malformed"

    def test_completed_batch_with_http_error_status_in_response_is_a_failure(self, provider, monkeypatch):
        item = {
            "id": "r",
            "custom_id": "x",
            "response": {"status_code": 503, "request_id": "q", "body": {"error": {"message": "upstream down"}}},
            "error": None,
        }
        _stub(provider, monkeypatch, _json(200, _completed([item])))
        (outcome,) = provider.fetch_batch("batch_1")
        assert not outcome.success and outcome.error.retryable is True
        assert "upstream down" in outcome.error.message

    def test_fetch_on_non_terminal_batch_raises_not_ready(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(200, _accepted(status="in_progress")))
        with pytest.raises(BatchNotReady) as exc:
            provider.fetch_batch("batch_1")
        assert exc.value.state is BatchState.IN_PROGRESS

    def test_failed_batch_without_results_raises_batch_failed_with_reason(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(200, _accepted(status="failed", error={"message": "bad envelope"})))
        with pytest.raises(BatchFailed, match="bad envelope"):
            provider.fetch_batch("batch_1")

    def test_requested_spelling_reported_in_metadata_when_it_differs(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(202, _accepted()), _json(200, _completed([_ok("a", "x")])))
        batch_id = provider.submit_batch(OPUS + ":batch", [BatchRequest("a", "p")])
        (outcome,) = provider.fetch_batch(batch_id)
        assert outcome.response.metadata["submitted_model"] == OPUS
        assert outcome.response.metadata["requested_model"] == OPUS + ":batch"


class TestFetchCompleteness:
    """One outcome per submitted request, whatever the provider actually returned."""

    def _submit_three(self, provider, monkeypatch, results: list[dict], **overrides: Any) -> list:
        reqs = [BatchRequest(c, "p") for c in ("a", "b", "c")]
        completed = _completed(results, **overrides)
        completed["request_counts"]["total"] = 3
        _stub(provider, monkeypatch, _json(202, _accepted()), _json(200, completed))
        batch_id = provider.submit_batch(OPUS, reqs)
        return provider.fetch_batch(batch_id)

    def test_missing_result_becomes_a_failed_outcome_for_that_request(self, provider, monkeypatch):
        outcomes = self._submit_three(provider, monkeypatch, [_ok("a", "x"), _ok("c", "z")])
        assert [o.custom_id for o in outcomes] == ["a", "b", "c"]
        by_id = {o.custom_id: o for o in outcomes}
        assert by_id["a"].success and by_id["c"].success
        assert by_id["b"].success is False
        assert by_id["b"].error.code == "missing_result"
        assert by_id["b"].error.retryable is False

    def test_duplicate_result_is_a_failed_outcome_not_a_silent_overwrite(self, provider, monkeypatch):
        outcomes = self._submit_three(
            provider, monkeypatch, [_ok("a", "first"), _ok("a", "second"), _ok("b", "y"), _ok("c", "z")]
        )
        by_id = {o.custom_id: o for o in outcomes}
        assert len(outcomes) == 3
        assert by_id["a"].success is False and by_id["a"].error.code == "duplicate_result"
        assert by_id["b"].success and by_id["c"].success

    def test_result_without_custom_id_does_not_stand_in_for_a_request(self, provider, monkeypatch):
        anonymous = _ok("zzz", "mystery")
        anonymous["custom_id"] = None
        outcomes = self._submit_three(provider, monkeypatch, [_ok("a", "x"), anonymous, _ok("c", "z")])
        by_id = {o.custom_id: o for o in outcomes}
        assert by_id["b"].success is False and by_id["b"].error.code == "missing_result"
        assert len(outcomes) == 3

    def test_result_for_an_unsubmitted_id_is_dropped(self, provider, monkeypatch):
        outcomes = self._submit_three(
            provider, monkeypatch, [_ok("a", "x"), _ok("b", "y"), _ok("c", "z"), _ok("q", "?")]
        )
        assert [o.custom_id for o in outcomes] == ["a", "b", "c"]

    def test_explicit_expected_ids_work_without_an_in_process_submission(self, provider, monkeypatch):
        completed = _completed([_ok("a", "x")])
        completed["request_counts"]["total"] = 2
        _stub(provider, monkeypatch, _json(200, completed))
        outcomes = provider.fetch_batch("batch_resumed", expected_custom_ids=["a", "b"])
        assert [(o.custom_id, o.success) for o in outcomes] == [("a", True), ("b", False)]

    def test_without_ids_a_shortfall_against_request_counts_is_an_error_carrying_the_partial_results(
        self, provider, monkeypatch
    ):
        completed = _completed([_ok("a", "x")])
        completed["request_counts"]["total"] = 2
        _stub(provider, monkeypatch, _json(200, completed))
        with pytest.raises(BatchIncompleteResults) as exc:
            provider.fetch_batch("batch_unknown")
        assert [o.custom_id for o in exc.value.outcomes] == ["a"]
        assert exc.value.retryable is False and exc.value.code == "incomplete_results"

    def test_complete_result_set_without_ids_is_returned_as_is(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(200, _completed([_ok("b", "y"), _ok("a", "x")])))
        assert {o.custom_id for o in provider.fetch_batch("batch_unknown")} == {"a", "b"}


class TestSubmittedModelMetadata:
    def test_metadata_reports_the_slug_actually_submitted_when_the_payload_omits_model(self, provider, monkeypatch):
        completed = _completed([_ok("a", "x")])
        del completed["model"]
        _stub(provider, monkeypatch, _json(202, _accepted()), _json(200, completed))
        batch_id = provider.submit_batch("opus:batch", [BatchRequest("a", "p")])
        (outcome,) = provider.fetch_batch(batch_id)
        # 14.13: `opus` resolves dynamically under batch, so the slug submitted is the latest batch-listed Opus.
        assert outcome.response.model_name == OPUS_LATEST
        assert outcome.response.metadata["submitted_model"] == OPUS_LATEST
        assert outcome.response.metadata["requested_model"] == "opus:batch"


class TestPollErrors:
    def test_429_is_typed_retryable_with_retry_after(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(429, {"error": {"message": "slow"}}, headers={"Retry-After": "7"}))
        with pytest.raises(BatchRateLimited) as exc:
            provider.poll_batch("batch_1")
        assert exc.value.retryable is True
        assert exc.value.retry_after == 7.0

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_5xx_is_retryable(self, provider, monkeypatch, status):
        _stub(provider, monkeypatch, _json(status, {"error": {"message": "oops"}}))
        with pytest.raises(BatchProviderError) as exc:
            provider.poll_batch("batch_1")
        assert exc.value.retryable is True and exc.value.status_code == status

    def test_transport_failure_is_retryable(self, provider, monkeypatch):
        def boom(method, path, content=None):
            raise httpx.ConnectError("down")

        monkeypatch.setattr(provider, "_batch_send", boom)
        with pytest.raises(BatchProviderError) as exc:
            provider.poll_batch("batch_1")
        assert exc.value.retryable is True

    def test_410_is_distinguishable_as_expired_results(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(410, {"error": {"message": "gone"}}))
        with pytest.raises(BatchResultsExpired) as exc:
            provider.fetch_batch("batch_1")
        assert exc.value.retryable is False

    def test_404_is_not_found(self, provider, monkeypatch):
        _stub(provider, monkeypatch, _json(404, {"error": {"message": "nope"}}))
        with pytest.raises(BatchNotFound):
            provider.poll_batch("batch_x")

    @pytest.mark.parametrize("status", [400, 402, 413])
    def test_other_4xx_not_retryable(self, provider, monkeypatch, status):
        _stub(provider, monkeypatch, _json(status, {"error": {"message": "no"}}))
        with pytest.raises(BatchProviderError) as exc:
            provider.poll_batch("batch_1")
        assert exc.value.retryable is False

    @pytest.mark.parametrize(
        "status, expect_retryable, expect_type",
        [(429, True, BatchRateLimited), (402, False, BatchProviderError), (413, False, BatchProviderError)],
    )
    def test_submit_errors_are_typed(self, provider, monkeypatch, status, expect_retryable, expect_type):
        _stub(provider, monkeypatch, _json(status, {"error": {"message": "limit"}}))
        with pytest.raises(expect_type) as exc:
            provider.submit_batch(OPUS, _two())
        assert exc.value.retryable is expect_retryable
        assert exc.value.status_code == status

    @pytest.mark.parametrize("exc_type", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError])
    def test_submit_transport_failure_is_typed_ambiguous_and_not_retried(self, provider, monkeypatch, exc_type):
        calls: list[str] = []

        def boom(method, path, content=None):
            calls.append(method)
            raise exc_type("timed out")

        monkeypatch.setattr(provider, "_batch_send", boom)
        with pytest.raises(BatchProviderError) as exc:
            provider.submit_batch(OPUS, _two())
        assert exc.value.code == "submission_outcome_unknown"
        assert exc.value.retryable is False
        assert "may or may not" in str(exc.value)
        assert calls == ["POST"]

    def test_submit_is_never_retried(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(503, {"error": {"message": "x"}}), _json(202, _accepted()))
        with pytest.raises(BatchProviderError):
            provider.submit_batch(OPUS, _two())
        assert len(rec.calls) == 1


class TestCancel:
    def test_cancelling_a_terminal_batch_reports_state_without_raising(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(200, _completed([_ok("a", "x")])))
        outcome = provider.cancel_batch("batch_1")
        assert outcome.state is CancelState.ALREADY_TERMINAL
        assert "completed" in outcome.detail
        assert [m for m, _, _ in rec.calls] == ["GET"]

    @pytest.mark.parametrize("state", ["validating", "in_progress", "finalizing", "cancelling"])
    def test_cancelling_a_running_batch_is_unsupported_and_issues_no_delete(self, provider, monkeypatch, state):
        rec = _stub(provider, monkeypatch, _json(200, _accepted(status=state)))
        outcome = provider.cancel_batch("batch_1")

        assert outcome.state is CancelState.UNSUPPORTED
        assert outcome.batch_continues is True
        assert "no cancel endpoint" in outcome.detail
        assert "billed" in outcome.detail
        assert [m for m, _, _ in rec.calls] == ["GET"]  # never DELETE

    @pytest.mark.parametrize("status", [404, 500])
    def test_cancel_failure_is_reported_as_a_typed_error_not_a_false_outcome(self, provider, monkeypatch, status):
        _stub(provider, monkeypatch, _json(status, {"error": {"message": "x"}}))
        with pytest.raises(BatchProviderError) as exc:
            provider.cancel_batch("batch_1")
        assert exc.value.status_code == status

    def test_cancel_transport_failure_is_typed(self, provider, monkeypatch):
        def boom(method, path, content=None):
            raise httpx.ReadTimeout("slow")

        monkeypatch.setattr(provider, "_batch_send", boom)
        with pytest.raises(BatchProviderError):
            provider.cancel_batch("batch_1")

    def test_no_code_path_issues_a_delete(self, provider, monkeypatch):
        rec = _stub(
            provider,
            monkeypatch,
            _json(202, _accepted()),
            _json(200, _accepted()),
            _json(200, _completed([_ok("a", "x")])),
            _json(200, _completed([_ok("a", "x")])),
        )
        batch_id = provider.submit_batch(OPUS, _two())
        provider.poll_batch(batch_id)
        provider.fetch_batch(batch_id)
        provider.cancel_batch(batch_id)
        assert "DELETE" not in {m for m, _, _ in rec.calls}


# ----------------------------------------------------------------------
# 3.7 slug mapping
# ----------------------------------------------------------------------


class TestSlugMapping:
    def test_both_spellings_converge_on_the_same_submission(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(202, _accepted("b1")), _json(202, _accepted("b2")))
        provider.submit_batch(OPUS, _two())
        provider.submit_batch(OPUS + ":batch", _two())

        first, second = (json.loads(c[2]) for c in rec.calls)
        assert first == second
        assert first["model"] == OPUS  # never the :batch spelling

    def test_suffix_is_case_insensitive(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(202, _accepted()))
        provider.submit_batch(OPUS + ":BATCH", _two())
        assert json.loads(rec.calls[0][2])["model"] == OPUS

    def test_model_without_batch_listing_is_rejected_not_remapped(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch)
        for spelling in (FLASH, FLASH + ":batch"):
            with pytest.raises(ValueError) as exc:
                provider.submit_batch(spelling, _two())
            assert FLASH in str(exc.value)
            assert "unavailable" in str(exc.value)
        assert rec.calls == []

    @pytest.mark.parametrize(
        "spelling",
        [
            OPUS + ":free",
            OPUS + ":online",
            OPUS + ":free:batch",
            OPUS + ":batch:free",
            OPUS + ":batch:batch",
            OPUS + ":BATCH:batch",
        ],
    )
    def test_variant_identifiers_are_rejected_locally_never_sent_as_the_model(self, provider, monkeypatch, spelling):
        rec = _stub(provider, monkeypatch)
        with pytest.raises(ValueError) as exc:
            provider.submit_batch(spelling, _two())
        assert "unavailable" in str(exc.value) and "variant" in str(exc.value)
        assert rec.calls == []

    def test_unknown_generic_model_is_rejected(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch)
        with pytest.raises(ValueError):
            provider.submit_batch("someone/brand-new-model:batch", _two())
        assert rec.calls == []

    def test_mapping_rule_is_documented_on_the_provider(self):
        doc = OpenRouterProvider.__doc__ or ""
        assert ":batch" in doc and "normal" in doc and "submitted_model" in doc


# ----------------------------------------------------------------------
# 3.8 alias resolution and restrictions
# ----------------------------------------------------------------------


class TestAliasesAndRestrictions:
    def test_alias_resolves_identically_to_the_canonical_name_it_floats_to(self, provider, monkeypatch):
        # 14.13: under batch `opus` is not the pinned slug; it is the latest batch-listed Opus, and a request
        # naming that slug directly is byte-for-byte the same submission. The synchronous resolver is untouched.
        rec = _stub(provider, monkeypatch, _json(202, _accepted("b1")), _json(202, _accepted("b2")))
        provider.submit_batch("opus", _two())
        provider.submit_batch(OPUS_LATEST, _two())
        by_alias, by_name = (json.loads(c[2]) for c in rec.calls)
        assert by_alias == by_name
        assert by_alias["model"] == OPUS_LATEST
        assert provider._resolve_model_name("opus") == OPUS  # the synchronous resolver still means the pin

    def test_alias_with_batch_suffix_also_converges(self, provider, monkeypatch):
        rec = _stub(provider, monkeypatch, _json(202, _accepted()))
        provider.submit_batch("opus:batch", _two())
        assert json.loads(rec.calls[0][2])["model"] == OPUS_LATEST

    def test_restricted_model_is_rejected_before_any_network_call(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_ALLOWED_MODELS", "anthropic/claude-sonnet-4.5")
        model_restrictions._restriction_service = None
        provider = OpenRouterProvider(api_key="test-key")
        rec = _stub(provider, monkeypatch)

        with pytest.raises(ValueError, match="not allowed"):
            provider.submit_batch(OPUS, _two())
        with pytest.raises(ValueError, match="not allowed"):
            provider.submit_batch("opus:batch", _two())
        assert rec.calls == []

    def test_allowed_model_still_submits_under_restrictions(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_ALLOWED_MODELS", "sonnet")
        model_restrictions._restriction_service = None
        provider = OpenRouterProvider(api_key="test-key")
        rec = _stub(provider, monkeypatch, _json(202, _accepted()))
        provider.submit_batch("sonnet", _two())
        # 14.13: an allow-list naming the alias admits every candidate, so the newest Sonnet is chosen.
        assert json.loads(rec.calls[0][2])["model"] == "anthropic/claude-sonnet-5.5"
