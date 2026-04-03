"""Tests for utils/circuit_breaker.py — state transitions, thread safety, and provider integration."""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from utils.circuit_breaker import CircuitBreaker, CircuitState, ProviderUnavailable

# -----------------------------------------------------------------------
# 8.1 State transition tests
# -----------------------------------------------------------------------


class TestCircuitBreakerStateTransitions:
    """CLOSED -> OPEN -> HALF_OPEN -> CLOSED and failure paths."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state is CircuitState.CLOSED

    def test_closed_to_open_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state is CircuitState.OPEN

    def test_failures_below_threshold_stay_closed(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state is CircuitState.CLOSED

    def test_open_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        time.sleep(0.02)
        assert cb.allow_request() is True
        assert cb.state is CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_request()  # transition to HALF_OPEN
        cb.record_success()
        assert cb.state is CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_request()  # transition to HALF_OPEN
        cb.record_failure()
        assert cb.state is CircuitState.OPEN

    def test_success_in_closed_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Failure count reset — need 3 more to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.CLOSED
        cb.record_failure()
        assert cb.state is CircuitState.OPEN


# -----------------------------------------------------------------------
# 8.2 allow_request tests
# -----------------------------------------------------------------------


class TestAllowRequest:
    def test_allows_in_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True

    def test_denies_in_open_before_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=100)
        cb.record_failure()
        assert cb.allow_request() is False

    def test_allows_in_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_request() is True

    def test_allows_first_call_in_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01, half_open_max_calls=1)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_request() is True  # first call
        assert cb.allow_request() is False  # second call blocked

    def test_half_open_max_calls_respected(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01, half_open_max_calls=2)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_request() is True  # 1st
        assert cb.allow_request() is True  # 2nd
        assert cb.allow_request() is False  # 3rd blocked


# -----------------------------------------------------------------------
# 8.3 Thread safety tests
# -----------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_failures_reach_threshold_exactly_once(self):
        cb = CircuitBreaker(failure_threshold=10)
        barrier = threading.Barrier(10)

        def fail():
            barrier.wait()
            cb.record_failure()

        threads = [threading.Thread(target=fail) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.state is CircuitState.OPEN
        status = cb.health_status()
        assert status["failure_count"] == 10

    def test_concurrent_allow_request_on_expired_open(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01, half_open_max_calls=1)
        cb.record_failure()
        time.sleep(0.02)

        results = []
        barrier = threading.Barrier(5)

        def check():
            barrier.wait()
            results.append(cb.allow_request())

        threads = [threading.Thread(target=check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 1 thread should get True (the probe), rest False
        assert results.count(True) == 1
        assert results.count(False) == 4


# -----------------------------------------------------------------------
# 8.4 ProviderUnavailable exception tests
# -----------------------------------------------------------------------


class TestProviderUnavailable:
    def test_default_message(self):
        exc = ProviderUnavailable("gemini", "open")
        assert "gemini" in str(exc)
        assert "open" in str(exc)

    def test_custom_message(self):
        exc = ProviderUnavailable("openai", "half_open", message="custom msg")
        assert str(exc) == "custom msg"

    def test_attributes(self):
        exc = ProviderUnavailable("xai", "open")
        assert exc.provider_name == "xai"
        assert exc.circuit_state == "open"

    def test_is_exception(self):
        with pytest.raises(ProviderUnavailable):
            raise ProviderUnavailable("test", "open")


# -----------------------------------------------------------------------
# 8.5 Provider integration tests
# -----------------------------------------------------------------------


class TestProviderIntegration:
    """Test circuit breaker integrated into ModelProvider._run_with_retries."""

    def _make_provider(self, failure_threshold=2):
        """Create a minimal ModelProvider subclass for testing."""
        from providers.base import ModelProvider
        from providers.shared import ProviderType

        class FakeProvider(ModelProvider):
            def get_provider_type(self):
                return ProviderType.CUSTOM

            def generate_content(self, prompt, model_name, **kwargs):
                pass

        with patch.dict(
            os.environ,
            {
                "CIRCUIT_FAILURE_THRESHOLD": str(failure_threshold),
                "CIRCUIT_RESET_TIMEOUT_SECONDS": "0.01",
            },
        ):
            return FakeProvider(api_key="test")

    def test_circuit_opens_after_threshold(self):
        provider = self._make_provider(failure_threshold=2)

        def failing_op():
            raise RuntimeError("boom")

        # Exhaust 2 failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                provider._run_with_retries(failing_op, max_attempts=1, log_prefix="test")

        # Now circuit should be open
        with pytest.raises(ProviderUnavailable):
            provider._run_with_retries(failing_op, max_attempts=1, log_prefix="test")

    def test_success_resets_circuit(self):
        provider = self._make_provider(failure_threshold=3)

        call_count = 0

        def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("connection refused")
            return "ok"

        # First two calls fail (retryable — "connection" in error string), attempt 3 succeeds
        result = provider._run_with_retries(flaky_op, max_attempts=3, log_prefix="test")
        assert result == "ok"
        # Circuit should still be closed since we got a success
        assert provider._circuit_breaker.state is CircuitState.CLOSED

    def test_provider_unavailable_raised_when_open(self):
        provider = self._make_provider(failure_threshold=1)

        def failing_op():
            raise RuntimeError("down")

        with pytest.raises(RuntimeError):
            provider._run_with_retries(failing_op, max_attempts=1, log_prefix="test")

        with pytest.raises(ProviderUnavailable) as exc_info:
            provider._run_with_retries(failing_op, max_attempts=1, log_prefix="test")

        assert exc_info.value.provider_name == "FakeProvider"


# -----------------------------------------------------------------------
# 8.6 Consensus graceful degradation tests
# -----------------------------------------------------------------------


class TestConsensusGracefulDegradation:
    """Test that consensus tool handles ProviderUnavailable correctly."""

    def test_consult_model_catches_provider_unavailable(self):
        """_consult_model should return error dict, not propagate."""
        from tools.consensus import ConsensusTool

        tool = ConsensusTool()
        model_config = {"model": "test-model", "stance": "neutral"}

        with patch.object(tool, "get_model_provider", side_effect=ProviderUnavailable("test", "open")):
            import asyncio

            request = MagicMock()
            request.relevant_files = []
            request.images = []
            result = asyncio.run(tool._consult_model(model_config, request))

        assert result["status"] == "error"
        assert result["error"] == "provider_unavailable"
        assert "test-model" in result["model"]

    def test_all_providers_unavailable_detected(self):
        """When all accumulated_responses are provider_unavailable, detect it."""
        responses = [
            {"model": "m1", "status": "error", "error": "provider_unavailable"},
            {"model": "m2", "status": "error", "error": "provider_unavailable"},
        ]
        successful = [r for r in responses if r.get("status") == "success"]
        unavailable = [r for r in responses if r.get("error") == "provider_unavailable"]

        assert len(successful) == 0
        assert len(unavailable) == 2

    def test_partial_unavailability_detected(self):
        """When some providers succeed and some are unavailable."""
        responses = [
            {"model": "m1", "status": "success", "verdict": "looks good"},
            {"model": "m2", "status": "error", "error": "provider_unavailable"},
        ]
        successful = [r for r in responses if r.get("status") == "success"]
        unavailable = [r for r in responses if r.get("error") == "provider_unavailable"]

        assert len(successful) == 1
        assert len(unavailable) == 1


# -----------------------------------------------------------------------
# 8.7 Configuration via environment variables
# -----------------------------------------------------------------------


class TestConfigurationFromEnv:
    def test_custom_threshold(self):
        with patch.dict(os.environ, {"CIRCUIT_FAILURE_THRESHOLD": "10"}):
            from providers.base import ModelProvider

            class FP(ModelProvider):
                def get_provider_type(self):
                    pass

                def generate_content(self, *a, **kw):
                    pass

            p = FP(api_key="k")
            assert p._circuit_breaker._failure_threshold == 10

    def test_custom_timeout(self):
        with patch.dict(os.environ, {"CIRCUIT_RESET_TIMEOUT_SECONDS": "120"}):
            from providers.base import ModelProvider

            class FP(ModelProvider):
                def get_provider_type(self):
                    pass

                def generate_content(self, *a, **kw):
                    pass

            p = FP(api_key="k")
            assert p._circuit_breaker._reset_timeout_seconds == 120.0

    def test_defaults_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove any circuit env vars
            for key in ["CIRCUIT_FAILURE_THRESHOLD", "CIRCUIT_RESET_TIMEOUT_SECONDS", "CIRCUIT_HALF_OPEN_MAX_CALLS"]:
                os.environ.pop(key, None)

            from providers.base import ModelProvider

            class FP(ModelProvider):
                def get_provider_type(self):
                    pass

                def generate_content(self, *a, **kw):
                    pass

            p = FP(api_key="k")
            assert p._circuit_breaker._failure_threshold == 5
            assert p._circuit_breaker._reset_timeout_seconds == 60.0
            assert p._circuit_breaker._half_open_max_calls == 1

    def test_invalid_env_value_uses_default(self):
        with patch.dict(os.environ, {"CIRCUIT_FAILURE_THRESHOLD": "not_a_number"}):
            from providers.base import ModelProvider

            class FP(ModelProvider):
                def get_provider_type(self):
                    pass

                def generate_content(self, *a, **kw):
                    pass

            p = FP(api_key="k")
            assert p._circuit_breaker._failure_threshold == 5


# -----------------------------------------------------------------------
# Health status tests
# -----------------------------------------------------------------------


class TestHealthStatus:
    def test_healthy_provider_status(self):
        cb = CircuitBreaker(provider_name="test-provider")
        status = cb.health_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["provider_name"] == "test-provider"
        assert status["last_failure_time"] is None

    def test_unhealthy_provider_status(self):
        cb = CircuitBreaker(failure_threshold=1, provider_name="bad-provider")
        cb.record_failure()
        status = cb.health_status()
        assert status["state"] == "open"
        assert status["failure_count"] == 1
        assert status["last_failure_time"] is not None
        assert "time_until_half_open_seconds" in status

    def test_half_open_status(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_request()
        status = cb.health_status()
        assert status["state"] == "half_open"
