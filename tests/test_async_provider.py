"""Tests for async provider interface and concurrent consensus dispatch."""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.shared import ModelResponse, ProviderType
from utils.circuit_breaker import ProviderUnavailable


def _make_model_response(content: str = "test response", model_name: str = "test-model") -> ModelResponse:
    return ModelResponse(
        content=content,
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        model_name=model_name,
        friendly_name="Test",
        provider=ProviderType.CUSTOM,
        metadata={},
    )


def _make_provider(failure_threshold: int = 5):
    """Create a minimal ModelProvider subclass for testing."""
    from providers.base import ModelProvider

    class FakeProvider(ModelProvider):
        def get_provider_type(self):
            return ProviderType.CUSTOM

        def generate_content(self, prompt, model_name, **kwargs):
            return _make_model_response(content=f"sync: {prompt}", model_name=model_name)

    with patch.dict(os.environ, {"CIRCUIT_FAILURE_THRESHOLD": str(failure_threshold)}):
        return FakeProvider(api_key="test")


# -----------------------------------------------------------------------
# 3.1 async_generate_content tests
# -----------------------------------------------------------------------


class TestAsyncGenerateContent:
    @pytest.mark.asyncio
    async def test_default_wraps_sync_generate_content(self):
        provider = _make_provider()
        result = await provider.async_generate_content("hello", "test-model")
        assert isinstance(result, ModelResponse)
        assert result.content == "sync: hello"
        assert result.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_passes_all_parameters(self):
        provider = _make_provider()
        with patch.object(provider, "generate_content", return_value=_make_model_response()) as mock:
            await provider.async_generate_content(
                "prompt",
                "model",
                system_prompt="sys",
                temperature=0.5,
                max_output_tokens=100,
                thinking_mode="high",
            )
            mock.assert_called_once_with(
                "prompt",
                "model",
                system_prompt="sys",
                temperature=0.5,
                max_output_tokens=100,
                thinking_mode="high",
            )

    @pytest.mark.asyncio
    async def test_exception_propagation_from_thread(self):
        provider = _make_provider()
        with patch.object(provider, "generate_content", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                await provider.async_generate_content("prompt", "model")


# -----------------------------------------------------------------------
# 3.2 Event loop not blocked test
# -----------------------------------------------------------------------


class TestEventLoopNotBlocked:
    @pytest.mark.asyncio
    async def test_async_generate_does_not_starve_event_loop(self):
        from providers.base import ModelProvider

        class SlowProvider(ModelProvider):
            def get_provider_type(self):
                return ProviderType.CUSTOM

            def generate_content(self, prompt, model_name, **kwargs):
                time.sleep(0.1)  # Simulate blocking I/O
                return _make_model_response()

        with patch.dict(os.environ, {"CIRCUIT_FAILURE_THRESHOLD": "5"}):
            provider = SlowProvider(api_key="test")

        sleep_completed = False

        async def short_sleep():
            nonlocal sleep_completed
            await asyncio.sleep(0.01)
            sleep_completed = True

        # Both should run concurrently — sleep should NOT be starved
        results = await asyncio.gather(
            provider.async_generate_content("hello", "model"),
            short_sleep(),
        )
        assert sleep_completed
        assert isinstance(results[0], ModelResponse)


# -----------------------------------------------------------------------
# 3.3 _run_with_retries_async tests
# -----------------------------------------------------------------------


class TestRunWithRetriesAsync:
    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        provider = _make_provider()
        result = await provider._run_with_retries_async(
            AsyncMock(return_value="ok"),
            max_attempts=3,
            log_prefix="test",
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        provider = _make_provider()
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection refused")
            return "recovered"

        result = await provider._run_with_retries_async(
            flaky,
            max_attempts=3,
            log_prefix="test",
        )
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self):
        provider = _make_provider()
        call_count = 0

        async def fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")  # Not retryable

        with pytest.raises(ValueError, match="bad input"):
            await provider._run_with_retries_async(
                fails,
                max_attempts=3,
                log_prefix="test",
            )
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        provider = _make_provider()

        async def always_fails():
            raise ConnectionError("connection refused")

        with pytest.raises(ConnectionError):
            await provider._run_with_retries_async(
                always_fails,
                max_attempts=2,
                log_prefix="test",
            )

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        provider = _make_provider(failure_threshold=1)

        async def fails():
            raise RuntimeError("down")

        with pytest.raises(RuntimeError):
            await provider._run_with_retries_async(fails, max_attempts=1, log_prefix="test")

        # Circuit should now be open
        with pytest.raises(ProviderUnavailable):
            await provider._run_with_retries_async(fails, max_attempts=1, log_prefix="test")


# -----------------------------------------------------------------------
# 3.4 _consult_models_concurrently tests
# -----------------------------------------------------------------------


class TestConsultModelsConcurrently:
    @pytest.mark.asyncio
    async def test_all_models_called_concurrently(self):
        from tools.consensus import ConsensusTool

        tool = ConsensusTool()
        call_order = []

        async def fake_consult(config, req):
            call_order.append(config["model"])
            await asyncio.sleep(0.01)
            return {"model": config["model"], "status": "success", "verdict": "ok"}

        with patch.object(tool, "_consult_model", side_effect=fake_consult):
            configs = [
                {"model": "m1", "stance": "neutral"},
                {"model": "m2", "stance": "for"},
                {"model": "m3", "stance": "against"},
            ]
            results = await tool._consult_models_concurrently(configs, MagicMock())

        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)
        assert {r["model"] for r in results} == {"m1", "m2", "m3"}

    @pytest.mark.asyncio
    async def test_per_model_error_isolation(self):
        from tools.consensus import ConsensusTool

        tool = ConsensusTool()

        async def fake_consult(config, req):
            if config["model"] == "m2":
                raise RuntimeError("m2 exploded")
            return {"model": config["model"], "status": "success", "verdict": "ok"}

        with patch.object(tool, "_consult_model", side_effect=fake_consult):
            configs = [
                {"model": "m1", "stance": "neutral"},
                {"model": "m2", "stance": "for"},
                {"model": "m3", "stance": "against"},
            ]
            results = await tool._consult_models_concurrently(configs, MagicMock())

        assert len(results) == 3
        success = [r for r in results if r["status"] == "success"]
        errors = [r for r in results if r["status"] == "error"]
        assert len(success) == 2
        assert len(errors) == 1
        assert errors[0]["model"] == "m2"

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        from tools.consensus import ConsensusTool

        tool = ConsensusTool()

        async def slow_consult(config, req):
            if config["model"] == "slow":
                await asyncio.sleep(999)  # Will be cancelled by timeout
            return {"model": config["model"], "status": "success", "verdict": "ok"}

        # Patch the timeout to be very short for testing
        with patch.object(tool, "_consult_model", side_effect=slow_consult):
            configs = [
                {"model": "fast", "stance": "neutral"},
                {"model": "slow", "stance": "for"},
            ]

            async def patched_concurrent(model_configs, request):
                per_model_timeout = 0.05  # 50ms timeout for test

                async def _consult_with_timeout(config):
                    try:
                        return await asyncio.wait_for(
                            tool._consult_model(config, request),
                            timeout=per_model_timeout,
                        )
                    except asyncio.TimeoutError:
                        return {
                            "model": config.get("model", "unknown"),
                            "stance": config.get("stance", "neutral"),
                            "status": "error",
                            "error": f"timeout after {per_model_timeout}s",
                        }

                tasks = [_consult_with_timeout(c) for c in model_configs]
                return await asyncio.gather(*tasks)

            results = await patched_concurrent(configs, MagicMock())

        assert len(results) == 2
        fast_result = next(r for r in results if r["model"] == "fast")
        slow_result = next(r for r in results if r["model"] == "slow")
        assert fast_result["status"] == "success"
        assert slow_result["status"] == "error"
        assert "timeout" in slow_result["error"]


# -----------------------------------------------------------------------
# 3.5 _consult_model calls async path
# -----------------------------------------------------------------------


class TestConsultModelUsesAsyncPath:
    @pytest.mark.asyncio
    async def test_consult_model_calls_async_generate_content(self):
        from tools.consensus import ConsensusTool

        tool = ConsensusTool()
        tool.original_proposal = "test proposal"
        tool.initial_prompt = "test proposal"

        mock_provider = MagicMock()
        mock_provider.get_provider_type.return_value = ProviderType.CUSTOM
        mock_provider.async_generate_content = AsyncMock(return_value=_make_model_response())

        request = MagicMock()
        request.relevant_files = []
        request.images = []

        with patch.object(tool, "get_model_provider", return_value=mock_provider):
            with patch.object(tool, "validate_and_correct_temperature", return_value=(0.3, [])):
                with patch.object(tool, "_get_stance_enhanced_prompt", return_value="sys prompt"):
                    result = await tool._consult_model({"model": "test", "stance": "neutral"}, request)

        mock_provider.async_generate_content.assert_awaited_once()
        assert result["status"] == "success"
