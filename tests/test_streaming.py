"""Tests for streaming provider interface, default wrapper, and MCP progress bridge."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ModelProvider, StreamChunk
from providers.shared import ModelResponse, ProviderType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_response(content: str = "full response", model_name: str = "test-model") -> ModelResponse:
    return ModelResponse(
        content=content,
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        model_name=model_name,
        friendly_name="Test",
        provider=ProviderType.CUSTOM,
        metadata={},
    )


def _make_provider():
    """Create a minimal ModelProvider subclass with default streaming."""

    class FakeProvider(ModelProvider):
        def get_provider_type(self):
            return ProviderType.CUSTOM

        def generate_content(self, prompt, model_name, **kwargs):
            return _make_model_response(content=f"response to: {prompt}")

    with patch.dict(os.environ, {"CIRCUIT_FAILURE_THRESHOLD": "5"}):
        return FakeProvider(api_key="test")


def _make_native_streaming_provider(chunks: list[StreamChunk]):
    """Provider that overrides generate_content_stream with pre-defined chunks."""

    class NativeStreamProvider(ModelProvider):
        def get_provider_type(self):
            return ProviderType.CUSTOM

        def generate_content(self, prompt, model_name, **kwargs):
            return _make_model_response(content="non-streaming fallback")

        def generate_content_stream(self, prompt, model_name, **kwargs):
            yield from chunks

    with patch.dict(os.environ, {"CIRCUIT_FAILURE_THRESHOLD": "5"}):
        return NativeStreamProvider(api_key="test")


def _make_error_streaming_provider(error_after: int = 2):
    """Provider that errors mid-stream after yielding some chunks."""

    class ErrorStreamProvider(ModelProvider):
        def get_provider_type(self):
            return ProviderType.CUSTOM

        def generate_content(self, prompt, model_name, **kwargs):
            return _make_model_response()

        def generate_content_stream(self, prompt, model_name, **kwargs):
            for i in range(error_after):
                yield StreamChunk(text=f"chunk{i} ", is_final=False)
            raise ConnectionError("Stream interrupted")

    with patch.dict(os.environ, {"CIRCUIT_FAILURE_THRESHOLD": "5"}):
        return ErrorStreamProvider(api_key="test")


# ---------------------------------------------------------------------------
# 7.1 Default single-chunk wrapper tests
# ---------------------------------------------------------------------------


class TestDefaultStreamWrapper:
    def test_yields_single_chunk_with_full_content(self):
        provider = _make_provider()
        chunks = list(provider.generate_content_stream("hello", "test-model"))
        assert len(chunks) == 1
        assert chunks[0].text == "response to: hello"
        assert chunks[0].is_final is True

    def test_usage_from_default_wrapper(self):
        provider = _make_provider()
        chunks = list(provider.generate_content_stream("hello", "test-model"))
        assert chunks[0].usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

    def test_default_wrapper_preserves_response_content(self):
        """Concatenated stream text == generate_content().content."""
        provider = _make_provider()
        non_stream = provider.generate_content("hello", "test-model")
        stream_text = "".join(c.text for c in provider.generate_content_stream("hello", "test-model"))
        assert stream_text == non_stream.content

    def test_default_wrapper_propagates_exceptions(self):
        """If generate_content() raises, generate_content_stream() propagates."""

        class FailProvider(ModelProvider):
            def get_provider_type(self):
                return ProviderType.CUSTOM

            def generate_content(self, prompt, model_name, **kwargs):
                raise ValueError("bad request")

        with patch.dict(os.environ, {"CIRCUIT_FAILURE_THRESHOLD": "5"}):
            provider = FailProvider(api_key="test")

        with pytest.raises(ValueError, match="bad request"):
            list(provider.generate_content_stream("hello", "test-model"))


# ---------------------------------------------------------------------------
# 7.2 Native streaming with mock provider
# ---------------------------------------------------------------------------


class TestNativeStreaming:
    def test_multiple_chunks_yielded_in_order(self):
        chunks_input = [
            StreamChunk(text="Hello ", is_final=False),
            StreamChunk(text="world", is_final=False),
            StreamChunk(text="!", is_final=True, usage={"total_tokens": 10}),
        ]
        provider = _make_native_streaming_provider(chunks_input)
        result = list(provider.generate_content_stream("test", "model"))

        assert len(result) == 3
        assert [c.text for c in result] == ["Hello ", "world", "!"]
        assert result[-1].is_final is True

    def test_final_chunk_has_is_final_true(self):
        chunks_input = [
            StreamChunk(text="only chunk", is_final=True, usage={"total_tokens": 5}),
        ]
        provider = _make_native_streaming_provider(chunks_input)
        result = list(provider.generate_content_stream("test", "model"))
        assert result[-1].is_final is True
        assert result[-1].usage == {"total_tokens": 5}


# ---------------------------------------------------------------------------
# 7.3 StreamProgressNotifier tests
# ---------------------------------------------------------------------------


class TestStreamProgressNotifier:
    @pytest.mark.asyncio
    async def test_notify_chunk_sends_progress(self):
        from utils.streaming import StreamProgressNotifier

        mock_session = AsyncMock()
        mock_server = MagicMock()
        mock_server.request_context = MagicMock()
        mock_server.request_context.session = mock_session

        notifier = StreamProgressNotifier(server=mock_server, progress_token="tok-1", min_chunk_size=1)

        await notifier.notify_chunk(StreamChunk(text="Hello", is_final=False))
        mock_session.send_progress_notification.assert_called()

    @pytest.mark.asyncio
    async def test_final_chunk_sends_notification(self):
        from utils.streaming import StreamProgressNotifier

        mock_session = AsyncMock()
        mock_server = MagicMock()
        mock_server.request_context = MagicMock()
        mock_server.request_context.session = mock_session

        notifier = StreamProgressNotifier(server=mock_server, progress_token="tok-1", min_chunk_size=1)

        await notifier.notify_chunk(StreamChunk(text="done", is_final=True))
        assert mock_session.send_progress_notification.call_count >= 1

    @pytest.mark.asyncio
    async def test_no_op_without_progress_token(self):
        from utils.streaming import StreamProgressNotifier

        mock_server = MagicMock()
        notifier = StreamProgressNotifier(server=mock_server, progress_token=None)

        # Should not raise
        await notifier.notify_chunk(StreamChunk(text="Hello", is_final=False))
        await notifier.notify_chunk(StreamChunk(text="", is_final=True))

    @pytest.mark.asyncio
    async def test_no_op_when_session_unavailable(self):
        from utils.streaming import StreamProgressNotifier

        mock_server = MagicMock()
        mock_server.request_context = property(lambda self: (_ for _ in ()).throw(LookupError))

        notifier = StreamProgressNotifier(server=mock_server, progress_token="tok-1")
        # Should not raise even though session lookup fails
        await notifier.notify_chunk(StreamChunk(text="test", is_final=True))


# ---------------------------------------------------------------------------
# 7.4 Rate limiting in StreamProgressNotifier
# ---------------------------------------------------------------------------


class TestStreamProgressNotifierRateLimiting:
    @pytest.mark.asyncio
    async def test_small_chunks_are_batched(self):
        from utils.streaming import StreamProgressNotifier

        mock_session = AsyncMock()
        mock_server = MagicMock()
        mock_server.request_context = MagicMock()
        mock_server.request_context.session = mock_session

        # High min_chunk_size so individual small chunks are buffered
        notifier = StreamProgressNotifier(
            server=mock_server,
            progress_token="tok-1",
            min_chunk_size=50,
            min_interval=10.0,  # large interval to prevent time-based flush
        )

        # Send many small chunks (each < 50 chars)
        for i in range(5):
            await notifier.notify_chunk(StreamChunk(text=f"w{i} ", is_final=False))

        # None should have triggered a flush since total is ~15 chars < 50
        assert mock_session.send_progress_notification.call_count == 0

        # Final chunk always flushes
        await notifier.notify_chunk(StreamChunk(text="end", is_final=True))
        assert mock_session.send_progress_notification.call_count >= 1


# ---------------------------------------------------------------------------
# 7.5 Assembled response matches non-streaming path
# ---------------------------------------------------------------------------


class TestResponseAssembly:
    def test_concatenated_chunks_match_full_response(self):
        full_text = "The quick brown fox jumps over the lazy dog."
        words = full_text.split(" ")
        chunks = []
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            text = word if is_last else word + " "
            chunks.append(StreamChunk(text=text, is_final=is_last, usage={"total_tokens": 10} if is_last else None))

        provider = _make_native_streaming_provider(chunks)
        assembled = "".join(c.text for c in provider.generate_content_stream("test", "model"))
        assert assembled == full_text


# ---------------------------------------------------------------------------
# 7.6 Error handling during streaming
# ---------------------------------------------------------------------------


class TestStreamingErrorHandling:
    def test_error_mid_stream_raises(self):
        provider = _make_error_streaming_provider(error_after=2)

        collected = []
        with pytest.raises(ConnectionError, match="Stream interrupted"):
            for chunk in provider.generate_content_stream("test", "model"):
                collected.append(chunk)

        # Chunks yielded before the error are valid
        assert len(collected) == 2
        assert collected[0].text == "chunk0 "
        assert collected[1].text == "chunk1 "

    def test_error_on_first_chunk(self):
        provider = _make_error_streaming_provider(error_after=0)
        with pytest.raises(ConnectionError, match="Stream interrupted"):
            list(provider.generate_content_stream("test", "model"))
