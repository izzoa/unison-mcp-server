"""
Test suite for conversation_store module.

Tests thread lifecycle management: creation, retrieval, turn management,
chain traversal, and UUID validation.
"""

from unittest.mock import Mock, patch

import pytest

from utils.conversation_store import (
    CONVERSATION_TIMEOUT_SECONDS,
    MAX_CONVERSATION_TURNS,
    ConversationTurn,
    ThreadContext,
    _is_valid_uuid,
    add_turn,
    create_thread,
    get_storage,
    get_thread,
    get_thread_chain,
)


class TestConversationStore:
    """Test thread lifecycle operations in conversation_store."""

    @patch("utils.conversation_store.get_storage")
    def test_create_thread(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client

        thread_id = create_thread("chat", {"prompt": "Hello"})

        assert thread_id is not None
        assert len(thread_id) == 36
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][0] == f"thread:{thread_id}"
        assert call_args[0][1] == CONVERSATION_TIMEOUT_SECONDS

    @patch("utils.conversation_store.get_storage")
    def test_create_thread_with_parent(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client

        parent_id = "11111111-1111-1111-1111-111111111111"
        thread_id = create_thread("chat", {"prompt": "Hello"}, parent_thread_id=parent_id)

        assert thread_id is not None
        call_args = mock_client.setex.call_args
        stored_json = call_args[0][2]
        context = ThreadContext.model_validate_json(stored_json)
        assert context.parent_thread_id == parent_id

    @patch("utils.conversation_store.get_storage")
    def test_create_thread_filters_non_serializable(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client

        create_thread("chat", {"prompt": "Hello", "temperature": 0.5, "model": "test", "continuation_id": "abc"})

        call_args = mock_client.setex.call_args
        stored_json = call_args[0][2]
        context = ThreadContext.model_validate_json(stored_json)
        assert "prompt" in context.initial_context
        assert "temperature" not in context.initial_context
        assert "model" not in context.initial_context
        assert "continuation_id" not in context.initial_context

    @patch("utils.conversation_store.get_storage")
    def test_get_thread_valid(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client

        thread_id = "11111111-1111-1111-1111-111111111111"
        context = ThreadContext(
            thread_id=thread_id,
            created_at="2024-01-01T00:00:00+00:00",
            last_updated_at="2024-01-01T00:00:00+00:00",
            tool_name="chat",
            turns=[],
            initial_context={},
        )
        mock_client.get.return_value = context.model_dump_json()

        result = get_thread(thread_id)
        assert result is not None
        assert result.thread_id == thread_id

    @patch("utils.conversation_store.get_storage")
    def test_get_thread_invalid_uuid(self, mock_storage):
        result = get_thread("not-a-uuid")
        assert result is None

    @patch("utils.conversation_store.get_storage")
    def test_get_thread_empty_id(self, mock_storage):
        result = get_thread("")
        assert result is None

    @patch("utils.conversation_store.get_storage")
    def test_get_thread_not_found(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client
        mock_client.get.return_value = None

        result = get_thread("11111111-1111-1111-1111-111111111111")
        assert result is None

    @patch("utils.conversation_store.get_storage")
    def test_add_turn_success(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client

        thread_id = "11111111-1111-1111-1111-111111111111"
        context = ThreadContext(
            thread_id=thread_id,
            created_at="2024-01-01T00:00:00+00:00",
            last_updated_at="2024-01-01T00:00:00+00:00",
            tool_name="chat",
            turns=[],
            initial_context={},
        )
        mock_client.get.return_value = context.model_dump_json()

        result = add_turn(thread_id, "user", "Hello", tool_name="chat")
        assert result is True
        mock_client.setex.assert_called_once()

    @patch("utils.conversation_store.get_storage")
    def test_add_turn_thread_not_found(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client
        mock_client.get.return_value = None

        result = add_turn("11111111-1111-1111-1111-111111111111", "user", "Hello")
        assert result is False

    @patch("utils.conversation_store.get_storage")
    def test_add_turn_max_turns_reached(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client

        thread_id = "11111111-1111-1111-1111-111111111111"
        turns = [
            ConversationTurn(role="user", content=f"Turn {i}", timestamp="2024-01-01T00:00:00+00:00")
            for i in range(MAX_CONVERSATION_TURNS)
        ]
        context = ThreadContext(
            thread_id=thread_id,
            created_at="2024-01-01T00:00:00+00:00",
            last_updated_at="2024-01-01T00:00:00+00:00",
            tool_name="chat",
            turns=turns,
            initial_context={},
        )
        mock_client.get.return_value = context.model_dump_json()

        result = add_turn(thread_id, "user", "One more")
        assert result is False

    @patch("utils.conversation_store.get_storage")
    def test_add_turn_with_metadata(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client

        thread_id = "11111111-1111-1111-1111-111111111111"
        context = ThreadContext(
            thread_id=thread_id,
            created_at="2024-01-01T00:00:00+00:00",
            last_updated_at="2024-01-01T00:00:00+00:00",
            tool_name="chat",
            turns=[],
            initial_context={},
        )
        mock_client.get.return_value = context.model_dump_json()

        result = add_turn(
            thread_id,
            "assistant",
            "Response",
            files=["/test.py"],
            images=["/img.png"],
            tool_name="analyze",
            model_provider="google",
            model_name="gemini-2.5-flash",
            model_metadata={"thinking": True},
        )
        assert result is True

    @patch("utils.conversation_store.get_storage")
    def test_get_thread_chain_single(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client

        thread_id = "11111111-1111-1111-1111-111111111111"
        context = ThreadContext(
            thread_id=thread_id,
            created_at="2024-01-01T00:00:00+00:00",
            last_updated_at="2024-01-01T00:00:00+00:00",
            tool_name="chat",
            turns=[],
            initial_context={},
        )
        mock_client.get.return_value = context.model_dump_json()

        chain = get_thread_chain(thread_id)
        assert len(chain) == 1
        assert chain[0].thread_id == thread_id

    @patch("utils.conversation_store.get_storage")
    def test_get_thread_chain_not_found(self, mock_storage):
        mock_client = Mock()
        mock_storage.return_value = mock_client
        mock_client.get.return_value = None

        chain = get_thread_chain("11111111-1111-1111-1111-111111111111")
        assert chain == []


class TestIsValidUuid:
    """Test UUID validation."""

    def test_valid_uuid(self):
        assert _is_valid_uuid("11111111-1111-1111-1111-111111111111") is True

    def test_invalid_uuid(self):
        assert _is_valid_uuid("not-a-uuid") is False

    def test_empty_string(self):
        assert _is_valid_uuid("") is False


class TestDataModels:
    """Test ConversationTurn and ThreadContext models."""

    def test_conversation_turn_minimal(self):
        turn = ConversationTurn(role="user", content="Hello", timestamp="2024-01-01T00:00:00+00:00")
        assert turn.role == "user"
        assert turn.files is None
        assert turn.images is None

    def test_conversation_turn_full(self):
        turn = ConversationTurn(
            role="assistant",
            content="Response",
            timestamp="2024-01-01T00:00:00+00:00",
            files=["/a.py"],
            images=["/b.png"],
            tool_name="analyze",
            model_provider="google",
            model_name="gemini-2.5-flash",
            model_metadata={"thinking": True},
        )
        assert turn.files == ["/a.py"]
        assert turn.model_metadata == {"thinking": True}

    def test_thread_context_serialization(self):
        context = ThreadContext(
            thread_id="test-id",
            created_at="2024-01-01T00:00:00+00:00",
            last_updated_at="2024-01-01T00:00:00+00:00",
            tool_name="chat",
            turns=[],
            initial_context={"prompt": "Hello"},
        )
        json_str = context.model_dump_json()
        restored = ThreadContext.model_validate_json(json_str)
        assert restored.thread_id == "test-id"
        assert restored.initial_context == {"prompt": "Hello"}
