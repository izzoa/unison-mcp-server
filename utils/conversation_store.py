"""
Conversation Store — Thread lifecycle management for conversation memory.

This module handles the CRUD operations for conversation threads:
- Creating threads with unique UUIDs and TTL
- Retrieving threads with automatic expiry enforcement
- Appending conversation turns with metadata
- Traversing thread chains for cross-tool continuation

Extracted from conversation_memory.py as part of the decomposition into
focused modules (conversation_store, context_reconstructor, facade).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel

from utils.env import get_env

logger = logging.getLogger(__name__)

# Configuration constants
# Get max conversation turns from environment, default to 20 turns (10 exchanges)
try:
    max_turns_raw = (get_env("MAX_CONVERSATION_TURNS", "50") or "50").strip()
    MAX_CONVERSATION_TURNS = int(max_turns_raw)
    if MAX_CONVERSATION_TURNS <= 0:
        logger.warning(f"Invalid MAX_CONVERSATION_TURNS value ({MAX_CONVERSATION_TURNS}), using default of 50 turns")
        MAX_CONVERSATION_TURNS = 50
except ValueError:
    logger.warning(
        f"Invalid MAX_CONVERSATION_TURNS value ('{get_env('MAX_CONVERSATION_TURNS')}'), using default of 50 turns"
    )
    MAX_CONVERSATION_TURNS = 50

# Get conversation timeout from environment (in hours), default to 3 hours
try:
    timeout_raw = (get_env("CONVERSATION_TIMEOUT_HOURS", "3") or "3").strip()
    CONVERSATION_TIMEOUT_HOURS = int(timeout_raw)
    if CONVERSATION_TIMEOUT_HOURS <= 0:
        logger.warning(
            f"Invalid CONVERSATION_TIMEOUT_HOURS value ({CONVERSATION_TIMEOUT_HOURS}), using default of 3 hours"
        )
        CONVERSATION_TIMEOUT_HOURS = 3
except ValueError:
    logger.warning(
        f"Invalid CONVERSATION_TIMEOUT_HOURS value ('{get_env('CONVERSATION_TIMEOUT_HOURS')}'), using default of 3 hours"
    )
    CONVERSATION_TIMEOUT_HOURS = 3

CONVERSATION_TIMEOUT_SECONDS = CONVERSATION_TIMEOUT_HOURS * 3600


class ConversationTurn(BaseModel):
    """
    Single turn in a conversation

    Represents one exchange in the AI-to-AI conversation, tracking both
    the content and metadata needed for cross-tool continuation.

    Attributes:
        role: "user" (Agent request) or "assistant" (model response)
        content: The actual message content/response
        timestamp: ISO timestamp when this turn was created
        files: List of file paths referenced in this specific turn
        images: List of image paths referenced in this specific turn
        tool_name: Which tool generated this turn (for cross-tool tracking)
        model_provider: Provider used (e.g., "google", "openai")
        model_name: Specific model used (e.g., "gemini-2.5-flash", "o3-mini")
        model_metadata: Additional model-specific metadata (e.g., thinking mode, token usage)
    """

    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    files: Optional[list[str]] = None  # Files referenced in this turn
    images: Optional[list[str]] = None  # Images referenced in this turn
    tool_name: Optional[str] = None  # Tool used for this turn
    model_provider: Optional[str] = None  # Model provider (google, openai, etc)
    model_name: Optional[str] = None  # Specific model used
    model_metadata: Optional[dict[str, Any]] = None  # Additional model info


class ThreadContext(BaseModel):
    """
    Complete conversation context for a thread

    Contains all information needed to reconstruct a conversation state
    across different tools and request cycles. This is the core data
    structure that enables cross-tool continuation.

    Attributes:
        thread_id: UUID identifying this conversation thread
        parent_thread_id: UUID of parent thread (for conversation chains)
        created_at: ISO timestamp when thread was created
        last_updated_at: ISO timestamp of last modification
        tool_name: Name of the tool that initiated this thread
        turns: List of all conversation turns in chronological order
        initial_context: Original request data that started the conversation
    """

    thread_id: str
    parent_thread_id: Optional[str] = None  # Parent thread for conversation chains
    created_at: str
    last_updated_at: str
    tool_name: str  # Tool that created this thread (preserved for attribution)
    turns: list[ConversationTurn]
    initial_context: dict[str, Any]  # Original request parameters


def _is_valid_uuid(val: str) -> bool:
    """
    Validate UUID format for security

    Ensures thread IDs are valid UUIDs to prevent injection attacks
    and malformed requests.

    Args:
        val: String to validate as UUID

    Returns:
        bool: True if valid UUID format, False otherwise
    """
    try:
        uuid.UUID(val)
        return True
    except ValueError:
        return False


def get_storage():
    """
    Get in-memory storage backend for conversation persistence.

    Returns:
        StorageBackend: Thread-safe storage backend
    """
    from .storage_backend import get_storage_backend

    return get_storage_backend()


def create_thread(tool_name: str, initial_request: dict[str, Any], parent_thread_id: Optional[str] = None) -> str:
    """
    Create new conversation thread and return thread ID

    Initializes a new conversation thread for AI-to-AI discussions.
    This is called when a tool wants to enable follow-up conversations
    or when Claude explicitly starts a multi-turn interaction.

    Args:
        tool_name: Name of the tool creating this thread (e.g., "analyze", "chat")
        initial_request: Original request parameters (will be filtered for serialization)
        parent_thread_id: Optional parent thread ID for conversation chains

    Returns:
        str: UUID thread identifier that can be used for continuation

    Note:
        - Thread expires after the configured timeout (default: 3 hours)
        - Non-serializable parameters are filtered out automatically
        - Thread can be continued by any tool using the returned UUID
        - Parent thread creates a chain for conversation history traversal
    """
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Filter out non-serializable parameters to avoid JSON encoding issues
    filtered_context = {
        k: v
        for k, v in initial_request.items()
        if k not in ["temperature", "thinking_mode", "model", "continuation_id"]
    }

    context = ThreadContext(
        thread_id=thread_id,
        parent_thread_id=parent_thread_id,  # Link to parent for conversation chains
        created_at=now,
        last_updated_at=now,
        tool_name=tool_name,  # Track which tool initiated this conversation
        turns=[],  # Empty initially, turns added via add_turn()
        initial_context=filtered_context,
    )

    # Store in memory with configurable TTL to prevent indefinite accumulation
    storage = get_storage()
    key = f"thread:{thread_id}"
    storage.setex(key, CONVERSATION_TIMEOUT_SECONDS, context.model_dump_json())

    logger.debug(f"[THREAD] Created new thread {thread_id} with parent {parent_thread_id}")

    return thread_id


def get_thread(thread_id: str) -> Optional[ThreadContext]:
    """
    Retrieve thread context from in-memory storage

    Fetches complete conversation context for cross-tool continuation.
    This is the core function that enables tools to access conversation
    history from previous interactions.

    Args:
        thread_id: UUID of the conversation thread

    Returns:
        ThreadContext: Complete conversation context if found
        None: If thread doesn't exist, expired, or invalid UUID

    Security:
        - Validates UUID format to prevent injection attacks
        - Handles storage connection failures gracefully
        - No error information leakage on failure
    """
    if not thread_id or not _is_valid_uuid(thread_id):
        return None

    try:
        storage = get_storage()
        key = f"thread:{thread_id}"
        data = storage.get(key)

        if data:
            return ThreadContext.model_validate_json(data)
        return None
    except Exception:
        # Silently handle errors to avoid exposing storage details
        return None


def add_turn(
    thread_id: str,
    role: str,
    content: str,
    files: Optional[list[str]] = None,
    images: Optional[list[str]] = None,
    tool_name: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    model_metadata: Optional[dict[str, Any]] = None,
) -> bool:
    """
    Add turn to existing thread with atomic file ordering.

    Appends a new conversation turn to an existing thread. This is the core
    function for building conversation history and enabling cross-tool
    continuation. Each turn preserves the tool and model that generated it.

    Args:
        thread_id: UUID of the conversation thread
        role: "user" (Agent request) or "assistant" (model response)
        content: The actual message/response content
        files: Optional list of files referenced in this turn
        images: Optional list of images referenced in this turn
        tool_name: Name of the tool adding this turn (for attribution)
        model_provider: Provider used (e.g., "google", "openai")
        model_name: Specific model used (e.g., "gemini-2.5-flash", "o3-mini")
        model_metadata: Additional model info (e.g., thinking mode, token usage)

    Returns:
        bool: True if turn was successfully added, False otherwise

    Failure cases:
        - Thread doesn't exist or expired
        - Maximum turn limit reached
        - Storage connection failure

    Note:
        - Refreshes thread TTL to configured timeout on successful update
        - Turn limits prevent runaway conversations
        - File references are preserved for cross-tool access with atomic ordering
        - Image references are preserved for cross-tool visual context
        - Model information enables cross-provider conversations
    """
    logger.debug(f"[FLOW] Adding {role} turn to {thread_id} ({tool_name})")

    context = get_thread(thread_id)
    if not context:
        logger.debug(f"[FLOW] Thread {thread_id} not found for turn addition")
        return False

    # Check turn limit to prevent runaway conversations
    if len(context.turns) >= MAX_CONVERSATION_TURNS:
        logger.debug(f"[FLOW] Thread {thread_id} at max turns ({MAX_CONVERSATION_TURNS})")
        return False

    # Create new turn with complete metadata
    turn = ConversationTurn(
        role=role,
        content=content,
        timestamp=datetime.now(timezone.utc).isoformat(),
        files=files,  # Preserved for cross-tool file context
        images=images,  # Preserved for cross-tool visual context
        tool_name=tool_name,  # Track which tool generated this turn
        model_provider=model_provider,  # Track model provider
        model_name=model_name,  # Track specific model
        model_metadata=model_metadata,  # Additional model info
    )

    context.turns.append(turn)
    context.last_updated_at = datetime.now(timezone.utc).isoformat()

    # Save back to storage and refresh TTL
    try:
        storage = get_storage()
        key = f"thread:{thread_id}"
        storage.setex(key, CONVERSATION_TIMEOUT_SECONDS, context.model_dump_json())  # Refresh TTL to configured timeout
        return True
    except Exception as e:
        logger.debug(f"[FLOW] Failed to save turn to storage: {type(e).__name__}")
        return False


def get_thread_chain(thread_id: str, max_depth: int = 20) -> list[ThreadContext]:
    """
    Traverse the parent chain to get all threads in conversation sequence.

    Retrieves the complete conversation chain by following parent_thread_id
    links. Returns threads in chronological order (oldest first).

    Args:
        thread_id: Starting thread ID
        max_depth: Maximum chain depth to prevent infinite loops

    Returns:
        list[ThreadContext]: All threads in chain, oldest first
    """
    chain = []
    current_id = thread_id
    seen_ids = set()

    # Build chain from current to oldest
    while current_id and len(chain) < max_depth:
        # Prevent circular references
        if current_id in seen_ids:
            logger.warning(f"[THREAD] Circular reference detected in thread chain at {current_id}")
            break

        seen_ids.add(current_id)

        context = get_thread(current_id)
        if not context:
            logger.debug(f"[THREAD] Thread {current_id} not found in chain traversal")
            break

        chain.append(context)
        current_id = context.parent_thread_id

    # Reverse to get chronological order (oldest first)
    chain.reverse()

    logger.debug(f"[THREAD] Retrieved chain of {len(chain)} threads for {thread_id}")
    return chain
