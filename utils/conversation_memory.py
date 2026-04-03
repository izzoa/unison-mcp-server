"""
Conversation Memory — Facade module for backward-compatible imports.

This module serves as a thin facade that re-exports all public symbols from
the decomposed conversation memory subsystem:

- ``conversation_store`` — Thread lifecycle: create, retrieve, add turns, chain traversal,
  data models (ConversationTurn, ThreadContext), and configuration constants.
- ``context_reconstructor`` — History building: file/image collection with newest-first
  deduplication, token-budget-aware history construction, and turn formatting.

All existing imports from ``utils.conversation_memory`` continue to work unchanged.
New code may import directly from ``utils.conversation_store`` or
``utils.context_reconstructor`` for clarity.

ARCHITECTURE NOTE:
The original ``conversation_memory.py`` (1,108 lines) was decomposed into focused
modules following the same pattern used in v10 for ``base_tool.py``. The circular
dependency with ``server.py`` (``from server import TOOLS``) was eliminated by
accepting a ``tool_formatter_fn`` callback in ``build_conversation_history()``.

For the original architecture documentation, see the module docstrings in
``conversation_store.py`` and ``context_reconstructor.py``.
"""

import uuid  # noqa: F401, I001 — re-exported for backward compat (tests access conversation_memory.uuid)

from utils.context_reconstructor import _default_turn_formatting  # noqa: F401
from utils.context_reconstructor import _get_tool_formatted_content  # noqa: F401
from utils.context_reconstructor import _plan_file_inclusion_by_size  # noqa: F401
from utils.context_reconstructor import (  # noqa: E402
    build_conversation_history,
    get_conversation_file_list,
    get_conversation_image_list,
)
from utils.conversation_store import _is_valid_uuid  # noqa: F401
from utils.conversation_store import (  # noqa: E402
    CONVERSATION_TIMEOUT_HOURS,
    CONVERSATION_TIMEOUT_SECONDS,
    MAX_CONVERSATION_TURNS,
    ConversationTurn,
    ThreadContext,
    add_turn,
    create_thread,
    get_storage,
    get_thread,
    get_thread_chain,
)

__all__ = [
    # Constants
    "MAX_CONVERSATION_TURNS",
    "CONVERSATION_TIMEOUT_HOURS",
    "CONVERSATION_TIMEOUT_SECONDS",
    # Data models
    "ConversationTurn",
    "ThreadContext",
    # Thread lifecycle (from conversation_store)
    "get_storage",
    "create_thread",
    "get_thread",
    "add_turn",
    "get_thread_chain",
    # Context reconstruction (from context_reconstructor)
    "get_conversation_file_list",
    "get_conversation_image_list",
    "build_conversation_history",
]
