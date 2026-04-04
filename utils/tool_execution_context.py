"""
Typed execution context passed from the MCP boundary to tool implementations.

Replaces the ad-hoc underscore-key convention (_model_context, _resolved_model_name,
_remaining_tokens, _original_user_prompt) with a single typed dataclass injected
at arguments["_context"].
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from utils.model_context import ModelContext

if TYPE_CHECKING:
    from providers.registry import ModelProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionContext:
    """Typed container for server-injected execution state.

    Created by server.py's handle_call_tool() and reconstruct_thread_context(),
    then consumed by tool base classes and individual tools via arguments["_context"].

    Attributes:
        model_context: ModelContext with token allocation and capability info.
        resolved_model_name: The resolved model name after alias/auto resolution.
        remaining_tokens: Token budget remaining after conversation history (0 if not a continuation).
        original_user_prompt: The user's original prompt before conversation history enhancement ("" if not a continuation).
        registry: The ModelProviderRegistry instance for this request (None for legacy/test paths).
    """

    model_context: ModelContext
    resolved_model_name: str
    remaining_tokens: int = 0
    original_user_prompt: str = ""
    registry: Optional["ModelProviderRegistry"] = field(default=None, repr=False)

    @classmethod
    def from_arguments(cls, args: dict) -> Optional["ToolExecutionContext"]:
        """Extract a ToolExecutionContext from a tool arguments dict.

        Checks for the new-style ``_context`` key first, then falls back to
        legacy underscore keys with a deprecation warning.

        Args:
            args: The tool arguments dict (may contain ``_context`` or legacy keys).

        Returns:
            ToolExecutionContext if found, None if neither new nor legacy keys are present.
        """
        # Preferred: new-style _context key
        ctx = args.get("_context")
        if isinstance(ctx, ToolExecutionContext):
            return ctx

        # Legacy fallback: individual underscore keys
        model_context = args.get("_model_context")
        if model_context is not None:
            logger.warning(
                "Legacy underscore keys detected in arguments. "
                "Migrate to arguments['_context'] (ToolExecutionContext)."
            )
            return cls(
                model_context=model_context,
                resolved_model_name=args.get("_resolved_model_name", ""),
                remaining_tokens=args.get("_remaining_tokens", 0),
                original_user_prompt=args.get("_original_user_prompt", ""),
            )

        return None
