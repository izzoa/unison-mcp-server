"""
Conversation handling utilities for Unison MCP Tools.

This module provides the ConversationHandler class which encapsulates
conversation-related functionality extracted from BaseTool, including
formatting conversation turns, validating prompt sizes, and providing
hooks for size validation content.
"""

import logging
from typing import Any, Optional

from config import MCP_PROMPT_SIZE_LIMIT
from utils.conversation_memory import ConversationTurn

logger = logging.getLogger(__name__)


class ConversationHandler:
    """
    Handles conversation-related operations for MCP tools.

    This class encapsulates conversation formatting and prompt size validation
    logic that was previously part of BaseTool. It can be instantiated independently
    with just a tool name.

    Attributes:
        tool_name: The name of the tool using this handler.
    """

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def format_conversation_turn(self, turn: ConversationTurn) -> list[str]:
        """
        Format a conversation turn for display in conversation history.

        Tools can override this to provide custom formatting for their responses
        while maintaining the standard structure for cross-tool compatibility.

        This method is called by build_conversation_history when reconstructing
        conversation context, allowing each tool to control how its responses
        appear in subsequent conversation turns.

        Args:
            turn: The conversation turn to format (from utils.conversation_memory)

        Returns:
            list[str]: Lines of formatted content for this turn

        Example:
            Default implementation returns:
            ["Files used in this turn: file1.py, file2.py", "", "Response content..."]

            Tools can override to add custom sections, formatting, or metadata display.
        """
        parts = []

        # Add files context if present
        if turn.files:
            parts.append(f"Files used in this turn: {', '.join(turn.files)}")
            parts.append("")  # Empty line for readability

        # Add the actual content
        parts.append(turn.content)

        return parts

    def get_prompt_content_for_size_validation(self, user_content: str) -> str:
        """
        Get the content that should be validated for MCP prompt size limits.

        This hook method allows tools to specify what content should be checked
        against the MCP transport size limit. By default, it returns the user content,
        but can be overridden to exclude conversation history when needed.

        Args:
            user_content: The user content that would normally be validated

        Returns:
            The content that should actually be validated for size limits
        """
        # Default implementation: validate the full user content
        return user_content

    def check_prompt_size(self, text: str) -> Optional[dict[str, Any]]:
        """
        Check if USER INPUT text is too large for MCP transport boundary.

        IMPORTANT: This method should ONLY be used to validate user input that crosses
        the CLI -> MCP Server transport boundary. It should NOT be used to limit
        internal MCP Server operations.

        Args:
            text: The user input text to check (NOT internal prompt content)

        Returns:
            Optional[Dict[str, Any]]: Response asking for file handling if too large, None otherwise
        """
        if text and len(text) > MCP_PROMPT_SIZE_LIMIT:
            return {
                "status": "resend_prompt",
                "content": (
                    f"MANDATORY ACTION REQUIRED: The prompt is too large for MCP's token limits (>{MCP_PROMPT_SIZE_LIMIT:,} characters). "
                    "YOU MUST IMMEDIATELY save the prompt text to a temporary file named 'prompt.txt' in the working directory. "
                    "DO NOT attempt to shorten or modify the prompt. SAVE IT AS-IS to 'prompt.txt'. "
                    "Then resend the request, passing the absolute file path to 'prompt.txt' as part of the tool call, "
                    "along with any other files you wish to share as context. Leave the prompt text itself empty or very brief in the new request. "
                    "This is the ONLY way to handle large prompts - you MUST follow these exact steps."
                ),
                "content_type": "text",
                "metadata": {
                    "prompt_size": len(text),
                    "limit": MCP_PROMPT_SIZE_LIMIT,
                    "instructions": "MANDATORY: Save prompt to 'prompt.txt' in current folder and provide full path when recalling this tool.",
                },
            }
        return None
