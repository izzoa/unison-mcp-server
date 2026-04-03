"""
Response formatting infrastructure for Unison MCP tools.

This module provides the ResponseFormatter class which handles post-processing
of AI model responses. The base implementation returns responses unchanged;
subclasses can override methods to add custom formatting, validation, or
additional context.
"""

from typing import Optional


class ResponseFormatter:
    """
    Handles formatting and parsing of AI model responses.

    The base implementation provides simple passthrough behavior.
    Subclasses can override methods to add tool-specific formatting.

    Attributes:
        tool_name: The name of the tool this formatter is associated with.
    """

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def format_response(self, response: str, request, model_info: dict = None) -> str:
        """
        Format the AI model's response for the user.

        This method allows tools to post-process the model's response,
        adding structure, validation, or additional context.

        The default implementation returns the response unchanged.
        Tools can override this method to add custom formatting.

        Args:
            response: Raw response from the AI model
            request: The original request object
            model_info: Optional model information and metadata

        Returns:
            str: Formatted response ready for the user
        """
        return response

    def _parse_response(self, raw_text: str, request, model_info: Optional[dict] = None):
        """Parse response - will be inherited for now.

        Subclasses must override this method to provide tool-specific
        response parsing logic.

        Args:
            raw_text: Raw text response from the AI model
            request: The original request object
            model_info: Optional model information and metadata

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement _parse_response method")
