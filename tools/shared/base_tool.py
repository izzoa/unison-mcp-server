"""
Core Tool Infrastructure for Unison MCP Tools

This module provides the fundamental base class for all tools:
- BaseTool: Abstract base class defining the tool interface

The BaseTool class defines the core contract that tools must implement and provides
common functionality for request validation, error handling, model management,
conversation handling, file processing, and response formatting.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from mcp.types import TextContent

if TYPE_CHECKING:
    from providers.shared import ModelCapabilities
    from tools.models import ToolModelCategory

from providers import ModelProvider, ModelProviderRegistry
from tools.shared.model_schema_builder import ModelSchemaBuilder
from utils.conversation_memory import ConversationTurn
from utils.env import get_env

# Import models from tools.models for compatibility
try:
    from tools.models import SPECIAL_STATUS_MODELS, ContinuationOffer, ToolOutput
except ImportError:
    # Fallback in case models haven't been set up yet
    SPECIAL_STATUS_MODELS = {}
    ContinuationOffer = None
    ToolOutput = None

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all Unison MCP tools.

    This class defines the interface that all tools must implement and provides
    common functionality for request handling, model creation, and response formatting.

    CONVERSATION-AWARE FILE PROCESSING:
    This base class implements the sophisticated dual prioritization strategy for
    conversation-aware file handling across all tools:

    1. FILE DEDUPLICATION WITH NEWEST-FIRST PRIORITY:
       - When same file appears in multiple conversation turns, newest reference wins
       - Prevents redundant file embedding while preserving most recent file state
       - Cross-tool file tracking ensures consistent behavior across analyze → codereview → debug

    2. CONVERSATION CONTEXT INTEGRATION:
       - All tools receive enhanced prompts with conversation history via reconstruct_thread_context()
       - File references from previous turns are preserved and accessible
       - Cross-tool knowledge transfer maintains full context without manual file re-specification

    3. TOKEN-AWARE FILE EMBEDDING:
       - Respects model-specific token allocation budgets from ModelContext
       - Prioritizes conversation history, then newest files, then remaining content
       - Graceful degradation when token limits are approached

    4. STATELESS-TO-STATEFUL BRIDGING:
       - Tools operate on stateless MCP requests but access full conversation state
       - Conversation memory automatically injected via continuation_id parameter
       - Enables natural AI-to-AI collaboration across tool boundaries

    To create a new tool:
    1. Create a new class that inherits from BaseTool
    2. Implement all abstract methods
    3. Define a request model that inherits from ToolRequest
    4. Register the tool in server.py's TOOLS dictionary
    """

    # Class-level cache for OpenRouter registry to avoid multiple loads
    _openrouter_registry_cache = None
    _custom_registry_cache = None

    @classmethod
    def _get_openrouter_registry(cls):
        """Get cached OpenRouter registry instance, creating if needed."""
        # Use BaseTool class directly to ensure cache is shared across all subclasses
        if BaseTool._openrouter_registry_cache is None:
            from providers.registries.openrouter import OpenRouterModelRegistry

            BaseTool._openrouter_registry_cache = OpenRouterModelRegistry()
            logger.debug("Created cached OpenRouter registry instance")
        return BaseTool._openrouter_registry_cache

    @classmethod
    def _get_custom_registry(cls):
        """Get cached custom-endpoint registry instance."""
        if BaseTool._custom_registry_cache is None:
            from providers.registries.custom import CustomEndpointModelRegistry

            BaseTool._custom_registry_cache = CustomEndpointModelRegistry()
            logger.debug("Created cached Custom registry instance")
        return BaseTool._custom_registry_cache

    def __init__(self):
        # Cache tool metadata at initialization to avoid repeated calls
        self.name = self.get_name()
        self.description = self.get_description()
        self.default_temperature = self.get_default_temperature()
        # Components (composition over inheritance)
        from tools.shared.conversation_handler import ConversationHandler
        from tools.shared.file_processor import FileProcessor
        from tools.shared.model_schema_builder import ModelSchemaBuilder
        from tools.shared.response_formatter import ResponseFormatter

        self._file_processor = FileProcessor(
            tool_name=self.name,
            include_line_numbers=self.wants_line_numbers_by_default(),
        )
        self._conversation_handler = ConversationHandler(tool_name=self.name)
        self._response_formatter = ResponseFormatter(tool_name=self.name)
        self._model_schema_builder = ModelSchemaBuilder(tool_name=self.name, tool=self)

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the unique name identifier for this tool.

        This name is used by MCP clients to invoke the tool and must be
        unique across all registered tools.

        Returns:
            str: The tool's unique name (e.g., "review_code", "analyze")
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Return a detailed description of what this tool does.

        This description is shown to MCP clients (like Claude / Codex / Gemini) to help them
        understand when and how to use the tool. It should be comprehensive
        and include trigger phrases.

        Returns:
            str: Detailed tool description with usage examples
        """
        pass

    @abstractmethod
    def get_input_schema(self) -> dict[str, Any]:
        """
        Return the JSON Schema that defines this tool's parameters.

        This schema is used by MCP clients to validate inputs before
        sending requests. It should match the tool's request model.

        Returns:
            Dict[str, Any]: JSON Schema object defining required and optional parameters
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt that configures the AI model's behavior.

        This prompt sets the context and instructions for how the model
        should approach the task. It's prepended to the user's request.

        Returns:
            str: System prompt with role definition and instructions
        """
        pass

    def get_capability_system_prompts(self, capabilities: Optional["ModelCapabilities"]) -> list[str]:
        """Return additional system prompt snippets gated on model capabilities.

        Subclasses can override this hook to append capability-specific
        instructions (for example, enabling code-generation exports when a
        model advertises support). The default implementation returns an empty
        list so no extra instructions are appended.

        Args:
            capabilities: The resolved capabilities for the active model.

        Returns:
            List of prompt fragments to append after the base system prompt.
        """

        return []

    def _augment_system_prompt_with_capabilities(
        self, base_prompt: str, capabilities: Optional["ModelCapabilities"]
    ) -> str:
        """Merge capability-driven prompt addenda with the base system prompt."""

        additions: list[str] = []
        if capabilities is not None:
            additions = [fragment.strip() for fragment in self.get_capability_system_prompts(capabilities) if fragment]

        if not additions:
            return base_prompt

        addition_text = "\n\n".join(additions)
        if not base_prompt:
            return addition_text

        suffix = "" if base_prompt.endswith("\n\n") else "\n\n"
        return f"{base_prompt}{suffix}{addition_text}"

    def get_annotations(self) -> Optional[dict[str, Any]]:
        """
        Return optional annotations for this tool.

        Annotations provide hints about tool behavior without being security-critical.
        They help MCP clients make better decisions about tool usage.

        Returns:
            Optional[dict]: Dictionary with annotation fields like readOnlyHint, destructiveHint, etc.
                           Returns None if no annotations are needed.
        """
        return None

    def requires_model(self) -> bool:
        """
        Return whether this tool requires AI model access.

        Tools that override execute() to do pure data processing (like planner)
        should return False to skip model resolution at the MCP boundary.

        Returns:
            bool: True if tool needs AI model access (default), False for data-only tools
        """
        return True

    def is_effective_auto_mode(self) -> bool:
        """Check if we're in effective auto mode. Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder.is_effective_auto_mode()

    def _should_require_model_selection(self, model_name: str) -> bool:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder._should_require_model_selection(model_name)

    def _get_available_models(self) -> list[str]:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder._get_available_models()

    def _format_available_models_list(self) -> str:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder._format_available_models_list()

    @staticmethod
    def _format_context_window(tokens: int) -> Optional[str]:
        """Delegates to ModelSchemaBuilder."""
        return ModelSchemaBuilder._format_context_window(tokens)

    def _collect_ranked_capabilities(self) -> list[tuple[int, str, Any]]:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder._collect_ranked_capabilities()

    @staticmethod
    def _normalize_model_identifier(name: str) -> str:
        """Delegates to ModelSchemaBuilder."""
        return ModelSchemaBuilder._normalize_model_identifier(name)

    def _get_ranked_model_summaries(self, limit: int = 5) -> tuple[list[str], int, bool]:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder._get_ranked_model_summaries(limit)

    def _get_restriction_note(self) -> Optional[str]:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder._get_restriction_note()

    def _build_model_unavailable_message(self, model_name: str) -> str:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder._build_model_unavailable_message(model_name)

    def _build_auto_mode_required_message(self) -> str:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder._build_auto_mode_required_message()

    def get_model_field_schema(self) -> dict[str, Any]:
        """Delegates to ModelSchemaBuilder."""
        return self._model_schema_builder.get_model_field_schema()

    def get_default_temperature(self) -> float:
        """
        Return the default temperature setting for this tool.

        Override this method to set tool-specific temperature defaults.
        Lower values (0.0-0.3) for analytical tasks, higher (0.7-1.0) for creative tasks.

        Returns:
            float: Default temperature between 0.0 and 1.0
        """
        return 0.5

    def wants_line_numbers_by_default(self) -> bool:
        """
        Return whether this tool wants line numbers added to code files by default.

        By default, ALL tools get line numbers for precise code references.
        Line numbers are essential for accurate communication about code locations.

        Returns:
            bool: True if line numbers should be added by default for this tool
        """
        return True  # All tools get line numbers by default for consistency

    def get_default_thinking_mode(self) -> str:
        """
        Return the default thinking mode for this tool.

        Thinking mode controls computational budget for reasoning.
        Override for tools that need more or less reasoning depth.

        Returns:
            str: One of "minimal", "low", "medium", "high", "max"
        """
        return "medium"  # Default to medium thinking for better reasoning

    def get_model_category(self) -> "ToolModelCategory":
        """
        Return the model category for this tool.

        Model category influences which model is selected in auto mode.
        Override to specify whether your tool needs extended reasoning,
        fast response, or balanced capabilities.

        Returns:
            ToolModelCategory: Category that influences model selection
        """
        from tools.models import ToolModelCategory

        return ToolModelCategory.BALANCED

    @abstractmethod
    def get_request_model(self):
        """
        Return the Pydantic model class used for validating requests.

        This model should inherit from ToolRequest and define all
        parameters specific to this tool.

        Returns:
            Type[ToolRequest]: The request model class
        """
        pass

    def validate_file_paths(self, request) -> Optional[str]:
        """
        Validate that all file paths in the request are absolute.

        This is a critical security function that prevents path traversal attacks
        and ensures all file access is properly controlled. All file paths must
        be absolute to avoid ambiguity and security issues.

        Args:
            request: The validated request object

        Returns:
            Optional[str]: Error message if validation fails, None if all paths are valid
        """
        # Only validate files/paths if they exist in the request
        file_fields = [
            "absolute_file_paths",
            "file",
            "path",
            "directory",
            "notebooks",
            "test_examples",
            "style_guide_examples",
            "files_checked",
            "relevant_files",
        ]

        for field_name in file_fields:
            if hasattr(request, field_name):
                field_value = getattr(request, field_name)
                if field_value is None:
                    continue

                # Handle both single paths and lists of paths
                paths_to_check = field_value if isinstance(field_value, list) else [field_value]

                for path in paths_to_check:
                    if path and not os.path.isabs(path):
                        return f"All file paths must be FULL absolute paths. Invalid path: '{path}'"

        return None

    def _validate_token_limit(self, content: str, content_type: str = "Content") -> None:
        """Delegates to FileProcessor."""
        return self._file_processor._validate_token_limit(content, content_type)

    def get_model_provider(self, model_name: str) -> ModelProvider:
        """
        Get the appropriate model provider for the given model name.

        This method performs runtime validation to ensure the requested model
        is actually available with the current API key configuration.

        Args:
            model_name: Name of the model to get provider for

        Returns:
            ModelProvider: The provider instance for the model

        Raises:
            ValueError: If the model is not available or provider not found
        """
        try:
            provider = ModelProviderRegistry.get_provider_for_model(model_name)
            if not provider:
                logger.error(f"No provider found for model '{model_name}' in {self.name} tool")
                raise ValueError(self._build_model_unavailable_message(model_name))

            return provider
        except Exception as e:
            logger.error(f"Failed to get provider for model '{model_name}' in {self.name} tool: {e}")
            raise

    # === CONVERSATION AND FILE HANDLING METHODS ===

    def get_conversation_embedded_files(self, continuation_id: Optional[str]) -> list[str]:
        """Delegates to FileProcessor."""
        return self._file_processor.get_conversation_embedded_files(continuation_id)

    def filter_new_files(self, requested_files: list[str], continuation_id: Optional[str]) -> list[str]:
        """Delegates to FileProcessor."""
        return self._file_processor.filter_new_files(requested_files, continuation_id)

    def format_conversation_turn(self, turn: ConversationTurn) -> list[str]:
        """Delegates to ConversationHandler."""
        return self._conversation_handler.format_conversation_turn(turn)

    def handle_prompt_file(self, files: Optional[list[str]]) -> tuple[Optional[str], Optional[list[str]]]:
        """Delegates to FileProcessor."""
        return self._file_processor.handle_prompt_file(files)

    def get_prompt_content_for_size_validation(self, user_content: str) -> str:
        """Delegates to ConversationHandler."""
        return self._conversation_handler.get_prompt_content_for_size_validation(user_content)

    def check_prompt_size(self, text: str) -> Optional[dict[str, Any]]:
        """Delegates to ConversationHandler."""
        return self._conversation_handler.check_prompt_size(text)

    def _prepare_file_content_for_prompt(
        self,
        request_files: list[str],
        continuation_id: Optional[str],
        context_description: str = "New files",
        max_tokens: Optional[int] = None,
        reserve_tokens: int = 1_000,
        remaining_budget: Optional[int] = None,
        arguments: Optional[dict] = None,
        model_context: Optional[Any] = None,
    ) -> tuple[str, list[str]]:
        """Delegates to FileProcessor."""
        return self._file_processor._prepare_file_content_for_prompt(
            request_files,
            continuation_id,
            context_description,
            max_tokens,
            reserve_tokens,
            remaining_budget,
            arguments,
            model_context,
        )

    def get_websearch_instruction(self, tool_specific: Optional[str] = None) -> str:
        """
        Generate standardized web search instruction.

        Args:
            tool_specific: Optional tool-specific search guidance

        Returns:
            str: Web search instruction to append to prompt
        """

        base_instruction = """

WEB SEARCH CAPABILITY: You can request the calling agent to perform web searches to enhance your analysis with current information!

IMPORTANT: When you identify areas where web searches would significantly improve your response (such as checking current documentation, finding recent solutions, verifying best practices, or gathering community insights), you MUST explicitly instruct the agent to perform specific web searches and then respond back using the continuation_id from this response to continue the analysis.

Use clear, direct language based on the value of the search:

For valuable supplementary information: "Please perform a web search on '[specific topic/query]' and then continue this analysis using the continuation_id from this response if you find relevant information."

For important missing information: "Please search for '[specific topic/query]' and respond back with the findings using the continuation_id from this response - this information is needed to provide a complete analysis."

For critical/essential information: "SEARCH REQUIRED: Please immediately perform a web search on '[specific topic/query]' and respond back with the results using the continuation_id from this response. Cannot provide accurate analysis without this current information."

This ensures you get the most current and comprehensive information while maintaining conversation context through the continuation_id."""

        if tool_specific:
            return f"""{base_instruction}

{tool_specific}

When recommending searches, be specific about what information you need and why it would improve your analysis."""

        # Default instruction for all tools
        return f"""{base_instruction}

Consider requesting searches for:
- Current documentation and API references
- Recent best practices and patterns
- Known issues and community solutions
- Framework updates and compatibility
- Security advisories and patches
- Performance benchmarks and optimizations

When recommending searches, be specific about what information you need and why it would improve your analysis. Always remember to instruct agent to use the continuation_id from this response when providing search results."""

    def get_language_instruction(self) -> str:
        """
        Generate language instruction based on LOCALE configuration.

        Returns:
            str: Language instruction to prepend to prompt, or empty string if
                 no locale set
        """
        # Read LOCALE directly from environment to support dynamic changes
        # Tests can monkeypatch LOCALE via the environment helper (or .env when override is enforced)

        locale = (get_env("LOCALE", "") or "").strip()

        if not locale:
            return ""

        # Simple language instruction
        return f"Always respond in {locale}.\n\n"

    # === ABSTRACT METHODS FOR SIMPLE TOOLS ===

    @abstractmethod
    async def prepare_prompt(self, request) -> str:
        """
        Prepare the complete prompt for the AI model.

        This method should construct the full prompt by combining:
        - System prompt from get_system_prompt()
        - File content from _prepare_file_content_for_prompt()
        - Conversation history from reconstruct_thread_context()
        - User's request and any tool-specific context

        Args:
            request: The validated request object

        Returns:
            str: Complete prompt ready for the AI model
        """
        pass

    def format_response(self, response: str, request, model_info: dict = None) -> str:
        """Delegates to ResponseFormatter."""
        return self._response_formatter.format_response(response, request, model_info)

    # === IMPLEMENTATION METHODS ===
    # These will be provided in a full implementation but are inherited from current base.py
    # for now to maintain compatibility.

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute the tool - will be inherited from existing base.py for now."""
        # This will be implemented by importing from the current base.py
        # for backward compatibility during the migration
        raise NotImplementedError("Subclasses must implement execute method")

    def _should_require_model_selection(self, model_name: str) -> bool:
        """
        Check if we should require the CLI to select a model at runtime.

        This is called during request execution to determine if we need
        to return an error asking the CLI to provide a model parameter.

        Args:
            model_name: The model name from the request or DEFAULT_MODEL

        Returns:
            bool: True if we should require model selection
        """
        # Case 1: Model is explicitly "auto"
        if model_name.lower() == "auto":
            return True

        # Case 2: Requested model is not available
        from providers.registry import ModelProviderRegistry

        provider = ModelProviderRegistry.get_provider_for_model(model_name)
        if not provider:
            logger.warning(f"Model '{model_name}' is not available with current API keys. Requiring model selection.")
            return True

        return False

    def _get_available_models(self) -> list[str]:
        """
        Get list of models available from enabled providers.

        Only returns models from providers that have valid API keys configured.
        This fixes the namespace collision bug where models from disabled providers
        were shown to the CLI, causing routing conflicts.

        Returns:
            List of model names from enabled providers only
        """
        from providers.registry import ModelProviderRegistry

        # Get models from enabled providers only (those with valid API keys)
        all_models = ModelProviderRegistry.get_available_model_names()

        # Add OpenRouter models and their aliases when OpenRouter is configured
        openrouter_key = get_env("OPENROUTER_API_KEY")
        if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
            try:
                registry = self._get_openrouter_registry()

                for alias in registry.list_aliases():
                    if alias not in all_models:
                        all_models.append(alias)
            except Exception as exc:  # pragma: no cover - logged for observability
                import logging

                logging.debug(f"Failed to add OpenRouter models to enum: {exc}")

        # Add custom models (and their aliases) when a custom endpoint is available
        custom_url = get_env("CUSTOM_API_URL")
        if custom_url:
            try:
                registry = self._get_custom_registry()
                for alias in registry.list_aliases():
                    if alias not in all_models:
                        all_models.append(alias)
            except Exception as exc:  # pragma: no cover - logged for observability
                import logging

                logging.debug(f"Failed to add custom models to enum: {exc}")

        # Remove duplicates while preserving insertion order
        seen: set[str] = set()
        unique_models: list[str] = []
        for model in all_models:
            if model not in seen:
                seen.add(model)
                unique_models.append(model)

        return unique_models

    def _resolve_model_context(self, arguments: dict, request) -> tuple[str, Any]:
        """
        Resolve model context and name using centralized logic.

        This method extracts the model resolution logic from execute() so it can be
        reused by tools that override execute() (like debug tool) without duplicating code.

        Args:
            arguments: Dictionary of arguments from the MCP client
            request: The validated request object

        Returns:
            tuple[str, ModelContext]: (resolved_model_name, model_context)

        Raises:
            ValueError: If model resolution fails or model selection is required
        """
        # MODEL RESOLUTION NOW HAPPENS AT MCP BOUNDARY
        # Extract pre-resolved model context from server.py via ToolExecutionContext
        from utils.tool_execution_context import ToolExecutionContext

        exec_ctx = ToolExecutionContext.from_arguments(arguments)
        model_context = exec_ctx.model_context if exec_ctx else None
        resolved_model_name = exec_ctx.resolved_model_name if exec_ctx else None

        if model_context and resolved_model_name:
            # Model was already resolved at MCP boundary
            model_name = resolved_model_name
            logger.debug(f"Using pre-resolved model '{model_name}' from MCP boundary")
        else:
            # Fallback for direct execute calls
            model_name = getattr(request, "model", None)
            if not model_name:
                from config import DEFAULT_MODEL

                model_name = DEFAULT_MODEL
            logger.debug(f"Using fallback model resolution for '{model_name}' (test mode)")

            # For tests: Check if we should require model selection (auto mode)
            if self._should_require_model_selection(model_name):
                # Build error message based on why selection is required
                if model_name.lower() == "auto":
                    error_message = self._build_auto_mode_required_message()
                else:
                    error_message = self._build_model_unavailable_message(model_name)
                raise ValueError(error_message)

            # Create model context for tests
            from utils.model_context import ModelContext

            model_context = ModelContext(model_name)

        return model_name, model_context

    def validate_and_correct_temperature(self, temperature: float, model_context: Any) -> tuple[float, list[str]]:
        """
        Validate and correct temperature for the specified model.

        This method ensures that the temperature value is within the valid range
        for the specific model being used. Different models have different temperature
        constraints (e.g., o1 models require temperature=1.0, GPT models support 0-2).

        Args:
            temperature: Temperature value to validate
            model_context: Model context object containing model name, provider, and capabilities

        Returns:
            Tuple of (corrected_temperature, warning_messages)
        """
        try:
            # Use model context capabilities directly - clean OOP approach
            capabilities = model_context.capabilities
            constraint = capabilities.temperature_constraint

            warnings = []
            if not constraint.validate(temperature):
                corrected = constraint.get_corrected_value(temperature)
                warning = (
                    f"Temperature {temperature} invalid for {model_context.model_name}. "
                    f"{constraint.get_description()}. Using {corrected} instead."
                )
                warnings.append(warning)
                return corrected, warnings

            return temperature, warnings

        except Exception as e:
            # If validation fails for any reason, use the original temperature
            # and log a warning (but don't fail the request)
            logger.warning(f"Temperature validation failed for {model_context.model_name}: {e}")
            return temperature, [f"Temperature validation failed: {e}"]

    def _validate_image_limits(
        self, images: Optional[list[str]], model_context: Optional[Any] = None, continuation_id: Optional[str] = None
    ) -> Optional[dict]:
        """Delegates to FileProcessor."""
        return self._file_processor._validate_image_limits(images, model_context, continuation_id)

    def _parse_response(self, raw_text: str, request, model_info: Optional[dict] = None):
        """Delegates to ResponseFormatter."""
        return self._response_formatter._parse_response(raw_text, request, model_info)
