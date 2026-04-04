"""
Tool Registry for Unison MCP Server

Manages tool definitions, lazy instantiation, availability filtering,
and schema generation. Replaces the inline TOOLS dictionary in server.py.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tools that cannot be disabled via DISABLED_TOOLS env var
ESSENTIAL_TOOLS = {"version", "listmodels"}

# Tool definitions: name -> (module_path, class_name, description)
# Module paths are relative to the tools package.
TOOL_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "chat": ("tools.chat", "ChatTool", "Interactive development chat and brainstorming"),
    "clink": ("tools.clink", "CLinkTool", "Bridge requests to configured AI CLIs"),
    "thinkdeep": ("tools.thinkdeep", "ThinkDeepTool", "Step-by-step deep thinking workflow with expert analysis"),
    "planner": ("tools.planner", "PlannerTool", "Interactive sequential planner using workflow architecture"),
    "consensus": ("tools.consensus", "ConsensusTool", "Step-by-step consensus workflow with multi-model analysis"),
    "codereview": (
        "tools.codereview",
        "CodeReviewTool",
        "Comprehensive step-by-step code review workflow with expert analysis",
    ),
    "precommit": ("tools.precommit", "PrecommitTool", "Step-by-step pre-commit validation workflow"),
    "debug": ("tools.debug", "DebugIssueTool", "Root cause analysis and debugging assistance"),
    "secaudit": (
        "tools.secaudit",
        "SecauditTool",
        "Comprehensive security audit with OWASP Top 10 and compliance coverage",
    ),
    "docgen": ("tools.docgen", "DocgenTool", "Step-by-step documentation generation with complexity analysis"),
    "analyze": ("tools.analyze", "AnalyzeTool", "General-purpose file and code analysis"),
    "refactor": (
        "tools.refactor",
        "RefactorTool",
        "Step-by-step refactoring analysis workflow with expert validation",
    ),
    "tracer": ("tools.tracer", "TracerTool", "Static call path prediction and control flow analysis"),
    "testgen": ("tools.testgen", "TestGenTool", "Step-by-step test generation workflow with expert validation"),
    "challenge": (
        "tools.challenge",
        "ChallengeTool",
        "Critical challenge prompt wrapper to avoid automatic agreement",
    ),
    "apilookup": ("tools.apilookup", "LookupTool", "Quick web/API lookup instructions"),
    "listmodels": ("tools.listmodels", "ListModelsTool", "List all available AI models by provider"),
    "version": ("tools.version", "VersionTool", "Display server version and system information"),
}


class ToolRegistry:
    """
    Manages tool definitions, lazy instantiation, and availability filtering.

    Tools are defined by their module path and class name but are only imported
    and instantiated when first requested via get_tool_instance().
    """

    def __init__(self) -> None:
        self._definitions: dict[str, tuple[str, str, str]] = dict(TOOL_DEFINITIONS)
        self._instances: dict[str, Any] = {}
        self._disabled: set[str] = self._parse_disabled_tools()
        self._validate_disabled_tools()
        self._log_configuration()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool_names(self) -> list[str]:
        """Return names of all registered tools (including disabled)."""
        return list(self._definitions.keys())

    def get_available_tools(self) -> dict[str, Any]:
        """
        Return tool instances for all enabled tools.

        Tools listed in the DISABLED_TOOLS env var (except essential ones)
        are excluded. Instances are lazily created on first access.
        """
        available: dict[str, Any] = {}
        for name in self._definitions:
            if name in ESSENTIAL_TOOLS or name not in self._disabled:
                available[name] = self.get_tool_instance(name)
        return available

    def get_tool_instance(self, tool_name: str) -> Any:
        """
        Lazily import and instantiate a tool, caching the instance.

        Args:
            tool_name: Registered tool name.

        Returns:
            The tool instance.

        Raises:
            KeyError: If tool_name is not in the registry.
        """
        if tool_name not in self._definitions:
            raise KeyError(f"Unknown tool: '{tool_name}'. Available: {sorted(self._definitions.keys())}")

        if tool_name not in self._instances:
            module_path, class_name, _desc = self._definitions[tool_name]
            self._instances[tool_name] = self._import_tool(module_path, class_name)
            logger.debug("Lazily instantiated tool '%s' from %s.%s", tool_name, module_path, class_name)

        return self._instances[tool_name]

    def get_tool_schema(self, tool_name: str) -> dict[str, Any]:
        """
        Return the MCP tool schema for a given tool.

        The schema includes name, description, and inputSchema fields
        matching the MCP protocol format.

        Args:
            tool_name: Registered tool name.

        Returns:
            Dict with 'name', 'description', 'inputSchema', and optional 'annotations'.
        """
        tool = self.get_tool_instance(tool_name)
        schema: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.get_input_schema(),
        }
        annotations = tool.get_annotations()
        if annotations:
            schema["annotations"] = annotations
        return schema

    def is_available(self, tool_name: str) -> bool:
        """Check whether a tool is registered and not disabled."""
        if tool_name not in self._definitions:
            return False
        return tool_name in ESSENTIAL_TOOLS or tool_name not in self._disabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _import_tool(module_path: str, class_name: str) -> Any:
        """Import a module and instantiate the tool class."""
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    def _parse_disabled_tools(self) -> set[str]:
        """Parse DISABLED_TOOLS env var into a set of lowercase names."""
        from utils.env import get_env

        raw = (get_env("DISABLED_TOOLS", "") or "").strip()
        if not raw:
            return set()
        return {t.strip().lower() for t in raw.split(",") if t.strip()}

    def _validate_disabled_tools(self) -> None:
        """Log warnings for invalid disabled-tool entries."""
        essential_disabled = self._disabled & ESSENTIAL_TOOLS
        if essential_disabled:
            logger.warning("Cannot disable essential tools: %s", sorted(essential_disabled))
        unknown = self._disabled - set(self._definitions.keys())
        if unknown:
            logger.warning("Unknown tools in DISABLED_TOOLS: %s", sorted(unknown))

    def _log_configuration(self) -> None:
        """Log the final tool configuration."""
        if not self._disabled:
            logger.info("All tools enabled (DISABLED_TOOLS not set)")
            return
        actual_disabled = self._disabled - ESSENTIAL_TOOLS
        if actual_disabled:
            logger.debug("Disabled tools: %s", sorted(actual_disabled))
            available = [n for n in self._definitions if n in ESSENTIAL_TOOLS or n not in self._disabled]
            logger.info("Active tools: %s", sorted(available))


# ---------------------------------------------------------------------------
# Standalone helpers (backward-compatible with server.py's original API)
# ---------------------------------------------------------------------------


def parse_disabled_tools_env() -> set[str]:
    """Parse the DISABLED_TOOLS environment variable into a set of tool names."""
    from utils.env import get_env

    raw = (get_env("DISABLED_TOOLS", "") or "").strip()
    if not raw:
        return set()
    return {t.strip().lower() for t in raw.split(",") if t.strip()}


def validate_disabled_tools(disabled_tools: set[str], all_tools: dict[str, Any]) -> None:
    """Log warnings for invalid disabled-tool entries."""
    essential_disabled = disabled_tools & ESSENTIAL_TOOLS
    if essential_disabled:
        logger.warning("Cannot disable essential tools: %s", sorted(essential_disabled))
    unknown = disabled_tools - set(all_tools.keys())
    if unknown:
        logger.warning("Unknown tools in DISABLED_TOOLS: %s", sorted(unknown))


def apply_tool_filter(all_tools: dict[str, Any], disabled_tools: set[str]) -> dict[str, Any]:
    """Apply the disabled tools filter, preserving essential tools."""
    enabled = {}
    for name, instance in all_tools.items():
        if name in ESSENTIAL_TOOLS or name not in disabled_tools:
            enabled[name] = instance
        else:
            logger.debug("Tool '%s' disabled via DISABLED_TOOLS", name)
    return enabled
