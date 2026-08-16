"""
Tool Registry for Unison MCP Server

Manages tool definitions, lazy instantiation, availability filtering,
and schema generation. Replaces the inline TOOLS dictionary in server.py.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Tools that cannot be disabled via DISABLED_TOOLS env var
ESSENTIAL_TOOLS = {"version", "listmodels"}

# Environment flag gating the opt-in local package scan. Default off: the scan
# imports modules eagerly, which is the operator's explicit trade, never the
# default behavior.
AUTODISCOVERY_ENV_VAR = "UNISON_TOOL_AUTODISCOVERY"

# Entry-point group third-party packages declare tools under. The entry-point
# KEY is the tool name: the shipped BaseTool contract is method-based
# (get_name() resolves during __init__), so a name cannot be read from a class
# without instantiating it.
PLUGIN_ENTRY_POINT_GROUP = "unison.tools"

#: Pending (name, class) pairs appended by @register_tool at import time.
#: The authoritative ToolRegistry is an instance constructed in server.py, so
#: import-time code cannot reach it directly; a registry drains this catalog
#: (through register() and full validation) when it loads dynamic sources.
_PENDING_DYNAMIC_TOOLS: list[tuple[str, type[Any]]] = []


class ToolQuarantinedError(RuntimeError):
    """Raised on direct access to a tool quarantined by instance-level validation.

    A controlled error rather than the tool's raw exception: the MCP handler
    maps this to a tool-level error response instead of surfacing an arbitrary
    constructor traceback.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(f"Tool '{tool_name}' is quarantined: {reason}")
        self.tool_name = tool_name
        self.reason = reason


def register_tool(name: str) -> Callable[[type[Any]], type[Any]]:
    """Mark a BaseTool subclass for registration under an explicit name.

    Appends ``(name, cls)`` to a module-level pending catalog when the
    decorated class's module is imported. The decorator imports nothing by
    itself — an unimported module's decorator never runs; modules arrive via
    entry points, the opt-in scan, or user import. A registry instance drains
    the catalog (through ``register()`` and full validation) at startup.
    """

    def decorator(cls: type[Any]) -> type[Any]:
        _PENDING_DYNAMIC_TOOLS.append((name, cls))
        return cls

    return decorator


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
        # Dynamically registered tool classes (entry points, decorator, opt-in
        # scan, direct register() calls). Kept separate from _definitions so the
        # built-in lazy-import path is untouched; instantiation is unified in
        # get_tool_instance().
        self._dynamic: dict[str, type[Any]] = {}
        # name -> reason, for tools that failed instance-level validation.
        self._quarantined: dict[str, str] = {}
        self._dynamic_skipped: int = 0
        self._frozen: bool = False
        self._validate_disabled_tools()
        self._log_configuration()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool_names(self) -> list[str]:
        """Return names of all registered tools (including disabled)."""
        return list(self._definitions.keys()) + list(self._dynamic.keys())

    # ------------------------------------------------------------------
    # Dynamic registration (entry points, decorator, opt-in scan)
    # ------------------------------------------------------------------

    def register(self, name: str, cls: type[Any]) -> bool:
        """Register a BaseTool CLASS under an explicit name.

        Structural validation only — registration never instantiates; the
        class flows through the same lazy instantiation, caching, and
        DISABLED_TOOLS handling as built-in definitions. Returns True when
        registered, False when refused (reason logged at WARNING).
        """
        if self._frozen:
            logger.warning("Registry is frozen; refusing late registration of tool '%s' (%r)", name, cls)
            self._dynamic_skipped += 1
            return False

        reason = self._structural_error(name, cls)
        if reason is not None:
            logger.warning("Refusing tool registration '%s' from %s: %s", name, getattr(cls, "__module__", "?"), reason)
            self._dynamic_skipped += 1
            return False

        self._dynamic[name] = cls
        logger.info("Discovered tool: %s from %s", name, getattr(cls, "__module__", "?"))
        return True

    def load_dynamic_sources(self) -> None:
        """Load all dynamic tool sources, then freeze the registry.

        Order: entry-point plugins, the opt-in local scan (imports may append
        decorator registrations), then a drain of the pending decorator
        catalog. Built-ins in TOOL_DEFINITIONS are present before any of this
        runs, so they win every name conflict by construction.
        """
        self.discover_plugins()
        self._autodiscover_local()
        self._drain_pending()
        self._frozen = True
        if self._dynamic or self._dynamic_skipped:
            logger.info(
                "Dynamic tool loading complete: %d registered, %d skipped (built-ins: %d)",
                len(self._dynamic),
                self._dynamic_skipped,
                len(self._definitions),
            )

    def discover_plugins(self) -> None:
        """Load third-party tools declared under the ``unison.tools`` entry-point group.

        The entry-point key is the tool name. ``entry.load()`` imports the
        plugin's module (inherent to the mechanism); instantiation remains
        lazy. Every failure is isolated per entry point.
        """
        import importlib.metadata

        try:
            entry_points = list(importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP))
        except Exception:
            logger.exception("Failed to enumerate '%s' entry points", PLUGIN_ENTRY_POINT_GROUP)
            return

        if not entry_points:
            logger.debug("Plugin discovery: no entry points found for '%s'", PLUGIN_ENTRY_POINT_GROUP)
            return

        loaded = 0
        for entry in entry_points:
            try:
                cls = entry.load()
            except Exception:
                logger.exception("Failed to load plugin entry point '%s' (%s)", entry.name, entry.value)
                self._dynamic_skipped += 1
                continue
            if self.register(entry.name, cls):
                loaded += 1
        logger.info(
            "Plugin discovery: %d loaded, %d skipped out of %d entry points",
            loaded,
            len(entry_points) - loaded,
            len(entry_points),
        )

    def _autodiscover_local(self) -> None:
        """Opt-in, import-only scan of the ``tools`` package.

        Gated behind UNISON_TOOL_AUTODISCOVERY=true (default off). Built-in
        module paths from TOOL_DEFINITIONS are excluded BEFORE import —
        skipping by name after import would already have destroyed the
        built-ins' lazy-import guarantee. The scan registers nothing itself:
        registration happens via @register_tool decorators executing at
        import time.
        """
        from utils.env import get_env

        enabled = (get_env(AUTODISCOVERY_ENV_VAR, "") or "").strip().lower() in ("1", "true", "yes")
        if not enabled:
            return

        import importlib
        import pkgutil

        import tools as tools_pkg

        builtin_modules = {module_path for module_path, _cls, _desc in self._definitions.values()}

        for module_info in pkgutil.walk_packages(tools_pkg.__path__, prefix="tools."):
            module_name = module_info.name
            if module_name in builtin_modules:
                continue
            parts = module_name.split(".")
            if "tests" in parts or "__pycache__" in parts:
                continue
            try:
                importlib.import_module(module_name)
            except Exception:
                logger.exception("Auto-discovery failed to import module '%s'; continuing", module_name)

    def _drain_pending(self) -> None:
        """Register pending @register_tool entries; repeat drains are idempotent."""
        for name, cls in list(_PENDING_DYNAMIC_TOOLS):
            if self._dynamic.get(name) is cls:
                continue  # already drained; first wins, repeats are no-ops
            self.register(name, cls)

    def _structural_error(self, name: str, cls: type[Any]) -> "str | None":
        """Return the reason a (name, class) pair fails structural validation, or None."""
        from tools.shared.base_tool import BaseTool

        if not isinstance(name, str) or not name.strip():
            return "tool name must be a non-empty string"
        if name in self._definitions or name in self._dynamic:
            return f"name '{name}' is already registered (first registration wins)"
        if not isinstance(cls, type):
            return f"expected a class, got {type(cls).__name__} (instances are not accepted)"
        if not issubclass(cls, BaseTool):
            return "class must be a concrete BaseTool subclass"
        if getattr(cls, "__abstractmethods__", None):
            return "class is abstract"
        # BaseTool.execute exists but only raises NotImplementedError, so mere
        # attribute presence proves nothing — require a genuine override or the
        # framework's workflow execution path.
        has_execute_override = cls.execute is not BaseTool.execute
        has_workflow_path = callable(getattr(cls, "execute_workflow", None))
        if not (has_execute_override or has_workflow_path):
            return "class declares no usable execution method (execute is BaseTool's NotImplementedError default)"
        return None

    def _quarantine(self, name: str, reason: str) -> None:
        if name not in self._quarantined:
            self._quarantined[name] = reason
            logger.warning("Quarantining tool '%s': %s", name, reason)

    def _instantiate_dynamic(self, name: str) -> Any:
        """Instantiate + instance-validate a dynamic tool, quarantining on failure."""
        cls = self._dynamic[name]
        try:
            instance = cls()
        except Exception as exc:
            logger.exception("Tool '%s' failed to instantiate", name)
            self._quarantine(name, f"constructor raised {type(exc).__name__}: {exc}")
            raise ToolQuarantinedError(name, self._quarantined[name]) from exc

        reported = getattr(instance, "name", None)
        if reported != name:
            self._quarantine(name, f"get_name() returned {reported!r}, registered as {name!r}")
            raise ToolQuarantinedError(name, self._quarantined[name])
        if not getattr(instance, "description", None):
            self._quarantine(name, "get_description() returned an empty description")
            raise ToolQuarantinedError(name, self._quarantined[name])
        try:
            schema = instance.get_input_schema()
        except Exception as exc:
            self._quarantine(name, f"get_input_schema() raised {type(exc).__name__}: {exc}")
            raise ToolQuarantinedError(name, self._quarantined[name]) from exc
        if not isinstance(schema, dict):
            self._quarantine(name, f"get_input_schema() returned {type(schema).__name__}, expected dict")
            raise ToolQuarantinedError(name, self._quarantined[name])
        return instance

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
        # Dynamic tools: instantiating as we list is the natural bulk hook for
        # instance-level validation; a failure quarantines the tool and it is
        # simply omitted here (warned once at quarantine time).
        for name in self._dynamic:
            if name in self._disabled or name in self._quarantined:
                continue
            try:
                available[name] = self.get_tool_instance(name)
            except ToolQuarantinedError:
                continue
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
        if tool_name in self._quarantined:
            raise ToolQuarantinedError(tool_name, self._quarantined[tool_name])

        if tool_name not in self._definitions and tool_name not in self._dynamic:
            raise KeyError(f"Unknown tool: '{tool_name}'. Available: {sorted(self.get_tool_names())}")

        if tool_name not in self._instances:
            if tool_name in self._definitions:
                module_path, class_name, _desc = self._definitions[tool_name]
                self._instances[tool_name] = self._import_tool(module_path, class_name)
                logger.debug("Lazily instantiated tool '%s' from %s.%s", tool_name, module_path, class_name)
            else:
                self._instances[tool_name] = self._instantiate_dynamic(tool_name)
                logger.debug("Lazily instantiated dynamic tool '%s'", tool_name)

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
        """Check whether a tool is registered, not disabled, and not quarantined."""
        if tool_name in self._quarantined:
            return False
        if tool_name not in self._definitions and tool_name not in self._dynamic:
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
