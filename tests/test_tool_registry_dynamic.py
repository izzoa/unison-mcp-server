"""Tests for ToolRegistry's dynamic registration surface.

Covers: register(name, cls) structural validation, lazy instantiation,
quarantine semantics, the @register_tool pending catalog with startup freeze,
entry-point plugin loading (mocked), the opt-in local scan gate, and
DISABLED_TOOLS applying to dynamic tools.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tools.registry import _PENDING_DYNAMIC_TOOLS, ToolQuarantinedError, ToolRegistry, register_tool
from tools.shared.base_tool import BaseTool

# ---------------------------------------------------------------------------
# Minimal dynamic tools for testing
# ---------------------------------------------------------------------------


class _GoodTool(BaseTool):
    """A valid dynamic tool exercising the full happy path."""

    instantiations = 0

    def __init__(self) -> None:
        type(self).instantiations += 1
        super().__init__()

    def get_name(self) -> str:
        return "goodtool"

    def get_description(self) -> str:
        return "A well-behaved dynamic tool"

    def get_input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    def get_system_prompt(self) -> str:
        return "test"

    def get_request_model(self):  # type: ignore[no-untyped-def]
        return None

    async def prepare_prompt(self, request) -> str:  # type: ignore[no-untyped-def]
        return ""

    async def execute(self, arguments: dict[str, Any]):  # type: ignore[no-untyped-def]
        return []


def _subclass(name_value: str, **overrides: Any) -> type[_GoodTool]:
    """Create a _GoodTool subclass with tweaked behavior."""
    ns: dict[str, Any] = {"get_name": lambda self: name_value}
    ns.update(overrides)
    return type("Tool_" + name_value, (_GoodTool,), ns)


@pytest.fixture()
def registry(monkeypatch) -> ToolRegistry:
    monkeypatch.delenv("DISABLED_TOOLS", raising=False)
    monkeypatch.delenv("UNISON_TOOL_AUTODISCOVERY", raising=False)
    return ToolRegistry()


@pytest.fixture(autouse=True)
def _clean_pending():
    saved = list(_PENDING_DYNAMIC_TOOLS)
    _PENDING_DYNAMIC_TOOLS.clear()
    yield
    _PENDING_DYNAMIC_TOOLS.clear()
    _PENDING_DYNAMIC_TOOLS.extend(saved)


# ---------------------------------------------------------------------------
# register(name, cls): structural validation and lazy instantiation
# ---------------------------------------------------------------------------


class TestRegister:
    def test_registered_class_appears_through_existing_surface(self, registry):
        cls = _subclass("goodtool")
        cls.instantiations = 0
        assert registry.register("goodtool", cls) is True
        assert "goodtool" in registry.get_tool_names()
        assert registry.is_available("goodtool")

        instance = registry.get_tool_instance("goodtool")
        assert instance.name == "goodtool"
        # cached: same instance, one construction
        assert registry.get_tool_instance("goodtool") is instance
        assert cls.instantiations == 1

    def test_registration_does_not_instantiate(self, registry):
        cls = _subclass("lazytool")
        cls.instantiations = 0
        registry.register("lazytool", cls)
        assert cls.instantiations == 0  # only first access instantiates

    def test_empty_name_rejected(self, registry):
        assert registry.register("", _subclass("x")) is False
        assert registry.register("   ", _subclass("y")) is False

    def test_duplicate_name_first_wins(self, registry, caplog):
        first = _subclass("duptool")
        second = _subclass("duptool")
        assert registry.register("duptool", first) is True
        with caplog.at_level(logging.WARNING):
            assert registry.register("duptool", second) is False
        assert "already registered" in caplog.text
        assert registry.get_tool_instance("duptool").__class__ is first

    def test_builtin_name_collision_rejected(self, registry):
        assert registry.register("chat", _subclass("chat")) is False

    def test_non_basetool_class_rejected(self, registry):
        class NotATool:
            pass

        assert registry.register("nottool", NotATool) is False

    def test_instance_rejected(self, registry):
        assert registry.register("insttool", _GoodTool()) is False  # type: ignore[arg-type]

    def test_inherited_default_execute_rejected(self, registry, caplog):
        # A concrete-looking class whose execute is still BaseTool's
        # NotImplementedError default and which has no workflow path.
        class NoExec(BaseTool):
            def get_name(self) -> str:
                return "noexec"

            def get_description(self) -> str:
                return "d"

            def get_input_schema(self) -> dict[str, Any]:
                return {}

            def get_system_prompt(self) -> str:
                return ""

            def get_request_model(self):  # type: ignore[no-untyped-def]
                return None

            async def prepare_prompt(self, request) -> str:  # type: ignore[no-untyped-def]
                return ""

        with caplog.at_level(logging.WARNING):
            assert registry.register("noexec", NoExec) is False
        assert "no usable execution method" in caplog.text

    def test_unknown_lookup_keeps_keyerror_contract(self, registry):
        with pytest.raises(KeyError):
            registry.get_tool_instance("nonexistent")

    def test_disabled_tools_applies_to_dynamic(self, monkeypatch):
        monkeypatch.setenv("DISABLED_TOOLS", "dyntool")
        reg = ToolRegistry()
        reg.register("dyntool", _subclass("dyntool"))
        assert not reg.is_available("dyntool")
        assert "dyntool" not in reg.get_available_tools()


# ---------------------------------------------------------------------------
# Quarantine semantics
# ---------------------------------------------------------------------------


class TestQuarantine:
    def test_constructor_failure_quarantines(self, registry):
        class Boom(_GoodTool):
            def __init__(self) -> None:
                raise RuntimeError("boom")

            def get_name(self) -> str:
                return "boomtool"

        registry.register("boomtool", Boom)
        with pytest.raises(ToolQuarantinedError) as excinfo:
            registry.get_tool_instance("boomtool")
        assert "boom" in excinfo.value.reason
        assert not registry.is_available("boomtool")
        assert "boomtool" not in registry.get_available_tools()
        # subsequent access stays controlled
        with pytest.raises(ToolQuarantinedError):
            registry.get_tool_instance("boomtool")

    def test_name_disagreement_quarantines(self, registry):
        cls = _subclass("i-say-something-else")
        registry.register("registered-name", cls)
        with pytest.raises(ToolQuarantinedError) as excinfo:
            registry.get_tool_instance("registered-name")
        assert "get_name()" in str(excinfo.value)

    def test_empty_description_quarantines(self, registry):
        cls = _subclass("nodesc", get_description=lambda self: "")
        registry.register("nodesc", cls)
        with pytest.raises(ToolQuarantinedError):
            registry.get_tool_instance("nodesc")

    def test_non_dict_schema_quarantines(self, registry):
        cls = _subclass("badschema", get_input_schema=lambda self: ["not", "a", "dict"])
        registry.register("badschema", cls)
        with pytest.raises(ToolQuarantinedError):
            registry.get_tool_instance("badschema")

    def test_bulk_listing_omits_quarantined_without_crashing(self, registry):
        class Boom(_GoodTool):
            def __init__(self) -> None:
                raise RuntimeError("boom")

            def get_name(self) -> str:
                return "boomtool"

        registry.register("boomtool", Boom)
        registry.register("goodtool", _subclass("goodtool"))
        available = registry.get_available_tools()
        assert "goodtool" in available
        assert "boomtool" not in available
        # built-ins unaffected
        assert "version" in available


# ---------------------------------------------------------------------------
# Decorator + pending catalog + freeze
# ---------------------------------------------------------------------------


class TestDecoratorAndFreeze:
    def test_decorator_appends_and_drain_registers(self, registry):
        cls = _subclass("decorated")
        register_tool("decorated")(cls)
        assert ("decorated", cls) in _PENDING_DYNAMIC_TOOLS

        registry.load_dynamic_sources()
        assert registry.is_available("decorated")

    def test_repeat_drain_is_idempotent(self, registry):
        cls = _subclass("once")
        register_tool("once")(cls)
        registry._drain_pending()
        registry._drain_pending()
        assert registry.get_tool_names().count("once") == 1

    def test_post_freeze_registration_refused(self, registry, caplog):
        registry.load_dynamic_sources()  # freezes
        with caplog.at_level(logging.WARNING):
            assert registry.register("latecomer", _subclass("latecomer")) is False
        assert "frozen" in caplog.text
        assert "latecomer" not in registry.get_tool_names()

    def test_decorated_class_still_validated(self, registry):
        class NotATool:
            pass

        register_tool("bogus")(NotATool)
        registry.load_dynamic_sources()
        assert "bogus" not in registry.get_tool_names()


# ---------------------------------------------------------------------------
# Entry points (mocked)
# ---------------------------------------------------------------------------


class TestEntryPoints:
    def _entry(self, name: str, cls_or_exc: Any) -> MagicMock:
        entry = MagicMock()
        entry.name = name
        entry.value = f"pkg:{name}"
        if isinstance(cls_or_exc, Exception):
            entry.load.side_effect = cls_or_exc
        else:
            entry.load.return_value = cls_or_exc
        return entry

    def test_entry_point_key_is_the_tool_name(self, registry):
        cls = _subclass("plug")
        with patch("importlib.metadata.entry_points", return_value=[self._entry("plug", cls)]):
            registry.discover_plugins()
        assert registry.is_available("plug")

    def test_duplicate_vs_builtin_skipped(self, registry):
        cls = _subclass("chat")
        with patch("importlib.metadata.entry_points", return_value=[self._entry("chat", cls)]):
            registry.discover_plugins()
        # built-in chat untouched: still resolvable via built-in path
        assert "chat" in registry.get_tool_names()
        assert registry._dynamic.get("chat") is None

    def test_load_failure_isolated_per_entry(self, registry):
        good = self._entry("okplug", _subclass("okplug"))
        bad = self._entry("badplug", ImportError("nope"))
        with patch("importlib.metadata.entry_points", return_value=[bad, good]):
            registry.discover_plugins()
        assert registry.is_available("okplug")
        assert "badplug" not in registry.get_tool_names()

    def test_no_entry_points_is_clean(self, registry):
        with patch("importlib.metadata.entry_points", return_value=[]):
            registry.discover_plugins()
        assert registry._dynamic == {}


# ---------------------------------------------------------------------------
# Opt-in local scan
# ---------------------------------------------------------------------------


class TestAutodiscoveryGate:
    def test_scan_off_by_default_imports_nothing(self, registry, monkeypatch):
        called = False

        def sentinel(*a, **k):
            nonlocal called
            called = True
            return []

        import pkgutil

        monkeypatch.setattr(pkgutil, "walk_packages", sentinel)
        registry._autodiscover_local()
        assert called is False

    def test_scan_excludes_builtin_modules_before_import(self, registry, monkeypatch):
        monkeypatch.setenv("UNISON_TOOL_AUTODISCOVERY", "true")
        imported: list[str] = []

        import importlib
        import pkgutil

        # Simulate a walk yielding one built-in module and one custom module
        builtin = MagicMock()
        builtin.name = "tools.chat"
        custom = MagicMock()
        custom.name = "tools.custom_probe"
        tests_mod = MagicMock()
        tests_mod.name = "tools.tests.fixture"
        monkeypatch.setattr(pkgutil, "walk_packages", lambda *a, **k: [builtin, custom, tests_mod])
        monkeypatch.setattr(importlib, "import_module", lambda name: imported.append(name))

        registry._autodiscover_local()
        assert imported == ["tools.custom_probe"]
        assert "tools.chat" not in imported  # built-in excluded BEFORE import
        assert "tools.tests.fixture" not in imported

    def test_default_registry_state_identical_without_dynamic_sources(self, monkeypatch):
        monkeypatch.delenv("DISABLED_TOOLS", raising=False)
        baseline = ToolRegistry()
        loaded = ToolRegistry()
        with patch("importlib.metadata.entry_points", return_value=[]):
            loaded.load_dynamic_sources()
        assert set(loaded.get_tool_names()) == set(baseline.get_tool_names())
        assert loaded._dynamic == {}
        assert loaded._quarantined == {}
