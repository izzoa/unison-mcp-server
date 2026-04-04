"""
Pytest configuration for Unison MCP Server tests
"""

import asyncio
import importlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

# On macOS, the default pytest temp dir is typically under /var (e.g. /private/var/folders/...).
# If /var is considered a dangerous system path, tests must use a safe temp root (like /tmp).
if sys.platform == "darwin":
    os.environ["TMPDIR"] = "/tmp"
    # tempfile caches the temp dir after first lookup; clear it so pytest fixtures pick up TMPDIR.
    tempfile.tempdir = None

# Ensure the parent directory is in the Python path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import utils.env as env_config  # noqa: E402

# Ensure tests operate with runtime environment rather than .env overrides during imports
env_config.reload_env({"UNISON_MCP_FORCE_ENV_OVERRIDE": "false"})

# Disable LiteLLM auto-discovery during tests so that only JSON-configured
# models appear in registries.  The litellm adapter tests re-enable it locally.
from providers.litellm_adapter import set_discovery_enabled  # noqa: E402

set_discovery_enabled(False)

# Set default model to a specific value for tests to avoid auto mode
# This prevents all tests from failing due to missing model parameter
os.environ["DEFAULT_MODEL"] = "gemini-2.5-flash"

# Force reload of config module to pick up the env var
import config  # noqa: E402

importlib.reload(config)

# Note: This creates a test sandbox environment
# Tests create their own temporary directories as needed

# Configure asyncio for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------------------------------------------------------------------
# Provider Registry Setup
# ---------------------------------------------------------------------------

from providers.gemini import GeminiModelProvider  # noqa: E402
from providers.openai import OpenAIModelProvider  # noqa: E402
from providers.registry import ModelProviderRegistry, set_default_registry  # noqa: E402
from providers.shared import ProviderType  # noqa: E402
from providers.xai import XAIModelProvider  # noqa: E402


def _make_test_registry() -> ModelProviderRegistry:
    """Create a fresh registry with standard test providers registered."""
    registry = ModelProviderRegistry(config={})
    registry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
    registry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
    registry.register_provider(ProviderType.XAI, XAIModelProvider)

    # Register CUSTOM provider if CUSTOM_API_URL is available (for integration tests)
    if os.getenv("CUSTOM_API_URL") and "test_prompt_regression.py" in os.getenv("PYTEST_CURRENT_TEST", ""):
        from providers.custom import CustomProvider

        def custom_provider_factory(api_key=None):
            base_url = os.getenv("CUSTOM_API_URL", "")
            return CustomProvider(api_key=api_key or "", base_url=base_url)

        registry.register_provider(ProviderType.CUSTOM, custom_provider_factory)

    return registry


# Create an initial default registry at module level so import-time code works
_initial_registry = _make_test_registry()
set_default_registry(_initial_registry)

# Force registry reload so providers pick up the disabled-discovery state
# (registries may have been lazily created during import with discovery still enabled)
GeminiModelProvider.reload_registry()
OpenAIModelProvider.reload_registry()
XAIModelProvider.reload_registry()


@pytest.fixture
def project_path(tmp_path):
    """
    Provides a temporary directory for tests.
    This ensures all file operations during tests are isolated.
    """
    # Create a subdirectory for this specific test
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir(parents=True, exist_ok=True)

    return test_dir


def _set_dummy_keys_if_missing():
    """Set dummy API keys only when they are completely absent."""
    for var in ("GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        if not os.environ.get(var):
            os.environ[var] = "dummy-key-for-tests"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "asyncio: mark test as async")
    config.addinivalue_line("markers", "no_mock_provider: disable automatic provider mocking")
    # Assume we need dummy keys until we learn otherwise
    config._needs_dummy_keys = True


def pytest_collection_modifyitems(session, config, items):
    """Hook that runs after test collection to check for no_mock_provider markers."""
    # Always set dummy keys if real keys are missing
    # This ensures tests work in CI even with no_mock_provider marker
    _set_dummy_keys_if_missing()


@pytest.fixture(autouse=True)
def _fresh_default_registry():
    """Create an isolated default registry for every test.

    This replaces the old ``ModelProviderRegistry.reset_for_testing()`` pattern.
    Each test gets its own registry instance so parallel execution is safe.
    """
    registry = _make_test_registry()
    set_default_registry(registry)
    yield registry
    # No cleanup needed — next test will set a new default.


@pytest.fixture(autouse=True)
def mock_provider_availability(request, monkeypatch, _fresh_default_registry):
    """
    Automatically mock provider availability for all tests to prevent
    effective auto mode from being triggered when DEFAULT_MODEL is unavailable.

    This fixture ensures that when tests run with dummy API keys,
    the tools don't require model selection unless explicitly testing auto mode.
    """
    # Skip this fixture for tests that need real providers
    if hasattr(request, "node"):
        marker = request.node.get_closest_marker("no_mock_provider")
        if marker:
            return

    registry = _fresh_default_registry

    # Ensure providers are registered
    from providers.shared import ProviderType

    if ProviderType.GOOGLE not in registry._providers:
        registry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
    if ProviderType.OPENAI not in registry._providers:
        registry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
    if ProviderType.XAI not in registry._providers:
        registry.register_provider(ProviderType.XAI, XAIModelProvider)

    # Ensure CUSTOM provider is registered if needed for integration tests
    if (
        os.getenv("CUSTOM_API_URL")
        and "test_prompt_regression.py" in os.getenv("PYTEST_CURRENT_TEST", "")
        and ProviderType.CUSTOM not in registry._providers
    ):
        from providers.custom import CustomProvider

        def custom_provider_factory(api_key=None):
            base_url = os.getenv("CUSTOM_API_URL", "")
            return CustomProvider(api_key=api_key or "", base_url=base_url)

        registry.register_provider(ProviderType.CUSTOM, custom_provider_factory)

    # Also mock is_effective_auto_mode for all BaseTool instances to return False
    # unless we're specifically testing auto mode behavior
    from tools.shared.base_tool import BaseTool

    def mock_is_effective_auto_mode(self):
        # If this is an auto mode test file or specific auto mode test, use the real logic
        test_file = request.node.fspath.basename if hasattr(request, "node") and hasattr(request.node, "fspath") else ""
        test_name = request.node.name if hasattr(request, "node") else ""

        # Allow auto mode for tests in auto mode files or with auto in the name
        if (
            "auto_mode" in test_file.lower()
            or "auto" in test_name.lower()
            or "intelligent_fallback" in test_file.lower()
            or "per_tool_model_defaults" in test_file.lower()
        ):
            # Call original method logic
            from config import DEFAULT_MODEL
            from providers.registry import get_default_registry

            if DEFAULT_MODEL.lower() == "auto":
                return True
            provider = get_default_registry().get_provider_for_model(DEFAULT_MODEL)
            return provider is None
        # For all other tests, return False to disable auto mode
        return False

    monkeypatch.setattr(BaseTool, "is_effective_auto_mode", mock_is_effective_auto_mode)


@pytest.fixture(autouse=True)
def clear_model_restriction_env(monkeypatch):
    """Ensure per-test isolation from user-defined model restriction env vars."""

    restriction_vars = [
        "OPENAI_ALLOWED_MODELS",
        "GOOGLE_ALLOWED_MODELS",
        "XAI_ALLOWED_MODELS",
        "OPENROUTER_ALLOWED_MODELS",
        "DIAL_ALLOWED_MODELS",
    ]

    for var in restriction_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def disable_force_env_override(monkeypatch):
    """Default tests to runtime environment visibility unless they explicitly opt in."""

    monkeypatch.setenv("UNISON_MCP_FORCE_ENV_OVERRIDE", "false")
    env_config.reload_env({"UNISON_MCP_FORCE_ENV_OVERRIDE": "false"})
    monkeypatch.setenv("DEFAULT_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("MAX_CONVERSATION_TURNS", "50")

    import importlib
    import sys

    import config
    import utils.context_reconstructor as context_reconstructor
    import utils.conversation_memory as conversation_memory
    import utils.conversation_store as conversation_store

    importlib.reload(config)
    importlib.reload(conversation_store)
    importlib.reload(context_reconstructor)
    importlib.reload(conversation_memory)

    # Sync MAX_CONVERSATION_TURNS to test modules that import it at module level
    for mod_name in ("tests.test_conversation_memory", "tests.test_conversation_store"):
        test_mod = sys.modules.get(mod_name)
        if test_mod is not None:
            test_mod.MAX_CONVERSATION_TURNS = conversation_store.MAX_CONVERSATION_TURNS

    try:
        yield
    finally:
        env_config.reload_env()
