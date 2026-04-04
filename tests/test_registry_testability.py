"""Tests for registry testability improvements: StorageBackend protocol and ModelProviderRegistry config injection."""

from unittest.mock import MagicMock

from providers.registry import ModelProviderRegistry, get_default_registry, set_default_registry


class TestStorageBackendProtocol:
    """Tests for StorageBackend protocol and reset/injection functions."""

    def test_in_memory_storage_satisfies_protocol(self):
        """InMemoryStorage should satisfy the StorageBackend protocol via isinstance."""
        from utils.storage_backend import InMemoryStorage, StorageBackend

        storage = InMemoryStorage()
        assert isinstance(storage, StorageBackend)
        storage.shutdown()

    def test_reset_storage_backend_clears_singleton(self):
        """reset_storage_backend() should clear the global singleton."""
        from utils.storage_backend import get_storage_backend, reset_storage_backend

        # Get a backend (creates singleton)
        backend1 = get_storage_backend()
        # Reset
        reset_storage_backend()
        # Next call creates a new instance
        backend2 = get_storage_backend()
        assert backend1 is not backend2
        backend1.shutdown()
        backend2.shutdown()
        reset_storage_backend()

    def test_get_storage_backend_with_custom_backend(self):
        """get_storage_backend(backend=mock) should install and return the mock."""
        from utils.storage_backend import get_storage_backend, reset_storage_backend

        reset_storage_backend()
        mock_backend = MagicMock()
        result = get_storage_backend(backend=mock_backend)
        assert result is mock_backend

        # Subsequent calls should return the same mock
        result2 = get_storage_backend()
        assert result2 is mock_backend
        reset_storage_backend()

    def test_reset_storage_backend_idempotent(self):
        """Calling reset_storage_backend() twice should not raise."""
        from utils.storage_backend import reset_storage_backend

        reset_storage_backend()
        reset_storage_backend()  # Should not raise


class TestRegistryIndependentInstances:
    """Tests for ModelProviderRegistry independent instance creation."""

    def test_new_registry_starts_empty(self):
        """A new registry should have no providers or initialized providers."""
        registry = ModelProviderRegistry(config={})
        assert len(registry._providers) == 0
        assert len(registry._initialized_providers) == 0

    def test_config_is_stored(self):
        """Config passed to __init__ should be stored on the instance."""
        config = {"GEMINI_API_KEY": "test-key"}
        registry = ModelProviderRegistry(config=config)
        assert registry._config == config

    def test_empty_config_is_valid(self):
        """An empty config dict should be accepted."""
        registry = ModelProviderRegistry(config={})
        assert registry._config == {}

    def test_separate_instances_are_independent(self):
        """Two registry instances should be completely independent."""
        from providers.gemini import GeminiModelProvider
        from providers.shared import ProviderType

        registry1 = ModelProviderRegistry(config={})
        registry2 = ModelProviderRegistry(config={})

        registry1.register_provider(ProviderType.GOOGLE, GeminiModelProvider)

        assert ProviderType.GOOGLE in registry1._providers
        assert ProviderType.GOOGLE not in registry2._providers


class TestGetSetDefaultRegistry:
    """Tests for get_default_registry() and set_default_registry() module-level functions."""

    def test_set_and_get_default_registry(self):
        """set_default_registry() should make the registry retrievable via get_default_registry()."""
        registry = ModelProviderRegistry(config={"key": "value"})
        set_default_registry(registry)

        result = get_default_registry()
        assert result is registry

    def test_set_default_replaces_previous(self):
        """Calling set_default_registry() again should replace the previous default."""
        registry1 = ModelProviderRegistry(config={})
        registry2 = ModelProviderRegistry(config={})

        set_default_registry(registry1)
        assert get_default_registry() is registry1

        set_default_registry(registry2)
        assert get_default_registry() is registry2

    def test_create_registry_with_config(self):
        """ModelProviderRegistry(config=...) should set config on the instance."""
        config = {"GEMINI_API_KEY": "test-key-123", "OPENAI_API_KEY": "sk-test"}
        registry = ModelProviderRegistry(config=config)
        assert registry._config == config

    def test_create_registry_starts_with_no_providers(self):
        """A new registry should start with no providers registered."""
        registry = ModelProviderRegistry(config={"GEMINI_API_KEY": "test"})
        assert len(registry._providers) == 0
