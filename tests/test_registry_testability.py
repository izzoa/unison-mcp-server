"""Tests for registry testability improvements: StorageBackend protocol and ModelProviderRegistry config injection."""

from unittest.mock import MagicMock

import pytest


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
        from utils.storage_backend import (
            get_storage_backend,
            reset_storage_backend,
        )

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
        from utils.storage_backend import (
            get_storage_backend,
            reset_storage_backend,
        )

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


class TestRegistryResetForTesting:
    """Tests for ModelProviderRegistry.reset_for_testing()."""

    def test_reset_produces_clean_registry(self):
        """reset_for_testing() should produce a registry with no providers or config."""
        from providers.registry import ModelProviderRegistry

        ModelProviderRegistry.reset_for_testing()
        registry = ModelProviderRegistry()
        assert len(registry._providers) == 0
        assert len(registry._initialized_providers) == 0
        assert registry._config is None
        ModelProviderRegistry.reset_for_testing()

    def test_reset_is_idempotent(self):
        """Calling reset_for_testing() twice should not raise."""
        from providers.registry import ModelProviderRegistry

        ModelProviderRegistry.reset_for_testing()
        ModelProviderRegistry.reset_for_testing()  # Should not raise

    def test_reset_clears_config(self):
        """reset_for_testing() should clear any injected config."""
        from providers.registry import ModelProviderRegistry

        config = {"GEMINI_API_KEY": "test-key"}
        ModelProviderRegistry.create_for_testing(config)
        registry = ModelProviderRegistry()
        assert registry._config is not None

        ModelProviderRegistry.reset_for_testing()
        registry2 = ModelProviderRegistry()
        assert registry2._config is None
        ModelProviderRegistry.reset_for_testing()


class TestCreateForTesting:
    """Tests for ModelProviderRegistry.create_for_testing()."""

    def test_create_for_testing_sets_config(self):
        """create_for_testing(config) should set config on the registry instance."""
        from providers.registry import ModelProviderRegistry

        config = {"GEMINI_API_KEY": "test-key-123", "OPENAI_API_KEY": "sk-test"}
        registry = ModelProviderRegistry.create_for_testing(config)
        assert registry._config == config
        ModelProviderRegistry.reset_for_testing()

    def test_create_for_testing_returns_clean_registry(self):
        """create_for_testing() should start with no providers registered."""
        from providers.registry import ModelProviderRegistry

        registry = ModelProviderRegistry.create_for_testing({"GEMINI_API_KEY": "test"})
        assert len(registry._providers) == 0
        ModelProviderRegistry.reset_for_testing()
