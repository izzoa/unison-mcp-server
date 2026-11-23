"""Consistency checks for the simulator test registry (no network, no server)."""

import pytest

from simulator_tests import TEST_REGISTRY


@pytest.mark.parametrize("registry_key", sorted(TEST_REGISTRY))
def test_simulator_test_name_matches_registry_key(registry_key):
    """communication_simulator_test.py selects tests by registry key and reports them by test_name.

    A class whose test_name differs from its key (ChatSimpleValidationTest used to return
    "_validation") shows up under a different name in logs and summaries than the one the
    caller asked for.
    """
    instance = TEST_REGISTRY[registry_key](verbose=False)
    assert instance.test_name == registry_key
