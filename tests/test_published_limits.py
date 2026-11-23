"""The published-limits lookup (D12, D13, D16): offline, conservative, min-only, one choke point.

Every test runs against a synthetic index or the pinned fixture; nothing reads the network and nothing reads the
packaged ``conf/litellm_published_limits.json`` except the structure tests that say so.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

import pytest

from providers.shared import ModelCapabilities, ProviderType
from providers.shared import published_limits as pl
from providers.shared.published_limits import PublishedIndex
from providers.shared.temperature import (
    DiscreteTemperatureConstraint,
    FixedTemperatureConstraint,
    RangeTemperatureConstraint,
)

ROOT = Path(__file__).resolve().parent.parent
PACKAGED = ROOT / "conf" / pl.EXTRACT_FILENAME


def index_of(entries: dict[str, dict[str, Any]]) -> PublishedIndex:
    return PublishedIndex.from_document({"entries": entries})


def caps(provider: ProviderType, name: str, **fields: Any) -> ModelCapabilities:
    base: dict[str, Any] = {"context_window": 200_000, "max_output_tokens": 64_000}
    base.update(fields)
    return ModelCapabilities(provider=provider, model_name=name, friendly_name=name, **base)


def install(entries: dict[str, dict[str, Any]]) -> None:
    pl.set_published_index(index_of(entries))


# ----------------------------------------------------------------------
# A published figure must be believable
# ----------------------------------------------------------------------


class TestValidOutputLimit:
    @pytest.mark.parametrize("value", [1024, 2048, 4096, 128_000, 10_000_000, 4096.0, 128_000.0])
    def test_a_believable_figure_is_accepted(self, value):
        assert pl.valid_output_limit(value) == int(value)

    @pytest.mark.parametrize(
        "value",
        [
            0,
            -1,
            -128_000,
            63,
            64,
            100,
            512,
            1023,
            10_000_001,
            10**12,
            math.nan,
            math.inf,
            -math.inf,
            4096.5,
            "4096",
            "big",
            True,
            False,
            None,
            [],
            {},
        ],
    )
    def test_a_garbled_figure_is_ignored_never_clamped(self, value):
        assert pl.valid_output_limit(value) is None

    def test_a_hostile_entry_in_the_extract_never_becomes_a_limit(self):
        index = index_of(
            {
                "zero": {"provider": "openai", "max_output_tokens": 0},
                "nan": {"provider": "openai", "max_output_tokens": float("nan")},
                "text": {"provider": "openai", "max_output_tokens": "128000"},
                "huge": {"provider": "openai", "max_output_tokens": 10**15},
                "no-provider": {"max_output_tokens": 4096},
                "not-a-dict": 7,
                "": {"provider": "openai", "max_output_tokens": 4096},
                "fine": {"provider": "openai", "max_output_tokens": 4096},
            }
        )
        assert set(index.entries) == {"fine"}

    def test_a_document_that_is_not_an_extract_is_refused(self):
        for document in ([], {}, {"entries": []}, {"entries": "x"}, "text", None):
            with pytest.raises(ValueError):
                PublishedIndex.from_document(document)


# ----------------------------------------------------------------------
# Matching is exact, tiered and provider-checked
# ----------------------------------------------------------------------


KEYS: dict[str, dict[str, Any]] = {
    "gpt-5": {"provider": "openai", "max_output_tokens": 128_000},
    "openai/gpt-4.1": {"provider": "openai", "max_output_tokens": 32_768},
    "azure/gpt-4o": {"provider": "azure", "max_output_tokens": 16_384},
    "gemini/gemini-3-pro-preview": {"provider": "gemini", "max_output_tokens": 65_536},
    "vertex_ai/gemini-3-pro-preview": {"provider": "vertex_ai-language-models", "max_output_tokens": 65_535},
    "xai/grok-4.3": {"provider": "xai", "max_output_tokens": 900_000},
    "claude-opus-4-5": {"provider": "anthropic", "max_output_tokens": 64_000},
    "claude-sonnet-4-6": {"provider": "anthropic", "max_output_tokens": 128_000},
    "claude-opus-4-7": {"provider": "anthropic", "max_output_tokens": 128_000, "supports_sampling_params": False},
    "claude-opus-4-1-20250805": {"provider": "anthropic", "max_output_tokens": 32_000},
    "mistral/mistral-medium-3-5": {"provider": "mistral", "max_output_tokens": 262_144},
    "moonshot/kimi-k3": {"provider": "moonshot", "max_output_tokens": 1_048_576},
    "zai/glm-5.1": {"provider": "zai", "max_output_tokens": 128_000},
    "openrouter/acme/robot": {"provider": "openrouter", "max_output_tokens": 8_192},
    # a foreign provider's entry under a bare key that is not the model's own
    "claude-3-haiku": {"provider": "bedrock", "max_output_tokens": 4_096},
}


@pytest.fixture
def keys():
    install(KEYS)
    return KEYS


class TestMatching:
    @pytest.mark.parametrize(
        ("provider", "name", "expected"),
        [
            (ProviderType.OPENAI, "gpt-5", 128_000),
            (ProviderType.OPENAI, "GPT-5", 128_000),
            (ProviderType.OPENAI, "gpt-4.1", 32_768),
            (ProviderType.AZURE, "gpt-4o", 16_384),
            (ProviderType.XAI, "grok-4.3", 900_000),
            (ProviderType.ANTHROPIC, "claude-opus-4-5", 64_000),
            (ProviderType.ANTHROPIC, "claude-opus-4.5", 64_000),  # the one documented dot-to-dash normalisation
            (ProviderType.OPENROUTER, "anthropic/claude-opus-4.5", 64_000),
            (ProviderType.OPENROUTER, "openai/gpt-5", 128_000),
            (ProviderType.OPENROUTER, "x-ai/grok-4.3", 900_000),
            (ProviderType.OPENROUTER, "mistralai/mistral-medium-3-5", 262_144),
            (ProviderType.OPENROUTER, "moonshotai/kimi-k3", 1_048_576),
            (ProviderType.OPENROUTER, "z-ai/glm-5.1", 128_000),
            (ProviderType.OPENROUTER, "acme/robot", 8_192),  # an exact ``openrouter/`` key
            (ProviderType.GOOGLE, "gemini-3-pro-preview", 65_535),  # two keys of one tier combine by min
            (ProviderType.OPENROUTER, "google/gemini-3-pro-preview", 65_535),
        ],
    )
    def test_an_exact_key_answers(self, keys, provider, name, expected):
        assert pl.litellm_limits(provider, name).output_cap == expected

    @pytest.mark.parametrize(
        ("provider", "name"),
        [
            (ProviderType.OPENAI, "gpt"),  # a prefix of a key
            (ProviderType.OPENAI, "gpt-5-mini"),  # another model that merely starts like one
            (ProviderType.OPENAI, "gpt-5-20260101"),  # a date suffix the key does not have
            (ProviderType.ANTHROPIC, "claude-opus-4-1"),  # the key carries a date suffix the name lacks
            (ProviderType.ANTHROPIC, "claude-opus-4.1"),
            (ProviderType.ANTHROPIC, "claude-opus"),
            (ProviderType.ANTHROPIC, "claude-3-haiku"),  # a bare key that belongs to another provider
            (ProviderType.OPENROUTER, "anthropic/claude-opus-4.1"),
            (ProviderType.OPENROUTER, "openai/gpt-5:free"),  # a colon variant is not the bare model
            (ProviderType.OPENROUTER, "openai/gpt-5:batch"),
            (ProviderType.OPENROUTER, "gpt-5"),  # no vendor prefix
            (ProviderType.OPENROUTER, "unknown/gpt-5"),  # a vendor whose native keys are unknown
            (ProviderType.CUSTOM, "gpt-5"),  # a custom endpoint's names are arbitrary
            (ProviderType.DIAL, "gpt-5"),
            (ProviderType.CUSTOM, "llama3.2:3b"),
            (ProviderType.OPENAI, "gpt 5"),
            (ProviderType.OPENAI, ""),
        ],
    )
    def test_nothing_else_attaches(self, keys, provider, name):
        limits = pl.litellm_limits(provider, name)
        assert limits.output_cap is None and not limits.matched and not limits.sampling_unsupported

    def test_a_colon_variant_never_matches_even_where_a_key_spells_it(self):
        install(
            {
                "openrouter/acme/robot:free": {"provider": "openrouter", "max_output_tokens": 111},
                "llama3.2:3b": {"provider": "openai", "max_output_tokens": 222},
            }
        )
        assert pl.litellm_limits(ProviderType.OPENROUTER, "acme/robot:free").output_cap is None
        assert pl.litellm_limits(ProviderType.OPENAI, "llama3.2:3b").output_cap is None

    @pytest.mark.parametrize(("prefixed", "native", "expected"), [(100_000, 50_000, 50_000), (50_000, 100_000, 50_000)])
    def test_every_tier_is_consulted_and_the_smallest_figure_wins(self, prefixed, native, expected):
        """A stale ``openrouter/<slug>`` key must not hide a smaller native key (or the other way round)."""

        install(
            {
                "openrouter/openai/gpt-5": {"provider": "openrouter", "max_output_tokens": prefixed},
                "gpt-5": {"provider": "openai", "max_output_tokens": native},
            }
        )
        limits = pl.litellm_limits(ProviderType.OPENROUTER, "openai/gpt-5")
        assert limits.output_cap == expected
        assert set(limits.matched) == {"openrouter/openai/gpt-5", "gpt-5"}

    def test_a_sampling_flag_in_a_later_tier_is_not_hidden_by_a_capped_earlier_tier(self):
        install(
            {
                "openrouter/anthropic/claude-x-1": {"provider": "openrouter", "max_output_tokens": 100_000},
                "claude-x-1": {"provider": "anthropic", "supports_sampling_params": False},
            }
        )
        limits = pl.litellm_limits(ProviderType.OPENROUTER, "anthropic/claude-x-1")
        assert limits.sampling_unsupported and limits.output_cap == 100_000

    def test_a_tier_without_a_hit_falls_through_to_the_next(self):
        install({"gpt-5": {"provider": "openai", "max_output_tokens": 50_000}})
        assert pl.litellm_limits(ProviderType.OPENROUTER, "openai/gpt-5").output_cap == 50_000

    def test_an_entry_of_another_provider_under_a_matching_key_is_ignored(self):
        install({"gpt-5": {"provider": "azure", "max_output_tokens": 1_000}})
        assert pl.litellm_limits(ProviderType.OPENAI, "gpt-5").output_cap is None

    def test_the_sampling_flag_is_reported_for_a_match(self, keys):
        assert pl.litellm_limits(ProviderType.ANTHROPIC, "claude-opus-4-7").sampling_unsupported
        assert pl.litellm_limits(ProviderType.OPENROUTER, "anthropic/claude-opus-4.7").sampling_unsupported
        assert not pl.litellm_limits(ProviderType.ANTHROPIC, "claude-sonnet-4-6").sampling_unsupported

    def test_a_missing_figure_with_the_flag_still_matches(self):
        install({"claude-x-1": {"provider": "anthropic", "supports_sampling_params": False}})
        limits = pl.litellm_limits(ProviderType.ANTHROPIC, "claude-x-1")
        assert limits.output_cap is None and limits.sampling_unsupported and limits.matched == ("claude-x-1",)

    def test_no_match_logs_one_debug_line_per_model(self, keys, caplog):
        with caplog.at_level(logging.DEBUG, logger=pl.__name__):
            for _ in range(3):
                pl.litellm_limits(ProviderType.OPENAI, "gpt-nope")
            pl.reset_published_limits()
            pl.litellm_limits(ProviderType.OPENAI, "gpt-other")
        lines = [r for r in caplog.records if "No LiteLLM published limit matches" in r.getMessage()]
        assert [r.levelno for r in lines] == [logging.DEBUG, logging.DEBUG]
        assert "gpt-nope" in lines[0].getMessage() and "gpt-other" in lines[1].getMessage()


# ----------------------------------------------------------------------
# Combining figures
# ----------------------------------------------------------------------


class TestCombine:
    def test_the_smallest_valid_figure_wins_and_the_earliest_source_reports_a_tie(self):
        assert pl.combine_output_caps([("live", 943_718), ("snapshot", 128_000), ("litellm", 128_000)]) == (
            128_000,
            "snapshot",
        )

    def test_an_invalid_figure_is_skipped_not_combined(self):
        assert pl.combine_output_caps([("live", 0), ("snapshot", None), ("litellm", 64_000)]) == (64_000, "litellm")
        assert pl.combine_output_caps([("live", math.nan), ("snapshot", "x"), ("litellm", True)]) is None

    def test_nothing_published_is_none_not_an_invented_figure(self):
        assert pl.combine_output_caps([]) is None


# ----------------------------------------------------------------------
# The one hook
# ----------------------------------------------------------------------


class TestApply:
    def test_a_published_cap_below_the_curated_cap_lowers_a_copy(self, keys):
        original = caps(ProviderType.OPENAI, "gpt-5", max_output_tokens=272_000, supports_temperature=True)
        result = pl.apply_published_limits(ProviderType.OPENAI, "gpt-5", original)

        assert result is not original and result.max_output_tokens == 128_000
        assert original.max_output_tokens == 272_000  # the shared registry object is never mutated

    def test_a_published_cap_above_the_curated_cap_changes_nothing_and_returns_the_same_object(self, keys):
        original = caps(
            ProviderType.ANTHROPIC,
            "claude-sonnet-4-6",
            max_output_tokens=64_000,
            temperature_constraint=RangeTemperatureConstraint(0.0, 1.0, 0.3),  # already Anthropic's documented range
        )
        assert pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-sonnet-4-6", original) is original

    def test_no_match_returns_the_same_object(self, keys):
        original = caps(ProviderType.OPENAI, "gpt-unknown")
        assert pl.apply_published_limits(ProviderType.OPENAI, "gpt-unknown", original) is original

    def test_the_copy_is_cached_per_object(self, keys):
        original = caps(ProviderType.OPENAI, "gpt-5", max_output_tokens=272_000)
        first = pl.apply_published_limits(ProviderType.OPENAI, "gpt-5", original)
        assert pl.apply_published_limits(ProviderType.OPENAI, "gpt-5", original) is first
        other = caps(ProviderType.OPENAI, "gpt-5", max_output_tokens=272_000)
        assert pl.apply_published_limits(ProviderType.OPENAI, "gpt-5", other) is not first

    def test_attributes_that_are_not_fields_survive_the_copy(self, keys):
        original = caps(ProviderType.OPENAI, "gpt-5", max_output_tokens=272_000)
        original._is_generic = True  # type: ignore[attr-defined]
        assert pl.apply_published_limits(ProviderType.OPENAI, "gpt-5", original)._is_generic is True  # type: ignore[attr-defined]

    def test_a_cap_that_is_not_set_is_not_invented(self, keys):
        original = caps(ProviderType.OPENAI, "gpt-5", max_output_tokens=0)
        assert pl.apply_published_limits(ProviderType.OPENAI, "gpt-5", original).max_output_tokens == 0

    def test_a_hostile_published_value_leaves_the_curated_cap_alone(self):
        install(
            {
                "gpt-5": {"provider": "openai", "max_output_tokens": -5},
                "gpt-4.1": {"provider": "openai", "max_output_tokens": 3},
            }
        )
        for name in ("gpt-5", "gpt-4.1"):
            original = caps(ProviderType.OPENAI, name, max_output_tokens=32_000)
            assert pl.apply_published_limits(ProviderType.OPENAI, name, original) is original

    def test_the_reduction_is_logged_once_per_model(self, keys, caplog):
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                pl.apply_published_limits(
                    ProviderType.OPENAI, "gpt-5", caps(ProviderType.OPENAI, "gpt-5", max_output_tokens=272_000)
                )
        reductions = [r for r in caplog.records if "above what" in r.getMessage()]
        assert len(reductions) == 1
        assert "gpt-5" in reductions[0].getMessage() and "128000" in reductions[0].getMessage()

    def test_every_provider_type_goes_through_the_one_hook(self, keys, monkeypatch):
        """``get_capabilities`` is the choke point: a provider that adds nothing of its own inherits the check."""

        from providers.openai import OpenAIModelProvider

        install({"gpt-5.2-pro": {"provider": "openai", "max_output_tokens": 128_000}})
        provider = OpenAIModelProvider(api_key="test-key")
        assert provider.get_capabilities("gpt-5.2-pro").max_output_tokens == 128_000
        assert provider.get_capabilities("gpt-5.2-pro") is provider.get_capabilities("gpt-5.2-pro")
        registry_entry = provider.get_all_model_capabilities()["gpt-5.2-pro"]
        assert registry_entry.max_output_tokens == 272_000  # the registry object stays curated


# ----------------------------------------------------------------------
# Temperature: published data, then the curated rule, then the labelled vendor fallback
# ----------------------------------------------------------------------


def _bounds(constraint: Any) -> tuple[float, float]:
    return constraint.min_temp, constraint.max_temp


class TestTemperature:
    def test_the_sampling_flag_fixes_an_anthropic_family_model(self, keys):
        curated = caps(ProviderType.ANTHROPIC, "claude-opus-4-7", supports_temperature=True)
        changes, provenance = pl.bound_temperature(ProviderType.ANTHROPIC, "claude-opus-4-7", curated)
        assert provenance == "litellm"
        assert changes["supports_temperature"] is False
        assert isinstance(changes["temperature_constraint"], FixedTemperatureConstraint)
        assert changes["temperature_constraint"].get_default() == 1.0

        result = pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-opus-4-7", curated)
        assert result.get_effective_temperature(0.4) is None  # nothing is sent

    def test_the_flag_applies_to_claude_on_openrouter(self, keys):
        curated = caps(ProviderType.OPENROUTER, "anthropic/claude-opus-4.7", supports_temperature=True)
        result = pl.apply_published_limits(ProviderType.OPENROUTER, "anthropic/claude-opus-4.7", curated)
        assert not result.supports_temperature and isinstance(result.temperature_constraint, FixedTemperatureConstraint)

    def test_an_already_fixed_curated_entry_is_returned_as_it_is(self, keys):
        curated = caps(
            ProviderType.ANTHROPIC,
            "claude-opus-4-7",
            supports_temperature=False,
            temperature_constraint=FixedTemperatureConstraint(1.0),
        )
        assert pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-opus-4-7", curated) is curated

    def test_the_flag_is_not_applied_to_other_vendors(self):
        install(
            {
                "gemini/gemini-3.5-flash": {
                    "provider": "gemini",
                    "supports_sampling_params": False,
                    "max_output_tokens": 65_536,
                }
            }
        )
        curated = caps(ProviderType.GOOGLE, "gemini-3.5-flash", max_output_tokens=65_536)
        assert pl.apply_published_limits(ProviderType.GOOGLE, "gemini-3.5-flash", curated) is curated

    @pytest.mark.parametrize(
        ("provider", "name", "expected"),
        [
            (ProviderType.ANTHROPIC, "claude-sonnet-4-6", (0.0, 1.0)),
            (ProviderType.OPENROUTER, "anthropic/claude-opus-4.5", (0.0, 1.0)),
            # a vendor outside the table: its "0..1" row is a guess, so a REVIEWED range is never narrowed by it
            (ProviderType.OPENROUTER, "qwen/qwen3-max", (0.0, 2.0)),
            # DIAL and custom name their vendor in an unambiguous model name
            (ProviderType.DIAL, "anthropic.claude-opus-4.1-20250805-v1:0", (0.0, 1.0)),
            (ProviderType.CUSTOM, "claude-sonnet-4-6", (0.0, 1.0)),
            (ProviderType.DIAL, "gemini-2.5-pro-preview-05-06", (0.0, 2.0)),
            (ProviderType.OPENROUTER, "openai/gpt-4.1", (0.0, 2.0)),
            (ProviderType.OPENROUTER, "google/gemini-3-pro-preview", (0.0, 2.0)),
            (ProviderType.OPENROUTER, "x-ai/grok-4.3", (0.0, 2.0)),
            (ProviderType.OPENAI, "gpt-4.1", (0.0, 2.0)),
            (ProviderType.AZURE, "gpt-4o", (0.0, 2.0)),
            (ProviderType.GOOGLE, "gemini-2.5-pro", (0.0, 2.0)),
            (ProviderType.XAI, "grok-4", (0.0, 2.0)),
            # no vendor can be told for these, so the curated range stands (nothing is guessed)
            (ProviderType.CUSTOM, "llama3.2", (0.0, 2.0)),
            (ProviderType.DIAL, "some-deployment", (0.0, 2.0)),
            (ProviderType.OPENROUTER, "no-vendor-prefix", (0.0, 2.0)),
        ],
    )
    def test_a_curated_range_is_intersected_with_the_vendor_fallback(self, provider, name, expected):
        install({})
        result = pl.apply_published_limits(provider, name, caps(provider, name, max_output_tokens=0))
        assert _bounds(result.temperature_constraint) == expected

    def test_a_generic_stand_in_takes_the_vendor_fallback_even_for_a_vendor_outside_the_table(self):
        """The fallback fills a gap: an uncurated slug's made-up 0..2 carries no review, so the table applies."""

        install({})
        generic = caps(ProviderType.OPENROUTER, "qwen/new-model", max_output_tokens=0)
        generic._is_generic = True
        result = pl.apply_published_limits(ProviderType.OPENROUTER, "qwen/new-model", generic)
        assert _bounds(result.temperature_constraint) == (0.0, 1.0)
        assert result._temperature_source == "vendor-fallback"
        assert vars(generic).get("_temperature_source") is None  # the shared object is never mutated

    def test_the_provenance_of_a_narrowed_range_is_logged_once_and_kept_on_the_capability(self, caplog):
        install({})
        original = caps(ProviderType.ANTHROPIC, "claude-sonnet-4-6", max_output_tokens=0)
        with caplog.at_level(logging.INFO, logger=pl.logger.name):
            result = pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-sonnet-4-6", original)
            # a different object (a fresh lookup) is not served from the apply cache, but still logs only once
            pl.apply_published_limits(
                ProviderType.ANTHROPIC, "claude-sonnet-4-6", caps(ProviderType.ANTHROPIC, "claude-sonnet-4-6")
            )
        assert result._temperature_source == "vendor-fallback"
        lines = [r.getMessage() for r in caplog.records if "temperature rules follow" in r.getMessage()]
        assert len(lines) == 1 and "vendor fallback table" in lines[0] and "[0.0, 1.0]" in lines[0]

    def test_the_sampling_flag_is_logged_with_its_own_provenance(self, caplog):
        install({"claude-x-1": {"provider": "anthropic", "supports_sampling_params": False}})
        with caplog.at_level(logging.INFO, logger=pl.logger.name):
            result = pl.apply_published_limits(
                ProviderType.ANTHROPIC, "claude-x-1", caps(ProviderType.ANTHROPIC, "claude-x-1", max_output_tokens=0)
            )
        assert result._temperature_source == "litellm"
        assert any("LiteLLM supports_sampling_params flag" in r.getMessage() for r in caplog.records)

    def test_the_intersection_only_narrows(self):
        install({})
        narrow = caps(
            ProviderType.OPENAI,
            "gpt-4.1",
            temperature_constraint=RangeTemperatureConstraint(0.2, 0.5, 0.3),
        )
        assert pl.apply_published_limits(ProviderType.OPENAI, "gpt-4.1", narrow) is narrow

    def test_a_disjoint_range_is_ignored_rather_than_shrunk_to_nothing(self):
        install({})
        odd = caps(
            ProviderType.ANTHROPIC,
            "claude-x",
            temperature_constraint=RangeTemperatureConstraint(1.2, 1.8, 1.5),
        )
        assert pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-x", odd) is odd

    def test_the_default_is_clamped_into_the_narrowed_range(self):
        install({})
        wide = caps(
            ProviderType.ANTHROPIC,
            "claude-x",
            temperature_constraint=RangeTemperatureConstraint(0.0, 2.0, 1.5),
        )
        result = pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-x", wide)
        assert result.temperature_constraint.get_default() == 1.0

    @pytest.mark.parametrize(
        "constraint",
        [FixedTemperatureConstraint(1.0), DiscreteTemperatureConstraint([0.0, 0.5, 1.0, 2.0], 0.5)],
    )
    def test_an_explicit_curated_constraint_is_kept(self, constraint):
        install({})
        curated = caps(ProviderType.ANTHROPIC, "claude-x", temperature_constraint=constraint)
        assert pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-x", curated) is curated

    def test_a_model_that_takes_no_temperature_is_left_alone(self):
        install({})
        curated = caps(ProviderType.ANTHROPIC, "claude-x", supports_temperature=False)
        assert pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-x", curated) is curated

    def test_equal_ranges_share_one_constraint_object(self):
        install({})
        first = pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-a", caps(ProviderType.ANTHROPIC, "claude-a"))
        second = pl.apply_published_limits(ProviderType.ANTHROPIC, "claude-b", caps(ProviderType.ANTHROPIC, "claude-b"))
        assert first.temperature_constraint is second.temperature_constraint
        assert first == pl.apply_published_limits(
            ProviderType.ANTHROPIC, "claude-a", caps(ProviderType.ANTHROPIC, "claude-a")
        )

    def test_a_vendor_the_table_does_not_name_gets_the_documented_default(self):
        from providers.shared.batch_catalog import VENDOR_TEMPERATURE_RANGES, vendor_temperature_range

        assert "qwen" not in VENDOR_TEMPERATURE_RANGES and vendor_temperature_range("qwen/x") == (0.0, 1.0)


# ----------------------------------------------------------------------
# The extract is optional, bounded and read once
# ----------------------------------------------------------------------


class TestTheExtractFile:
    def test_a_missing_file_disables_the_check_with_one_warning(self, tmp_path, caplog):
        pl.set_published_index(tmp_path / "missing.json")
        original = caps(ProviderType.OPENAI, "gpt-5", max_output_tokens=272_000)
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                assert pl.litellm_limits(ProviderType.OPENAI, "gpt-5").output_cap is None
                assert pl.apply_published_limits(ProviderType.OPENAI, "gpt-5", original).max_output_tokens == 272_000
        assert len([r for r in caplog.records if "cannot be used" in r.getMessage()]) == 1

    @pytest.mark.parametrize("text", ["not json", "[]", '{"entries": 3}', "", "\x00\x01"])
    def test_a_corrupt_file_disables_the_check_with_one_warning(self, tmp_path, caplog, text):
        path = tmp_path / "bad.json"
        path.write_text(text)
        pl.set_published_index(path)
        with caplog.at_level(logging.WARNING):
            assert pl.litellm_limits(ProviderType.OPENAI, "gpt-5").output_cap is None
            assert pl.litellm_limits(ProviderType.OPENAI, "gpt-4.1").output_cap is None
        assert len([r for r in caplog.records if "cannot be used" in r.getMessage()]) == 1

    def test_an_oversized_file_is_not_parsed(self, tmp_path, monkeypatch, caplog):
        path = tmp_path / "big.json"
        path.write_text('{"entries": {}}')
        monkeypatch.setattr(pl, "MAX_EXTRACT_BYTES", 5)
        pl.set_published_index(path)
        with caplog.at_level(logging.WARNING):
            assert pl.litellm_limits(ProviderType.OPENAI, "gpt-5").output_cap is None
        assert any("over the" in r.getMessage() for r in caplog.records)

    def test_a_good_file_loads_once_across_many_threads(self, tmp_path, monkeypatch):
        path = tmp_path / "ok.json"
        path.write_text(json.dumps({"entries": {"gpt-5": {"provider": "openai", "max_output_tokens": 128_000}}}))
        loads: list[Path] = []
        original = PublishedIndex.from_path.__func__  # type: ignore[attr-defined]

        def counting(cls, target):
            loads.append(target)
            return original(cls, target)

        monkeypatch.setattr(PublishedIndex, "from_path", classmethod(counting))
        pl.set_published_index(path)
        barrier = threading.Barrier(16)
        seen: list[int | None] = []

        def work() -> None:
            barrier.wait()
            seen.append(pl.litellm_limits(ProviderType.OPENAI, "gpt-5").output_cap)

        threads = [threading.Thread(target=work) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert seen == [128_000] * 16 and loads == [path]

    def test_the_lookup_makes_no_network_call(self, keys, monkeypatch):
        import socket

        def refuse(*args, **kwargs):
            raise AssertionError("the published-limits lookup touched the network")

        monkeypatch.setattr(socket, "create_connection", refuse)
        monkeypatch.setattr(socket.socket, "connect", refuse)
        assert pl.litellm_limits(ProviderType.OPENAI, "gpt-5").output_cap == 128_000
        pl.apply_published_limits(
            ProviderType.OPENAI, "gpt-5", caps(ProviderType.OPENAI, "gpt-5", max_output_tokens=272_000)
        )


class TestThePackagedExtract:
    """Structure of ``conf/litellm_published_limits.json``; no figure in it is asserted (a weekly refresh rewrites it)."""

    def test_it_is_valid_sorted_and_small(self):
        document = json.loads(PACKAGED.read_text())
        entries = document["entries"]
        assert document["format"] == 1 and isinstance(document["fetched_at"], str)
        assert list(entries) == sorted(entries)
        assert PACKAGED.stat().st_size < 200_000
        index = PublishedIndex.from_path(PACKAGED)
        assert len(index.entries) == len(entries)  # every entry survives the loader's own validation
        for key, record in entries.items():
            assert set(record) <= {"provider", "max_output_tokens", "supports_sampling_params"}, key
            assert pl.valid_output_limit(record.get("max_output_tokens", 64)) is not None, key

    def test_it_loads_as_the_default_index(self):
        pl.set_published_index(None)  # the packaged file
        assert pl.litellm_limits(ProviderType.OPENAI, "gpt-5.2-pro").output_cap is not None

    def test_the_pinned_test_fixture_is_a_subset_of_real_entries(self):
        fixture = json.loads((ROOT / "tests" / "fixtures" / pl.EXTRACT_FILENAME).read_text())
        assert fixture["entries"] and all(isinstance(key, str) for key in fixture["entries"])
        PublishedIndex.from_document(fixture)


class TestBuildExtract:
    CATALOG = {
        "gpt-5": {"litellm_provider": "openai", "mode": "chat", "max_output_tokens": 128_000, "max_tokens": 4},
        "claude-opus-4-7": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "max_output_tokens": 128_000,
            "supports_sampling_params": False,
        },
        "legacy-only": {"litellm_provider": "openai", "mode": "chat", "max_tokens": 8192, "max_input_tokens": 100_000},
        "legacy-is-window": {
            "litellm_provider": "openai",
            "mode": "chat",
            "max_tokens": 100_000,
            "max_input_tokens": 100_000,
        },
        "vertex_ai/gemini-x": {
            "litellm_provider": "vertex_ai-language-models",
            "mode": "chat",
            "max_output_tokens": 65_535,
        },
        "embedding": {"litellm_provider": "openai", "mode": "embedding", "max_output_tokens": 8192},
        "bedrock/claude": {"litellm_provider": "bedrock", "mode": "chat", "max_output_tokens": 4096},
        "garbled": {"litellm_provider": "openai", "mode": "chat", "max_output_tokens": -3},
        "flag-false-only": {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": False},
        "flag-true": {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": True},
        "not-a-dict": 3,
    }

    def test_only_what_the_checks_read_and_only_for_servable_providers(self):
        document = pl.build_extract(self.CATALOG, fetched_at="2026-11-22")
        assert document["fetched_at"] == "2026-11-22" and document["format"] == 1
        assert document["entries"] == {
            "claude-opus-4-7": {
                "provider": "anthropic",
                "max_output_tokens": 128_000,
                "supports_sampling_params": False,
            },
            "flag-false-only": {"provider": "anthropic", "supports_sampling_params": False},
            "gpt-5": {"provider": "openai", "max_output_tokens": 128_000},
            "legacy-only": {"provider": "openai", "max_output_tokens": 8192},  # the legacy figure stands in
            "vertex_ai/gemini-x": {"provider": "vertex_ai-language-models", "max_output_tokens": 65_535},
        }

    def test_the_output_is_deterministic(self):
        first = pl.build_extract(self.CATALOG, fetched_at="d")
        second = pl.build_extract(dict(reversed(list(self.CATALOG.items()))), fetched_at="d")
        assert json.dumps(first) == json.dumps(second)

    def test_the_refresh_script_writes_the_same_bytes_twice_and_only_on_a_change(self, tmp_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "refresh_litellm_models", ROOT / "scripts" / "refresh_litellm_models.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = tmp_path / "extract.json"

        assert module.refresh_published_limits(self.CATALOG, fetched_at="2026-11-22", path=target) is True
        first = target.read_bytes()
        assert module.refresh_published_limits(self.CATALOG, fetched_at="2099-01-01", path=target) is False
        assert target.read_bytes() == first  # an unchanged catalog leaves the file (and the weekly PR) alone
        assert json.loads(first)["entries"] == pl.build_extract(self.CATALOG, fetched_at="x")["entries"]
        changed = dict(self.CATALOG, gpt_new={"litellm_provider": "openai", "mode": "chat", "max_output_tokens": 4096})
        assert module.refresh_published_limits(changed, fetched_at="2026-11-29", path=target) is True
        PublishedIndex.from_path(target)  # what the script wrote is what the loader reads


def test_the_suite_never_lets_litellm_fetch_its_model_map_at_import():
    """Importing ``litellm`` fetches its map from GitHub unless this is set; the conftest sets it before any import."""

    import os

    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"
