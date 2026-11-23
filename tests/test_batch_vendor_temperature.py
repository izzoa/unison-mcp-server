"""A derived entry clamps temperature to its vendor's range (add-batch-model-support, D10).

A listing says whether a model takes ``temperature``, never the range. The range comes from the vendor prefix of the
resolved slug: ``anthropic`` 0..1, ``openai``, ``google`` and ``x-ai`` 0..2, and every other vendor 0..1 (the
conservative fallback). The table is one validated constant beside the family rules.

No network and no API key (the fixture under ``tests/fixtures`` and the in-process batch service).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from providers.openrouter import OpenRouterProvider
from providers.registries.openrouter import OpenRouterModelRegistry
from providers.shared import batch_catalog as bc
from providers.shared.batch import BatchRequest
from providers.shared.batch_catalog import (
    DEFAULT_TEMPERATURE_RANGE,
    VENDOR_TEMPERATURE_RANGES,
    BatchCatalogFormatError,
    validate_vendor_temperature_ranges,
    vendor_temperature_range,
)
from providers.shared.batch_resolution import derive_capabilities
from tests.batch_test_helpers import OpenRouterBatchService, fixture_listings
from utils.model_context import ModelContext

REPO = Path(__file__).resolve().parent.parent

#: The 13 vendors of the committed snapshot's batch slugs (2026-11-22) and the range each one gets.
EXPECTED = {
    "openai": (0.0, 2.0),
    "google": (0.0, 2.0),
    "x-ai": (0.0, 2.0),
    "anthropic": (0.0, 1.0),
    "deepseek": (0.0, 1.0),
    "qwen": (0.0, 1.0),
    "z-ai": (0.0, 1.0),
    "moonshotai": (0.0, 1.0),
    "morph": (0.0, 1.0),
    "meta": (0.0, 1.0),
    "thinkingmachines": (0.0, 1.0),
    "minimax": (0.0, 1.0),
    "mistralai": (0.0, 1.0),
}


def _uncurated_slug_per_vendor() -> dict[str, str]:
    """For each vendor, the first batch slug whose listing names ``temperature`` and that is not a reviewed entry."""

    registry = OpenRouterModelRegistry()
    snapshot = json.loads((REPO / "conf" / "openrouter_batch_catalog.json").read_text(encoding="utf-8"))
    listings = fixture_listings()
    chosen: dict[str, str] = {}
    for slug in sorted(snapshot["batch_models"]):
        vendor = slug.partition("/")[0]
        if vendor in chosen or "temperature" not in listings[f"{slug}:batch"]["supported_parameters"]:
            continue
        if OpenRouterProvider._is_reviewed_entry(registry.get_capabilities(slug), slug):
            continue
        chosen[vendor] = slug
    return chosen


SLUG_BY_VENDOR = _uncurated_slug_per_vendor()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import utils.model_restrictions as model_restrictions
    from utils import batch_store as bs

    for name in ("BATCH_STORE_PATH", "OPENROUTER_ALLOWED_MODELS", "OPENROUTER_BATCH_LIVE_CATALOG"):
        monkeypatch.delenv(name, raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


@pytest.fixture(autouse=True)
def _no_synchronous_openrouter_call(monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("the synchronous provider path ran for a batch request")

    monkeypatch.setattr(OpenRouterProvider, "generate_content", boom)
    monkeypatch.setattr(OpenRouterProvider, "async_generate_content", boom, raising=False)


@pytest.fixture
def provider() -> OpenRouterProvider:
    return OpenRouterProvider(api_key="test-key")


@pytest.fixture
def service() -> OpenRouterBatchService:
    return OpenRouterBatchService()


def _sent_temperature(provider: OpenRouterProvider, service: Any, slug: str, asked: float) -> float | None:
    provider._batch_send = service
    provider.submit_batch(slug, [BatchRequest(custom_id="a", prompt="hello", temperature=asked)])
    return service.posts[-1]["envelope"]["requests"][0]["body"].get("temperature")


# ----------------------------------------------------------------------
# The table
# ----------------------------------------------------------------------


class TestTheTable:
    def test_it_names_exactly_four_vendors_and_the_fallback_is_the_conservative_range(self):
        assert dict(VENDOR_TEMPERATURE_RANGES) == {
            "anthropic": (0.0, 1.0),
            "openai": (0.0, 2.0),
            "google": (0.0, 2.0),
            "x-ai": (0.0, 2.0),
        }
        assert DEFAULT_TEMPERATURE_RANGE == (0.0, 1.0)

    def test_the_table_cannot_be_edited_at_run_time(self):
        with pytest.raises(TypeError):
            VENDOR_TEMPERATURE_RANGES["qwen"] = (0.0, 2.0)  # type: ignore[index]

    def test_the_snapshots_vendors_are_the_thirteen_the_decision_covers(self):
        snapshot = json.loads((REPO / "conf" / "openrouter_batch_catalog.json").read_text(encoding="utf-8"))
        assert {slug.partition("/")[0] for slug in snapshot["batch_models"]} == set(EXPECTED)

    @pytest.mark.parametrize(("vendor", "expected"), sorted(EXPECTED.items()))
    def test_every_vendor_of_the_snapshot_maps_to_its_range(self, vendor, expected):
        assert vendor_temperature_range(f"{vendor}/some-model-1") == expected

    @pytest.mark.parametrize(
        "slug",
        ["acme/robot-2", "newvendor/x", "OpenAI/GPT-4.1", " google/gemini-9 ", "ANTHROPIC/claude-x", "plain-name", ""],
    )
    def test_case_and_whitespace_are_ignored_and_an_unknown_vendor_or_a_missing_prefix_gets_the_fallback(self, slug):
        expected = {"OpenAI/GPT-4.1": (0.0, 2.0), " google/gemini-9 ": (0.0, 2.0), "ANTHROPIC/claude-x": (0.0, 1.0)}
        assert vendor_temperature_range(slug) == expected.get(slug, DEFAULT_TEMPERATURE_RANGE)

    def test_a_bare_vendor_name_is_not_a_slug_and_gets_the_fallback(self):
        assert vendor_temperature_range("openai") == DEFAULT_TEMPERATURE_RANGE  # no ``vendor/`` prefix

    def test_a_custom_table_is_read_the_same_way(self):
        assert vendor_temperature_range("acme/x", {"acme": (0.0, 1.5)}) == (0.0, 1.5)
        assert vendor_temperature_range("other/x", {"acme": (0.0, 1.5)}) == DEFAULT_TEMPERATURE_RANGE


class TestTheTableIsValidatedAtLoad:
    def test_the_shipped_table_validates(self):
        validate_vendor_temperature_ranges(VENDOR_TEMPERATURE_RANGES)

    @pytest.mark.parametrize(
        ("table", "needle"),
        [
            ({"openai": (0.0, 2.5)}, "openai"),  # above the ceiling
            ({"openai": (-0.1, 1.0)}, "openai"),  # below zero
            ({"openai": (1.0, 1.0)}, "openai"),  # an empty range
            ({"openai": (2.0, 1.0)}, "openai"),  # reversed
            ({"openai": (0.0, float("nan"))}, "openai"),
            ({"openai": (0.0, float("inf"))}, "openai"),
            ({"openai": (True, 2.0)}, "openai"),  # a bool is not a number here
            ({"openai": ("0", "2")}, "openai"),
            ({"openai": (0.0,)}, "openai"),  # not a pair
            ({"openai": 2.0}, "openai"),  # not a pair
            ({"OpenAI": (0.0, 2.0)}, "OpenAI"),  # not lower case
            ({"open/ai": (0.0, 2.0)}, "open/ai"),  # a slash is not part of a vendor
            ({"": (0.0, 2.0)}, "''"),
            ({" openai": (0.0, 2.0)}, "openai"),
        ],
    )
    def test_a_bad_table_is_a_load_error_naming_the_vendor(self, table, needle):
        with pytest.raises(BatchCatalogFormatError) as excinfo:
            validate_vendor_temperature_ranges(table)  # type: ignore[arg-type]
        assert needle in str(excinfo.value)

    def test_validation_runs_when_the_module_loads(self):
        source = (REPO / "providers" / "shared" / "batch_catalog.py").read_text(encoding="utf-8")
        assert "\nvalidate_vendor_temperature_ranges(VENDOR_TEMPERATURE_RANGES)\n" in source

    def test_there_is_one_table_and_no_second_derived_range_constant(self):
        source = (REPO / "providers" / "shared" / "batch_resolution.py").read_text(encoding="utf-8")
        assert "_DERIVED_TEMPERATURE =" not in source and "RangeTemperatureConstraint(0.0, 1.0" not in source
        assert "vendor_temperature_range(" in source


# ----------------------------------------------------------------------
# Derived entries clamp to their vendor's range
# ----------------------------------------------------------------------


class TestEveryVendorOfTheSnapshot:
    def test_a_slug_that_lists_temperature_was_found_for_every_vendor(self):
        assert set(SLUG_BY_VENDOR) == set(EXPECTED)

    @pytest.mark.parametrize("vendor", sorted(EXPECTED))
    def test_the_resolved_entry_validates_inside_and_clamps_outside_its_vendors_range(self, provider, vendor):
        slug = SLUG_BY_VENDOR[vendor]
        low, high = EXPECTED[vendor]

        resolution = provider._resolve_batch_model(slug)
        capabilities = resolution.capabilities
        constraint = capabilities.temperature_constraint

        assert resolution.derived and capabilities.supports_temperature
        assert constraint.validate(low) and constraint.validate(high) and constraint.validate((low + high) / 2)
        assert not constraint.validate(high + 0.01) and not constraint.validate(low - 0.01)
        assert constraint.get_corrected_value(99.0) == high and constraint.get_corrected_value(-5.0) == low

    @pytest.mark.parametrize("vendor", sorted(EXPECTED))
    def test_the_request_sends_a_temperature_clamped_to_that_range(self, provider, service, vendor):
        slug = SLUG_BY_VENDOR[vendor]
        _, high = EXPECTED[vendor]

        assert _sent_temperature(provider, service, slug, 1.5) == min(1.5, high)
        assert _sent_temperature(provider, service, slug, 3.0) == high
        assert _sent_temperature(provider, service, slug, 0.4) == 0.4

    def test_a_vendor_missing_from_the_table_is_clamped_to_the_conservative_range(self, provider, service):
        # derive a capability for a vendor that is in no table, from a listing that names temperature
        capabilities = derive_capabilities(
            {
                "context_length": 100_000,
                "max_completion_tokens": 8_000,
                "supported_parameters": ["temperature", "max_tokens"],
                "input_modalities": None,
            },
            provider._lookup_capabilities("acme/robot-2"),
            slug="acme/robot-2",
        )

        assert capabilities.temperature_constraint.validate(1.0)
        assert not capabilities.temperature_constraint.validate(1.01)

    def test_a_listing_without_temperature_sends_none_whatever_the_vendor(self, provider, service):
        assert "temperature" not in fixture_listings()["openai/gpt-5.5:batch"]["supported_parameters"]

        assert _sent_temperature(provider, service, "openai/gpt-5.5", 1.5) is None

    def test_two_derivations_of_one_listing_give_equal_capabilities(self, provider):
        slug = SLUG_BY_VENDOR["openai"]
        assert provider._resolve_batch_model(slug).capabilities == provider._resolve_batch_model(slug).capabilities

    def test_a_curated_entry_keeps_its_own_range_narrowed_by_the_vendor_fallback(self, provider, service):
        """D13/D16 (amends D10): a curated RANGE is intersected with the labelled vendor fallback, never widened."""

        curated = provider._registry.get_capabilities("anthropic/claude-opus-4.5")
        assert curated is not None and curated.temperature_constraint.max_temp == 2.0  # reviewed 0..2
        resolution = provider._resolve_batch_model("anthropic/claude-opus-4.5")

        constraint = resolution.capabilities.temperature_constraint
        assert (constraint.min_temp, constraint.max_temp) == (0.0, 1.0)  # Anthropic's documented range

        gpt5 = provider._registry.get_capabilities("openai/gpt-5")
        assert gpt5 is not None
        assert provider._resolve_batch_model("openai/gpt-5").capabilities.temperature_constraint is (
            gpt5.temperature_constraint
        )  # a 0..2 vendor keeps the curated object

    def test_an_alias_that_floats_to_an_openai_slug_takes_openais_range(self, provider, service):
        # gpt5 -> openai/gpt-5.5, whose batch listing names no temperature: none is sent; the range is moot
        assert _sent_temperature(provider, service, "gpt5", 1.5) is None
        # grok -> x-ai/grok-4.7 lists temperature: 0..2
        assert provider._resolve_batch_model("grok").capabilities.temperature_constraint.validate(2.0)
        assert _sent_temperature(provider, service, "grok", 1.5) == 1.5


# ----------------------------------------------------------------------
# A clamp is a WARNING, a model that takes no temperature is INFO
# ----------------------------------------------------------------------


def _support_records(caplog: Any) -> list[logging.LogRecord]:
    """What ``log_batch_temperature_warnings`` logged (other tests may leave the root logger at DEBUG)."""

    return [record for record in caplog.records if record.name == "tools.shared.batch_support"]


class TestAMalformedTemperatureIsRefusedBeforeAnythingIsSent:
    """A range clamp turns NaN into the upper bound and passes a ``bool`` through at a 0..2 vendor (``min(2.0, True)``
    is ``True``, sent as the JSON literal ``true``); a string raises a bare ``TypeError``. None of that is acceptable at
    a vendor whose range the clamp now widened, so the request is refused locally with a ``ValueError``."""

    @pytest.mark.parametrize("bad", [float("nan"), True, False, "0.5", [0.5], object()])
    @pytest.mark.parametrize("vendor", ["openai", "x-ai", "anthropic", "qwen"])
    def test_it_is_refused_with_no_network_call(self, provider, service, vendor, bad):
        provider._batch_send = service

        with pytest.raises(ValueError, match="invalid temperature"):
            provider.submit_batch(SLUG_BY_VENDOR[vendor], [BatchRequest(custom_id="a", prompt="hi", temperature=bad)])

        assert service.posts == []

    def test_the_message_names_the_request(self, provider, service):
        provider._batch_send = service

        with pytest.raises(ValueError, match=r"'row-7'.*nan"):
            provider.submit_batch(
                SLUG_BY_VENDOR["openai"],
                [
                    BatchRequest(custom_id="row-1", prompt="hi", temperature=0.2),
                    BatchRequest(custom_id="row-7", prompt="hi", temperature=float("nan")),
                ],
            )

    @pytest.mark.parametrize("vendor", ["openai", "anthropic"])
    def test_a_default_an_int_and_an_infinity_are_still_accepted_and_clamped(self, provider, service, vendor):
        high = EXPECTED[vendor][1]
        slug = SLUG_BY_VENDOR[vendor]

        assert _sent_temperature(provider, service, slug, 1) == 1  # an int is a number
        assert _sent_temperature(provider, service, slug, float("inf")) == high
        assert _sent_temperature(provider, service, slug, float("-inf")) == 0.0

        provider.submit_batch(slug, [BatchRequest(custom_id="b", prompt="hi")])  # None: the provider default
        assert service.posts[-1]["envelope"]["requests"][0]["body"]["temperature"] == 0.3


class TestTheLogLevelOfWhatTheValidatorSays:
    def _validate(self, provider: OpenRouterProvider, slug: str, asked: float) -> tuple[float, list[str]]:
        from tools.chat import ChatTool

        context = ModelContext(slug, provider, provider._resolve_batch_model(slug).capabilities)
        return ChatTool().validate_and_correct_temperature(asked, context)

    @pytest.mark.parametrize("vendor", ["openai", "google", "x-ai"])
    def test_a_clamp_at_a_zero_to_two_vendor_still_reaches_logger_warning(self, provider, caplog, vendor):
        from tools.shared.batch_support import BatchResumableToolMixin

        corrected, warnings = self._validate(provider, SLUG_BY_VENDOR[vendor], 2.5)
        assert corrected == 2.0 and len(warnings) == 1
        assert "does not accept a temperature" not in warnings[0]  # the text the INFO split keys on

        with caplog.at_level(logging.INFO):
            BatchResumableToolMixin.log_batch_temperature_warnings(warnings)

        assert [(r.levelno, r.getMessage()) for r in _support_records(caplog)] == [(logging.WARNING, warnings[0])]
        assert "2.5" in warnings[0] and "Using 2.0" in warnings[0] and "[0.0, 2.0]" in warnings[0]

    @pytest.mark.parametrize("vendor", ["anthropic", "qwen", "deepseek"])
    def test_a_clamp_at_a_zero_to_one_vendor_is_a_warning_too(self, provider, caplog, vendor):
        from tools.shared.batch_support import BatchResumableToolMixin

        corrected, warnings = self._validate(provider, SLUG_BY_VENDOR[vendor], 1.5)
        assert corrected == 1.0

        with caplog.at_level(logging.INFO):
            BatchResumableToolMixin.log_batch_temperature_warnings(warnings)

        assert [r.levelno for r in _support_records(caplog)] == [logging.WARNING]

    def test_a_temperature_inside_the_vendors_range_says_nothing(self, provider):
        assert self._validate(provider, SLUG_BY_VENDOR["openai"], 1.5) == (1.5, [])

    def test_an_unsupported_temperature_stays_info(self, provider, caplog):
        from tools.shared.batch_support import BatchResumableToolMixin

        _, warnings = self._validate(provider, "openai/gpt-5.5", 0.3)  # an openai listing that names no temperature
        assert len(warnings) == 1 and "does not accept a temperature" in warnings[0]

        with caplog.at_level(logging.INFO):
            BatchResumableToolMixin.log_batch_temperature_warnings(warnings)

        assert [r.levelno for r in _support_records(caplog)] == [logging.INFO]

    def test_the_tool_schemas_accept_up_to_the_widest_range_and_the_model_then_clamps(self):
        """D13 (replaces the 0..1 pin of D10): the tools accept 0..2 and the per-model rule clamps where the model is known."""

        from pydantic import ValidationError

        from tools.chat import ChatRequest

        assert ChatRequest(prompt="p", working_directory_absolute_path="/tmp", temperature=1.5).temperature == 1.5
        assert ChatRequest(prompt="p", working_directory_absolute_path="/tmp", temperature=2.0).temperature == 2.0
        with pytest.raises(ValidationError):
            ChatRequest(prompt="p", working_directory_absolute_path="/tmp", temperature=2.01)
        with pytest.raises(ValidationError):
            ChatRequest(prompt="p", working_directory_absolute_path="/tmp", temperature=-0.1)


def test_the_module_exports_the_table_and_its_reader():
    assert "VENDOR_TEMPERATURE_RANGES" in bc.__all__ and "vendor_temperature_range" in bc.__all__
