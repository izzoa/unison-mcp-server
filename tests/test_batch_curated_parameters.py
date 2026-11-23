"""A curated slug sends the parameters its ``:batch`` listing names (add-batch-model-support, D9).

Before D9 a slug the reviewed ``conf/openrouter_models.json`` lists kept its curated request construction under
batch: ``openai/gpt-5`` sent ``temperature``, ``top_p`` and ``stop`` although its ``:batch`` listing names only
``seed`` and ``max_tokens``, and ``openai/o3`` (curated as taking no temperature) sent nothing but the messages, so a
caller's ``max_output_tokens`` and ``seed`` were silently dropped although the listing names both. Now the body carries
exactly the parameters the listing names (live, then snapshot), and the curated LIMITS stay.

No network and no API key: catalogs are the fixture under ``tests/fixtures`` (the 2026-11-22 list) or a hand-built
variant of it, served by an injected fetcher; the batch API is the in-process ``OpenRouterBatchService``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from providers.openrouter import OpenRouterProvider
from providers.registries.openrouter import OpenRouterModelRegistry
from providers.shared import batch_catalog as bc
from providers.shared.batch import BatchRequest
from providers.shared.batch_catalog import BatchCatalog, BatchCatalogSource, load_snapshot
from providers.shared.batch_resolution import PARAMETERS_FIELD, PUBLISHED_CAP_FIELD, apply_listed_parameters
from tests.batch_test_helpers import (
    OpenRouterBatchService,
    RecordingFetcher,
    batch_world_factory,
    fixture_listings,
    install_batch_catalog,
    parameter_source,
    payload_with,
)
from tests.test_batch_dynamic_resolution import _chat_args, _outer

#: Every optional name the request builder can forward (the temperature is decided separately).
FORWARDED = ("top_p", "seed", "stop", "frequency_penalty", "presence_penalty")
EXTRA = {"top_p": 0.9, "seed": 7, "stop": ["END"], "frequency_penalty": 0.1, "presence_penalty": 0.2}

#: The curated slugs the section-16 review named, and what their batch listings name (2026-11-22).
UNLISTED_TEMPERATURE = ("openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5.2")
NO_TEMPERATURE_CURATED = ("openai/o3", "openai/o3-mini", "openai/o4-mini", "openai/gpt-5.2-pro")


def _curated_slugs_with_a_batch_listing() -> list[str]:
    """Every reviewed curated slug whose ``<slug>:batch`` listing the fixture (a copy of the live list) carries."""

    registry = OpenRouterModelRegistry()
    listings = fixture_listings()
    slugs = set()
    for name in registry.list_models():
        config = registry.resolve(name)
        if config is not None and not config.auto_discovered and f"{config.model_name}:batch" in listings:
            slugs.add(config.model_name)
    return sorted(slugs)


CURATED = _curated_slugs_with_a_batch_listing()


def _listing(slug: str) -> set[str]:
    return set(fixture_listings()[f"{slug}:batch"]["supported_parameters"])


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import utils.model_restrictions as model_restrictions
    from utils import batch_store as bs

    for name in (
        "BATCH_STORE_PATH",
        "BATCH_RETENTION_DAYS",
        "OPENROUTER_ALLOWED_MODELS",
        "OPENROUTER_BATCH_LIVE_CATALOG",
        "OPENROUTER_BATCH_CATALOG_TTL_SECONDS",
    ):
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
def service() -> OpenRouterBatchService:
    return OpenRouterBatchService()


@pytest.fixture
def provider() -> OpenRouterProvider:
    return OpenRouterProvider(api_key="test-key")


@pytest.fixture
def boot(tmp_path, service):
    started = []
    factory = batch_world_factory(tmp_path, service)

    def make():
        world = factory()
        started.append(world)
        return world

    yield make
    for world in started:
        try:
            world.close()
        except Exception:
            pass


def _body(provider: OpenRouterProvider, service: Any, model: str, **request: Any) -> dict[str, Any]:
    provider._batch_send = service
    provider.submit_batch(
        model,
        [
            BatchRequest(
                custom_id="a",
                prompt="hello",
                system_prompt="sys",
                **{"temperature": 0.5, "max_output_tokens": 500, "extra": dict(EXTRA), **request},
            )
        ],
    )
    return service.posts[-1]["envelope"]["requests"][0]["body"]


def _expected_keys(slug: str, curated_supports_temperature: bool) -> set[str]:
    """What a body must carry, worked out from the LISTING alone (not from anything the code under test computes)."""

    listed = _listing(slug)
    keys = {"messages"}
    if curated_supports_temperature and "temperature" in listed:
        keys.add("temperature")
    if "max_tokens" in listed:
        keys.add("max_tokens")
    elif "max_completion_tokens" in listed:
        keys.add("max_completion_tokens")
    keys.update(name for name in FORWARDED if name in listed)
    return keys


# ----------------------------------------------------------------------
# Every curated slug in the real list: only listed parameters are sent
# ----------------------------------------------------------------------


class TestEveryCuratedSlugSendsOnlyWhatItsListingNames:
    def test_the_set_under_test_is_not_vacuous(self):
        assert set(UNLISTED_TEMPERATURE + NO_TEMPERATURE_CURATED) <= set(CURATED)
        assert {"anthropic/claude-opus-4.5", "anthropic/claude-opus-4.1", "anthropic/claude-sonnet-4.5"} <= set(CURATED)
        # the premise of D9: curated slugs exist whose listing lacks temperature, top_p or seed
        assert any("temperature" not in _listing(slug) for slug in CURATED)
        assert any("top_p" not in _listing(slug) for slug in CURATED)
        assert any("seed" not in _listing(slug) for slug in CURATED)

    @pytest.mark.parametrize("slug", CURATED)
    def test_the_body_carries_exactly_the_listed_parameters(self, provider, service, slug):
        config = OpenRouterModelRegistry().get_capabilities(slug)
        assert config is not None

        body = _body(provider, service, slug)

        assert set(body) == _expected_keys(slug, config.supports_temperature), slug
        listed = _listing(slug)
        for name in FORWARDED:
            assert (name in body) == (name in listed), (slug, name)
        if "max_tokens" in listed:  # the spelling the listing uses
            assert body["max_tokens"] == 500 and "max_completion_tokens" not in body
        elif "max_completion_tokens" in listed:
            assert body["max_completion_tokens"] == 500 and "max_tokens" not in body
        else:
            assert "max_tokens" not in body and "max_completion_tokens" not in body

    @pytest.mark.parametrize("slug", UNLISTED_TEMPERATURE)
    def test_the_gpt5_family_no_longer_sends_temperature_top_p_or_stop(self, provider, service, slug):
        assert not {"temperature", "top_p", "stop"} & _listing(slug)  # the premise, from the listing

        body = _body(provider, service, slug)

        assert "temperature" not in body and "top_p" not in body and "stop" not in body
        assert body["seed"] == 7 and body["max_tokens"] == 500

    @pytest.mark.parametrize("slug", ("anthropic/claude-opus-4.5", "anthropic/claude-opus-4.1"))
    def test_the_anthropic_opus_entries_no_longer_send_seed_or_top_p(self, provider, service, slug):
        body = _body(provider, service, slug)

        assert "seed" not in body and "top_p" not in body
        assert body["temperature"] == 0.5 and body["max_tokens"] == 500 and body["stop"] == ["END"]

    def test_sonnet_4_5_sends_top_p_but_no_seed(self, provider, service):
        body = _body(provider, service, "anthropic/claude-sonnet-4.5")

        assert body["top_p"] == 0.9 and "seed" not in body and body["temperature"] == 0.5

    @pytest.mark.parametrize("slug", NO_TEMPERATURE_CURATED)
    def test_a_model_that_takes_no_temperature_now_keeps_its_token_cap_and_seed(self, provider, service, slug):
        assert "temperature" not in _listing(slug) and {"max_tokens", "seed"} <= _listing(slug)

        body = _body(provider, service, slug)

        assert "temperature" not in body and "top_p" not in body and "stop" not in body
        assert body["max_tokens"] == 500 and body["seed"] == 7

    def test_the_token_cap_is_spelled_the_way_the_listing_spells_it(self, provider, service):
        # a curated slug whose listing names only max_completion_tokens
        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with(
                    {
                        "openai/gpt-5:batch": lambda entry: entry.update(
                            supported_parameters=["max_completion_tokens", "seed"]
                        )
                    }
                )
            )
        )

        body = _body(provider, service, "openai/gpt-5")

        assert body["max_completion_tokens"] == 500 and "max_tokens" not in body and body["seed"] == 7

    def test_a_listing_naming_a_parameter_never_has_it_dropped(self, provider, service):
        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with(
                    {
                        "openai/gpt-5:batch": lambda entry: entry.update(
                            supported_parameters=["temperature", "top_p", "seed", "max_tokens", "stop"]
                        )
                    }
                )
            )
        )

        body = _body(provider, service, "openai/gpt-5")

        # gpt-5's curated constraint is the fixed one, so the temperature sent is that constraint's value
        assert {"temperature", "top_p", "seed", "max_tokens", "stop"} <= set(body)


# ----------------------------------------------------------------------
# Limits stay curated; the temperature support is never widened
# ----------------------------------------------------------------------


#: The output cap a curated slug ends with under D12: the smallest of the reviewed cap and what OpenRouter's listing and
#: LiteLLM publish. Only these two of the eleven differ from the reviewed cap (the committed snapshot and the pinned
#: LiteLLM extract both say so): opus-4.1 is listed at 32,000 and gpt-5.2-pro at 128,000.
BOUNDED_CAP = {"anthropic/claude-opus-4.1": 32_000, "openai/gpt-5.2-pro": 128_000}
#: The Anthropic slugs' reviewed 0..2 range is narrowed to Anthropic's documented 0..1 (D13, the vendor fallback).
NARROWED_TO_ONE = {"anthropic/claude-opus-4.1", "anthropic/claude-opus-4.5", "anthropic/claude-sonnet-4.5"}


class TestCuratedLimitsStay:
    @pytest.mark.parametrize("slug", CURATED)
    def test_the_resolution_keeps_the_reviewed_limits_and_the_window_stays_enforced(self, provider, slug):
        curated = provider._registry.get_capabilities(slug)
        assert curated is not None

        resolution = provider._resolve_batch_model(slug)

        assert not resolution.derived  # derived is NOT the "send only listed parameters" switch
        assert resolution.window_is_listed()  # a reviewed window is enforced
        capabilities = resolution.capabilities
        assert capabilities.context_window == curated.context_window  # the window is never merged (D16)
        # D12: the output cap is the smallest of the reviewed cap and every published one, never above the reviewed one
        assert capabilities.max_output_tokens == BOUNDED_CAP.get(slug, curated.max_output_tokens)
        assert capabilities.max_output_tokens <= curated.max_output_tokens
        assert capabilities.model_name == curated.model_name
        if slug in NARROWED_TO_ONE:
            constraint = capabilities.temperature_constraint
            assert (constraint.min_temp, constraint.max_temp) == (0.0, 1.0)
        else:
            assert capabilities.temperature_constraint is curated.temperature_constraint
        assert capabilities.supports_function_calling == curated.supports_function_calling
        assert capabilities.supported_parameters == tuple(sorted(_listing(slug) & bc.LISTED_PARAMETERS))
        assert capabilities.supports_temperature == (curated.supports_temperature and "temperature" in _listing(slug))
        assert parameter_source(resolution) == "live"
        assert resolution.sources.get(PUBLISHED_CAP_FIELD) in ("live", "snapshot", "litellm", None)

    def test_a_curated_entry_that_lacks_temperature_never_gains_it_from_a_listing(self, provider, service):
        assert "temperature" not in _listing("openai/o3")
        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with(
                    {
                        "openai/o3:batch": lambda entry: entry.update(
                            supported_parameters=["temperature", "max_tokens", "seed"]
                        )
                    }
                )
            )
        )

        assert provider._resolve_batch_model("openai/o3").capabilities.supports_temperature is False
        body = _body(provider, service, "openai/o3")
        assert "temperature" not in body and body["max_tokens"] == 500 and body["seed"] == 7

    def test_that_listing_does_not_make_the_validator_correct_a_value_that_is_never_sent(self, provider):
        """The stored list leaves ``temperature`` out, so the tools' validator says "unsupported" (INFO), not a clamp."""

        from tools.chat import ChatTool
        from utils.model_context import ModelContext

        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with(
                    {
                        "openai/o3:batch": lambda entry: entry.update(
                            supported_parameters=["temperature", "max_tokens", "seed"]
                        )
                    }
                )
            )
        )
        capabilities = provider._resolve_batch_model("openai/o3").capabilities

        assert "temperature" not in capabilities.supported_parameters
        _, warnings = ChatTool().validate_and_correct_temperature(
            0.3, ModelContext("openai/o3", provider, capabilities)
        )
        assert len(warnings) == 1 and "does not accept a temperature" in warnings[0]
        assert "Using" not in warnings[0]

    def test_the_output_cap_is_imposed_on_a_curated_entry_at_submission(self, provider, service):
        """D12 (replaces ``..._is_not_imposed_...``): a curated entry's cap, bounded by what is published, clamps."""

        curated = OpenRouterModelRegistry().get_capabilities("anthropic/claude-opus-4.5")
        assert curated is not None and curated.max_output_tokens == 64_000

        body = _body(provider, service, "anthropic/claude-opus-4.5", max_output_tokens=500_000)

        assert body["max_tokens"] == 64_000
        # a request at or below the cap is sent as given, and opus-4.1 is bounded by OpenRouter's listing (32,000)
        assert _body(provider, service, "anthropic/claude-opus-4.5", max_output_tokens=1_000)["max_tokens"] == 1_000
        assert _body(provider, service, "anthropic/claude-opus-4.1", max_output_tokens=500_000)["max_tokens"] == 32_000

    def test_a_synchronous_capability_query_is_untouched(self, provider):
        capabilities = provider.get_capabilities("openai/gpt-5")

        assert capabilities.supported_parameters is None and capabilities.supports_temperature


# ----------------------------------------------------------------------
# A family alias that lands on a reviewed curated slug follows the listing too
# ----------------------------------------------------------------------


class TestAFloatingAliasLandingOnACuratedSlug:
    """``opus`` lands on ``claude-opus-4.5`` and ``gpt5`` on ``openai/gpt-5`` when nothing newer is allowed or listed.

    That is the floating-alias branch of ``_resolve_batch_model``, a different call site from the full-slug branch the
    class above drives, so D9 is pinned here separately (a mutant that built the legacy resolution on this branch
    passed every other test).
    """

    @staticmethod
    def _pinned(monkeypatch, *slugs: str) -> None:
        import utils.model_restrictions as model_restrictions

        monkeypatch.setenv("OPENROUTER_ALLOWED_MODELS", ",".join(slugs))
        model_restrictions._restriction_service = None

    def test_opus_landing_on_opus_4_5_sends_no_seed_or_top_p(self, provider, service, monkeypatch):
        self._pinned(monkeypatch, "anthropic/claude-opus-4.5")
        install_batch_catalog()  # the whole fixture: a live catalog

        resolution = provider._resolve_batch_model("opus")
        body = _body(provider, service, "opus")

        assert (resolution.slug, resolution.floated, resolution.derived) == ("anthropic/claude-opus-4.5", True, False)
        assert parameter_source(resolution) == "live"
        assert set(body) == _expected_keys("anthropic/claude-opus-4.5", True)
        assert "seed" not in body and "top_p" not in body and body["temperature"] == 0.5

    def test_gpt5_landing_on_gpt_5_sends_no_temperature_top_p_or_stop(self, provider, service, monkeypatch):
        self._pinned(monkeypatch, "openai/gpt-5")
        install_batch_catalog()

        resolution = provider._resolve_batch_model("gpt5")
        body = _body(provider, service, "gpt5")

        assert (resolution.slug, resolution.floated, resolution.derived) == ("openai/gpt-5", True, False)
        assert set(body) == _expected_keys("openai/gpt-5", True) == {"messages", "max_tokens", "seed"}

    def test_the_landing_keeps_the_curated_limits_and_the_listing_outranks_the_snapshot(
        self, provider, service, monkeypatch
    ):
        self._pinned(monkeypatch, "anthropic/claude-opus-4.5")
        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with(
                    {"anthropic/claude-opus-4.5:batch": lambda e: e.update(supported_parameters=["max_tokens", "seed"])}
                )
            )
        )

        resolution = provider._resolve_batch_model("opus")
        body = _body(provider, service, "opus")

        assert resolution.capabilities.max_output_tokens == 64_000 and resolution.window_is_listed()
        assert parameter_source(resolution) == "live"
        assert set(body) == {"messages", "max_tokens", "seed"}  # the live list, not the snapshot's

    def test_offline_the_snapshot_answers_for_the_landing(self, provider, service, monkeypatch):
        self._pinned(monkeypatch, "openai/gpt-5")
        install_batch_catalog(batch_slugs=["openai/gpt-5"])  # slugs only: no views, so the snapshot speaks

        resolution = provider._resolve_batch_model("gpt5")

        assert resolution.floated and parameter_source(resolution) == "snapshot"
        assert set(_body(provider, service, "gpt5")) == {"messages", "max_tokens", "seed"}


# ----------------------------------------------------------------------
# Source order, the fallback, and the record
# ----------------------------------------------------------------------


def _view(*names: str) -> dict[str, Any]:
    return {
        "context_length": None,
        "max_completion_tokens": None,
        "supported_parameters": sorted(names) or None,
        "input_modalities": None,
        "output_modalities": None,
    }


def _snapshot(views: dict[str, dict[str, Any]]) -> BatchCatalog:
    return BatchCatalog(frozenset(views), origin="snapshot", views=views)


class TestTheSourceOfTheParameterList:
    def test_the_live_listing_outranks_the_snapshot(self, provider):
        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with({"openai/gpt-5:batch": lambda e: e.update(supported_parameters=["top_p", "max_tokens"])})
            )
        )

        resolution = provider._resolve_batch_model("openai/gpt-5")

        assert resolution.capabilities.supported_parameters == ("max_tokens", "top_p")
        assert parameter_source(resolution) == "live"

    def test_a_live_listing_without_a_usable_list_falls_back_to_the_snapshot_for_that_field(self, provider):
        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with({"openai/gpt-5:batch": lambda e: e.update(supported_parameters="temperature")})  # garbled
            )
        )
        committed = load_snapshot()
        assert committed is not None and committed.view_for("openai/gpt-5") is not None

        resolution = provider._resolve_batch_model("openai/gpt-5")

        assert parameter_source(resolution) == "snapshot"
        assert resolution.capabilities.supported_parameters == tuple(
            committed.view_for("openai/gpt-5")["supported_parameters"]
        )

    def test_with_the_live_fetch_failing_the_committed_snapshot_speaks_and_builds_the_same_body(
        self, provider, service
    ):
        live_body = _body(provider, service, "openai/gpt-5")
        bc.set_batch_catalog_source(BatchCatalogSource(fetcher=RecordingFetcher(error=TimeoutError("down"))))

        resolution = provider._resolve_batch_model("openai/gpt-5")
        snapshot_body = _body(provider, OpenRouterBatchService(), "openai/gpt-5")

        assert parameter_source(resolution) == "snapshot" and resolution.catalog_origin == "snapshot"
        assert snapshot_body == live_body

    def test_no_listing_anywhere_leaves_the_curated_entry_to_decide_with_one_warning(self, provider, service, caplog):
        bc.set_batch_catalog_source(
            BatchCatalogSource(fetcher=RecordingFetcher(error=TimeoutError("down")), snapshot=lambda: _snapshot({}))
        )
        with caplog.at_level(logging.WARNING):
            first = provider._resolve_batch_model("openai/gpt-5")
            provider._resolve_batch_model("openai/gpt-5")
            body = _body(provider, service, "openai/gpt-5")

        assert first.capabilities.supported_parameters is None
        assert parameter_source(first) == "curated"
        # the curated entry decides: gpt-5 supports temperature, so the old temperature-gated body is built
        assert {"temperature", "max_tokens", "top_p", "seed", "stop"} <= set(body)
        warnings = [r.getMessage() for r in caplog.records if "lists no batch parameters" in r.getMessage()]
        assert len(warnings) == 1 and "openai/gpt-5" in warnings[0]

    def test_not_even_a_readable_snapshot_is_the_same_fallback(self, provider, service):
        bc.set_batch_catalog_source(
            BatchCatalogSource(fetcher=RecordingFetcher(error=TimeoutError("down")), snapshot=lambda: None)
        )

        resolution = provider._resolve_batch_model("openai/gpt-5")  # admitted by its committed supports_batch flag

        assert parameter_source(resolution) == "curated" and resolution.capabilities.supported_parameters is None

    def test_apply_listed_parameters_is_pure_and_never_raises_on_a_hostile_view(self):
        curated = OpenRouterModelRegistry().get_capabilities("openai/gpt-5")
        assert curated is not None
        for views in ([], [("live", None)], [("live", {"supported_parameters": "temperature"})], [("live", {})]):
            result, source = apply_listed_parameters(curated, slug="openai/gpt-5", sources=views)  # type: ignore[arg-type]
            assert result is curated and source == "curated"
        result, source = apply_listed_parameters(
            curated, slug="openai/gpt-5", sources=[("live", _view("seed")), ("snapshot", _view("top_p"))]
        )
        assert source == "live" and result.supported_parameters == ("seed",) and not result.supports_temperature
        assert curated.supported_parameters is None  # the registry's entry is never mutated


class TestTheRecordCarriesTheListAndItsSource:
    def test_to_record_stores_parameters_and_their_source_for_a_curated_slug(self, provider):
        record = provider._resolve_batch_model("openai/gpt-5").to_record()

        assert record["derived"] is False and {
            k: v for k, v in record["sources"].items() if k != PUBLISHED_CAP_FIELD
        } == {PARAMETERS_FIELD: "live"}
        assert set(record["parameters"]) == _listing("openai/gpt-5") & bc.LISTED_PARAMETERS
        assert json.loads(json.dumps(record)) == record

    def test_the_fallback_is_recorded_as_curated_without_a_list(self, provider):
        bc.set_batch_catalog_source(
            BatchCatalogSource(fetcher=RecordingFetcher(error=TimeoutError("down")), snapshot=lambda: _snapshot({}))
        )

        record = provider._resolve_batch_model("openai/gpt-5").to_record()

        assert {k: v for k, v in record["sources"].items() if k != PUBLISHED_CAP_FIELD} == {
            PARAMETERS_FIELD: "curated"
        } and "parameters" not in record

    @pytest.mark.asyncio
    async def test_the_child_record_keeps_what_the_body_was_built_from(self, boot, service, tmp_path):
        world = boot()

        running = _outer(await world.chat.execute(_chat_args(tmp_path, "openai/gpt-5", wait_seconds=0)))

        handle = json.loads(running["content"])["batch_handle"]
        record = world.store.get(handle)
        assert record is not None
        stored = record.children[0].resolution
        assert {k: v for k, v in stored["sources"].items() if k != PUBLISHED_CAP_FIELD} == {
            PARAMETERS_FIELD: "live"
        } and "seed" in stored["parameters"]
        assert "temperature" not in stored["parameters"]
        body = service.posts[-1]["envelope"]["requests"][0]["body"]
        assert "temperature" not in body and "top_p" not in body and "stop" not in body

    @pytest.mark.asyncio
    async def test_a_default_temperature_call_to_a_curated_slug_without_temperature_logs_info_not_a_warning(
        self, boot, service, tmp_path, caplog
    ):
        world = boot()

        with caplog.at_level(logging.INFO):
            await world.chat.execute(_chat_args(tmp_path, "openai/gpt-5", wait_seconds=0))

        body = service.posts[-1]["envelope"]["requests"][0]["body"]
        assert "temperature" not in body
        notices = [r for r in caplog.records if "does not accept a temperature" in r.getMessage()]
        assert notices and all(r.levelno == logging.INFO for r in notices)


# ----------------------------------------------------------------------
# The snapshot is narrower than a live listing, and says so
# ----------------------------------------------------------------------


class TestTheSnapshotIsNarrowerThanALiveListing:
    @pytest.mark.parametrize(
        ("slug", "live_only"),
        [
            ("openai/gpt-5", {"reasoning_effort", "verbosity", "tool_choice"}),
            ("openai/gpt-5-mini", {"reasoning_effort", "verbosity", "tool_choice"}),
            ("openai/gpt-5.2", {"reasoning_effort", "verbosity", "tool_choice"}),
            ("anthropic/claude-opus-4.5", {"tool_choice"}),
            ("anthropic/claude-sonnet-4.5", {"tool_choice", "top_k"}),
        ],
    )
    def test_the_names_a_live_listing_has_and_the_snapshot_omits(self, slug, live_only):
        committed = load_snapshot()
        assert committed is not None
        view = committed.view_for(slug)
        assert view is not None
        snapshot_names = set(view["supported_parameters"])

        assert live_only <= _listing(slug)  # the live listing names them ...
        assert not live_only & snapshot_names  # ... the snapshot's reduction does not
        assert snapshot_names == _listing(slug) & bc.LISTED_PARAMETERS  # it is exactly the allowlisted reduction
        assert not live_only & bc.LISTED_PARAMETERS  # none of them is a name the server forwards

    def test_the_guide_and_the_design_say_so(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parent.parent / "docs" / "tools" / "batch.md").read_text(encoding="utf-8")
        assert "narrower than a live listing" in text
        for name in ("reasoning_effort", "verbosity", "tool_choice", "top_k"):
            assert name in text
