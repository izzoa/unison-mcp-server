"""Live-derived capabilities of a resolved batch slug (add-batch-model-support, section 15, D8).

Under ``execution_mode="batch"`` the limits applied to a call are those of the slug the name resolved to, derived
from that slug's own ``:batch`` listing (live catalog, then the committed snapshot, then the curated entry), resolved
once at admission, carried to submission, and persisted with the child record.

No network and no API key: catalogs are the fixture under ``tests/fixtures`` (or a hand-built variant of it) served
by an injected fetcher, and the batch API is the in-process ``OpenRouterBatchService``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

import httpx
import pytest

from providers.openrouter import BatchModelRefused, OpenRouterProvider
from providers.shared import ProviderType
from providers.shared import batch_catalog as bc
from providers.shared.batch import BatchRequest
from providers.shared.batch_catalog import BatchCatalog, BatchCatalogSource, parse_catalog_payload
from providers.shared.batch_resolution import (
    PUBLISHED_CAP_FIELD,
    BatchModelResolution,
    derive_capabilities,
    limits_from_record,
    note_from_record,
)
from providers.shared.temperature import RangeTemperatureConstraint
from tests.batch_test_helpers import (
    OpenRouterBatchService,
    RecordingFetcher,
    batch_world_factory,
    catalog_fixture_payload,
    install_batch_catalog,
    parameter_source,
    payload_with,
)
from tests.test_batch_dynamic_resolution import (
    GEMINI_FLASH,
    GPT_55,
    GROK_47,
    HAIKU_55,
    OPUS_55,
    SONNET_55,
    _chat_args,
    _consensus_args,
    _entry,
    _outer,
)
from tools.shared.exceptions import ToolExecutionError
from utils.model_context import ModelContext

CURATED_OPUS = "anthropic/claude-opus-4.5"
ANTHROPIC_LISTING_TEMPERATURE = (
    "anthropic/claude-opus-4.6"  # an uncurated Anthropic slug whose batch listing names temperature
)
ARGON = "google/gemini-4-argon"


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


@pytest.fixture
def provider() -> OpenRouterProvider:
    return OpenRouterProvider(api_key="test-key")


def _generic(provider: OpenRouterProvider) -> Any:
    capabilities = provider._lookup_capabilities("some/uncurated-model")
    assert capabilities is not None
    return capabilities


def _snapshot_with(views: dict[str, dict[str, Any]]) -> BatchCatalog:
    return BatchCatalog(frozenset(views), origin="snapshot", views=views)


def _opus_56(**fields: Any) -> dict[str, dict[str, Any]]:
    return {"anthropic/claude-opus-5.6": fields}


def _warnings(caplog: Any, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and needle in r.getMessage()]


def _limits(world: Any, handle: str) -> list[dict[str, int] | None]:
    record = world.store.get(handle)
    assert record is not None
    return [limits_from_record(child.resolution) for child in record.children]


# ----------------------------------------------------------------------
# 15.4 derivation and the per-field chain
# ----------------------------------------------------------------------


class TestCapabilityDerivation:
    def test_a_listing_without_temperature_derives_a_model_that_takes_none(self, provider):
        view = {
            "context_length": 1_000_000,
            "max_completion_tokens": 128_000,
            "supported_parameters": ["max_tokens", "reasoning", "tools"],
            "input_modalities": ["image", "text"],
        }

        capabilities = derive_capabilities(view, _generic(provider), slug=OPUS_55)

        assert capabilities.model_name == OPUS_55 and capabilities.aliases == [] and capabilities.batch_family is None
        assert capabilities.context_window == 1_000_000 and capabilities.max_output_tokens == 128_000
        assert capabilities.supports_temperature is False and capabilities.get_effective_temperature(0.3) is None
        assert capabilities.supports_function_calling and capabilities.supports_extended_thinking
        assert capabilities.supports_images and not capabilities.supports_json_mode
        assert capabilities.supported_parameters == ("max_tokens", "reasoning", "tools")
        assert capabilities.accepts_parameter("max_tokens") is True and capabilities.accepts_parameter("seed") is False

    def test_a_listed_temperature_gets_the_zero_to_one_range_not_the_family_entrys(self, provider):
        capabilities = derive_capabilities({"supported_parameters": ["temperature"]}, _generic(provider), slug="a/b")

        assert capabilities.supports_temperature and isinstance(
            capabilities.temperature_constraint, RangeTemperatureConstraint
        )
        assert capabilities.get_effective_temperature(1.5) == 1.0 and capabilities.get_effective_temperature(0.3) == 0.3

    def test_a_view_with_no_parameters_leaves_the_fallbacks_flags_alone(self, provider):
        curated = provider._registry.get_capabilities(CURATED_OPUS)
        assert curated is not None

        capabilities = derive_capabilities({"context_length": 5_000}, curated, slug="a/b")

        assert capabilities.context_window == 5_000 and capabilities.supported_parameters is None
        assert capabilities.supports_temperature == curated.supports_temperature
        assert capabilities.accepts_parameter("temperature") is None  # unknown, not "no"
        assert capabilities.max_output_tokens == 5_000  # the curated 64,000 is never above the new window

    def test_the_output_cap_is_never_above_the_window_whatever_the_sources(self, provider):
        capabilities = derive_capabilities(
            {"context_length": 4_096, "max_completion_tokens": 9_999}, _generic(provider), slug="a/b"
        )
        assert capabilities.max_output_tokens == 4_096

    def test_the_same_body_parsed_live_and_read_from_the_snapshot_derives_equal_capabilities_for_all_92(self, provider):
        live = parse_catalog_payload(catalog_fixture_payload(), origin="live")
        snapshot = bc.load_snapshot()
        assert snapshot is not None and len(live.slugs) == len(snapshot.slugs) == 92
        fallback = _generic(provider)

        for slug in sorted(live.slugs):
            from_live = derive_capabilities(dict(live.views[slug]), fallback, slug=slug)
            from_snapshot = derive_capabilities(dict(snapshot.views[slug]), fallback, slug=slug)
            assert from_live == from_snapshot, slug
            assert from_live.max_output_tokens <= from_live.context_window

    def test_one_bad_live_field_falls_back_alone_and_the_rest_stay_live(self, provider, caplog):
        def break_context(entry: dict[str, Any]) -> None:
            entry["context_length"] = "big"

        body = payload_with({OPUS_55 + ":batch": break_context, OPUS_55: break_context})
        snapshot = _snapshot_with(
            {OPUS_55: {"context_length": 777_777, "max_completion_tokens": 1, "supported_parameters": ["seed"]}}
        )
        bc.set_batch_catalog_source(BatchCatalogSource(fetcher=RecordingFetcher(body), snapshot=lambda: snapshot))

        with caplog.at_level(logging.DEBUG, logger="providers.shared.batch_resolution"):
            resolution = provider._resolve_batch_model("opus")

        assert resolution.slug == OPUS_55
        assert {k: v for k, v in resolution.sources.items() if k != PUBLISHED_CAP_FIELD} == {
            "context_length": "snapshot",
            "max_completion_tokens": "live",
            "supported_parameters": "live",
            "input_modalities": "live",
        }
        capabilities = resolution.capabilities
        assert capabilities.context_window == 777_777 and capabilities.max_output_tokens == 128_000
        assert capabilities.accepts_parameter("seed") is False  # the live parameter list, not the snapshot's
        assert any("lacked" in r.getMessage() and "context_length" in r.getMessage() for r in caplog.records)
        assert not _warnings(caplog, "partly taken")  # the snapshot answered; the curated entry was not needed

    def test_live_unreachable_uses_the_snapshot_for_every_field(self, provider):
        bc.set_batch_catalog_source(BatchCatalogSource(fetcher=RecordingFetcher(error=httpx.ConnectError("down"))))

        resolution = provider._resolve_batch_model("opus")

        assert resolution.catalog_origin == "snapshot" and resolution.slug == OPUS_55
        assert set(resolution.sources.values()) == {"snapshot"}
        assert resolution.capabilities.context_window == 1_000_000
        assert resolution.capabilities.max_output_tokens == 128_000
        assert resolution.capabilities.supports_temperature is False

    def test_live_disabled_by_the_environment_uses_the_snapshot_and_makes_no_request(self, provider, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BATCH_LIVE_CATALOG", "false")
        fetcher = install_batch_catalog()

        resolution = provider._resolve_batch_model("opus")

        assert fetcher.calls == [] and set(resolution.sources.values()) == {"snapshot"}

    def test_live_and_snapshot_give_the_same_capabilities_for_the_same_slug(self, provider):
        live = provider._resolve_batch_model("opus")
        bc.set_batch_catalog_source(BatchCatalogSource(fetcher=RecordingFetcher(error=httpx.ConnectError("x"))))
        offline = provider._resolve_batch_model("opus")

        assert live.catalog_origin == "live" and offline.catalog_origin == "snapshot"
        assert live.capabilities == offline.capabilities and live.slug == offline.slug

    def test_a_slug_no_source_describes_takes_the_curated_entry_with_one_warning_across_two_calls(
        self, provider, caplog
    ):
        install_batch_catalog(fetcher=RecordingFetcher(payload_with(extra_batch=_opus_56())))

        with caplog.at_level(logging.WARNING):
            first = provider._resolve_batch_model("opus")
            second = provider._resolve_batch_model("opus")

        assert first.slug == "anthropic/claude-opus-5.6"
        assert dict(first.sources) == dict.fromkeys(first.sources, "curated") and len(first.sources) == 4
        assert first.capabilities.context_window == 200_000 and first.capabilities.max_output_tokens == 64_000
        assert second.capabilities == first.capabilities
        warned = _warnings(caplog, "partly taken")
        assert len(warned) == 1
        assert "anthropic/claude-opus-5.6" in warned[0] and "anthropic/claude-opus-4.5" in warned[0]
        for name in ("context_length", "max_completion_tokens", "supported_parameters", "input_modalities"):
            assert name in warned[0]

        # a different set of fields is a different situation: a second warning
        install_batch_catalog(
            fetcher=RecordingFetcher(payload_with(extra_batch=_opus_56(supported_parameters=["temperature"])))
        )
        with caplog.at_level(logging.WARNING):
            third = provider._resolve_batch_model("opus")
        assert third.capabilities.supports_temperature is True  # the one field it had is believed
        warned = _warnings(caplog, "partly taken")
        assert len(warned) == 2 and "supported_parameters" not in warned[1].split("because")[1]

    def test_a_full_uncurated_slug_with_no_listing_data_falls_back_to_generic_defaults_with_a_warning(
        self, provider, caplog
    ):
        install_batch_catalog(fetcher=RecordingFetcher(payload_with(extra_batch={"newco/new-1": {}})))

        with caplog.at_level(logging.WARNING):
            resolution = provider._resolve_batch_model("newco/new-1")

        assert resolution.capabilities.context_window == 32_768 == resolution.capabilities.max_output_tokens
        assert set(resolution.sources.values()) == {"generic"}
        assert _warnings(caplog, "generic defaults")

    def test_an_output_cap_above_the_context_is_never_produced_even_across_sources(self, provider):
        live = {"context_length": 4_096, "top_provider": {"max_completion_tokens": "x"}, "supported_parameters": []}
        snapshot = _snapshot_with({"anthropic/claude-opus-5.6": {"max_completion_tokens": 128_000}})
        bc.set_batch_catalog_source(
            BatchCatalogSource(
                fetcher=RecordingFetcher(payload_with(extra_batch=_opus_56(**live))), snapshot=lambda: snapshot
            )
        )

        capabilities = provider._resolve_batch_model("opus").capabilities

        assert capabilities.context_window == 4_096 and capabilities.max_output_tokens == 4_096

    def test_a_hostile_body_never_raises_and_never_yields_an_unsafe_value(self, provider):
        garbage = {
            "context_length": -1,
            "top_provider": {"max_completion_tokens": 10**12},
            "supported_parameters": "temperature",
            "architecture": {"input_modalities": 5, "output_modalities": ["x"] * 99},
        }
        install_batch_catalog(fetcher=RecordingFetcher(payload_with(extra_batch=_opus_56(**garbage))))

        capabilities = provider._resolve_batch_model("opus").capabilities

        assert capabilities.context_window > 0 and 0 < capabilities.max_output_tokens <= capabilities.context_window
        assert capabilities.supported_parameters is None  # no list believed, so none is invented


# ----------------------------------------------------------------------
# 15.5 resolve once, carry, consume
# ----------------------------------------------------------------------


class TestEachResolvedSlugCarriesItsOwnLimits:
    def test_opus_under_batch_gets_the_limits_of_opus_5_5_not_of_the_curated_4_5_entry(self, provider):
        curated = provider._registry.get_capabilities(CURATED_OPUS)
        assert curated is not None and curated.context_window == 200_000 and curated.max_output_tokens == 64_000

        resolution = provider._resolve_batch_model("opus")

        assert resolution.slug == OPUS_55 and resolution.derived and resolution.floated
        assert resolution.capabilities.context_window == 1_000_000
        assert resolution.capabilities.max_output_tokens == 128_000
        assert resolution.capabilities.model_name == OPUS_55 and resolution.requested == "opus"

    def test_a_full_uncurated_slug_gets_its_own_listing_not_the_generic_32k_defaults(self, provider):
        assert provider.get_capabilities(GEMINI_FLASH).context_window == 32_768  # the synchronous path: unchanged

        resolution = provider._resolve_batch_model(GEMINI_FLASH)

        assert resolution.derived and not resolution.floated and resolution.slug == GEMINI_FLASH
        assert resolution.capabilities.context_window == 1_048_576
        assert resolution.capabilities.max_output_tokens == 65_536
        assert not resolution.capabilities.supports_temperature  # its batch listing takes none

    def test_a_curated_slug_batched_by_full_name_keeps_its_curated_entry(self, provider):
        resolution = provider._resolve_batch_model(CURATED_OPUS)

        # Changed by D9: the limits stay the reviewed entry's (so ``derived`` stays false and the window is enforced), but the
        # request parameters are the ones its ``:batch`` listing names, and the record says where that list came from.
        assert not resolution.derived and parameter_source(resolution) == "live"
        assert resolution.capabilities.context_window == 200_000 and resolution.capabilities.max_output_tokens == 64_000
        listed = resolution.capabilities.supported_parameters
        assert listed is not None and {"temperature", "max_tokens", "stop"} <= set(listed)
        assert not {"top_p", "seed"} & set(listed)  # opus-4.5's batch listing names neither

    def test_a_version_pinned_alias_resolves_to_the_curated_entry_too(self, provider):
        resolution = provider._resolve_batch_model("opus4.5")
        assert resolution.slug == CURATED_OPUS and not resolution.derived and not resolution.floated

    def test_every_floating_alias_is_described_by_its_own_listing(self, provider):
        expected = {
            "opus": (OPUS_55, 1_000_000, 128_000),
            "sonnet": (SONNET_55, 1_000_000, 128_000),
            "haiku": (HAIKU_55, 1_000_000, 128_000),
            "flash": (GEMINI_FLASH, 1_048_576, 65_536),
            "gpt5": (GPT_55, 1_050_000, 128_000),
            "grok": (GROK_47, 500_000, 450_000),
        }
        for alias, (slug, window, output) in expected.items():
            resolution = provider._resolve_batch_model(alias)
            assert (resolution.slug, resolution.capabilities.context_window) == (slug, window), alias
            assert resolution.capabilities.max_output_tokens == output, alias

    def test_capability_queries_and_the_synchronous_path_never_fetch_and_are_unchanged(self, provider):
        fetcher = install_batch_catalog()

        sync_opus = provider.get_capabilities("opus")
        provider.get_capabilities(GEMINI_FLASH)
        provider.validate_model_name("flash")
        provider.list_models()
        provider.batch_family_preview("opus")
        context = ModelContext("opus", provider)

        assert sync_opus.model_name == CURATED_OPUS and sync_opus.context_window == 200_000
        assert context.capabilities.context_window == 200_000 and context.capabilities.supported_parameters is None
        assert fetcher.calls == []

    @pytest.mark.asyncio
    async def test_listmodels_makes_no_fetch(self):
        import os
        from unittest.mock import patch

        from providers.registry import get_default_registry
        from tools.listmodels import ListModelsTool

        fetcher = install_batch_catalog()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key", "DEFAULT_MODEL": "auto"}, clear=True):
            get_default_registry().register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
            await ListModelsTool().execute({})

        assert fetcher.calls == []


def _count_tokens_by_length(monkeypatch: Any) -> None:
    monkeypatch.setattr(OpenRouterProvider, "count_tokens", lambda self, text, model_name: len(text) // 4)


def _prompt_of(world: Any, characters: int) -> None:
    async def assembled(request: Any, arguments: Any) -> str:
        return "x" * characters

    world.chat._assemble_batch_prompt = assembled


class TestInputSizeUsesTheResolvedWindow:
    @pytest.mark.asyncio
    async def test_a_prompt_over_the_curated_window_but_inside_the_resolved_one_is_admitted(
        self, boot, service, tmp_path, monkeypatch
    ):
        _count_tokens_by_length(monkeypatch)
        world = boot()
        _prompt_of(world, 1_000_000)  # about 250,000 tokens: over the curated 4.5 window (200,000), inside 1,000,000

        await world.chat.execute(_chat_args(tmp_path, "opus", wait_seconds=0))

        assert service.submitted_models == [OPUS_55]

    @pytest.mark.asyncio
    async def test_the_same_prompt_is_refused_for_the_curated_slug_naming_its_own_window(
        self, boot, service, tmp_path, monkeypatch
    ):
        _count_tokens_by_length(monkeypatch)
        world = boot()
        _prompt_of(world, 1_000_000)

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.chat.execute(_chat_args(tmp_path, CURATED_OPUS, wait_seconds=0))

        content = json.loads(excinfo.value.payload)["content"]
        assert CURATED_OPUS in content and "200,000" in content and service.posts == []

    @pytest.mark.asyncio
    async def test_a_prompt_beyond_the_resolved_window_is_refused_before_any_batch_exists(
        self, boot, service, tmp_path, monkeypatch
    ):
        _count_tokens_by_length(monkeypatch)
        world = boot()
        _prompt_of(world, 4_400_000)  # about 1,100,000 tokens

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.chat.execute(_chat_args(tmp_path, "opus", wait_seconds=0))

        content = json.loads(excinfo.value.payload)["content"]
        assert OPUS_55 in content and "1,000,000" in content and "Nothing was submitted" in content
        assert service.posts == [] and world.store.needs_attention() == []

    @pytest.mark.asyncio
    async def test_two_consensus_entries_each_use_their_own_limits(self, boot, service):
        world = boot()
        seen: list[tuple[str, int, int]] = []
        original = world.consensus.check_batch_input_fits

        def spy(model_context: Any, slug: str, texts: Any, **kwargs: Any) -> None:
            capabilities = model_context.capabilities
            seen.append((slug, capabilities.context_window, capabilities.max_output_tokens))
            original(model_context, slug, texts, **kwargs)

        world.consensus.check_batch_input_fits = spy

        run = await world.consensus.execute(
            _consensus_args([_entry("opus", "for"), _entry("gemini", "against")], wait_seconds=0)
        )

        assert sorted(set(seen)) == [
            ("anthropic/claude-opus-5.5", 1_000_000, 128_000),
            ("google/gemini-3.1-pro-preview", 1_048_576, 65_536),
        ]
        handle = json.loads(json.loads(run[0].text)["content"])["batch_handle"]
        assert _limits(world, handle) == [
            {"context_window": 1_000_000, "max_output_tokens": 128_000},
            {"context_window": 1_048_576, "max_output_tokens": 65_536},
        ]


class TestTheResolutionIsMadeOnceAndCarried:
    @pytest.mark.asyncio
    async def test_a_catalog_swapped_between_admission_and_submission_changes_neither_slug_nor_limits(
        self, boot, service, tmp_path
    ):
        world = boot()
        newer = payload_with(
            extra_batch={
                "anthropic/claude-opus-6.0": {
                    "context_length": 2_000_000,
                    "top_provider": {"max_completion_tokens": 200_000},
                    "supported_parameters": ["temperature", "max_tokens"],
                }
            }
        )
        swapped = RecordingFetcher(newer)
        original = world.chat.run_batch_submission

        async def swap_then_submit(*args: Any, **kwargs: Any) -> Any:
            bc.set_batch_catalog_source(BatchCatalogSource(fetcher=swapped))  # admission is over; the world moved on
            return await original(*args, **kwargs)

        world.chat.run_batch_submission = swap_then_submit

        running = _outer(await world.chat.execute(_chat_args(tmp_path, "opus", wait_seconds=0)))

        assert service.submitted_models == [OPUS_55]  # not claude-opus-6.0
        handle = json.loads(running["content"])["batch_handle"]
        record = world.store.get(handle)
        assert record is not None and record.children[0].submitted_model == OPUS_55
        assert _limits(world, handle) == [{"context_window": 1_000_000, "max_output_tokens": 128_000}]
        assert swapped.calls == []  # submission asked no catalog anything: the carried resolution was used

    @pytest.mark.asyncio
    async def test_the_coordinator_hands_the_admission_resolution_to_the_provider(self, boot, service, tmp_path):
        world = boot()
        seen: list[Any] = []
        original = type(world.provider).submit_batch

        def spy(self: Any, model_name: str, requests: Any, resolution: Any = None) -> str:
            seen.append(resolution)
            return original(self, model_name, requests, resolution=resolution)

        type(world.provider).submit_batch = spy  # type: ignore[method-assign]
        try:
            await world.chat.execute(_chat_args(tmp_path, "flash", wait_seconds=0))
        finally:
            type(world.provider).submit_batch = original  # type: ignore[method-assign]

        assert len(seen) == 1 and isinstance(seen[0], BatchModelResolution)
        assert seen[0].slug == GEMINI_FLASH and seen[0].requested == "flash"

    def test_submit_batch_refuses_a_resolution_made_for_another_model(self, provider):
        resolution = provider._resolve_batch_model("opus")
        provider._batch_send = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no request"))  # type: ignore[assignment]

        with pytest.raises(ValueError, match="resolution is for 'opus'"):
            provider.submit_batch("sonnet", [BatchRequest(custom_id="a", prompt="p")], resolution=resolution)


# ----------------------------------------------------------------------
# 15.6 request construction honours the listing
# ----------------------------------------------------------------------


def _bodies(service: Any) -> list[dict[str, Any]]:
    return [post["envelope"]["requests"][0]["body"] for post in service.posts]


def _submit(provider: Any, service: Any, model: str, **request: Any) -> dict[str, Any]:
    provider._batch_send = service
    provider.submit_batch(model, [BatchRequest(custom_id="a", prompt="hello", system_prompt="sys", **request)])
    return _bodies(service)[-1]


class TestRequestConstructionHonoursTheListing:
    def test_a_listing_without_temperature_yields_a_body_without_it_but_keeps_the_token_cap(self, provider, service):
        body = _submit(provider, service, "opus", temperature=0.3, max_output_tokens=1_000)

        assert "temperature" not in body  # opus-5.5's batch listing takes none
        assert body["max_tokens"] == 1_000  # the old coupling would have dropped it with the temperature
        assert body["messages"][-1] == {"role": "user", "content": "hello"}

    def test_only_the_listed_passthrough_parameters_are_sent(self, provider, service):
        # gpt-5.5's batch listing names seed and max_tokens, and neither temperature nor top_p
        body = _submit(
            provider, service, "gpt5", temperature=0.3, max_output_tokens=500, extra={"top_p": 0.9, "seed": 7}
        )

        assert body["seed"] == 7 and "top_p" not in body and "temperature" not in body
        assert body["max_tokens"] == 500

    def test_a_requested_temperature_above_one_is_sent_as_one_where_the_vendor_stops_at_one(self, provider, service):
        # Amended by D10 (it named grok, an x-ai model, now 0..2): an Anthropic listing that names temperature.
        body = _submit(provider, service, ANTHROPIC_LISTING_TEMPERATURE, temperature=1.5)

        assert body["temperature"] == 1.0

    def test_a_requested_temperature_above_one_is_sent_as_asked_where_the_vendor_takes_two(self, provider, service):
        body = _submit(
            provider, service, "grok", temperature=1.5
        )  # grok-4.7: x-ai, 0..2, its listing names temperature
        assert body["temperature"] == 1.5
        assert _submit(provider, service, "grok", temperature=2.0)["temperature"] == 2.0

    def test_a_parameter_the_listing_names_is_not_dropped(self, provider, service):
        body = _submit(provider, service, "grok", temperature=0.5, max_output_tokens=900, extra={"top_p": 0.8})

        assert body["temperature"] == 0.5 and body["top_p"] == 0.8 and body["max_tokens"] == 900

    def test_an_output_cap_above_the_models_is_reduced_to_it(self, provider, service):
        body = _submit(provider, service, "opus", max_output_tokens=500_000)
        assert body["max_tokens"] == 128_000

    def test_a_cap_that_is_a_stand_in_clamps_nothing(self, provider, service):
        """A derived entry whose output cap no listing gave carries the generic 32768; it must not clamp a request."""

        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with(
                    extra_batch={"some/new-model": {"supported_parameters": ["temperature", "max_tokens", "seed"]}}
                )
            )
        )

        resolution = provider._resolve_batch_model("some/new-model")
        assert resolution.derived and resolution.sources["max_completion_tokens"] == "generic"
        assert resolution.output_cap_bound() is None  # nobody stated a cap for this model: nothing is invented

        body = _submit(provider, service, "some/new-model", temperature=0.5, max_output_tokens=200_000)
        assert body["max_tokens"] == 200_000

    def test_a_cap_the_listing_gave_still_clamps(self, provider, service):
        resolution = provider._resolve_batch_model("opus")
        assert resolution.sources["max_completion_tokens"] in ("live", "snapshot")
        assert resolution.output_cap_bound() is not None

    def test_a_listing_that_names_only_max_completion_tokens_gets_that_key(self, provider, service):
        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_with(
                    extra_batch=_opus_56(
                        context_length=100_000,
                        top_provider={"max_completion_tokens": 8_000},
                        supported_parameters=["max_completion_tokens"],
                    )
                )
            )
        )

        body = _submit(provider, service, "opus", temperature=0.3, max_output_tokens=2_000)

        assert body.get("max_completion_tokens") == 2_000 and "max_tokens" not in body and "temperature" not in body

    def test_a_curated_resolution_produces_the_body_it_always_did(self, provider, service):
        from providers.openrouter_batch_wire import build_chat_body

        request = BatchRequest(
            custom_id="a",
            prompt="hello",
            system_prompt="sys",
            temperature=0.3,
            max_output_tokens=800,
            extra={"top_p": 0.9, "seed": 1},
        )
        provider._batch_send = service

        provider.submit_batch(CURATED_OPUS, [request])

        legacy = build_chat_body(request, temperature=0.3, include_max_tokens=True)  # the old signature, no listing
        # The builder with no listing is unchanged byte for byte ...
        assert legacy == {
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
            "temperature": 0.3,
            "max_tokens": 800,
            "top_p": 0.9,
            "seed": 1,
        }
        # ... but a curated slug now has a listing (D9): opus-4.5's names temperature, stop and max_tokens, so the
        # ``top_p`` and ``seed`` the legacy body carried are no longer sent.
        assert _bodies(service)[0] == {
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}],
            "temperature": 0.3,
            "max_tokens": 800,
        }

    def test_a_curated_slug_with_no_listing_anywhere_keeps_the_body_it_always_did_and_warns_once(
        self, provider, service, caplog
    ):
        from providers.openrouter_batch_wire import build_chat_body

        # live unreachable and the snapshot cannot speak for the slug: nothing names its parameters
        bc.set_batch_catalog_source(
            BatchCatalogSource(
                fetcher=RecordingFetcher(error=TimeoutError("down")), snapshot=lambda: _snapshot_with({})
            )
        )
        request = BatchRequest(
            custom_id="a",
            prompt="hello",
            system_prompt="sys",
            temperature=0.3,
            max_output_tokens=800,
            extra={"top_p": 0.9, "seed": 1},
        )
        provider._batch_send = service
        with caplog.at_level(logging.WARNING):
            resolution = provider._resolve_batch_model(CURATED_OPUS)
            provider._resolve_batch_model(CURATED_OPUS)
            provider.submit_batch(CURATED_OPUS, [request], resolution=resolution)

        assert resolution.capabilities.supported_parameters is None
        assert parameter_source(resolution) == "curated"
        assert "parameters" not in resolution.to_record()
        assert _bodies(service)[0] == build_chat_body(request, temperature=0.3, include_max_tokens=True)
        assert len(_warnings(caplog, "lists no batch parameters")) == 1

    def test_without_a_listing_a_missing_temperature_still_drops_the_other_parameters(self):
        from providers.openrouter_batch_wire import build_chat_body

        request = BatchRequest(custom_id="a", prompt="p", max_output_tokens=10, extra={"top_p": 0.9})
        assert build_chat_body(request, temperature=None, include_max_tokens=True) == {
            "messages": [{"role": "user", "content": "p"}]
        }

    def test_the_validator_reports_unsupported_instead_of_clamping(self, provider):
        from tools.chat import ChatTool

        derived = provider._resolve_batch_model("opus").capabilities
        context = ModelContext("opus", provider, derived)

        corrected, warnings = ChatTool().validate_and_correct_temperature(0.3, context)

        assert corrected == 0.3  # not clamped to a fixed 1.0
        assert len(warnings) == 1 and "does not accept a temperature" in warnings[0]
        # a model whose listing takes temperature validates the way every ranged entry does
        listed = ModelContext("grok", provider, provider._resolve_batch_model("grok").capabilities)
        assert ChatTool().validate_and_correct_temperature(0.3, listed) == (0.3, [])
        assert ChatTool().validate_and_correct_temperature(1.5, listed) == (1.5, [])  # x-ai: 0..2 (D10)
        assert ChatTool().validate_and_correct_temperature(2.5, listed)[0] == 2.0
        capped = ModelContext(
            ANTHROPIC_LISTING_TEMPERATURE,
            provider,
            provider._resolve_batch_model(ANTHROPIC_LISTING_TEMPERATURE).capabilities,
        )
        assert ChatTool().validate_and_correct_temperature(1.5, capped)[0] == 1.0  # anthropic: 0..1

    @pytest.mark.asyncio
    async def test_the_batched_request_a_chat_sends_follows_the_resolved_listing(self, boot, service, tmp_path):
        world = boot()

        await world.chat.execute(_chat_args(tmp_path, "opus", temperature=0.9, wait_seconds=0))
        await world.chat.execute(_chat_args(tmp_path, "grok", temperature=0.9, wait_seconds=0))

        opus_body, grok_body = _bodies(service)
        assert "temperature" not in opus_body and grok_body["temperature"] == 0.9


# ----------------------------------------------------------------------
# 15.7 the resolution is persisted and a resume reads it
# ----------------------------------------------------------------------


def _forbid_resolution(monkeypatch: Any) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a resumption resolved a model again")

    for name in (
        "_resolve_batch_model",
        "resolve_batch_model",
        "_batch_slug",
        "batch_submission_model",
        "_batch_catalog",
    ):
        monkeypatch.setattr(OpenRouterProvider, name, forbidden)


class TestTheResolutionSurvivesRestartAndResume:
    def test_the_stored_note_never_exceeds_its_cap(self, provider):
        many = tuple(f"vendor/model-{n:02d}" for n in range(25))
        resolution = BatchModelResolution(
            requested="x",
            slug="vendor/pick",
            capabilities=_generic(provider),
            unclassified=many,
            unclassified_omitted=3,
        )

        record = resolution.to_record()

        assert len(record["unclassified_newer_batch_models"]) == 10
        assert record["unclassified_newer_batch_models_omitted"] == 3 + 15
        # a tampered row is re-bounded on the way out
        tampered = {"unclassified_newer_batch_models": [f"v/m{n}" for n in range(50)]}
        assert len(note_from_record(tampered)["unclassified_newer_batch_models"]) == 10
        assert note_from_record(tampered)["unclassified_newer_batch_models_omitted"] == 40
        assert note_from_record({"unclassified_newer_batch_models": "nope"}) == {}
        assert note_from_record({"unclassified_newer_batch_models": ["a:batch", 5, ""]}) == {}
        assert note_from_record(None) == {} and limits_from_record(None) is None
        assert limits_from_record({"limits": {"context_window": 0, "max_output_tokens": 5}}) is None
        assert limits_from_record({"limits": {"context_window": True, "max_output_tokens": 5}}) is None

    @pytest.mark.asyncio
    async def test_a_handle_resumed_under_a_catalog_with_other_limits_reports_the_recorded_values(
        self, boot, service, tmp_path, monkeypatch
    ):
        world = boot()
        running = _outer(await world.chat.execute(_chat_args(tmp_path, "flash", wait_seconds=0)))
        handle = json.loads(running["content"])["batch_handle"]
        world.close()

        changed = payload_with(
            {GEMINI_FLASH + ":batch": lambda entry: entry.update(context_length=2_000_000)},
            extra_batch={
                "google/gemini-5-zinc": {"context_length": 3_000_000, "supported_parameters": ["temperature"]},
                "google/gemini-5-xenon": {"context_length": 3_000_000, "supported_parameters": ["temperature"]},
            },
        )
        newer = install_batch_catalog(fetcher=RecordingFetcher(changed))
        _forbid_resolution(monkeypatch)
        service.finish("or_batch_1", {"chat": "Answered."})

        second = boot()
        done = _outer(await second.chat.execute({"batch_handle": handle, "wait_seconds": 0}))

        assert newer.calls == [] and len(service.posts) == 1  # no catalog access, nothing resubmitted
        assert done["metadata"]["model_resolved"] == GEMINI_FLASH
        assert done["metadata"]["unclassified_newer_batch_models"] == [
            "google/gemini-4-argon",
            "google/gemini-4-carbon",
            "google/gemini-4-hydrogen",
            "google/gemini-4-neon",
        ]  # the note recorded at submission, not the two newer ones the catalog lists now
        assert _limits(second, handle) == [{"context_window": 1_048_576, "max_output_tokens": 65_536}]

    @pytest.mark.asyncio
    async def test_a_running_status_names_the_recorded_limits_and_note(self, boot, service, tmp_path):
        world = boot()
        slow = _outer(await world.chat.execute(_chat_args(tmp_path, "flash", wait_seconds=0)))

        child = json.loads(slow["content"])["children"][0]

        assert child["limits"] == {"context_window": 1_048_576, "max_output_tokens": 65_536}
        assert len(child["unclassified_newer_batch_models"]) == 4 and child["model_resolved"] == GEMINI_FLASH

    @pytest.mark.asyncio
    async def test_a_null_row_resumes_with_no_note_and_no_limits(self, boot, service, tmp_path, monkeypatch):
        world = boot()
        slow = _outer(await world.chat.execute(_chat_args(tmp_path, "flash", wait_seconds=0)))
        handle = json.loads(slow["content"])["batch_handle"]
        world.close()
        raw = sqlite3.connect(str(tmp_path / "durable" / "batches.db"))
        raw.execute("UPDATE batch_children SET resolution = NULL")
        raw.commit()
        raw.close()
        service.finish("or_batch_1", {"chat": "Answered."})

        second = boot()
        done = _outer(await second.chat.execute({"batch_handle": handle, "wait_seconds": 0}))

        assert done["metadata"]["model_resolved"] == GEMINI_FLASH  # the recorded slug, resolution or not
        assert "unclassified_newer_batch_models" not in done["metadata"]
        assert _limits(second, handle) == [None]

    @pytest.mark.asyncio
    async def test_a_null_row_running_status_carries_neither(self, boot, service, tmp_path):
        world = boot()
        slow = _outer(await world.chat.execute(_chat_args(tmp_path, "flash", wait_seconds=0)))
        handle = json.loads(slow["content"])["batch_handle"]
        world.close()
        raw = sqlite3.connect(str(tmp_path / "durable" / "batches.db"))
        raw.execute("UPDATE batch_children SET resolution = NULL")
        raw.commit()
        raw.close()

        second = boot()
        child = json.loads(_outer(await second.chat.execute({"batch_handle": handle, "wait_seconds": 0}))["content"])[
            "children"
        ][0]

        assert "limits" not in child and "unclassified_newer_batch_models" not in child

    @pytest.mark.asyncio
    async def test_each_child_of_a_multi_batch_operation_keeps_its_own_resolution(self, boot, service):
        world = boot()
        install_batch_catalog(
            fetcher=RecordingFetcher(payload_with(extra_batch={f"openai/gpt-7-{n}": {} for n in "abcdefghijkl"}))
        )

        run = await world.consensus.execute(
            _consensus_args([_entry("flash", "for"), _entry("gpt5", "against"), _entry("opus")], wait_seconds=0)
        )

        handle = json.loads(json.loads(run[0].text)["content"])["batch_handle"]
        record = world.store.get(handle)
        assert record is not None and len(record.children) == 3
        by_model = {child.requested_model: child.resolution for child in record.children}
        assert by_model["flash"] is not None and by_model["gpt5"] is not None and by_model["opus"] is not None
        assert by_model["flash"]["slug"] == GEMINI_FLASH and by_model["gpt5"]["slug"] == GPT_55
        assert by_model["opus"]["slug"] == OPUS_55
        assert by_model["flash"]["limits"]["max_output_tokens"] == 65_536
        assert by_model["gpt5"]["limits"] == {"context_window": 1_050_000, "max_output_tokens": 128_000}
        assert len(by_model["flash"]["unclassified_newer_batch_models"]) == 4
        assert len(by_model["gpt5"]["unclassified_newer_batch_models"]) == 10  # capped on the record
        assert by_model["gpt5"]["unclassified_newer_batch_models_omitted"] == 12
        assert "unclassified_newer_batch_models" not in by_model["opus"]


# ----------------------------------------------------------------------
# 15.10 a codename model is chosen by typing its full slug
# ----------------------------------------------------------------------


class TestACodenameIsChosenByItsFullSlug:
    @pytest.mark.asyncio
    async def test_the_full_slug_submits_under_that_slug_with_its_own_limits_and_no_note(self, boot, service, tmp_path):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id, {"chat": "Answered."})

        done = _outer(await world.chat.execute(_chat_args(tmp_path, ARGON, wait_seconds=30)))

        assert service.submitted_models == [ARGON]
        assert "unclassified_newer_batch_models" not in done["metadata"]
        assert "model_requested" not in done["metadata"]  # typed in full: nothing was resolved
        resolution = world.provider._resolve_batch_model(ARGON)
        assert resolution.derived and not resolution.floated
        assert resolution.capabilities.context_window == 1_048_576
        assert resolution.capabilities.max_output_tokens == 943_718  # its own listing, not flash's 65,536
        assert resolution.capabilities.supports_temperature  # the codename's batch listing takes temperature

    def test_a_full_slug_the_catalog_does_not_list_is_still_refused(self, provider):
        for slug in ("google/gemini-2.5-pro", "google/gemini-4-unlisted"):
            with pytest.raises(BatchModelRefused, match="no batch listing"):
                provider._resolve_batch_model(slug)
            assert not provider.can_batch_model(slug)

    def test_the_refusal_for_a_family_emptied_by_restrictions_is_unchanged(self, provider, monkeypatch):
        import utils.model_restrictions as model_restrictions

        monkeypatch.setenv("OPENROUTER_ALLOWED_MODELS", "openai/gpt-5.5")
        model_restrictions._restriction_service = None

        with pytest.raises(BatchModelRefused) as excinfo:
            provider._resolve_batch_model("opus")

        assert str(excinfo.value) == (
            "Batch execution is unavailable for 'opus': OpenRouter lists Claude Opus batch models (per its live "
            "catalog), but the restriction policy does not allow any of them: not allowed by restriction policy, "
            "so nothing was chosen (a different family is never substituted). The same call without "
            "execution_mode runs it synchronously; or name a specific batch-capable model."
        )
        model_restrictions._restriction_service = None


def test_a_catalog_fixture_slug_set_is_what_these_tests_assume():
    catalog = parse_catalog_payload(catalog_fixture_payload(), origin="live")
    assert len(catalog.slugs) == 92 and OPUS_55 in catalog.slugs and ARGON in catalog.slugs
