"""Batch resolution and submission honour what is published (D12, D13, D16) for curated AND derived entries.

The catalog under test is the COMMITTED snapshot ``conf/openrouter_batch_catalog.json`` (the 2026-11-22 copy of
OpenRouter's listings, 92 batch slugs), served with the live fetch failing so nothing touches the network; LiteLLM's
figures come from the pinned fixture ``tests/fixtures/litellm_published_limits.json``. The oracle below reads both
files directly and applies the documented rule with its own, simpler code, so a bug in the implementation cannot also
hide in the expectation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from providers.registries.openrouter import OpenRouterModelRegistry
from providers.shared import published_limits as pl
from providers.shared.batch import BatchRequest
from providers.shared.batch_resolution import PUBLISHED_CAP_FIELD, limits_from_record
from tests.batch_test_helpers import (
    OpenRouterBatchService,
    RecordingFetcher,
    batch_world_factory,
    install_batch_catalog,
)
from tests.test_batch_dynamic_resolution import _chat_args, _outer

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = json.loads((ROOT / "conf" / "openrouter_batch_catalog.json").read_text())
LITELLM = json.loads((ROOT / "tests" / "fixtures" / "litellm_published_limits.json").read_text())["entries"]
HUGE = 50_000_000


# ----------------------------------------------------------------------
# The oracle: the documented rule, written independently of the implementation
# ----------------------------------------------------------------------

_NATIVE = {
    "openai": lambda n: [n, f"openai/{n}"],
    "google": lambda n: [f"gemini/{n}", f"vertex_ai/{n}"],
    "x-ai": lambda n: [f"xai/{n}"],
    "mistralai": lambda n: [f"mistral/{n}"],
    "moonshotai": lambda n: [f"moonshot/{n}"],
    "z-ai": lambda n: [f"zai/{n}"],
    "anthropic": lambda n: [n, re.sub(r"(?<=\d)\.(?=\d)", "-", n)],
}


def litellm_for(slug: str) -> tuple[int | None, bool]:
    """``(output cap, sampling flag)`` LiteLLM publishes for an OpenRouter slug: the ``openrouter/`` key if present,
    else the vendor's native keys, hits combined by ``min``."""

    vendor, _, name = slug.partition("/")
    for tier in ([f"openrouter/{slug}"], _NATIVE.get(vendor, lambda n: [])(name)):
        hits = [LITELLM[key] for key in tier if key in LITELLM]
        if hits:
            caps = [h["max_output_tokens"] for h in hits if "max_output_tokens" in h]
            return (min(caps) if caps else None), any(h.get("supports_sampling_params") is False for h in hits)
    return None, False


def expected_cap(slug: str, curated_cap: int | None) -> int | None:
    """The bound a request's ``max_output_tokens`` meets, or ``None`` when nobody stated one."""

    view = SNAPSHOT["model_data"].get(slug, {})
    published = [v for v in (view.get("max_completion_tokens"), litellm_for(slug)[0]) if v]
    if curated_cap is not None:
        return min([curated_cap, *published])
    if not published:
        return None
    window = view.get("context_length")
    return min([*published, window] if window else published)


def sent_cap(body: dict[str, Any]) -> int | None:
    return body.get("max_tokens", body.get("max_completion_tokens"))


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import utils.model_restrictions as model_restrictions
    from utils import batch_store as bs

    for name in (
        "BATCH_STORE_PATH",
        "BATCH_RETENTION_DAYS",
        "OPENROUTER_ALLOWED_MODELS",
        "OPENROUTER_BATCH_LIVE_CATALOG",
    ):
        monkeypatch.delenv(name, raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


@pytest.fixture(autouse=True)
def _the_committed_snapshot_is_the_catalog():
    """Live unreachable: every call resolves against the committed snapshot, exactly the offline path."""

    install_batch_catalog(fetcher=RecordingFetcher(error=httpx.ConnectError("down")))


@pytest.fixture
def service() -> OpenRouterBatchService:
    return OpenRouterBatchService()


def _fresh_provider_class() -> Any:
    """The provider class as the module defines it NOW. Another test file reloads ``providers.openrouter``; a class
    imported at collection time would then carry the old module's sentinel default and refuse to resolve."""

    import importlib

    return importlib.import_module("providers.openrouter").OpenRouterProvider


@pytest.fixture
def provider() -> Any:
    return _fresh_provider_class()(api_key="test-key")


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


def submit(provider: Any, service: Any, model: str, **request: Any) -> dict[str, Any]:
    provider._batch_send = service
    provider.submit_batch(
        model,
        [BatchRequest(custom_id="a", prompt="hello", system_prompt="sys", **{"max_output_tokens": HUGE, **request})],
    )
    return service.posts[-1]["envelope"]["requests"][0]["body"]


def curated_batchable() -> list[str]:
    registry = OpenRouterModelRegistry()
    out = []
    for name in registry.list_models():
        config = registry.resolve(name)
        if (
            config is not None
            and not config.auto_discovered
            and config.supports_batch
            and config.model_name in SNAPSHOT["batch_models"]
        ):
            out.append(config.model_name)
    return sorted(out)


def alias_targets() -> dict[str, str]:
    """Every alias that floats under batch, and the slug the committed snapshot resolves it to."""

    install_batch_catalog(fetcher=RecordingFetcher(error=httpx.ConnectError("down")))
    provider = _fresh_provider_class()(api_key="test-key")
    registry = OpenRouterModelRegistry()
    return {
        alias: provider._resolve_batch_model(alias).slug
        for alias in sorted(registry.list_aliases())
        if provider._floating_family(alias) is not None
    }


CURATED = curated_batchable()
ALIASES = alias_targets()


def test_the_sets_under_test_are_not_empty():
    assert len(CURATED) == 11, CURATED
    assert {"opus", "sonnet", "haiku", "gemini", "flash", "gpt5", "grok"} <= set(ALIASES), sorted(ALIASES)


# ----------------------------------------------------------------------
# D12: the output cap of every curated batchable slug and every alias target
# ----------------------------------------------------------------------


class TestEveryCuratedSlugAndAliasTargetIsBounded:
    @pytest.mark.parametrize("slug", CURATED)
    def test_a_curated_slug_is_bounded_by_the_smallest_published_cap(self, provider, service, slug):
        curated = OpenRouterModelRegistry().get_capabilities(slug)
        assert curated is not None
        bound = expected_cap(slug, curated.max_output_tokens)
        assert bound is not None and bound <= curated.max_output_tokens

        resolution = provider._resolve_batch_model(slug)
        assert resolution.capabilities.max_output_tokens == bound
        assert resolution.output_cap_bound() is not None and resolution.output_cap_bound()[0] == bound

        body = submit(provider, service, slug)
        if "max_tokens" in body or "max_completion_tokens" in body:
            assert sent_cap(body) == bound  # never above any published limit
            assert sent_cap(submit(provider, service, slug, max_output_tokens=bound)) == bound  # at the cap: as given
            assert sent_cap(submit(provider, service, slug, max_output_tokens=100)) == 100  # below: untouched

    @pytest.mark.parametrize("alias", sorted(ALIASES))
    def test_an_alias_target_is_bounded_by_the_smallest_published_cap(self, provider, service, alias):
        slug = ALIASES[alias]
        registry = OpenRouterModelRegistry().get_capabilities(slug)
        reviewed = registry is not None and not registry.auto_discovered and registry.model_name.lower() == slug.lower()
        bound = expected_cap(slug, registry.max_output_tokens if reviewed and registry is not None else None)

        resolution = provider._resolve_batch_model(alias)
        assert resolution.slug == slug
        if bound is None:
            assert resolution.output_cap_bound() is None  # nothing stated: nothing invented
        else:
            assert resolution.output_cap_bound() is not None
            assert resolution.capabilities.max_output_tokens == bound

        body = submit(provider, service, alias)
        if sent_cap(body) is not None and bound is not None:
            assert sent_cap(body) == bound

    def test_the_published_source_that_bounds_the_cap_is_recorded(self, provider):
        record = provider._resolve_batch_model("openai/gpt-5.2-pro").to_record()
        assert record["sources"][PUBLISHED_CAP_FIELD] in ("snapshot", "litellm")
        assert record["limits"]["max_output_tokens"] == 128_000
        assert json.loads(json.dumps(record)) == record

    def test_opus_4_1_is_caught_by_openrouters_listing_alone(self, provider, service):
        """LiteLLM has no bare key for it; the committed listing says 32,000 against the reviewed 64,000."""

        assert litellm_for("anthropic/claude-opus-4.1") == (None, False)
        resolution = provider._resolve_batch_model("anthropic/claude-opus-4.1")
        assert resolution.sources[PUBLISHED_CAP_FIELD] == "snapshot"
        assert sent_cap(submit(provider, service, "anthropic/claude-opus-4.1")) == 32_000

    def test_a_stand_in_cap_is_replaced_by_a_published_figure_never_combined_with_it(self, provider):
        """``anthropic/claude-opus-5.5`` has no reviewed entry: the curated family's 64,000 is a guess about another model."""

        resolution = provider._resolve_batch_model("opus")
        published, _ = litellm_for(resolution.slug)
        view = SNAPSHOT["model_data"][resolution.slug]
        assert resolution.capabilities.max_output_tokens == min(
            v for v in (published, view["max_completion_tokens"]) if v
        )
        assert resolution.capabilities.max_output_tokens > 64_000  # not lowered to the family entry's figure

    def test_a_model_nobody_published_a_cap_for_sends_the_callers_figure_and_logs_once(self, provider, service, caplog):
        install_batch_catalog(
            fetcher=RecordingFetcher(
                {
                    "data": [
                        {
                            "id": "acme/robot-9:batch",
                            "supported_parameters": ["temperature", "max_tokens"],
                        }
                    ]
                }
            )
        )
        with caplog.at_level(logging.WARNING):
            assert sent_cap(submit(provider, service, "acme/robot-9")) == HUGE
            assert sent_cap(submit(provider, service, "acme/robot-9")) == HUGE
        assert len([r for r in caplog.records if "No published output limit" in r.getMessage()]) == 1

    def test_a_published_figure_above_a_stand_in_cap_is_used_not_the_stand_in(self, provider, service):
        """The listing gives no cap, so the generic 32,768 is a guess: LiteLLM's real 200,000 replaces it."""

        listing = {"id": "acme/robot-9:batch", "context_length": 1_000_000, "supported_parameters": ["max_tokens"]}
        install_batch_catalog(fetcher=RecordingFetcher({"data": [listing]}))
        pl.set_published_index(
            pl.PublishedIndex.from_document(
                {"entries": {"openrouter/acme/robot-9": {"provider": "openrouter", "max_output_tokens": 200_000}}}
            )
        )
        resolution = provider._resolve_batch_model("acme/robot-9")
        assert resolution.sources["max_completion_tokens"] == "generic"
        assert resolution.sources[PUBLISHED_CAP_FIELD] == "litellm"
        assert sent_cap(submit(provider, service, "acme/robot-9")) == 200_000

    def test_a_reviewed_cap_bounds_a_curated_slug_when_nothing_at_all_is_published(self, provider, service):
        pl.set_published_index(pl.PublishedIndex.from_document({"entries": {}}))
        install_batch_catalog(fetcher=RecordingFetcher(error=httpx.ConnectError("down")), snapshot=lambda: None)

        resolution = provider._resolve_batch_model("anthropic/claude-opus-4.5")
        assert PUBLISHED_CAP_FIELD not in resolution.sources
        assert resolution.output_cap_bound() == (64_000, "curated")
        assert sent_cap(submit(provider, service, "anthropic/claude-opus-4.5")) == 64_000

    def test_one_warning_per_model_and_cap_for_a_reduction(self, provider, service, caplog):
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                submit(provider, service, "anthropic/claude-opus-4.5")
        reductions = [r for r in caplog.records if "is reduced to it" in r.getMessage()]
        assert len(reductions) == 1 and "claude-opus-4.5" in reductions[0].getMessage()


# ----------------------------------------------------------------------
# D17: the snapshot bounds the batch path, and only the batch path
# ----------------------------------------------------------------------


class TestTheSnapshotStillBoundsABatchRequest:
    """D17 takes the committed snapshot off the synchronous path (``tests/test_published_limits_sync.py::
    TestTheSnapshotDoesNotBoundASynchronousRequest``); the batch path keeps it: live ``:batch``, snapshot, LiteLLM,
    curated, smallest valid figure."""

    def test_a_snapshot_figure_below_litellms_bounds_the_batch_request(self, provider, service):
        slug = "moonshotai/kimi-k3"
        snapshot_cap = SNAPSHOT["model_data"][slug]["max_completion_tokens"]
        litellm_cap, _ = litellm_for(slug)
        assert snapshot_cap == 16_384 and litellm_cap == 1_048_576  # the premise: the snapshot is the low figure

        resolution = provider._resolve_batch_model(slug)
        assert resolution.sources[PUBLISHED_CAP_FIELD] == "snapshot"
        assert resolution.capabilities.max_output_tokens == 16_384
        assert sent_cap(submit(provider, service, slug)) == 16_384

    def test_the_same_slug_on_the_synchronous_path_is_not_bounded_by_it(self):
        from providers.openrouter import OpenRouterProvider
        from providers.shared import ProviderType

        assert pl.published_output_caps(ProviderType.OPENROUTER, "moonshotai/kimi-k3") == [("litellm", 1_048_576)]
        capabilities = OpenRouterProvider(api_key="test-key").get_capabilities("moonshotai/kimi-k3")
        assert getattr(capabilities, "_published_output_cap", None) == 1_048_576

    def test_a_curated_slug_on_the_batch_path_is_still_caught_by_the_snapshot(self, provider, service):
        assert sent_cap(submit(provider, service, "anthropic/claude-opus-4.1")) == 32_000


# ----------------------------------------------------------------------
# Hostile published values fall back safely
# ----------------------------------------------------------------------


class TestHostilePublishedValues:
    @pytest.mark.parametrize(
        "bad", [0, -1, -128_000, float("nan"), float("inf"), "128000", "big", True, 10**15, 3, None]
    )
    def test_a_garbled_litellm_figure_is_ignored_for_a_curated_slug(self, provider, service, bad):
        pl.set_published_index(
            pl.PublishedIndex.from_document({"entries": {"gpt-5": {"provider": "openai", "max_output_tokens": bad}}})
        )
        # the listing and the reviewed cap still bound it: 128,000 for openai/gpt-5
        assert sent_cap(submit(provider, service, "openai/gpt-5")) == 128_000

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), "big", True, 10**15, 3, None])
    def test_a_garbled_litellm_figure_is_ignored_for_a_derived_slug(self, provider, service, bad):
        slug = "openai/gpt-5.5"
        reference = sent_cap(submit(provider, service, slug))
        pl.set_published_index(
            pl.PublishedIndex.from_document({"entries": {"gpt-5.5": {"provider": "openai", "max_output_tokens": bad}}})
        )
        assert sent_cap(submit(provider, service, slug)) == reference

    def test_a_garbled_listing_figure_falls_back_to_litellm_then_to_the_stand_in(self, provider, service):
        listing = {
            "id": "acme/robot-9:batch",
            "context_length": 100_000,
            "top_provider": {"max_completion_tokens": -5},
            "supported_parameters": ["max_tokens"],
        }
        install_batch_catalog(fetcher=RecordingFetcher({"data": [listing]}))
        pl.set_published_index(
            pl.PublishedIndex.from_document(
                {"entries": {"openrouter/acme/robot-9": {"provider": "openrouter", "max_output_tokens": 7_000}}}
            )
        )
        assert sent_cap(submit(provider, service, "acme/robot-9")) == 7_000
        pl.set_published_index(pl.PublishedIndex.from_document({"entries": {}}))
        assert sent_cap(submit(provider, service, "acme/robot-9")) == HUGE  # stand-in only: nothing invented

    def test_disagreeing_sources_take_the_smallest(self, provider, service):
        listing = {
            "id": "acme/robot-9:batch",
            "context_length": 2_000_000,
            "top_provider": {"max_completion_tokens": 900_000},
            "supported_parameters": ["max_tokens"],
        }
        install_batch_catalog(fetcher=RecordingFetcher({"data": [listing]}))
        pl.set_published_index(
            pl.PublishedIndex.from_document(
                {"entries": {"openrouter/acme/robot-9": {"provider": "openrouter", "max_output_tokens": 128_000}}}
            )
        )
        assert sent_cap(submit(provider, service, "acme/robot-9")) == 128_000  # LiteLLM smaller than OpenRouter
        pl.set_published_index(
            pl.PublishedIndex.from_document(
                {"entries": {"openrouter/acme/robot-9": {"provider": "openrouter", "max_output_tokens": 1_500_000}}}
            )
        )
        assert sent_cap(submit(provider, service, "acme/robot-9")) == 900_000  # OpenRouter smaller than LiteLLM

    def test_a_missing_extract_leaves_batch_as_it_was(self, provider, service, tmp_path):
        pl.set_published_index(tmp_path / "missing.json")
        assert (
            sent_cap(submit(provider, service, "anthropic/claude-opus-4.5")) == 64_000
        )  # the reviewed cap still bounds
        assert (
            sent_cap(submit(provider, service, "openai/gpt-5.2-pro")) == 128_000
        )  # the committed listing still bounds


# ----------------------------------------------------------------------
# D13: temperature per model, on the batch path
# ----------------------------------------------------------------------


def listed_temperature(slug: str) -> bool:
    return "temperature" in SNAPSHOT["model_data"][slug]["supported_parameters"]


class TestTemperatureOfEverySlugAndAliasTarget:
    @pytest.mark.parametrize("slug", sorted(set(CURATED) | set(ALIASES.values())))
    def test_a_temperature_is_corrected_to_the_models_published_rule(self, provider, service, slug):
        vendor = slug.partition("/")[0]
        _, flagged = litellm_for(slug)
        registry = OpenRouterModelRegistry().get_capabilities(slug)
        reviewed = registry is not None and not registry.auto_discovered and registry.model_name.lower() == slug.lower()
        takes = listed_temperature(slug) and not (vendor == "anthropic" and flagged)
        if reviewed and registry is not None:
            takes = takes and registry.supports_temperature

        high = submit(provider, service, slug, temperature=1.5).get("temperature")
        mid = submit(provider, service, slug, temperature=0.7).get("temperature")

        if not takes:
            assert high is None and mid is None  # nothing is sent for a model that takes none
            return
        assert mid == 0.7  # within every range: unchanged
        assert high == (1.5 if vendor in ("openai", "google", "x-ai") else 1.0)

    def test_the_vendor_table_is_what_decides_where_nothing_is_published(self):
        from providers.shared.batch_catalog import VENDOR_TEMPERATURE_RANGES

        assert dict(VENDOR_TEMPERATURE_RANGES) == {
            "anthropic": (0.0, 1.0),
            "openai": (0.0, 2.0),
            "google": (0.0, 2.0),
            "x-ai": (0.0, 2.0),
        }


# ----------------------------------------------------------------------
# Resume and reconcile read the record, never the data
# ----------------------------------------------------------------------


class TestAHandleReportsWhatWasTrueAtSubmission:
    @pytest.mark.asyncio
    async def test_the_bound_is_recorded_and_a_later_change_of_the_data_cannot_move_it(self, boot, service, tmp_path):
        world = boot()
        running = _outer(await world.chat.execute(_chat_args(tmp_path, "anthropic/claude-opus-4.1", wait_seconds=0)))
        handle = json.loads(running["content"])["batch_handle"]

        record = world.store.get(handle)
        assert record is not None
        stored = record.children[0].resolution
        assert stored["limits"]["max_output_tokens"] == 32_000
        assert stored["sources"][PUBLISHED_CAP_FIELD] == "snapshot"
        assert limits_from_record(stored) == {"context_window": 200_000, "max_output_tokens": 32_000}

        world.close()

        # the world moves on: the extract now says something else, and the catalog is unreachable
        pl.set_published_index(
            pl.PublishedIndex.from_document(
                {"entries": {"claude-opus-4-1": {"provider": "anthropic", "max_output_tokens": 999}}}
            )
        )
        newer = install_batch_catalog(fetcher=RecordingFetcher(error=httpx.ConnectError("down")), snapshot=lambda: None)
        service.finish("or_batch_1", {"chat": "Answered."})

        second = boot()
        done = _outer(await second.chat.execute({"batch_handle": handle, "wait_seconds": 0}))

        assert newer.calls == [] and len(service.posts) == 1  # no catalog access, nothing resubmitted
        assert "Answered." in json.dumps(done)
        again = second.store.get(handle)
        assert again is not None and again.children[0].resolution == stored  # a resume derives nothing

    def test_an_older_row_without_the_new_key_is_read_as_before(self):
        assert limits_from_record(
            {
                "limits": {"context_window": 200_000, "max_output_tokens": 64_000},
                "sources": {"supported_parameters": "live"},
            }
        ) == {
            "context_window": 200_000,
            "max_output_tokens": 64_000,
        }
