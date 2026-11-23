"""Newer batch listings no family rule classifies are reported, never chosen (add-batch-model-support, D7).

The family rules are version-number rules, so OpenRouter's newer codename listings (``google/gemini-4-argon``,
``openai/gpt-5.6-sol``) match none and an alias ignores them. What is added is visibility: a bounded note in the
resolution metadata and one warning per distinct situation. The note never changes the pick.

No network and no API key (catalogs are the fixture under ``tests/fixtures`` or a variant of it).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from providers.openrouter import OpenRouterProvider
from providers.registries.openrouter import OpenRouterModelRegistry
from providers.shared import ModelResponse, ProviderType
from providers.shared import batch_catalog as bc
from providers.shared.batch_catalog import (
    BatchCatalog,
    BatchFamilyRule,
    parse_catalog_payload,
    unclassified_newer_listings,
    validate_view,
)
from tests.batch_test_helpers import (
    OpenRouterBatchService,
    RecordingFetcher,
    batch_world_factory,
    catalog_fixture_payload,
    install_batch_catalog,
    payload_with,
)
from tests.test_batch_dynamic_resolution import (
    GEMINI_FLASH,
    GEMINI_PRO,
    GPT_55,
    OPUS_55,
    _chat_args,
    _consensus_args,
    _entry,
    _outer,
)

ARGONS = [f"google/gemini-4-{name}" for name in ("argon", "carbon", "hydrogen", "neon")]
#: Twelve more distinct codename GPT listings: with the ten the fixture has, gpt5 has twenty-two candidates, which is
#: more than the note's cap of ten.
MORE_GPT = {f"openai/gpt-7-{name}": {} for name in "abcdefghijkl"}
NOTE = "unclassified_newer_batch_models"
OMITTED = "unclassified_newer_batch_models_omitted"


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


def all_rules() -> list[BatchFamilyRule]:
    registry = OpenRouterModelRegistry()
    rules = []
    for name in registry.list_models():
        config = registry.resolve(name)
        if config is not None and config.batch_family is not None:
            rules.append(config.batch_family)
    return rules


def rule_for(alias: str) -> BatchFamilyRule:
    config = OpenRouterModelRegistry().resolve(alias)
    assert config is not None and config.batch_family is not None, alias
    return config.batch_family


def fixture_catalog() -> BatchCatalog:
    return parse_catalog_payload(catalog_fixture_payload(), origin="live")


def with_slugs(base: BatchCatalog, views: dict[str, dict[str, Any] | None]) -> BatchCatalog:
    """``base`` plus extra listed slugs; a ``None`` view means the catalog carries no data for that slug."""

    merged = dict(base.views)
    for slug, view in views.items():
        if view is not None:
            merged[slug] = validate_view(view)
    return BatchCatalog(base.slugs | frozenset(views), origin=base.origin, views=merged)


def unclassified(
    alias: str, catalog: BatchCatalog | None = None, *, allowed: Any = None, limit: int = 10
) -> tuple[tuple[str, ...], int]:
    catalog = catalog or fixture_catalog()
    rule = rule_for(alias)
    pick = rule.select([slug for slug in catalog.slugs if rule.is_member(slug)])
    assert pick is not None
    return unclassified_newer_listings(
        pick=pick, rule=rule, catalog=catalog, rules=all_rules(), allowed=allowed or (lambda slug: True), limit=limit
    )


# ----------------------------------------------------------------------
# 15.8 the classification (pure)
# ----------------------------------------------------------------------


class TestTheClassification:
    @pytest.mark.parametrize("alias", ["flash", "gemini"])
    def test_flash_and_gemini_give_the_four_gemini_4_codenames(self, alias):
        assert unclassified(alias) == (tuple(ARGONS), 0)

    def test_gpt5_gives_ten_distinct_models_with_the_5_6_codenames_named_and_no_pro_duplicates(self):
        shown, omitted = unclassified("gpt5")

        assert omitted == 0 and len(shown) == 10
        assert not any(slug.endswith("-pro") for slug in shown)  # openai/gpt-6.2-sol-pro folds into gpt-6.2-sol
        assert {"openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "openai/gpt-5.6-luna"} <= set(shown)  # the named ones
        assert shown[0] == "openai/gpt-6.2-astra" and shown[-3:] == tuple(
            sorted(("openai/gpt-5.6-luna", "openai/gpt-5.6-sol", "openai/gpt-5.6-terra"))
        )  # newest version first, then slug

    def test_a_long_candidate_list_is_cut_to_the_cap_newest_first_and_the_rest_counted(self):
        catalog = with_slugs(fixture_catalog(), dict(MORE_GPT))

        shown, omitted = unclassified("gpt5", catalog)

        assert len(shown) == 10 and omitted == 12
        assert all(slug.startswith("openai/gpt-7-") for slug in shown)  # version 7 outranks every 6.x
        top, cut = unclassified("gpt5", catalog, limit=4)
        assert top == shown[:4] and cut == 18  # 22 candidates, four shown

    def test_a_pro_listing_is_folded_into_its_plain_sibling_only_when_the_sibling_is_reported(self):
        catalog = with_slugs(fixture_catalog(), {"openai/gpt-9-nova-pro": None})  # no plain gpt-9-nova

        shown, _ = unclassified("gpt5", catalog)

        assert "openai/gpt-9-nova-pro" in shown  # its only mention
        assert "openai/gpt-6.2-sol-pro" not in shown and "openai/gpt-6.2-sol" in shown

    def test_the_note_is_scoped_to_the_product_line_a_newer_fable_is_not_reported_for_opus_nor_a_codename_with_no_lead(
        self,
    ):
        # D11 (decided 2026-11-22, "product line only"): the documented limits of the note, pinned.
        catalog = with_slugs(
            fixture_catalog(),
            {
                "anthropic/claude-fable-9.9": None,  # a NEWER model, but another product than ``claude-opus-``
                "google/gemini-nova": None,  # a codename with no numeric lead: cannot be compared with the pick
                "openai/gpt-astra": None,
            },
        )

        assert unclassified("opus", catalog) == ((), 0)
        assert "google/gemini-nova" not in unclassified("flash", catalog)[0]
        assert "openai/gpt-astra" not in unclassified("gpt5", catalog, limit=1_000)[0]
        assert (
            "google/gemini-4-argon" in unclassified("flash", catalog)[0]
        )  # a codename WITH a numeric lead is reported

    @pytest.mark.parametrize("alias", ["opus", "sonnet", "haiku", "grok"])
    def test_families_with_nothing_newer_give_nothing(self, alias):
        assert unclassified(alias) == ((), 0)

    def test_same_release_tier_variants_older_models_and_other_products_are_never_reported(self):
        everything, _ = unclassified("gpt5", limit=1_000)
        for never in (
            "openai/gpt-5.5-pro",  # the same release as the pick
            "openai/gpt-5.4-mini",  # older
            "openai/gpt-5.4-nano",
            "openai/gpt-oss-120b",  # no lead version: a different product
            "openai/gpt-4o",  # lead '4o' runs into letters, and is older anyway
            "openai/o3",  # outside the product line
        ):
            assert never not in everything, never
        gemini, _ = unclassified("gemini", limit=1_000)
        assert "google/gemini-3.5-flash-lite" not in gemini  # newer than the pick, but 'flash' is a classified tier
        assert "google/gemini-3.6-flash" not in gemini and "google/gemini-3.8-flash" not in gemini  # classified

    def test_a_newer_listing_a_rule_matches_is_picked_not_reported(self):
        catalog = with_slugs(fixture_catalog(), {"google/gemini-3.9-pro": None})

        names, _ = unclassified("gemini", catalog)
        rule = rule_for("gemini")
        pick = rule.select([slug for slug in catalog.slugs if rule.is_member(slug)])

        assert pick == "google/gemini-3.9-pro" and "google/gemini-3.9-pro" not in names
        assert names == tuple(ARGONS)  # the codenames are still newer than 3.9

    def test_modalities_exclude_embeddings_images_and_audio_but_not_a_listing_with_no_data(self):
        text = {"output_modalities": ["text"]}
        catalog = with_slugs(
            fixture_catalog(),
            {
                "google/gemini-4-pixel": {"output_modalities": ["image", "text"]},
                "google/gemini-4-embed": {"output_modalities": ["embeddings"]},
                "google/gemini-4-speech": {"output_modalities": ["audio"]},
                "google/gemini-4-plain": text,
                "google/gemini-4-bare": None,  # no modality data at all: there is no signal to exclude it by
            },
        )

        names, _ = unclassified("flash", catalog)

        assert "google/gemini-4-pixel" not in names and "google/gemini-4-embed" not in names
        assert "google/gemini-4-speech" not in names
        assert "google/gemini-4-plain" in names and "google/gemini-4-bare" in names

    def test_a_slug_the_restriction_policy_forbids_is_excluded(self):
        names, _ = unclassified("flash", allowed=lambda slug: slug != "google/gemini-4-neon")
        assert names == tuple(ARGONS[:3])

    def test_a_newer_version_orders_before_a_codename_of_an_older_one(self):
        catalog = with_slugs(fixture_catalog(), {"google/gemini-4.1-aardvark": None, "google/gemini-10-zeta": None})

        names, _ = unclassified("flash", catalog)

        assert names[:2] == ("google/gemini-10-zeta", "google/gemini-4.1-aardvark")
        assert names[2:] == tuple(ARGONS)

    def test_a_tier_variant_of_a_classified_tier_is_not_a_new_model(self):
        catalog = with_slugs(fixture_catalog(), {"google/gemini-4-flash-lite": None, "google/gemini-4-pro-max": None})

        names, _ = unclassified("flash", catalog)

        assert "google/gemini-4-flash-lite" not in names  # the tier word 'flash' is one a rule already covers
        assert "google/gemini-4-pro-max" not in names  # and so is 'pro'
        assert names == tuple(ARGONS)  # while a word no rule covers (argon, neon ...) is a new model

    def test_a_newer_listing_a_rule_matches_displaces_the_codenames_it_outdates(self):
        catalog = with_slugs(fixture_catalog(), {"google/gemini-4-flash": None})  # the flash rule matches this one

        names, _ = unclassified("flash", catalog)
        rule = rule_for("flash")

        assert rule.select([slug for slug in catalog.slugs if rule.is_member(slug)]) == "google/gemini-4-flash"
        assert names == ()  # the codenames are the same release, not newer than the pick

    def test_the_stem_is_the_product_line_not_the_vendor(self):
        assert rule_for("gpt5").line_stem == "openai/gpt-"
        assert rule_for("opus").line_stem == "anthropic/claude-opus-"
        assert rule_for("gemini").line_stem == rule_for("flash").line_stem == "google/gemini-"
        assert rule_for("grok").line_stem == "x-ai/grok-"
        catalog = with_slugs(fixture_catalog(), {"openai/o9-codename": None, "openai/zzz-9": None})
        names, _ = unclassified("gpt5", catalog, limit=1_000)
        assert "openai/o9-codename" not in names and "openai/zzz-9" not in names

    def test_a_rule_whose_pattern_has_no_literal_stem_reports_nothing(self):
        odd = BatchFamilyRule(aliases=("odd",), pattern=r"^(?:a|b)/m-(?P<version>\d+)$")
        assert odd.line_stem is None
        names, omitted = unclassified_newer_listings(
            pick="a/m-1", rule=odd, catalog=fixture_catalog(), rules=[odd], allowed=lambda slug: True
        )
        assert (names, omitted) == ((), 0)

    def test_an_empty_catalog_reports_nothing(self):
        rule = rule_for("opus")
        assert unclassified_newer_listings(
            pick=OPUS_55, rule=rule, catalog=BatchCatalog(frozenset()), rules=all_rules(), allowed=lambda slug: True
        ) == ((), 0)

    def test_the_result_is_independent_of_set_iteration_order(self):
        base = fixture_catalog()
        shuffled = BatchCatalog(frozenset(sorted(base.slugs, reverse=True)), origin="live", views=base.views)
        assert unclassified("gpt5", shuffled) == unclassified("gpt5", base)


# ----------------------------------------------------------------------
# 15.9 the wiring
# ----------------------------------------------------------------------


def _finish(service: OpenRouterBatchService) -> None:
    service.on_get = lambda batch_id: service.finish(batch_id)


class TestTheNoteReachesTheCaller:
    @pytest.mark.asyncio
    async def test_a_chat_batch_for_flash_carries_the_note_in_its_resolution_metadata(self, boot, service, tmp_path):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id, {"chat": "Answered."})

        done = _outer(await world.chat.execute(_chat_args(tmp_path, "flash", wait_seconds=30)))

        metadata = done["metadata"]
        assert metadata["model_requested"] == "flash" and metadata["model_resolved"] == GEMINI_FLASH
        assert metadata[NOTE] == ARGONS and OMITTED not in metadata  # nothing was cut

    @pytest.mark.asyncio
    async def test_a_consensus_run_carries_a_separate_note_per_entry(self, boot, service):
        world = boot()
        _finish(service)
        install_batch_catalog(fetcher=RecordingFetcher(payload_with(extra_batch=MORE_GPT)))

        body = json.loads(
            (await world.consensus.execute(_consensus_args([_entry("flash", "for"), _entry("gpt5", "against")])))[
                0
            ].text
        )

        by_model = {r["model"]: r["metadata"] for r in body["accumulated_responses"]}
        assert by_model["flash"][NOTE] == ARGONS and OMITTED not in by_model["flash"]
        assert len(by_model["gpt5"][NOTE]) == 10 and by_model["gpt5"][OMITTED] == 12
        assert by_model["gpt5"]["model_resolved"] == GPT_55
        assert not set(by_model["flash"][NOTE]) & set(by_model["gpt5"][NOTE])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["opus", "opus4.5", "google/gemini-3.8-flash", "anthropic/claude-opus-4.5"])
    async def test_the_key_is_absent_for_an_alias_with_nothing_newer_a_pinned_alias_and_a_full_slug(
        self, boot, service, tmp_path, model
    ):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id, {"chat": "Answered."})

        done = _outer(await world.chat.execute(_chat_args(tmp_path, model, wait_seconds=30)))

        assert NOTE not in done["metadata"] and OMITTED not in done["metadata"]

    def test_the_note_is_absent_not_empty_when_the_catalog_has_no_entries(self, provider):
        bc.set_batch_catalog_source(
            bc.BatchCatalogSource(fetcher=RecordingFetcher(error=OSError("down")), snapshot=lambda: None)
        )

        resolution = provider._resolve_batch_model("opus")  # the flags-and-pins fallback: nothing floats

        assert not resolution.floated and resolution.note() == {} and resolution.unclassified == ()

    @pytest.mark.asyncio
    async def test_a_synchronous_flash_carries_no_note_and_never_touches_the_catalog(self, boot, tmp_path, monkeypatch):
        world = boot()
        fetcher = install_batch_catalog()

        def reply(self, prompt, model_name, system_prompt=None, temperature=0.3, **kwargs):
            return ModelResponse(
                content="Sync answer.",
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                model_name="google/gemini-2.5-flash",
                friendly_name="OpenRouter",
                provider=ProviderType.OPENROUTER,
                metadata={},
            )

        monkeypatch.setattr(OpenRouterProvider, "generate_content", reply)
        arguments = {k: v for k, v in _chat_args(tmp_path, "flash").items() if k != "execution_mode"}

        done = _outer(await world.chat.execute(arguments))

        assert done["metadata"]["model_requested"] == "flash"  # the synchronous resolution report is unchanged
        assert NOTE not in done["metadata"] and OMITTED not in done["metadata"]
        assert fetcher.calls == []


class TestTheWarningIsOncePerSituation:
    def test_one_warning_across_two_identical_calls_naming_alias_slug_and_candidates(self, provider, caplog):
        with caplog.at_level(logging.WARNING, logger="providers.shared.batch_catalog"):
            first = provider._resolve_batch_model("flash")
            second = provider._resolve_batch_model("flash")
            provider.can_batch_model("flash")

        warned = [r.getMessage() for r in caplog.records if "no family rule classifies" in r.getMessage()]
        assert len(warned) == 1 and first.unclassified == second.unclassified == tuple(ARGONS)
        for needle in ("'flash'", GEMINI_FLASH, *ARGONS, "full slug"):
            assert needle in warned[0], needle

    def test_a_different_candidate_set_or_alias_is_a_new_situation(self, provider, caplog):
        with caplog.at_level(logging.WARNING, logger="providers.shared.batch_catalog"):
            provider._resolve_batch_model("flash")
            install_batch_catalog(fetcher=RecordingFetcher(payload_with(extra_batch={"google/gemini-4-xenon": {}})))
            provider._resolve_batch_model("flash")  # one more codename is now listed
            provider._resolve_batch_model("gemini")  # another alias, same candidates as before plus xenon

        warned = [r.getMessage() for r in caplog.records if "no family rule classifies" in r.getMessage()]
        assert len(warned) == 3
        assert "google/gemini-4-xenon" in warned[1] and "'gemini'" in warned[2]

    def test_the_warning_cuts_a_long_list_and_counts_the_rest(self, provider, caplog):
        install_batch_catalog(fetcher=RecordingFetcher(payload_with(extra_batch=MORE_GPT)))
        with caplog.at_level(logging.WARNING, logger="providers.shared.batch_catalog"):
            provider._resolve_batch_model("gpt5")

        warned = [r.getMessage() for r in caplog.records if "no family rule classifies" in r.getMessage()]
        assert len(warned) == 1 and "(and 12 more)" in warned[0]

    def test_the_seen_set_is_bounded(self):
        bc.reset_warn_once()
        for number in range(400):
            bc.warn_once(("k", number), "message %s", number)
        assert len(bc._warn_once_seen) == 256
        assert bc.warn_once(("k", 399), "again") is False  # a recent key is still remembered
        assert bc.warn_once(("k", 0), "forgotten") is True  # the oldest was forgotten
        bc.reset_warn_once()
        assert len(bc._warn_once_seen) == 0

    def test_resolving_for_listmodels_or_a_preview_warns_nothing(self, provider, caplog):
        with caplog.at_level(logging.WARNING, logger="providers.shared.batch_catalog"):
            assert provider.batch_family_preview("flash") == GEMINI_FLASH
        assert not [r for r in caplog.records if "no family rule classifies" in r.getMessage()]


class TestTheNoteNeverChangesTheChoice:
    def test_the_pick_is_identical_when_the_note_computation_fails(self, provider, monkeypatch):
        expected = {alias: provider._resolve_batch_model(alias).slug for alias in ("opus", "flash", "gemini", "gpt5")}

        def boom(**kwargs: Any) -> Any:
            raise RuntimeError("the note exploded")

        monkeypatch.setattr("providers.openrouter.unclassified_newer_listings", boom)

        for alias, slug in expected.items():
            resolution = provider._resolve_batch_model(alias)
            assert resolution.slug == slug and resolution.note() == {}

    def test_the_pick_is_the_same_with_and_without_codenames_in_the_catalog(self, provider):
        with_codenames = {alias: provider._resolve_batch_model(alias).slug for alias in ("flash", "gemini", "gpt5")}
        stripped = payload_with(drop=[f"{slug}:batch" for slug in (*ARGONS, "openai/gpt-6-sol", "openai/gpt-5.6-sol")])
        install_batch_catalog(fetcher=RecordingFetcher(stripped))

        assert {alias: provider._resolve_batch_model(alias).slug for alias in ("flash", "gemini", "gpt5")} == (
            with_codenames
        )

    def test_the_default_pick_for_gemini_is_unchanged(self, provider):
        assert provider._resolve_batch_model("gemini").slug == GEMINI_PRO
