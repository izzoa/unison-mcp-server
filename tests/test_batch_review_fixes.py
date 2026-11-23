"""Regressions for the review of section 15 (add-batch-model-support, D7 and D8).

Each test fails without the fix it names. No network and no API key: catalogs are the fixture under
``tests/fixtures`` (or a variant of it) served by an injected fetcher; the batch API is the in-process service.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from providers.openrouter import OpenRouterProvider
from providers.shared import batch_catalog as bc
from providers.shared.batch import BatchRequest
from providers.shared.batch_catalog import (
    MAX_SLUG_CHARS,
    BatchCatalog,
    parse_catalog_payload,
    parse_models_payload,
    reduce_listing,
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
from tests.test_batch_dynamic_resolution import GEMINI_FLASH, GPT_55, OPUS_55, _chat_args
from tests.test_batch_unclassified_newer import fixture_catalog, unclassified, with_slugs
from utils.batch_store import BatchStore

REPO = Path(__file__).resolve().parent.parent
NOTE = "unclassified_newer_batch_models"


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


@pytest.fixture
def provider() -> OpenRouterProvider:
    return OpenRouterProvider(api_key="test-key")


@pytest.fixture
def service() -> OpenRouterBatchService:
    return OpenRouterBatchService()


@pytest.fixture
def refresh():
    scripts = str(REPO / "scripts")
    sys.path.insert(0, scripts)
    try:
        import refresh_openrouter_batch_catalog as module
    finally:
        sys.path.remove(scripts)
    return module


def _auto_discovered(provider: OpenRouterProvider, slug: str, **changes: Any) -> None:
    """Make the registry hold a LiteLLM auto-discovered entry for ``slug`` (what an installed LiteLLM may add)."""

    generic = provider._lookup_capabilities("some/uncurated-model")
    assert generic is not None
    entry = dataclasses.replace(
        generic, model_name=slug, auto_discovered=True, supports_batch=False, supports_temperature=True, **changes
    )
    original = provider._registry.get_capabilities

    def get_capabilities(name: str) -> Any:
        return entry if name.strip().lower() == slug.lower() else original(name)

    provider._registry.get_capabilities = get_capabilities  # type: ignore[method-assign]


def _body(provider: OpenRouterProvider, service: Any, model: str, **request: Any) -> dict[str, Any]:
    provider._batch_send = service
    provider.submit_batch(model, [BatchRequest(custom_id="a", prompt="hello", system_prompt="sys", **request)])
    return service.posts[-1]["envelope"]["requests"][0]["body"]


# ----------------------------------------------------------------------
# An auto-discovered registry entry is not a reviewed curated entry
# ----------------------------------------------------------------------


class TestAnAutoDiscoveredEntryDoesNotShadowTheListing:
    def test_a_full_slug_with_an_auto_discovered_entry_gets_live_derived_capabilities(self, provider, service):
        _auto_discovered(provider, GEMINI_FLASH, context_window=200_000)

        resolution = provider._resolve_batch_model(GEMINI_FLASH)

        assert resolution.derived and resolution.slug == GEMINI_FLASH
        assert resolution.capabilities.context_window == 1_048_576  # the listing's, not LiteLLM's 200,000
        assert not resolution.capabilities.supports_temperature  # the listing names none
        body = _body(provider, service, GEMINI_FLASH, temperature=0.5, max_output_tokens=500, extra={"top_p": 0.9})
        assert "temperature" not in body and "top_p" not in body and body["max_tokens"] == 500

    def test_a_floating_alias_that_lands_on_an_auto_discovered_entry_is_derived_too(self, provider, service):
        _auto_discovered(provider, OPUS_55, context_window=200_000)

        resolution = provider._resolve_batch_model("opus")

        assert resolution.slug == OPUS_55 and resolution.derived and resolution.floated
        assert resolution.capabilities.context_window == 1_000_000
        assert "temperature" not in _body(provider, service, "opus", temperature=0.3)

    def test_a_reviewed_curated_entry_keeps_its_limits_but_not_its_unlisted_parameters(self, provider, service):
        # Amended by D9 (was ``..._is_still_kept_as_it_is``): the LIMITS stay the reviewed entry's; the request
        # parameters are the ones its ``:batch`` listing names (opus-4.5's names no ``top_p`` and no ``seed``).
        resolution = provider._resolve_batch_model("anthropic/claude-opus-4.5")
        assert not resolution.derived and resolution.capabilities.context_window == 200_000
        body = _body(provider, service, "anthropic/claude-opus-4.5", temperature=0.5, extra={"top_p": 0.9, "seed": 3})
        assert body["temperature"] == 0.5 and "top_p" not in body and "seed" not in body


# ----------------------------------------------------------------------
# The refresh script refuses a body that drops ONE family of fields everywhere
# ----------------------------------------------------------------------


def _committed(refresh, tmp_path, capsys) -> tuple[Path, Path, bytes]:
    target = tmp_path / "snapshot.json"
    models = REPO / "conf" / "openrouter_models.json"
    assert refresh.main([], fetch=catalog_fixture_payload, snapshot_path=target, models_path=models) == 0
    capsys.readouterr()
    return target, models, target.read_bytes()


class TestTheRefreshGuardIsPerField:
    @pytest.mark.parametrize(
        ("field", "strip"),
        [
            ("supported_parameters", lambda entry: entry.pop("supported_parameters", None)),
            ("max_completion_tokens", lambda entry: entry.pop("top_provider", None)),
            ("input_modalities", lambda entry: entry.pop("architecture", None)),
            ("context_length", lambda entry: entry.pop("context_length", None)),
        ],
    )
    def test_a_body_that_drops_one_field_for_every_slug_is_refused_and_the_file_untouched(
        self, refresh, tmp_path, capsys, field, strip
    ):
        target, models, before = _committed(refresh, tmp_path, capsys)
        body = copy.deepcopy(catalog_fixture_payload())
        for entry in body["data"]:
            strip(entry)  # the batch listings and the plain ones alike

        assert refresh.main([], fetch=lambda: body, snapshot_path=target, models_path=models) == 1

        assert target.read_bytes() == before
        err = capsys.readouterr().err
        assert "capability fields" in err and field in err
        assert refresh.main(["--allow-shrink"], fetch=lambda: body, snapshot_path=target, models_path=models) == 0

    def test_a_few_slugs_losing_one_field_is_an_ordinary_refresh(self, refresh, tmp_path, capsys):
        target, models, _ = _committed(refresh, tmp_path, capsys)
        some = [f"{slug}:batch" for slug in json.loads(target.read_text())["batch_models"][:5]]
        body = payload_with(dict.fromkeys(some, lambda entry: entry.pop("supported_parameters", None)))

        assert refresh.main(["--dry-run"], fetch=lambda: body, snapshot_path=target, models_path=models) == 0


# ----------------------------------------------------------------------
# A refusal is never made against a guessed window
# ----------------------------------------------------------------------


class TestNoRefusalAgainstAGuessedWindow:
    @pytest.mark.asyncio
    async def test_a_slug_with_no_listed_window_admits_a_prompt_the_generic_default_would_refuse(
        self, tmp_path, service, monkeypatch
    ):
        monkeypatch.setattr(OpenRouterProvider, "count_tokens", lambda self, text, model_name: len(text) // 4)
        started = []
        world = batch_world_factory(tmp_path, service)()
        started.append(world)
        try:
            install_batch_catalog(
                fetcher=RecordingFetcher(
                    payload_with(
                        extra_batch={
                            "acme/new-model": {
                                "context_length": None,
                                "supported_parameters": ["temperature", "max_tokens"],
                            }
                        }
                    )
                )
            )

            async def assembled(request: Any, arguments: Any) -> str:
                return "x" * 160_000  # about 40,000 tokens: over the generic 32,768, a guess

            world.chat._assemble_batch_prompt = assembled
            resolution = OpenRouterProvider(api_key="k")._resolve_batch_model("acme/new-model")
            assert resolution.sources["context_length"] == "generic" and not resolution.window_is_listed()

            await world.chat.execute(_chat_args(tmp_path, "acme/new-model", wait_seconds=0))

            assert service.submitted_models == ["acme/new-model"]
        finally:
            world.close()

    def test_a_listed_window_counts_as_listed(self, provider):
        assert provider._resolve_batch_model("opus").window_is_listed()
        assert provider._resolve_batch_model("anthropic/claude-opus-4.5").window_is_listed()  # curated, reviewed


# ----------------------------------------------------------------------
# The snapshot's batch-derived value outranks the plain listing of a live body
# ----------------------------------------------------------------------


class TestSnapshotOutranksThePlainListing:
    def test_a_missing_live_batch_number_comes_from_the_snapshot_not_the_plain_listing(self, provider):
        def drop(entry: dict[str, Any]) -> None:
            entry.pop("context_length", None)

        def widen(entry: dict[str, Any]) -> None:
            entry["context_length"] = 2_000_000  # the plain listing disagrees with the batch listing's real window

        install_batch_catalog(fetcher=RecordingFetcher(payload_with({OPUS_55 + ":batch": drop, OPUS_55: widen})))

        resolution = provider._resolve_batch_model("opus")

        assert resolution.capabilities.context_window == 1_000_000
        assert resolution.sources["context_length"] == "snapshot"

    def test_the_plain_listing_still_answers_when_neither_batch_source_does(self, provider):
        payload = payload_with(extra_batch={"newco/new-1": {"context_length": None}})
        payload["data"].append({"id": "newco/new-1", "context_length": 300_000})
        install_batch_catalog(fetcher=RecordingFetcher(payload))

        resolution = provider._resolve_batch_model("newco/new-1")

        assert resolution.capabilities.context_window == 300_000 and resolution.sources["context_length"] == "live"

    def test_a_catalog_without_layers_keeps_its_merged_view(self):
        catalog = BatchCatalog(frozenset({"a/b"}), views={"a/b": {"context_length": 5_000}})
        assert catalog.view_layers("a/b") == ({"context_length": 5_000}, None)


# ----------------------------------------------------------------------
# A tiny output cap is not believed
# ----------------------------------------------------------------------


class TestATinyOutputCapIsRejected:
    @pytest.mark.parametrize("tiny", [1, 5, 63, 64, 100, 512, 1023])
    def test_a_listing_with_a_tiny_cap_falls_back_instead_of_capping_requests_to_a_few_tokens(self, tiny):
        view = reduce_listing({"context_length": 100_000, "top_provider": {"max_completion_tokens": tiny}})
        assert view["max_completion_tokens"] is None
        assert (
            validate_view({"context_length": 100_000, "max_completion_tokens": tiny})["max_completion_tokens"] is None
        )

    def test_the_floor_itself_is_accepted(self):
        assert (
            reduce_listing({"context_length": 100_000, "top_provider": {"max_completion_tokens": 1024}})[
                "max_completion_tokens"
            ]
            == 1024
        )

    def test_the_committed_snapshot_has_no_cap_below_the_floor(self):
        document = json.loads((REPO / "conf" / "openrouter_batch_catalog.json").read_text())
        caps = [v["max_completion_tokens"] for v in document["model_data"].values() if v["max_completion_tokens"]]
        assert caps and min(caps) >= bc.MIN_MAX_COMPLETION


# ----------------------------------------------------------------------
# Two processes upgrading one old store file
# ----------------------------------------------------------------------


class _RacingConnection:
    """Wraps a connection so that, just before our ALTER runs, 'another process' has already added the column."""

    def __init__(self, real: sqlite3.Connection, path: str, connect: Any) -> None:
        self._real = real
        self._path = path
        self._connect = connect

    def execute(self, sql: str, *args: Any) -> Any:
        if sql.startswith("ALTER TABLE batch_children ADD COLUMN resolution"):
            other = self._connect(self._path)
            other.execute(sql)
            other.commit()
            other.close()
        return self._real.execute(sql, *args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class TestTheStoreUpgradeIsRaceTolerant:
    def test_a_concurrent_alter_by_another_process_is_not_an_error(self, tmp_path, monkeypatch):
        path = tmp_path / "old.db"
        connection = sqlite3.connect(path)
        connection.executescript("""
            CREATE TABLE schema_version (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (1, 1);
            CREATE TABLE batches (handle TEXT PRIMARY KEY, tool_name TEXT, continuation_id TEXT, payload TEXT,
                state TEXT NOT NULL, state_detail TEXT, unwound INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, expired_at REAL);
            CREATE TABLE batch_children (handle TEXT NOT NULL, idx INTEGER NOT NULL, provider TEXT NOT NULL,
                provider_batch_id TEXT, requested_model TEXT NOT NULL, submitted_model TEXT NOT NULL,
                endpoint TEXT NOT NULL, request_total INTEGER NOT NULL, status TEXT NOT NULL, custom_ids TEXT NOT NULL,
                intent_window_start REAL NOT NULL, intent_window_end REAL, claimed INTEGER NOT NULL DEFAULT 0,
                consumed TEXT, detail TEXT, PRIMARY KEY (handle, idx));
            """)
        connection.commit()
        connection.close()

        real_connect = sqlite3.connect

        def racing_connect(database: Any, *args: Any, **kwargs: Any) -> Any:
            connection = real_connect(database, *args, **kwargs)
            return (
                _RacingConnection(connection, str(database), real_connect) if str(database) == str(path) else connection
            )

        monkeypatch.setattr("utils.batch_store.sqlite3.connect", racing_connect)

        store = BatchStore(str(path))  # must not raise 'duplicate column name'
        store.close()

        columns = {row[1] for row in real_connect(path).execute("PRAGMA table_info(batch_children)").fetchall()}
        assert "resolution" in columns


# ----------------------------------------------------------------------
# The note advertises only what the caller could type
# ----------------------------------------------------------------------


class TestTheNoteHonoursTheRestrictionPolicyByFullSlug:
    def _restrict(self, monkeypatch, value: str) -> None:
        import utils.model_restrictions as model_restrictions

        monkeypatch.setenv("OPENROUTER_ALLOWED_MODELS", value)
        model_restrictions._restriction_service = None

    def test_an_allow_list_naming_only_the_alias_advertises_no_codename(self, provider, monkeypatch):
        self._restrict(monkeypatch, "gpt5")
        catalog = fixture_catalog()

        resolution = provider._resolve_batch_model("gpt5", catalog)

        assert resolution.slug == GPT_55 and resolution.unclassified == () and resolution.note() == {}
        with pytest.raises(ValueError, match="not allowed"):
            provider._resolve_batch_model("openai/gpt-6.2-sol", catalog)  # what the note would have advised

    def test_a_slug_the_allow_list_names_in_full_is_the_only_one_advertised(self, provider, monkeypatch):
        self._restrict(monkeypatch, "gpt5,openai/gpt-6-sol")

        resolution = provider._resolve_batch_model("gpt5", fixture_catalog())

        assert resolution.unclassified == ("openai/gpt-6-sol",)
        provider._resolve_batch_model("openai/gpt-6-sol", fixture_catalog())  # and it really can be typed


# ----------------------------------------------------------------------
# Classification details
# ----------------------------------------------------------------------


class TestClassificationDetails:
    def test_a_plain_newer_slug_with_no_tier_word_is_reported(self):
        catalog = with_slugs(fixture_catalog(), {"openai/gpt-6.5": None})

        shown, _ = unclassified("gpt5", catalog)

        assert "openai/gpt-6.5" in shown

    def test_a_pro_variant_is_folded_into_its_plain_sibling(self):
        shown, omitted = unclassified("gpt5")
        assert not [slug for slug in shown if slug.endswith("-pro")] and omitted == 0

    def test_a_colon_spelling_is_never_reported_as_a_base_slug(self):
        catalog = with_slugs(
            fixture_catalog(), {"openai/gpt-6-sol:free": None, "openai/gpt-6-sol:batch": None, "openai/gpt-8:x": None}
        )

        shown, _ = unclassified("gpt5", catalog)

        assert not [slug for slug in shown if ":" in slug]

    def test_an_overlong_slug_is_never_reported(self):
        long_slug = "openai/gpt-6-" + "a" * 200_000
        catalog = with_slugs(fixture_catalog(), {long_slug: None})

        shown, omitted = unclassified("gpt5", catalog)

        assert long_slug not in shown and all(len(slug) <= MAX_SLUG_CHARS for slug in shown)

    def test_the_catalog_parser_drops_an_overlong_slug(self):
        long_slug = "openai/gpt-6-" + "a" * 200_000
        payload = payload_with(extra_batch={long_slug: {}, "openai/gpt-6-fine": {}})

        slugs = parse_models_payload(payload)

        assert "openai/gpt-6-fine" in slugs and long_slug not in slugs
        assert long_slug not in parse_catalog_payload(payload).slugs

    def test_the_note_of_a_hostile_catalog_stays_small(self, provider):
        payload = payload_with(extra_batch={"openai/gpt-6-" + "a" * 100_000: {}, "openai/gpt-7-x:free": {}})
        catalog = parse_catalog_payload(payload)

        resolution = provider._resolve_batch_model("gpt5", catalog)

        assert all(len(slug) <= MAX_SLUG_CHARS and ":" not in slug for slug in resolution.unclassified)
        assert len(json.dumps(resolution.note())) < 2_000


# ----------------------------------------------------------------------
# A model that takes no temperature is not a warning on every call
# ----------------------------------------------------------------------


class TestNoTemperatureIsNotAWarningPerCall:
    @pytest.mark.asyncio
    async def test_a_default_temperature_chat_batch_to_opus_logs_no_warning_about_it(self, tmp_path, service, caplog):
        import logging

        world = batch_world_factory(tmp_path, service)()
        try:
            with caplog.at_level(logging.INFO):
                await world.chat.execute(_chat_args(tmp_path, "opus", wait_seconds=0))
        finally:
            world.close()

        records = [r for r in caplog.records if "does not accept a temperature" in r.getMessage()]
        assert records and all(r.levelno == logging.INFO for r in records)

    def test_a_real_correction_stays_a_warning(self, caplog):
        import logging

        from tools.shared.batch_support import BatchResumableToolMixin

        with caplog.at_level(logging.INFO):
            BatchResumableToolMixin.log_batch_temperature_warnings(["Temperature 3 clamped to 2"])

        assert [r.levelno for r in caplog.records] == [logging.WARNING]
