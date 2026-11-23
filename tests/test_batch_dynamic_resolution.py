"""Dynamic batch model resolution (add-batch-model-support, section 14).

Under ``execution_mode="batch"`` a family alias (``opus``, ``gemini``, ``flash`` ...) resolves to the latest
member of its family that OpenRouter lists as ``:batch``, and a full slug the catalog lists is batch-eligible
even when ``conf/openrouter_models.json`` does not carry it.

No network and no API key: the catalog is the trimmed fixture under ``tests/fixtures`` served by an injected
fetcher (the harness guard in ``tests/conftest.py`` fails any test that reaches the real one), and the batch
API is the in-process ``OpenRouterBatchService``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

import utils.model_restrictions as model_restrictions
from providers.openrouter import BatchModelRefused, OpenRouterProvider
from providers.registries.openrouter import OpenRouterModelRegistry
from providers.registry import set_default_registry
from providers.shared import ModelCapabilities, ProviderType
from providers.shared import batch_catalog as bc
from providers.shared.batch_catalog import (
    BatchCatalog,
    BatchCatalogFormatError,
    BatchCatalogSource,
    BatchFamilyRule,
    compare_key,
    load_snapshot,
    merge_views,
    parse_catalog_payload,
    parse_models_payload,
    parse_version,
    reduce_listing,
    validate_view,
)
from tests.batch_test_helpers import (
    CATALOG_FIXTURE_PATH,
    FakeClock,
    FakeProvider,
    FakeRemote,
    OpenRouterBatchService,
    RecordingFetcher,
    batch_world_factory,
    catalog_fixture_payload,
    fixture_listings,
    install_batch_catalog,
    make_executor,
    make_plan,
    payload_for,
    payload_with,
)
from tests.test_batch_execution_selection import _registry
from tools.chat import ChatTool
from tools.shared.batch_support import select_batch_provider
from tools.shared.exceptions import ToolExecutionError
from utils import batch_store as bs
from utils.batch_execution import build_status_payload, result_message

REPO = Path(__file__).resolve().parent.parent

#: The real HTTP fetch, bound at import, before the harness guard (tests/conftest.py) replaces the module attribute.
real_http_fetch = bc._http_fetch
SNAPSHOT_PATH = REPO / "conf" / "openrouter_batch_catalog.json"

OPUS_55 = "anthropic/claude-opus-5.5"
SONNET_55 = "anthropic/claude-sonnet-5.5"
HAIKU_55 = "anthropic/claude-haiku-5.5"
GEMINI_PRO = "google/gemini-3.1-pro-preview"
GEMINI_FLASH = "google/gemini-3.8-flash"
GPT_55 = "openai/gpt-5.5"
GROK_47 = "x-ai/grok-4.7"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
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


def _restrict(monkeypatch, allowed: str) -> None:
    monkeypatch.setenv("OPENROUTER_ALLOWED_MODELS", allowed)
    model_restrictions._restriction_service = None


def fixture_slugs() -> frozenset[str]:
    return parse_models_payload(catalog_fixture_payload())


def rule_for(alias: str) -> BatchFamilyRule:
    config = OpenRouterModelRegistry().resolve(alias)
    assert config is not None and config.batch_family is not None, alias
    return config.batch_family


def tick_clock(start: float = 1000.0) -> Any:
    class Clock:
        now = start

        def __call__(self) -> float:
            return self.now

    return Clock()


# ----------------------------------------------------------------------
# 14.1 the committed offline data
# ----------------------------------------------------------------------


class TestCommittedCatalogData:
    def test_snapshot_loads_offline_with_92_sorted_base_slugs(self):
        document = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        slugs = document["batch_models"]

        assert len(slugs) == 92
        assert slugs == sorted(slugs)
        assert not [slug for slug in slugs if slug.endswith(":batch")]
        assert "anthropic/claude-opus-5.5" in slugs and "google/gemini-3.8-flash" in slugs
        assert document["fetched_at"] == "2026-11-22" and document["source"]

        loaded = load_snapshot()
        assert loaded is not None and loaded.origin == "snapshot" and loaded.slugs == frozenset(slugs)

    def test_trimmed_fixture_carries_every_batch_listing_and_the_named_plain_entries(self):
        payload = catalog_fixture_payload()
        ids = [entry["id"] for entry in payload["data"]]
        batch_ids = [i for i in ids if i.endswith(":batch")]

        assert len(batch_ids) == 92
        assert {i[:-6] for i in batch_ids} == set(json.loads(SNAPSHOT_PATH.read_text())["batch_models"])
        for plain in ("anthropic/claude-opus-5.5", "google/gemini-3.8-flash", "google/gemini-3.1-flash-lite"):
            assert plain in ids

    def test_an_unreadable_or_missing_snapshot_is_none_not_an_error(self, tmp_path):
        assert load_snapshot(tmp_path / "absent.json") is None
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        assert load_snapshot(broken) is None
        wrong = tmp_path / "wrong.json"
        wrong.write_text(json.dumps({"batch_models": [1, 2]}))
        assert load_snapshot(wrong) is None

    def test_snapshot_ships_in_the_package_data(self):
        text = (REPO / "pyproject.toml").read_text()
        assert "conf/*.json" in text and "conf/openrouter_batch_catalog.json" in text


# ----------------------------------------------------------------------
# 14.2 version ordering and family selection (pure)
# ----------------------------------------------------------------------


def _rule(pattern: str, **extra: Any) -> BatchFamilyRule:
    return BatchFamilyRule.from_config(
        {"aliases": ["x"], "pattern": pattern, **extra}, model_name="v/x", entry_aliases=["x"]
    )


PRO = r"^v/x-(?P<version>\d+(?:\.\d+)*)-pro(?P<preview>-preview)?$"
PLAIN = r"^v/x-(?P<version>\d+(?:\.\d+)*)$"


class TestVersionOrdering:
    def test_versions_compare_numerically_component_wise_with_missing_components_zero(self):
        assert (
            compare_key((5, 5), preview=False) > compare_key((5,), preview=False) > compare_key((4, 8), preview=False)
        )
        assert compare_key((4, 10), preview=False) > compare_key((4, 9), preview=False)
        assert compare_key((5, 0), preview=False) == compare_key((5,), preview=False)
        assert parse_version("5.5") == (5, 5) and parse_version("gpt") is None and parse_version("5.") is None

    def test_the_latest_member_wins_and_numeric_beats_lexical(self):
        rule = _rule(PLAIN)
        assert rule.select(["v/x-4.8", "v/x-5", "v/x-5.5"]) == "v/x-5.5"
        assert rule.select(["v/x-4.9", "v/x-4.10"]) == "v/x-4.10"
        assert rule.select(["v/x-5.5", "v/x-5", "v/x-4.8"][::-1]) == "v/x-5.5"  # input order never matters

    def test_ga_beats_preview_at_an_equal_version_but_a_newer_preview_beats_an_older_ga(self):
        rule = _rule(PRO)
        assert rule.select(["v/x-3.1-pro-preview", "v/x-3.1-pro"]) == "v/x-3.1-pro"
        assert rule.select(["v/x-3.1-pro", "v/x-3.1-pro-preview"]) == "v/x-3.1-pro"
        assert rule.select(["v/x-3.1-pro-preview", "v/x-3-pro"]) == "v/x-3.1-pro-preview"

    def test_a_slug_with_no_numeric_version_is_never_chosen(self):
        rule = _rule(PRO)
        assert rule.select(["google/gemini-4-argon", "v/x-argon-pro"]) is None
        assert rule.select(["google/gemini-4-argon", "v/x-3-pro"]) == "v/x-3-pro"

    def test_an_explicit_order_list_wins_over_numeric_order_and_is_the_only_way_in_for_a_codename(self):
        rule = _rule(PLAIN, order=["v/x-argon", "v/x-4"])
        assert rule.select(["v/x-5", "v/x-4", "v/x-argon"]) == "v/x-argon"
        assert rule.select(["v/x-5", "v/x-4"]) == "v/x-4"  # the first listed one that is available
        assert rule.select(["v/x-5", "v/x-6"]) == "v/x-6"  # none listed is available: numeric again

    def test_an_empty_family_selects_nothing(self):
        assert _rule(PLAIN).select([]) is None
        assert _rule(PLAIN).select(["other/model-1"]) is None


# ----------------------------------------------------------------------
# 14.3 the curated family rules
# ----------------------------------------------------------------------


class TestFamilyRules:
    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("opus", OPUS_55),
            ("claude-opus", OPUS_55),
            ("sonnet", SONNET_55),
            ("haiku", HAIKU_55),
            ("gemini", GEMINI_PRO),
            ("gemini-pro", GEMINI_PRO),
            ("pro", GEMINI_PRO),
            ("flash", GEMINI_FLASH),
            ("gemini-flash", GEMINI_FLASH),
            ("gpt5", GPT_55),
            ("grok", GROK_47),
        ],
    )
    def test_each_family_alias_picks_the_latest_batch_listed_member_of_its_tier(self, alias, expected):
        assert rule_for(alias).floats(alias)
        assert rule_for(alias).select(fixture_slugs()) == expected

    def test_a_tier_is_part_of_the_pattern(self):
        slugs = fixture_slugs()
        chosen_flash = rule_for("flash").select(slugs)
        assert chosen_flash == GEMINI_FLASH and "lite" not in chosen_flash and "pro" not in chosen_flash
        assert "google/gemini-3.1-flash-lite" in slugs  # present, and still not chosen
        assert rule_for("gemini").select(slugs) == GEMINI_PRO and "flash" not in rule_for("gemini").select(slugs)
        gpt = rule_for("gpt5").select(slugs)
        assert gpt == GPT_55 and not gpt.endswith(("-pro", "-mini", "-nano")) and "luna" not in gpt
        assert "openai/gpt-6.2-sol" in slugs  # a newer generation exists; gpt5 stays in generation 5

    def test_a_newer_release_of_another_tier_never_wins(self):
        """The fixture's lite and pro models are older than the Flash pick, so prove it with newer ones."""

        slugs = fixture_slugs() | {
            "google/gemini-9.9-flash-lite",
            "google/gemini-9.9-pro",
            "google/gemini-9.9-pro-preview",
            "openai/gpt-5.9-pro",
            "openai/gpt-5.9-mini",
            "anthropic/claude-opus-9",
            "anthropic/claude-haiku-9",
        }
        assert rule_for("flash").select(slugs) == GEMINI_FLASH
        assert rule_for("gemini").select(slugs) == "google/gemini-9.9-pro"  # GA beats its own preview
        assert rule_for("gpt5").select(slugs) == GPT_55
        assert rule_for("opus").select(slugs) == "anthropic/claude-opus-9"
        assert rule_for("haiku").select(slugs) == "anthropic/claude-haiku-9"

    def test_version_pinned_aliases_and_full_slugs_do_not_float(self):
        registry = OpenRouterModelRegistry()
        for pinned in (
            "opus4.5",
            "opus4.1",
            "sonnet4.5",
            "o3",
            "gpt5.2",
            "gpt5mini",
            "gemini3",
            "gemini-2.5",
            "grok-4",
        ):
            config = registry.resolve(pinned)
            assert config is not None, pinned
            rule = config.batch_family
            assert rule is None or not rule.floats(pinned), pinned
        for config in registry.model_map.values():
            if config.batch_family is not None:
                assert not config.batch_family.floats(config.model_name)  # a full slug never floats

    def test_every_rule_alias_belongs_to_its_own_entry(self):
        registry = OpenRouterModelRegistry()
        ruled = [c for c in registry.model_map.values() if c.batch_family is not None]
        assert {c.model_name for c in ruled} >= {
            "anthropic/claude-opus-4.5",
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-3.5-haiku",
            "google/gemini-3-pro-preview",
            "google/gemini-2.5-flash",
            "openai/gpt-5",
            "x-ai/grok-4",
        }
        for config in ruled:
            owned = {a.lower() for a in config.aliases}
            assert {a.lower() for a in config.batch_family.aliases} <= owned

    @pytest.mark.parametrize(
        ("rule", "problem"),
        [
            ({"aliases": ["x"], "pattern": "(unclosed"}, "does not compile"),
            ({"aliases": ["x"], "pattern": r"^v/x-(\d+)$"}, "'version'"),
            ({"aliases": ["stranger"], "pattern": PLAIN}, "not an alias of this entry"),
            ({"aliases": [], "pattern": PLAIN}, "non-empty list"),
            ({"aliases": ["x"], "pattern": PLAIN, "bogus": 1}, "unsupported keys"),
        ],
    )
    def test_an_invalid_rule_fails_naming_the_entry(self, rule, problem):
        with pytest.raises(BatchCatalogFormatError) as excinfo:
            ModelCapabilities(
                provider=ProviderType.OPENROUTER,
                model_name="vendor/some-model",
                friendly_name="x",
                aliases=["x"],
                batch_family=rule,  # type: ignore[arg-type]
            )
        assert "vendor/some-model" in str(excinfo.value) and problem in str(excinfo.value)

    def test_an_invalid_rule_fails_the_registry_load_naming_the_entry(self, tmp_path):
        config = tmp_path / "models.json"
        config.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "model_name": "vendor/broken-entry",
                            "aliases": ["brk"],
                            "context_window": 1000,
                            "max_output_tokens": 100,
                            "batch_family": {"aliases": ["brk"], "pattern": "no version group"},
                        }
                    ]
                }
            )
        )
        with pytest.raises(ValueError, match="vendor/broken-entry"):
            OpenRouterModelRegistry(config_path=str(config))

    def test_the_field_is_documented_in_the_catalog_readme(self):
        readme = json.loads((REPO / "conf" / "openrouter_models.json").read_text())["_README"]["field_descriptions"]
        assert "batch_family" in readme and "version" in readme["batch_family"]


# ----------------------------------------------------------------------
# 14.4 the live source
# ----------------------------------------------------------------------


def _snap(*slugs: str) -> Any:
    return lambda: BatchCatalog(frozenset(slugs), origin="snapshot")


def _source(fetcher: Any, clock: Any = None, **kwargs: Any) -> BatchCatalogSource:
    kwargs.setdefault("snapshot", _snap("snap/model"))
    kwargs.setdefault("ttl_seconds", 3600.0)
    if clock is not None:
        kwargs["clock"] = clock
    return BatchCatalogSource(fetcher=fetcher, **kwargs)


class TestLiveSource:
    def test_one_fetch_serves_two_calls_inside_the_ttl_and_a_later_call_refetches(self):
        fetcher, clock = RecordingFetcher(), tick_clock()
        source = _source(fetcher, clock)

        first, second = source.current(), source.current()
        assert first is second and first.origin == "live" and len(fetcher.calls) == 1
        assert first.lists("anthropic/claude-opus-5.5")

        clock.now += 3599
        assert source.current() is first and len(fetcher.calls) == 1
        clock.now += 2
        third = source.current()
        assert len(fetcher.calls) == 2 and third is not first

    def test_the_ttl_comes_from_the_environment_when_not_given(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BATCH_CATALOG_TTL_SECONDS", "10")
        fetcher, clock = RecordingFetcher(), tick_clock()
        source = BatchCatalogSource(fetcher=fetcher, clock=clock, snapshot=_snap("s/m"))

        source.current()
        clock.now += 11
        source.current()
        assert len(fetcher.calls) == 2

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(httpx.ReadTimeout("slow"), id="timeout"),
            pytest.param(
                httpx.HTTPStatusError("502", request=httpx.Request("GET", "http://x"), response=httpx.Response(502)),
                id="error-status",
            ),
            pytest.param(ConnectionError("down"), id="connection"),
        ],
    )
    def test_a_transport_failure_falls_back_to_the_snapshot_without_raising(self, failure):
        source = _source(RecordingFetcher(error=failure))

        catalog = source.current()

        assert catalog is not None and catalog.origin == "snapshot" and catalog.lists("snap/model")

    @pytest.mark.parametrize("body", [{"error": "nope"}, [], "html", {"data": []}, {"data": [1, 2]}, 7])
    def test_a_malformed_body_falls_back_to_the_snapshot_without_raising(self, body):
        source = _source(RecordingFetcher(body))

        assert source.current().origin == "snapshot"

    def test_a_stalled_fetch_is_abandoned_at_the_timeout(self):
        gate = threading.Event()

        def stalled(timeout: float) -> Any:
            gate.wait(5)
            return catalog_fixture_payload()

        source = _source(stalled, timeout_seconds=0.05)
        started = time.monotonic()
        try:
            catalog = source.current()
        finally:
            gate.set()
        assert time.monotonic() - started < 2 and catalog.origin == "snapshot"

    def test_a_fetcher_that_raises_a_baseexception_never_escapes(self):
        def dying(timeout: float) -> Any:
            raise SystemExit(3)

        assert _source(dying).current().origin == "snapshot"

    def test_a_failed_refresh_keeps_the_earlier_catalog(self):
        fetcher, clock = RecordingFetcher(), tick_clock()
        source = _source(fetcher, clock)
        good = source.current()

        fetcher.error = httpx.ConnectError("down")
        clock.now += 4000
        assert source.current() is good  # stale, but a good catalog beats the snapshot

    def test_a_failure_is_remembered_for_the_negative_ttl_then_retried(self):
        fetcher, clock = RecordingFetcher(error=httpx.ConnectError("down")), tick_clock()
        source = _source(fetcher, clock)

        for _ in range(5):
            assert source.current().origin == "snapshot"
        assert len(fetcher.calls) == 1  # an outage does not add a fetch to every call

        clock.now += 301
        source.current()
        assert len(fetcher.calls) == 2

        fetcher.error = None
        clock.now += 301
        assert source.current().origin == "live"

    def test_the_opt_out_makes_no_fetch(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BATCH_LIVE_CATALOG", "false")
        fetcher = RecordingFetcher()
        source = BatchCatalogSource(fetcher=fetcher, snapshot=_snap("snap/model"))

        assert source.current().origin == "snapshot" and fetcher.calls == []

        monkeypatch.setenv("OPENROUTER_BATCH_LIVE_CATALOG", "true")
        assert source.current().origin == "live" and len(fetcher.calls) == 1

    def test_a_missing_snapshot_after_a_failed_fetch_is_none_and_the_provider_uses_flags_and_pins(self):
        install = RecordingFetcher(error=httpx.ConnectError("down"))
        bc.set_batch_catalog_source(BatchCatalogSource(fetcher=install, snapshot=lambda: None))
        provider = OpenRouterProvider(api_key="test-key")

        assert bc.get_batch_catalog_source().current() is None
        # the pre-decision behaviour: curated pins and committed flags, no dynamic resolution
        assert provider._resolve_batch_model("opus").slug == "anthropic/claude-opus-4.5"
        assert provider._resolve_batch_model("gpt5").slug == "openai/gpt-5"
        with pytest.raises(BatchModelRefused):
            provider._resolve_batch_model("flash")  # the pin has no batch flag, and nothing floats it
        with pytest.raises(BatchModelRefused):
            provider._resolve_batch_model(GEMINI_FLASH)  # not curated, so nothing admits it either

    def test_importing_the_module_and_creating_the_provider_make_no_fetch(self):
        fetcher = install_batch_catalog()
        provider = OpenRouterProvider(api_key="test-key")
        bc.get_batch_catalog_source()
        assert provider is not None and fetcher.calls == []

        script = (
            "import socket\n"
            "def refuse(*a, **k):\n"
            "    raise AssertionError('network access at import or construction')\n"
            "socket.getaddrinfo = refuse\n"
            "socket.socket.connect = refuse\n"
            "import providers.shared.batch_catalog as bc\n"
            "from providers.openrouter import OpenRouterProvider\n"
            "bc.get_batch_catalog_source()\n"
            "OpenRouterProvider(api_key='k')\n"
            "print('quiet')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=180, check=False
        )
        assert proc.returncode == 0 and "quiet" in proc.stdout, proc.stderr

    def test_concurrent_callers_share_one_fetch(self):
        calls: list[float] = []

        def slow(timeout: float) -> Any:
            calls.append(timeout)
            time.sleep(0.15)
            return catalog_fixture_payload()

        source = _source(slow)
        results: list[Any] = []
        threads = [threading.Thread(target=lambda: results.append(source.current())) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert len(calls) == 1 and len(results) == 8 and all(r is results[0] for r in results)

    @pytest.mark.asyncio
    async def test_calling_through_to_thread_does_not_block_the_event_loop(self):
        def slow(timeout: float) -> Any:
            time.sleep(0.4)
            return catalog_fixture_payload()

        source = _source(slow)
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            first, second = await asyncio.gather(asyncio.to_thread(source.current), asyncio.to_thread(source.current))
        finally:
            beat.cancel()
        assert first is second and ticks >= 5

    def test_a_pin_holds_one_catalog_for_the_whole_operation_across_threads(self):
        counter = {"n": 0}

        def changing(timeout: float) -> Any:
            counter["n"] += 1
            return payload_for([f"anthropic/claude-opus-5.{counter['n']}"])

        clock = tick_clock()
        source = _source(changing, clock, ttl_seconds=1.0)
        seen: list[BatchCatalog] = []
        with source.pin():
            seen.append(source.current())
            clock.now += 100  # the TTL lapses mid-operation
            worker = threading.Thread(target=lambda: seen.append(source.current()))
            worker.start()
            worker.join(5)
            seen.append(source.current())
        assert seen[0] is seen[1] is seen[2] and counter["n"] == 1

        after = source.current()  # released: the next call may refresh
        assert after is not seen[0] and counter["n"] == 2

    def test_loaded_never_fetches(self):
        fetcher = RecordingFetcher()
        source = _source(fetcher)
        assert source.loaded().origin == "snapshot" and fetcher.calls == []
        source.current()
        assert source.loaded().origin == "live" and len(fetcher.calls) == 1


class TestAdmissionStaysOffTheEventLoop:
    @pytest.mark.asyncio
    async def test_a_slow_catalog_fetch_during_admission_does_not_block_the_loop(self, boot, service, tmp_path):
        def slow(timeout: float) -> Any:
            time.sleep(0.5)
            return catalog_fixture_payload()

        bc.set_batch_catalog_source(BatchCatalogSource(fetcher=slow))
        world = boot()
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        try:
            await world.chat.execute(_chat_args(tmp_path, "opus", wait_seconds=0))
        finally:
            beat.cancel()

        assert service.submitted_models == [OPUS_55] and ticks >= 5


# ----------------------------------------------------------------------
# 14.5 unit tests never fetch
# ----------------------------------------------------------------------


class TestHarnessGuard:
    def test_an_unstubbed_fetch_raises_inside_the_guard_and_the_source_degrades(self, batch_catalog_guard):
        source = BatchCatalogSource(snapshot=_snap("snap/model"))  # the real fetcher, unstubbed

        catalog = source.current()

        assert batch_catalog_guard, "the guard did not see the attempt"
        assert catalog.origin == "snapshot"  # the guard's error was absorbed like any failure
        with pytest.raises(AssertionError, match="live OpenRouter model list"):
            bc._http_fetch(1.0)
        batch_catalog_guard.clear()  # this test provoked the guard on purpose

    def test_every_test_starts_with_a_fixture_backed_source(self):
        catalog = bc.get_batch_catalog_source().current()
        assert catalog is not None and catalog.origin == "live" and catalog.lists(GEMINI_FLASH)

    def test_no_python_test_names_the_live_endpoint(self):
        needle = "openrouter.ai/api/v1/" + "models"
        offenders = [
            str(path.relative_to(REPO))
            for path in (REPO / "tests").rglob("*.py")
            if needle in path.read_text(encoding="utf-8")
        ]
        assert offenders == []
        assert needle in CATALOG_FIXTURE_PATH.read_text(encoding="utf-8")  # only the fixture documents it


# ----------------------------------------------------------------------
# 14.6 full-slug eligibility
# ----------------------------------------------------------------------


@pytest.fixture
def provider() -> OpenRouterProvider:
    return OpenRouterProvider(api_key="test-key")


class TestFullSlugEligibility:
    def test_a_slug_the_catalog_lists_is_eligible_though_the_curated_file_lacks_it(self, provider):
        assert OpenRouterModelRegistry().resolve(GEMINI_FLASH) is None  # genuinely uncurated

        resolution = provider._resolve_batch_model(GEMINI_FLASH)

        assert resolution.slug == GEMINI_FLASH and resolution.capabilities is not None
        assert provider.can_batch_model(GEMINI_FLASH) and provider.can_batch_model(GEMINI_FLASH + ":batch")

    def test_a_slug_in_neither_source_is_refused(self, provider):
        for slug in ("google/gemini-2.5-pro", "someone/brand-new-model", "mistralai/mistral-large-2411"):
            with pytest.raises(BatchModelRefused, match="no batch listing"):
                provider._resolve_batch_model(slug)
            assert not provider.can_batch_model(slug)

    def test_a_committed_true_flag_still_admits_a_model_the_catalog_does_not_list(self, provider):
        install_batch_catalog([GEMINI_FLASH])  # a catalog without opus 4.1, whose committed flag is true
        assert provider._resolve_batch_model("anthropic/claude-opus-4.1").slug == "anthropic/claude-opus-4.1"

    def test_a_committed_true_flag_admits_when_the_catalog_cannot_be_read_at_all(self, provider):
        bc.set_batch_catalog_source(
            BatchCatalogSource(fetcher=RecordingFetcher(error=httpx.ConnectError("x")), snapshot=lambda: None)
        )
        assert provider._resolve_batch_model("anthropic/claude-opus-4.1").slug == "anthropic/claude-opus-4.1"

    def test_a_capability_query_never_fetches(self, provider):
        fetcher = install_batch_catalog()

        provider.get_capabilities(GEMINI_FLASH)
        provider.get_capabilities("opus")
        provider.list_models()
        provider.get_all_model_capabilities()
        provider.validate_model_name("flash")
        provider.batch_family_preview("opus")

        assert fetcher.calls == []

    @pytest.mark.asyncio
    async def test_listmodels_never_fetches_and_shows_what_the_aliases_mean_under_batch(self, monkeypatch):
        import os
        from unittest.mock import patch

        from providers.registry import get_default_registry
        from tools.listmodels import ListModelsTool

        fetcher = install_batch_catalog()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key", "DEFAULT_MODEL": "auto"}, clear=True):
            get_default_registry().register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
            result = await ListModelsTool().execute({})

        content = json.loads(result[0].text)["content"]
        assert fetcher.calls == []
        opus = next(line for line in content.splitlines() if "`anthropic/claude-opus-4.5`" in line)
        assert "batch: opus, claude-opus → " + OPUS_55 in opus
        gemini = next(line for line in content.splitlines() if line.startswith("- `google/gemini-3-pro-preview`"))
        assert "batch: gemini, gemini-pro, pro → " + GEMINI_PRO in gemini
        flash = next(line for line in content.splitlines() if line.startswith("- `google/gemini-2.5-flash`"))
        assert "→ " + GEMINI_FLASH in flash
        assert "full slug" in content  # any <slug>:batch model can be batched by its full slug

    @pytest.mark.asyncio
    async def test_listmodels_reflects_a_catalog_already_loaded_but_still_does_not_fetch(self):
        import os
        from unittest.mock import patch

        from providers.registry import get_default_registry
        from tools.listmodels import ListModelsTool

        fetcher = install_batch_catalog(["anthropic/claude-opus-4.8", "google/gemini-3.8-flash"])
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key", "DEFAULT_MODEL": "auto"}, clear=True):
            get_default_registry().register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
            provider = get_default_registry().get_provider(ProviderType.OPENROUTER)
            provider.can_batch_model("opus")  # an admission loads the live catalog (one fetch)
            assert len(fetcher.calls) == 1
            result = await ListModelsTool().execute({})

        content = json.loads(result[0].text)["content"]
        assert len(fetcher.calls) == 1  # listing added none
        assert "batch: opus, claude-opus → anthropic/claude-opus-4.8" in content


# ----------------------------------------------------------------------
# 14.7 alias expansion under batch, restrictions, the synchronous path
# ----------------------------------------------------------------------


class TestAliasExpansion:
    def test_opus_under_batch_is_the_latest_batch_listed_opus(self, provider):
        assert provider._resolve_batch_model("opus").slug == OPUS_55
        assert provider._resolve_batch_model("opus:batch").slug == OPUS_55
        assert provider._resolve_batch_model("  sonnet  ").slug == SONNET_55
        assert provider.batch_submission_model("opus") == OPUS_55

    def test_a_newer_release_is_picked_up_without_any_code_change(self, provider):
        install_batch_catalog([OPUS_55, "anthropic/claude-opus-5.6", "anthropic/claude-opus-4.8"])
        assert provider._resolve_batch_model("opus").slug == "anthropic/claude-opus-5.6"

    def test_an_allow_list_naming_only_opus_4_8_makes_opus_resolve_to_4_8(self, provider, monkeypatch):
        _restrict(monkeypatch, "anthropic/claude-opus-4.8")
        assert provider._resolve_batch_model("opus").slug == "anthropic/claude-opus-4.8"
        assert provider.batch_submission_model("opus") == "anthropic/claude-opus-4.8"

    def test_an_allow_list_naming_the_alias_admits_the_newest(self, provider, monkeypatch):
        _restrict(monkeypatch, "opus")
        assert provider._resolve_batch_model("opus").slug == OPUS_55

    def test_restrictions_are_applied_before_choosing_the_newest(self, provider, monkeypatch):
        _restrict(monkeypatch, "anthropic/claude-opus-4.7,anthropic/claude-opus-4.6")
        assert provider._resolve_batch_model("opus").slug == "anthropic/claude-opus-4.7"

    def test_synchronous_resolution_is_untouched_and_never_consults_the_catalog(self, provider):
        fetcher = install_batch_catalog()
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)

        assert provider._resolve_model_name("opus") == "anthropic/claude-opus-4.5"
        assert provider.get_capabilities("opus").model_name == "anthropic/claude-opus-4.5"
        assert provider._resolve_model_name("flash") == "google/gemini-2.5-flash"
        assert registry.get_provider_for_model("opus").get_provider_type() is ProviderType.OPENROUTER
        assert registry.get_provider_for_model("flash").get_provider_type() is ProviderType.GOOGLE  # native, sync
        assert registry.get_provider_for_model("gpt5").get_provider_type() is ProviderType.OPENAI
        assert fetcher.calls == []

    def test_gemini_and_flash_under_batch_select_openrouter_where_native_google_cannot_batch(self):
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)
        set_default_registry(registry)

        for alias, expected in (("gemini", GEMINI_PRO), ("flash", GEMINI_FLASH), ("opus", OPUS_55), ("gpt5", GPT_55)):
            chosen = select_batch_provider(alias, registry)
            assert chosen.get_provider_type() is ProviderType.OPENROUTER, alias
            assert chosen.batch_submission_model(alias) == expected, alias

        # a full slug the live list carries (and the curated file does not) selects OpenRouter as well
        assert select_batch_provider(GEMINI_FLASH, registry).get_provider_type() is ProviderType.OPENROUTER

    def test_the_restriction_policy_decides_selection_not_only_the_unresolved_alias(self, monkeypatch):
        _restrict(monkeypatch, "anthropic/claude-opus-4.8")
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)
        provider = select_batch_provider("opus", registry)
        assert provider.batch_submission_model("opus") == "anthropic/claude-opus-4.8"

    def test_a_batch_naming_the_alias_cannot_carry_requests_for_a_different_slug(self, provider, monkeypatch):
        from providers.shared import BatchRequest

        monkeypatch.setattr(provider, "_batch_send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sent")))
        mixed = [
            BatchRequest("a", "x", extra={"model": "opus"}),
            BatchRequest("b", "y", extra={"model": "anthropic/claude-opus-4.5"}),
        ]
        with pytest.raises(ValueError, match="one model"):
            provider.submit_batch("opus", mixed)
        # a full-slug batch (4.5) no longer accepts requests written for the alias (now the latest, 5.5)
        with pytest.raises(ValueError, match="one model"):
            provider.submit_batch("anthropic/claude-opus-4.5", [BatchRequest("a", "x", extra={"model": "opus"})])


# ----------------------------------------------------------------------
# 14.8 reporting, resolve once, the recorded slug governs
# ----------------------------------------------------------------------


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


@pytest.fixture(autouse=True)
def _no_synchronous_openrouter_call(monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("the synchronous provider path ran for a batch request")

    monkeypatch.setattr(OpenRouterProvider, "generate_content", boom)
    monkeypatch.setattr(OpenRouterProvider, "async_generate_content", boom, raising=False)


def _chat_args(tmp_path, model: str, **extra: Any) -> dict[str, Any]:
    return {
        "prompt": "my question",
        "working_directory_absolute_path": str(tmp_path),
        "model": model,
        "execution_mode": "batch",
        **extra,
    }


def _outer(blocks: Any) -> dict[str, Any]:
    return json.loads(blocks[0].text)


def _consensus_args(models: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "step": "Should we cache?",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "findings",
        "models": models,
        **extra,
    }


def _entry(model: str, stance: str = "neutral") -> dict[str, Any]:
    return {"model": model, "stance": stance, "execution_mode": "batch"}


class TestReportingAndDeterminism:
    @pytest.mark.asyncio
    async def test_a_chat_batch_reports_what_the_alias_resolved_to(self, boot, service, tmp_path):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id, {"chat": "Answered."})

        done = _outer(await world.chat.execute(_chat_args(tmp_path, "opus", wait_seconds=30)))

        assert done["metadata"]["model_requested"] == "opus"
        assert done["metadata"]["model_resolved"] == OPUS_55 and done["metadata"]["model_used"] == OPUS_55
        assert service.submitted_models == [OPUS_55]

    @pytest.mark.asyncio
    async def test_a_full_slug_reports_no_resolution(self, boot, service, tmp_path):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id, {"chat": "Answered."})

        done = _outer(await world.chat.execute(_chat_args(tmp_path, GEMINI_FLASH, wait_seconds=30)))

        assert "model_requested" not in done["metadata"] and "model_resolved" not in done["metadata"]
        assert service.submitted_models == [GEMINI_FLASH]  # the D5 scenario: an uncurated full slug batches

    @pytest.mark.asyncio
    async def test_a_running_chat_batch_names_the_resolved_model_before_it_finishes(self, boot, service, tmp_path):
        world = boot()

        running = _outer(await world.chat.execute(_chat_args(tmp_path, "sonnet", wait_seconds=0)))

        body = json.loads(running["content"])
        assert body["children"][0]["model"] == "sonnet" and body["children"][0]["model_resolved"] == SONNET_55

    @pytest.mark.asyncio
    async def test_a_consensus_batch_reports_the_resolution_for_each_entry(self, boot, service):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id)

        body = json.loads(
            (await world.consensus.execute(_consensus_args([_entry("gemini", "for"), _entry("opus", "against")])))[
                0
            ].text
        )

        by_model = {r["model"]: r["metadata"] for r in body["accumulated_responses"]}
        assert by_model["gemini"]["model_requested"] == "gemini"
        assert by_model["gemini"]["model_resolved"] == GEMINI_PRO
        assert by_model["opus"]["model_resolved"] == OPUS_55
        assert sorted(service.submitted_models) == sorted([GEMINI_PRO, OPUS_55])  # one batch per model

    @pytest.mark.asyncio
    async def test_a_handle_submitted_under_one_catalog_resumes_under_a_newer_one_on_the_recorded_slug(
        self, boot, service, tmp_path, monkeypatch
    ):
        world = boot()
        slow = _outer(await world.chat.execute(_chat_args(tmp_path, "opus", wait_seconds=0)))
        handle = json.loads(slow["content"])["batch_handle"]
        assert service.submitted_models == [OPUS_55]
        world.close()

        newer = install_batch_catalog([OPUS_55, "anthropic/claude-opus-6.0"])
        service.finish("or_batch_1", {"chat": "From the recorded model."})

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a resumption resolved a model name again")

        for name in ("_resolve_batch_model", "_batch_slug", "batch_submission_model", "_batch_catalog"):
            monkeypatch.setattr(OpenRouterProvider, name, forbidden)

        second = boot()
        done = _outer(await second.chat.execute({"batch_handle": handle}))

        assert done["metadata"]["model_used"] == OPUS_55  # the recorded slug, not the newer Opus
        assert done["metadata"]["model_requested"] == "opus" and done["metadata"]["model_resolved"] == OPUS_55
        assert newer.calls == [] and len(service.posts) == 1  # no catalog access, nothing resubmitted

    @pytest.mark.asyncio
    async def test_cancel_reads_the_record_and_never_resolves(self, boot, service, tmp_path, monkeypatch):
        world = boot()
        slow = _outer(await world.chat.execute(_chat_args(tmp_path, "opus", wait_seconds=0)))
        handle = json.loads(slow["content"])["batch_handle"]
        fetcher = install_batch_catalog([OPUS_55, "anthropic/claude-opus-6.0"])

        cancelled = _outer(await world.chat.execute({"batch_handle": handle, "cancel": True}))

        assert json.loads(cancelled["content"])["status"] == "cancelled" and fetcher.calls == []

    @pytest.mark.asyncio
    async def test_every_model_of_one_submission_resolves_against_one_catalog(self, boot, service):
        """The catalog changes on every fetch and the TTL is zero; only the pin keeps one operation consistent."""

        counter = {"n": 0}

        def moving(timeout: float) -> Any:
            counter["n"] += 1
            n = counter["n"]
            return payload_for([f"anthropic/claude-opus-5.{n}", f"anthropic/claude-sonnet-5.{n}"])

        bc.set_batch_catalog_source(BatchCatalogSource(fetcher=moving, ttl_seconds=0.0))
        world = boot()

        run = await world.consensus.execute(
            _consensus_args([_entry("opus", "for"), _entry("sonnet", "against")], wait_seconds=0)
        )

        handle = json.loads(json.loads(run[0].text)["content"])["batch_handle"]
        opus_wire, sonnet_wire = (
            next(p["envelope"]["model"] for p in service.posts if f"claude-{family}" in p["envelope"]["model"])
            for family in ("opus", "sonnet")
        )
        assert opus_wire.rsplit(".", 1)[1] == sonnet_wire.rsplit(".", 1)[1]  # the same catalog served both
        record = world.store.get(handle)
        assert [c.submitted_model for c in record.children] == [opus_wire, sonnet_wire]  # intent == the wire


# ----------------------------------------------------------------------
# 14.9 failure modes
# ----------------------------------------------------------------------


class TestFailureModes:
    @pytest.mark.asyncio
    async def test_an_alias_whose_family_has_no_batch_member_is_refused_before_any_batch(self, boot, service, tmp_path):
        install_batch_catalog([GEMINI_FLASH, SONNET_55])  # no Opus at all
        world = boot()

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.chat.execute(_chat_args(tmp_path, "opus"))

        content = json.loads(excinfo.value.payload)["content"]
        assert "'opus'" in content and "Claude Opus" in content and "without execution_mode" in content
        assert "sonnet" not in content.lower()  # no other family is offered or used
        assert service.posts == []

    @pytest.mark.asyncio
    async def test_a_family_emptied_by_restrictions_is_refused_naming_the_policy(
        self, boot, service, tmp_path, monkeypatch
    ):
        _restrict(monkeypatch, "anthropic/claude-sonnet-4.5")
        world = boot()

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.chat.execute(_chat_args(tmp_path, "opus"))

        content = json.loads(excinfo.value.payload)["content"]
        assert "'opus'" in content and "restriction policy" in content and "without execution_mode" in content
        assert service.posts == []

    @pytest.mark.asyncio
    async def test_a_consensus_roster_with_one_empty_family_submits_nothing(self, boot, service):
        install_batch_catalog([GEMINI_FLASH, SONNET_55])
        world = boot()

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.consensus.execute(_consensus_args([_entry("sonnet"), _entry("opus")]))

        assert "'opus'" in json.loads(excinfo.value.payload)["content"] and service.posts == []

    def test_a_floating_alias_miss_never_resolves_to_a_different_family(self, provider):
        install_batch_catalog([SONNET_55, GEMINI_FLASH, "openai/gpt-5.5"])
        for alias in ("opus", "haiku", "grok", "gemini"):
            with pytest.raises(BatchModelRefused):
                provider._resolve_batch_model(alias)
            assert not provider.can_batch_model(alias)

    def test_an_unreachable_catalog_degrades_to_the_snapshot_with_one_warning_per_failure_window(self, caplog):
        clock = tick_clock()
        fetcher = RecordingFetcher(error=httpx.ConnectError("down"))
        source = _source(fetcher, clock)

        with caplog.at_level(logging.WARNING, logger=bc.logger.name):
            for _ in range(6):
                assert source.current().origin == "snapshot"

            def warnings() -> list[logging.LogRecord]:
                return [r for r in caplog.records if r.levelno == logging.WARNING and r.name == bc.logger.name]

            assert len(warnings()) == 1 and "snapshot" in warnings()[0].getMessage()

            clock.now += 301  # the next failure window
            source.current()
            assert len(warnings()) == 2

    def test_the_snapshot_still_resolves_families_while_the_live_catalog_is_down(self, provider):
        bc.set_batch_catalog_source(BatchCatalogSource(fetcher=RecordingFetcher(error=httpx.ConnectError("down"))))
        assert provider._resolve_batch_model("opus").slug == OPUS_55  # from the committed snapshot
        assert provider._resolve_batch_model("flash").slug == GEMINI_FLASH


# ----------------------------------------------------------------------
# 14.10 the refresh script
# ----------------------------------------------------------------------


@pytest.fixture
def refresh():
    scripts = str(REPO / "scripts")
    sys.path.insert(0, scripts)
    try:
        import refresh_openrouter_batch_catalog as module
    finally:
        sys.path.remove(scripts)
    return module


class TestRefreshScript:
    def test_it_writes_a_sorted_snapshot_and_a_second_identical_run_is_unchanged(self, refresh, tmp_path, capsys):
        target = tmp_path / "snapshot.json"
        models = REPO / "conf" / "openrouter_models.json"
        payload = payload_for(["z/last-1", "a/first-1", OPUS_55], extra_ids=["a/plain"])

        code = refresh.main([], fetch=lambda: payload, snapshot_path=target, models_path=models, today="2030-01-02")

        assert code == 0
        document = json.loads(target.read_text())
        assert document["batch_models"] == sorted(["z/last-1", "a/first-1", OPUS_55])
        assert document["fetched_at"] == "2030-01-02" and not any(
            s.endswith(":batch") for s in document["batch_models"]
        )
        before = target.read_bytes()

        again = refresh.main([], fetch=lambda: payload, snapshot_path=target, models_path=models, today="2031-01-01")
        assert again == 2 and target.read_bytes() == before
        assert "No change" in capsys.readouterr().out

    def test_a_transport_error_exits_1_and_leaves_the_file_alone(self, refresh, tmp_path, capsys):
        target = tmp_path / "snapshot.json"
        target.write_text('{"batch_models": ["keep/me"]}')
        before = target.read_bytes()

        def failing() -> Any:
            raise httpx.ConnectError("down")

        code = refresh.main(
            [], fetch=failing, snapshot_path=target, models_path=REPO / "conf" / "openrouter_models.json"
        )

        assert code == 1 and target.read_bytes() == before
        assert "ERROR" in capsys.readouterr().err

    def test_a_malformed_body_exits_1_and_dry_run_writes_nothing(self, refresh, tmp_path):
        target = tmp_path / "snapshot.json"
        models = REPO / "conf" / "openrouter_models.json"
        assert refresh.main([], fetch=lambda: {"error": "x"}, snapshot_path=target, models_path=models) == 1
        assert (
            refresh.main(["--dry-run"], fetch=lambda: payload_for([OPUS_55]), snapshot_path=target, models_path=models)
            == 0
        )
        assert not target.exists()

    def test_the_drift_report_names_a_flag_that_disagrees_with_the_catalog(self, refresh, tmp_path, capsys):
        config = {
            "models": [
                {"model_name": "vendor/stale", "aliases": ["st"], "supports_batch": True},
                {"model_name": "vendor/unflagged", "aliases": ["un"], "supports_batch": False},
                {"model_name": "vendor/fine", "aliases": ["fi"], "supports_batch": True},
                {
                    "model_name": "vendor/old-1",
                    "aliases": ["fam"],
                    "supports_batch": False,
                    "batch_family": {"aliases": ["fam"], "pattern": r"^vendor/old-(?P<version>\d+(?:\.\d+)*)$"},
                },
                {
                    "model_name": "vendor/gone-1",
                    "aliases": ["gone"],
                    "supports_batch": False,
                    "batch_family": {"aliases": ["gone"], "pattern": r"^vendor/gone-(?P<version>\d+)$"},
                },
            ]
        }
        models = tmp_path / "models.json"
        models.write_text(json.dumps(config))
        listed = payload_for(["vendor/unflagged", "vendor/fine", "vendor/old-1", "vendor/old-2"])

        refresh.main(["--dry-run"], fetch=lambda: listed, snapshot_path=tmp_path / "s.json", models_path=models)
        out = capsys.readouterr().out

        assert "`vendor/stale`: supports_batch is true but OpenRouter no longer lists" in out
        assert "`vendor/unflagged`: OpenRouter lists" in out and "supports_batch is false" in out
        assert "vendor/fine" not in out.split("### Drift")[1]  # an entry whose flag agrees is silent
        assert "`vendor/old-1`" in out and "`vendor/old-2`" in out  # the pin is behind the family's latest
        assert "`vendor/gone-1`: the batch_family rule" in out and "matches no batch-listed slug" in out

    def test_the_committed_catalog_has_no_flag_or_rule_drift_against_the_committed_snapshot(self, refresh):
        slugs = load_snapshot().slugs
        config = json.loads((REPO / "conf" / "openrouter_models.json").read_text())
        flagged = [
            line for line in refresh.drift_report(slugs, config) if "supports_batch" in line or "matches no" in line
        ]
        assert flagged == []  # pins behind the latest are expected and only informational

    def test_output_summary_is_written(self, refresh, tmp_path):
        summary = tmp_path / "summary.md"
        refresh.main(
            ["--dry-run", "--output-summary", str(summary)],
            fetch=lambda: payload_for([OPUS_55]),
            snapshot_path=tmp_path / "s.json",
            models_path=REPO / "conf" / "openrouter_models.json",
        )
        assert "OpenRouter batch catalog refresh" in summary.read_text()

    # -- section 15.2: the capability view in the snapshot -------------------------------------------------

    def test_it_writes_the_capability_view_deterministically_and_small(self, refresh, tmp_path):
        target = tmp_path / "snapshot.json"
        models = REPO / "conf" / "openrouter_models.json"
        body = catalog_fixture_payload()

        assert refresh.main([], fetch=lambda: body, snapshot_path=target, models_path=models, today="2030-01-02") == 0
        first = target.read_bytes()
        assert len(first) < 40_000
        document = json.loads(first)
        assert document["format"] == 2 and sorted(document["model_data"]) == document["batch_models"]
        opus = document["model_data"][OPUS_55]
        assert opus["context_length"] == 1_000_000 and opus["max_completion_tokens"] == 128_000
        assert "temperature" not in opus["supported_parameters"]
        assert list(opus) == sorted(opus)  # keys sorted

        # a rewrite of the same body to a new path is byte-identical (nothing depends on iteration order)
        other = tmp_path / "again.json"
        assert refresh.main([], fetch=lambda: body, snapshot_path=other, models_path=models, today="2030-01-02") == 0
        assert other.read_bytes() == first
        assert refresh.main([], fetch=lambda: body, snapshot_path=target, models_path=models, today="2031-05-05") == 2
        assert target.read_bytes() == first

    def test_the_drift_report_names_a_slug_whose_temperature_support_changed(self, refresh, tmp_path, capsys):
        target = tmp_path / "snapshot.json"
        models = REPO / "conf" / "openrouter_models.json"
        assert refresh.main([], fetch=catalog_fixture_payload, snapshot_path=target, models_path=models) == 0
        capsys.readouterr()

        def add_temperature(entry: dict[str, Any]) -> None:
            entry["supported_parameters"] = sorted([*entry["supported_parameters"], "temperature"])

        def shrink_context(entry: dict[str, Any]) -> None:
            entry["context_length"] = 500_000

        changed = payload_with({OPUS_55 + ":batch": add_temperature, GEMINI_FLASH + ":batch": shrink_context})
        assert refresh.main([], fetch=lambda: changed, snapshot_path=target, models_path=models) == 0
        out = capsys.readouterr().out

        section = out.split("### Capability drift")[1].split("### Drift against")[0]
        assert f"`{OPUS_55}`: temperature support changed (false -> true)" in section
        assert f"`{GEMINI_FLASH}`: context_length changed (1048576 -> 500000)" in section
        assert "anthropic/claude-sonnet-5.5" not in section  # an unchanged model is silent

    def test_a_body_that_drops_the_capability_fields_exits_1_and_leaves_the_file_untouched(
        self, refresh, tmp_path, capsys
    ):
        target = tmp_path / "snapshot.json"
        models = REPO / "conf" / "openrouter_models.json"
        assert refresh.main([], fetch=catalog_fixture_payload, snapshot_path=target, models_path=models) == 0
        before = target.read_bytes()
        capsys.readouterr()

        stripped = payload_for(json.loads(before)["batch_models"])  # every slug still listed, no capability fields
        assert refresh.main([], fetch=lambda: stripped, snapshot_path=target, models_path=models) == 1
        assert target.read_bytes() == before
        assert "capability fields" in capsys.readouterr().err

        # a few slugs losing fields is an ordinary refresh; --allow-shrink accepts a real, wholesale change
        def drop_fields(entry: dict[str, Any]) -> None:
            for key in ("context_length", "supported_parameters", "architecture", "top_provider"):
                entry.pop(key, None)

        some = [f"{slug}:batch" for slug in json.loads(before)["batch_models"][:5]]
        mild = payload_with(dict.fromkeys(some, drop_fields))
        assert refresh.main(["--dry-run"], fetch=lambda: mild, snapshot_path=target, models_path=models) == 0
        assert refresh.main(["--allow-shrink"], fetch=lambda: stripped, snapshot_path=target, models_path=models) == 0
        assert json.loads(target.read_text())["model_data"][OPUS_55]["context_length"] is None


# ----------------------------------------------------------------------
# 14.11 decision D3: the provider batch id is logged, never returned, on cancel
# ----------------------------------------------------------------------


def _start(executor: Any, plan: Any) -> str:
    return executor.submit_and_wait("chat", [plan], wait_seconds=0).handle


class TestCancelLogsTheProviderIdOnly:
    def test_cancel_unsupported_logs_the_id_and_the_response_never_carries_it(self, tmp_path, caplog):
        remote, clock = FakeRemote(), FakeClock()
        provider = FakeProvider(remote)  # no cancel mechanism: unsupported
        from utils.batch_store import BatchStore

        store = BatchStore(str(tmp_path / "d" / "b.db"))
        try:
            executor = make_executor(store, {"openrouter": provider}, clock)
            handle = _start(executor, make_plan(provider))

            with caplog.at_level(logging.INFO, logger="utils.batch_execution"):
                result = executor.cancel(handle)

            logged = [r for r in caplog.records if "prov_batch_1" in r.getMessage()]
            assert logged and any(r.levelno == logging.WARNING for r in logged)
            assert handle in logged[0].getMessage() and "openrouter" in logged[0].getMessage()
            surfaced = json.dumps(build_status_payload(result, tool_name="chat")) + result_message(
                result, tool_name="chat"
            )
            surfaced += " ".join(report.detail for report in result.cancel_reports)
            assert "prov_batch_1" not in surfaced
        finally:
            store.close()

    def test_a_successful_cancel_also_logs_the_id_server_side(self, tmp_path, caplog):
        from providers.shared import CancelOutcome, CancelState
        from utils.batch_store import BatchStore

        remote, clock = FakeRemote(), FakeClock()
        provider = FakeProvider(remote, cancel=CancelOutcome(CancelState.CANCELLED))
        store = BatchStore(str(tmp_path / "d" / "b.db"))
        try:
            executor = make_executor(store, {"openrouter": provider}, clock)
            handle = _start(executor, make_plan(provider))
            with caplog.at_level(logging.INFO, logger="utils.batch_execution"):
                result = executor.cancel(handle)
            assert any("prov_batch_1" in r.getMessage() for r in caplog.records)
            assert "prov_batch_1" not in json.dumps(build_status_payload(result, tool_name="chat"))
        finally:
            store.close()

    def test_the_orphaned_batch_error_is_the_one_exception_and_still_names_provider_and_id(self, tmp_path, monkeypatch):
        from providers.shared import CancelOutcome, CancelState
        from tests.test_batch_coordinator import FakeProvider as CoordinatorProvider
        from tests.test_batch_coordinator import _Clock, _coordinator, _fail_first_write, _plan
        from utils.batch_coordinator import BatchSubmissionError
        from utils.batch_store import BatchStore

        clock = _Clock()
        store = BatchStore(str(tmp_path / "d" / "o.db"), clock=clock)
        try:
            provider = CoordinatorProvider("openrouter", cancel=CancelOutcome(CancelState.UNSUPPORTED, "no cancel"))
            _fail_first_write(store, monkeypatch)
            with pytest.raises(BatchSubmissionError) as excinfo:
                _coordinator(store, {"openrouter": provider}, clock).submit("chat", [_plan(provider)])
            text = str(excinfo.value)
            assert "openrouter" in text and "prov_openrouter_1" in text
        finally:
            store.close()


# ----------------------------------------------------------------------
# 14.12 decision D4: variants are refused under batch, never stripped
# ----------------------------------------------------------------------


class TestVariantsUnderBatch:
    @pytest.mark.parametrize("variant", ["free", "online", "thinking", "exacto", "nitro"])
    def test_a_variant_is_refused_naming_it_and_the_synchronous_alternative(self, provider, variant):
        identifier = f"openai/gpt-5:{variant}"

        with pytest.raises(BatchModelRefused) as excinfo:
            provider._resolve_batch_model(identifier)

        text = str(excinfo.value)
        assert f"':{variant}'" in text and identifier in text and "without execution_mode" in text
        assert "not stripped" in text
        assert not provider.can_batch_model(identifier)
        assert provider.explain_batch_refusal(identifier) == text

    def test_a_variant_on_an_alias_is_refused_too(self, provider):
        with pytest.raises(BatchModelRefused, match="':free'"):
            provider._resolve_batch_model("opus:free")

    def test_the_batch_suffix_and_the_base_spelling_converge(self, provider):
        for base in (OPUS_55, GEMINI_FLASH, "opus", "gemini"):
            assert provider._resolve_batch_model(base).slug == provider._resolve_batch_model(base + ":batch").slug
            assert provider.batch_submission_model(base) == provider.batch_submission_model(base + ":batch")

    def test_a_doubled_batch_suffix_is_a_variant(self, provider):
        with pytest.raises(BatchModelRefused, match="':batch'"):
            provider._resolve_batch_model(OPUS_55 + ":batch:batch")

    def test_admission_explains_a_variant_by_name_and_the_synchronous_way_out(self):
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)
        from tools.shared.batch_support import BatchRequestRejected

        with pytest.raises(BatchRequestRejected) as excinfo:
            select_batch_provider("openai/gpt-5:free", registry)

        assert excinfo.value.code == "batch_unsupported_model"
        assert "':free'" in str(excinfo.value) and "without execution_mode" in str(excinfo.value)

    def test_without_execution_mode_every_variant_forwards_byte_for_byte(self, provider):
        from utils.model_resolution import trim_model_identifier

        for identifier in ("openai/gpt-5:free", "openai/gpt-5:online", OPUS_55 + ":batch", "x/y:thinking"):
            assert trim_model_identifier(f"  {identifier}  ") == identifier
            assert provider._resolve_model_name(identifier) == identifier  # the synchronous resolver: untouched
        assert provider.get_capabilities("openai/gpt-5:free").model_name == "openai/gpt-5:free"

    def test_a_genuinely_unbatchable_model_is_refused_by_name(self, provider):
        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)
        from tools.shared.batch_support import BatchRequestRejected

        with pytest.raises(BatchRequestRejected) as excinfo:
            select_batch_provider("mistral", registry)

        assert "'mistral'" in str(excinfo.value) and "without execution_mode" in str(excinfo.value)


# ----------------------------------------------------------------------
# end to end: the exact user scenarios
# ----------------------------------------------------------------------


class TestUserScenarios:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("typed", "wire"),
        [
            ("opus", OPUS_55),
            ("sonnet", SONNET_55),
            ("haiku", HAIKU_55),
            ("gemini", GEMINI_PRO),
            ("flash", GEMINI_FLASH),
            ("gpt5", GPT_55),
            (GEMINI_FLASH, GEMINI_FLASH),
            ("anthropic/claude-opus-4.5", "anthropic/claude-opus-4.5"),  # a full slug is used as given
        ],
    )
    async def test_chat_under_batch_submits_the_expected_slug(self, boot, service, tmp_path, typed, wire):
        world = boot()

        await world.chat.execute(_chat_args(tmp_path, typed, wait_seconds=0))

        assert service.submitted_models == [wire]

    @pytest.mark.asyncio
    async def test_use_unison_consensus_with_gemini_and_opus_batch_models_submits_one_batch_per_model(
        self, boot, service
    ):
        world = boot()

        run = await world.consensus.execute(
            _consensus_args([_entry("gemini", "for"), _entry("opus", "against")], wait_seconds=0)
        )

        body = json.loads(json.loads(run[0].text)["content"])
        assert body["status"] == "running" and body["batch_handle"].startswith("batch_")
        assert sorted(service.submitted_models) == sorted([GEMINI_PRO, OPUS_55])
        assert len(service.posts) == 2
        assert [m["model"] for m in world.store.get(body["batch_handle"]).payload["models"]] == ["gemini", "opus"]

    @pytest.mark.asyncio
    async def test_the_boundary_resolves_a_handle_less_start_work_call_through_dynamic_resolution(
        self, boot, service, tmp_path
    ):
        from handlers.tool_handlers import build_handlers
        from server import tool_registry

        world = boot()
        chat = tool_registry.get_tool_instance("chat")
        chat._batch_executor = world.executor
        handlers = build_handlers(tool_registry)

        await handlers.handle_call_tool("chat", _chat_args(tmp_path, "opus", wait_seconds=0))

        assert service.submitted_models == [OPUS_55]
        _ = ChatTool  # the same tool class the boundary instantiates


# ----------------------------------------------------------------------
# review fixes: a degraded body never empties the catalog
# ----------------------------------------------------------------------

DEGRADED_BODY = {"data": [{"id": "openai/gpt-5"}, {"id": OPUS_55}, {"id": "google/gemini-3.8-flash"}]}


class TestADegradedBodyNeverEmptiesTheCatalog:
    def test_a_model_list_with_no_batch_entry_is_rejected(self):
        with pytest.raises(BatchCatalogFormatError, match="no usable ':batch' listing"):
            parse_models_payload(DEGRADED_BODY)

    def test_the_first_fetch_being_degraded_falls_back_to_the_snapshot_not_an_empty_live_catalog(self):
        source = _source(RecordingFetcher(DEGRADED_BODY), snapshot=_snap("snap/model"))

        catalog = source.current()

        assert catalog.origin == "snapshot" and catalog.lists("snap/model")

    def test_a_later_degraded_refresh_keeps_the_last_good_live_catalog(self):
        fetcher, clock = RecordingFetcher(), tick_clock()
        source = _source(fetcher, clock)
        good = source.current()

        fetcher.payload = DEGRADED_BODY
        clock.now += 4000
        assert source.current() is good and good.lists(OPUS_55)

    def test_aliases_and_uncurated_slugs_still_resolve_after_a_degraded_body(self, provider):
        install_batch_catalog(fetcher=RecordingFetcher(DEGRADED_BODY))

        assert provider._resolve_batch_model("opus").slug == OPUS_55  # from the committed snapshot
        assert provider._resolve_batch_model(GEMINI_FLASH).slug == GEMINI_FLASH

    def test_the_refresh_script_refuses_a_degraded_body_and_keeps_the_snapshot(self, refresh, tmp_path, capsys):
        target = tmp_path / "snapshot.json"
        target.write_text(json.dumps({"batch_models": ["keep/me"]}))
        before = target.read_bytes()

        code = refresh.main(
            [], fetch=lambda: DEGRADED_BODY, snapshot_path=target, models_path=REPO / "conf" / "openrouter_models.json"
        )

        assert code == 1 and target.read_bytes() == before
        assert "no usable ':batch' listing" in capsys.readouterr().err

    def test_the_refresh_script_refuses_a_list_that_shrinks_the_snapshot_to_under_half(self, refresh, tmp_path, capsys):
        models = REPO / "conf" / "openrouter_models.json"
        target = tmp_path / "snapshot.json"
        full = [f"vendor/model-{n}" for n in range(10)]
        target.write_text(json.dumps({"batch_models": sorted(full)}))
        before = target.read_bytes()

        code = refresh.main([], fetch=lambda: payload_for(full[:3]), snapshot_path=target, models_path=models)
        assert code == 1 and target.read_bytes() == before
        assert "--allow-shrink" in capsys.readouterr().err

        # a modest shrink is an ordinary refresh, and --allow-shrink accepts a real delisting
        assert refresh.main([], fetch=lambda: payload_for(full[:7]), snapshot_path=target, models_path=models) == 0
        assert (
            refresh.main(
                ["--allow-shrink"], fetch=lambda: payload_for(full[:2]), snapshot_path=target, models_path=models
            )
            == 0
        )
        assert json.loads(target.read_text())["batch_models"] == sorted(full[:2])


# ----------------------------------------------------------------------
# review fixes: the real HTTP fetch is bounded, credential-free, and covered
# ----------------------------------------------------------------------


def _mock(handler: Any) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


class TestTheRealFetch:
    def test_it_sends_an_unauthenticated_get_and_returns_the_decoded_body(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=catalog_fixture_payload())

        body = real_http_fetch(5.0, transport=_mock(handler))

        assert body["data"] == catalog_fixture_payload()["data"]
        (request,) = seen
        assert request.method == "GET" and str(request.url) == bc.BATCH_CATALOG_URL
        lowered = {name.lower() for name in request.headers}
        assert "authorization" not in lowered and "x-api-key" not in lowered and "cookie" not in lowered
        assert "sk-or-secret" not in repr(dict(request.headers)) and not request.content

    def test_a_body_declaring_more_than_the_cap_is_refused_before_it_is_read(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-length": "999999"}, content=b"x" * 10)

        with pytest.raises(BatchCatalogFormatError, match="declares 999999 bytes"):
            real_http_fetch(5.0, transport=_mock(handler), max_bytes=1000)

    def test_a_body_that_does_not_declare_its_size_is_cut_off_at_the_cap(self):
        read = []

        def chunks():
            for index in range(1000):
                read.append(index)
                yield b"x" * 100

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=chunks())

        with pytest.raises(BatchCatalogFormatError, match="larger than 1000 bytes"):
            real_http_fetch(5.0, transport=_mock(handler), max_bytes=1000)
        assert len(read) < 20  # it stopped reading, it did not buffer the whole stream

    def test_a_slow_drip_download_is_cut_off_by_the_total_deadline(self):
        clock = tick_clock()

        def chunks():
            for _ in range(100):
                clock.now += 1.0  # each read is individually quick, together they are slow
                yield b" "

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=chunks())

        with pytest.raises(TimeoutError, match="outlasted 5s"):
            real_http_fetch(5.0, transport=_mock(handler), clock=clock)

    def test_an_error_status_raises(self):
        with pytest.raises(httpx.HTTPStatusError):
            real_http_fetch(5.0, transport=_mock(lambda request: httpx.Response(503)))

    def test_an_oversized_body_leaves_the_source_on_the_snapshot_without_raising(self):
        def fetcher(timeout: float) -> Any:
            return real_http_fetch(
                timeout, transport=_mock(lambda request: httpx.Response(200, content=b"[" + b"1," * 600)), max_bytes=100
            )

        assert _source(fetcher).current().origin == "snapshot"

    def test_the_server_and_the_refresh_script_both_go_through_the_one_guarded_entry_point(
        self, batch_catalog_guard, refresh
    ):
        with pytest.raises(AssertionError, match="live OpenRouter model list"):
            bc.fetch_models_payload(2.0)
        with pytest.raises(AssertionError, match="live OpenRouter model list"):
            refresh.fetch_models()
        assert len(batch_catalog_guard) == 2
        batch_catalog_guard.clear()  # provoked on purpose


# ----------------------------------------------------------------------
# review fixes: readers never wait for a fetch
# ----------------------------------------------------------------------


class TestReadersNeverWaitBehindAFetch:
    @staticmethod
    def _blocked_fetch():
        entered, gate = threading.Event(), threading.Event()

        def fetcher(timeout: float) -> Any:
            entered.set()
            gate.wait(10)
            return catalog_fixture_payload()

        source = _source(fetcher, timeout_seconds=20.0)
        worker = threading.Thread(target=source.current, daemon=True)
        worker.start()
        assert entered.wait(5), "the fetch never started"
        return source, gate, worker

    def _within(self, seconds: float, action: Any) -> bool:
        done = threading.Event()
        threading.Thread(target=lambda: (action(), done.set()), daemon=True).start()
        return done.wait(seconds)

    def test_loaded_answers_from_the_snapshot_while_another_thread_is_fetching(self):
        source, gate, worker = self._blocked_fetch()
        try:
            results: list[Any] = []
            assert self._within(2.0, lambda: results.append(source.loaded()))
            assert results[0].origin == "snapshot"
        finally:
            gate.set()
            worker.join(5)

    def test_entering_a_pin_does_not_wait_for_a_fetch_in_flight(self):
        source, gate, worker = self._blocked_fetch()
        try:

            def enter_and_leave() -> None:
                with source.pin():
                    pass

            assert self._within(2.0, enter_and_leave)
        finally:
            gate.set()
            worker.join(5)

    def test_a_second_current_still_waits_for_and_shares_the_one_fetch(self):
        calls: list[float] = []
        gate = threading.Event()

        def fetcher(timeout: float) -> Any:
            calls.append(timeout)
            gate.wait(5)
            return catalog_fixture_payload()

        source = _source(fetcher, timeout_seconds=20.0)
        results: list[Any] = []
        threads = [threading.Thread(target=lambda: results.append(source.current())) for _ in range(4)]
        for thread in threads:
            thread.start()
        time.sleep(0.2)
        gate.set()
        for thread in threads:
            thread.join(5)

        assert len(calls) == 1 and len(results) == 4 and all(r is results[0] for r in results)

    @pytest.mark.asyncio
    async def test_listmodels_is_not_held_up_by_an_admission_that_is_fetching(self):
        source, gate, worker = self._blocked_fetch()
        bc.set_batch_catalog_source(source)
        try:
            from tools.listmodels import ListModelsTool

            started = time.monotonic()
            await asyncio.wait_for(ListModelsTool().execute({}), timeout=5)
            assert time.monotonic() - started < 3
        finally:
            gate.set()
            worker.join(5)


# ----------------------------------------------------------------------
# review fixes: a slow catalog lookup is spent from the wait budget
# ----------------------------------------------------------------------


class TestAdmissionIsChargedToTheWaitBudget:
    @staticmethod
    async def _run(admission_seconds: float, wait: Any) -> dict[str, Any]:
        tool = ChatTool()
        captured: dict[str, Any] = {}

        def slow_admission(arguments: dict[str, Any]) -> dict[str, Any]:
            time.sleep(admission_seconds)
            return {}

        async def start(arguments: dict[str, Any], providers: Any) -> list[Any]:
            captured.update(arguments)
            return []

        tool.admit_batch_request = slow_admission  # type: ignore[method-assign]
        tool.execute_batch_start = start  # type: ignore[method-assign]
        arguments: dict[str, Any] = {"prompt": "hi", "model": "opus", "execution_mode": "batch"}
        if wait is not None:
            arguments["wait_seconds"] = wait
        await tool.execute(arguments)
        return captured

    @pytest.mark.asyncio
    async def test_a_slow_admission_is_subtracted_from_an_explicit_wait(self):
        captured = await self._run(0.3, 20)
        assert 19.0 < captured["wait_seconds"] <= 19.7

    @pytest.mark.asyncio
    async def test_a_slow_admission_is_subtracted_from_the_default_wait(self):
        from utils.batch_execution import DEFAULT_WAIT_POLICY

        captured = await self._run(0.3, None)
        assert (
            DEFAULT_WAIT_POLICY.default_seconds - 0.7
            < captured["wait_seconds"]
            <= DEFAULT_WAIT_POLICY.default_seconds - 0.3
        )

    @pytest.mark.asyncio
    async def test_an_admission_longer_than_the_wait_leaves_none_and_never_goes_negative(self):
        captured = await self._run(0.3, 0.1)
        assert captured["wait_seconds"] == 0.0

    @pytest.mark.asyncio
    async def test_a_fast_admission_leaves_the_arguments_exactly_as_sent(self):
        assert (await self._run(0.0, 20))["wait_seconds"] == 20
        assert "wait_seconds" not in await self._run(0.0, None)

    def test_a_wait_that_is_not_a_plain_number_is_left_for_validation_to_reject(self):
        from tools.shared.batch_support import BatchResumableToolMixin

        arguments = {"wait_seconds": "soon"}
        assert BatchResumableToolMixin._charge_admission_to_wait(arguments, 5.0) is arguments
        flagged = {"wait_seconds": True}
        assert BatchResumableToolMixin._charge_admission_to_wait(flagged, 5.0) is flagged


# ----------------------------------------------------------------------
# review fixes: environment switches fail safe
# ----------------------------------------------------------------------


class TestEnvironmentSwitches:
    @pytest.mark.parametrize("value", ["inf", "-inf", "nan", "Infinity", "0", "-5", "soon"])
    def test_a_ttl_that_is_not_positive_and_finite_uses_the_default_with_a_warning(self, monkeypatch, caplog, value):
        monkeypatch.setenv("OPENROUTER_BATCH_CATALOG_TTL_SECONDS", value)
        with caplog.at_level(logging.WARNING, logger=bc.logger.name):
            assert bc._ttl_from_env() == bc.DEFAULT_TTL_SECONDS
        assert "positive, finite" in caplog.text

    def test_a_good_ttl_is_used(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BATCH_CATALOG_TTL_SECONDS", "90.5")
        assert bc._ttl_from_env() == 90.5

    @pytest.mark.parametrize("value", ["false", "FALSE", " no ", "0", "off", "disabled", "disable"])
    def test_the_recognised_off_words_disable_the_fetch(self, monkeypatch, value):
        monkeypatch.setenv("OPENROUTER_BATCH_LIVE_CATALOG", value)
        assert bc.live_catalog_enabled() is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", "enabled", "", "   "])
    def test_the_recognised_on_words_and_an_empty_value_leave_the_fetch_on(self, monkeypatch, value):
        monkeypatch.setenv("OPENROUTER_BATCH_LIVE_CATALOG", value)
        assert bc.live_catalog_enabled() is True

    def test_an_unset_variable_leaves_the_fetch_on(self):
        assert bc.live_catalog_enabled() is True

    @pytest.mark.parametrize("value", ["fasle", "nope", "garbage", "2"])
    def test_an_unrecognised_value_fails_closed_with_one_warning(self, monkeypatch, caplog, value):
        monkeypatch.setenv("OPENROUTER_BATCH_LIVE_CATALOG", value)
        bc._warned_values.clear()
        with caplog.at_level(logging.WARNING, logger=bc.logger.name):
            assert bc.live_catalog_enabled() is False
            assert bc.live_catalog_enabled() is False
        assert caplog.text.count("not a recognised value") == 1

    def test_an_unrecognised_value_makes_no_fetch(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_BATCH_LIVE_CATALOG", "disable-it")
        fetcher = RecordingFetcher()
        source = BatchCatalogSource(fetcher=fetcher, snapshot=_snap("snap/model"))

        assert source.current().origin == "snapshot" and fetcher.calls == []


# ----------------------------------------------------------------------
# review fixes: a garbled or hostile slug is never trusted
# ----------------------------------------------------------------------

OPUS_RULE = r"^anthropic/claude-opus-(?P<version>\d+(?:\.\d+)*)$"


class TestHostileSlugs:
    def test_a_trailing_newline_is_not_a_family_member(self):
        rule = BatchFamilyRule(aliases=("opus",), pattern=OPUS_RULE)
        assert rule.is_member(OPUS_55) and not rule.is_member(OPUS_55 + "\n")
        assert rule.select([OPUS_55, "anthropic/claude-opus-9\n"]) == OPUS_55

    def test_a_non_ascii_digit_is_not_a_version(self):
        rule = BatchFamilyRule(aliases=("opus",), pattern=OPUS_RULE)
        assert rule.rank("anthropic/claude-opus-\u0665") is None  # an Arabic-Indic five
        assert parse_version("\u0665") is None and parse_version("5.\u0665") is None

    @pytest.mark.parametrize("text", ["9" * 30, "99999", "1.2.3.4.5", "1..2", "", ".", "5.", "-1", "1e3", "0x5"])
    def test_an_implausible_version_has_no_value(self, text):
        assert parse_version(text) is None

    def test_a_giant_version_never_outranks_a_real_release(self):
        rule = BatchFamilyRule(aliases=("opus",), pattern=OPUS_RULE)
        assert rule.select([OPUS_55, "anthropic/claude-opus-" + "9" * 23]) == OPUS_55

    def test_real_versions_still_parse(self):
        assert parse_version("5.5") == (5, 5) and parse_version("4.10") == (4, 10) and parse_version("0") == (0,)
        assert parse_version("1.2.3.4") == (1, 2, 3, 4)

    def test_a_batch_listing_with_an_odd_base_slug_is_dropped_from_the_catalog(self):
        payload = payload_for(
            [OPUS_55, "anthropic/claude-opus-9\n", "has space/model", "no-vendor-slash", "vendor/\u0665", "/x"]
        )
        assert parse_models_payload(payload) == frozenset({OPUS_55})

    def test_a_body_whose_only_batch_slugs_are_odd_is_degraded(self):
        with pytest.raises(BatchCatalogFormatError):
            parse_models_payload(payload_for(["a b/c", "x/y\n"], extra_ids=["ok/model"]))

    def test_a_tampered_slug_is_never_submitted_for_an_alias(self, provider):
        install_batch_catalog(
            fetcher=RecordingFetcher(
                payload_for([OPUS_55, "anthropic/claude-opus-" + "9" * 23, "anthropic/claude-opus-9\n"])
            )
        )
        assert provider._resolve_batch_model("opus").slug == OPUS_55

    def test_every_real_batch_slug_in_the_fixture_and_snapshot_is_still_trusted(self):
        assert parse_models_payload(catalog_fixture_payload()) == fixture_slugs()
        snapshot = load_snapshot().slugs
        assert all(bc._SLUG_RE.fullmatch(slug) for slug in snapshot)
        assert parse_models_payload(payload_for(sorted(snapshot))) == snapshot


# ----------------------------------------------------------------------
# review fixes: a colon name OpenRouter does not serve is not a variant of ours
# ----------------------------------------------------------------------


class TestColonNamesThatAreNotOpenRouters:
    @pytest.mark.parametrize("name", ["llama3:8b", "local-llama:3b", "qwen2.5-coder:7b-instruct"])
    def test_an_ollama_style_tag_is_not_explained_as_an_openrouter_variant(self, provider, name):
        assert provider.explain_batch_refusal(name) is None
        assert not provider.can_batch_model(name)
        with pytest.raises(ValueError, match="not served by OpenRouter"):
            provider._resolve_batch_model(name)

    def test_admission_gives_the_generic_refusal_for_such_a_name(self):
        from tools.shared.batch_support import BatchRequestRejected

        registry = _registry(ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER)
        with pytest.raises(BatchRequestRejected) as excinfo:
            select_batch_provider("llama3:8b", registry)

        text = str(excinfo.value)
        assert "variant" not in text and "name the base model" not in text and "'llama3:8b'" in text

    def test_a_variant_of_a_catalog_listed_uncurated_slug_is_still_refused_by_name(self, provider):
        assert OpenRouterModelRegistry().resolve(GEMINI_FLASH) is None
        with pytest.raises(BatchModelRefused, match="':free'"):
            provider._resolve_batch_model(GEMINI_FLASH + ":free")

    def test_the_variant_refusal_says_the_same_call_forwards_the_identifier_unchanged(self, provider):
        with pytest.raises(BatchModelRefused) as excinfo:
            provider._resolve_batch_model("openai/gpt-5:free")
        assert "forwards the identifier unchanged" in str(excinfo.value)
        assert "runs it synchronously" not in str(excinfo.value)


# ----------------------------------------------------------------------
# 15.1 the reduced listing view and its validators (pure)
# ----------------------------------------------------------------------


def _listing(slug: str, *, batch: bool = True) -> dict[str, Any]:
    return fixture_listings()[slug + (":batch" if batch else "")]


def _view(slug: str) -> dict[str, Any]:
    listings = fixture_listings()
    return reduce_listing(listings[slug + ":batch"], listings.get(slug))


class TestListingView:
    def test_a_batch_listing_reduces_to_its_own_limits_and_parameters(self):
        view = _view(OPUS_55)

        assert view["context_length"] == 1_000_000 and view["max_completion_tokens"] == 128_000
        assert "temperature" not in view["supported_parameters"] and "max_tokens" in view["supported_parameters"]
        assert view["input_modalities"] == ["file", "image", "text"] and view["output_modalities"] == ["text"]
        assert view["supported_parameters"] == sorted(view["supported_parameters"])

    def test_parameters_come_from_the_batch_listing_only(self):
        listings = fixture_listings()
        assert "temperature" in listings[GEMINI_FLASH]["supported_parameters"]  # the base listing takes it
        assert "temperature" not in listings[GEMINI_FLASH + ":batch"]["supported_parameters"]

        view = _view(GEMINI_FLASH)

        assert "temperature" not in view["supported_parameters"]
        assert view["context_length"] == 1_048_576 and view["max_completion_tokens"] == 65_536

    def test_a_slug_with_no_base_listing_reduces_from_its_batch_listing_alone(self):
        assert "openai/gpt-4-turbo" not in fixture_listings()  # no base listing at all

        view = reduce_listing(_listing("openai/gpt-4-turbo"), None)

        assert view["context_length"] == 128_000 and view["max_completion_tokens"] == 4_096
        assert "temperature" in view["supported_parameters"] and view["input_modalities"] == ["image", "text"]

    def test_numbers_and_modalities_fall_back_to_the_base_listing_but_parameters_never_do(self):
        batch = {"id": "x/y:batch", "supported_parameters": ["seed"]}  # nothing else
        base = {
            "id": "x/y",
            "context_length": 200_000,
            "top_provider": {"max_completion_tokens": 8_000},
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["temperature", "top_p"],
        }

        view = reduce_listing(batch, base)

        assert view == {
            "context_length": 200_000,
            "max_completion_tokens": 8_000,
            "supported_parameters": ["seed"],
            "input_modalities": ["text"],
            "output_modalities": ["text"],
        }
        assert reduce_listing({"id": "x/y:batch"}, base)["supported_parameters"] is None

    def test_parameters_are_allowlisted_sorted_and_deduplicated_and_an_empty_list_is_valid(self):
        view = reduce_listing({"supported_parameters": ["zeta", "top_p", "temperature", "top_p", "logit_bias"]})
        assert view["supported_parameters"] == ["temperature", "top_p"]
        assert reduce_listing({"supported_parameters": []})["supported_parameters"] == []  # "send none", not "unknown"

    @pytest.mark.parametrize(
        "bad",
        [
            -1,
            0,
            1_023,
            10_000_001,
            10**12,
            1e12,
            "big",
            "1000000",
            True,
            False,
            None,
            [],
            {},
            float("nan"),
            float("inf"),
        ],
    )
    def test_a_hostile_context_length_is_none_for_that_field_alone(self, bad):
        entry = _listing(OPUS_55) | {"context_length": bad}
        view = reduce_listing(entry, None)
        good = reduce_listing(_listing(OPUS_55), None)

        assert view["context_length"] is None  # never clamped to a bound, never repaired
        assert view["supported_parameters"] == good["supported_parameters"]
        assert view["input_modalities"] == good["input_modalities"]
        # the output cap is still checked against the ceiling for an unknown context, and 128,000 is within it
        assert view["max_completion_tokens"] == 128_000

    @pytest.mark.parametrize("bad", [-5, 0, "lots", True, float("nan"), float("inf"), 10**12, 1_000_001, [], None])
    def test_a_hostile_output_cap_is_none_and_an_output_cap_above_the_context_is_rejected(self, bad):
        entry = _listing(OPUS_55)
        entry = entry | {"top_provider": {"max_completion_tokens": bad}}

        view = reduce_listing(entry, None)

        assert view["max_completion_tokens"] is None
        assert view["context_length"] == 1_000_000 and view["supported_parameters"] is not None

    def test_an_output_cap_equal_to_the_context_is_valid_and_one_above_is_not(self):
        base = {"context_length": 4_096, "top_provider": {"max_completion_tokens": 4_096}}
        assert reduce_listing(base, None)["max_completion_tokens"] == 4_096
        over = {"context_length": 4_096, "top_provider": {"max_completion_tokens": 4_097}}
        assert reduce_listing(over, None)["max_completion_tokens"] is None

    @pytest.mark.parametrize(
        "bad",
        ["temperature", {"temperature": True}, ["temperature", 5], ["temperature", None], [""], ["x" * 65], None, 7],
    )
    def test_malformed_supported_parameters_are_none(self, bad):
        view = reduce_listing(_listing(OPUS_55) | {"supported_parameters": bad}, None)
        assert view["supported_parameters"] is None
        assert view["context_length"] == 1_000_000 and view["max_completion_tokens"] == 128_000

    def test_ten_thousand_parameters_are_rejected_whole(self):
        view = reduce_listing({"supported_parameters": [f"p{n}" for n in range(10_000)]})
        assert view["supported_parameters"] is None
        assert reduce_listing({"supported_parameters": ["temperature"] * 256})["supported_parameters"] == [
            "temperature"
        ]
        assert reduce_listing({"supported_parameters": ["temperature"] * 257})["supported_parameters"] is None

    @pytest.mark.parametrize(
        "bad", ["text", [1], ["text", None], ["x" * 33], [f"m{n}" for n in range(17)], {"a": 1}, None]
    )
    def test_malformed_modalities_are_none(self, bad):
        entry = {"architecture": {"input_modalities": bad, "output_modalities": bad}}
        view = reduce_listing(entry, None)
        assert view["input_modalities"] is None and view["output_modalities"] is None

    @pytest.mark.parametrize("entry", [None, 5, "x", [], {}, {"top_provider": 3}, {"architecture": []}])
    def test_nothing_raises_whatever_the_entry_is(self, entry):
        assert reduce_listing(entry, entry) == {
            "context_length": None,
            "max_completion_tokens": None,
            "supported_parameters": None,
            "input_modalities": None,
            "output_modalities": None,
        }

    def test_a_stored_view_is_revalidated_with_the_same_rules(self):
        stored = {
            "context_length": 10**12,
            "max_completion_tokens": 5000,
            "supported_parameters": ["temperature", "nope"],
            "input_modalities": "text",
            "output_modalities": ["text"],
        }
        assert validate_view(stored) == {
            "context_length": None,
            "max_completion_tokens": 5000,
            "supported_parameters": ["temperature"],
            "input_modalities": None,
            "output_modalities": ["text"],
        }
        assert validate_view("not a mapping")["context_length"] is None

    def test_views_merge_field_by_field_in_preference_order(self):
        live = {"context_length": None, "max_completion_tokens": 10, "supported_parameters": []}
        snapshot = {"context_length": 5_000, "max_completion_tokens": 99, "supported_parameters": ["seed"]}

        merged, origin = merge_views([("live", live), ("snapshot", snapshot)])

        assert merged["context_length"] == 5_000 and origin["context_length"] == "snapshot"
        assert merged["max_completion_tokens"] == 10 and origin["max_completion_tokens"] == "live"
        assert merged["supported_parameters"] == [] and origin["supported_parameters"] == "live"  # empty is a value
        assert origin["input_modalities"] == "none" and merged["input_modalities"] is None


# ----------------------------------------------------------------------
# 15.2 / 15.3 the snapshot carries the same view
# ----------------------------------------------------------------------


class TestSnapshotCapabilityView:
    def test_the_committed_model_data_has_one_entry_per_batch_slug_and_equals_the_fixture_reduction(self):
        document = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        assert document["format"] == 2
        assert len(document["batch_models"]) == 92 and sorted(document["model_data"]) == document["batch_models"]

        listings = fixture_listings()
        checked = 0
        for slug, view in document["model_data"].items():
            assert view == reduce_listing(listings[slug + ":batch"], listings.get(slug)), slug
            checked += 1
        assert checked == 92

    def test_the_committed_file_is_small_and_deterministic_in_shape(self):
        text = SNAPSHOT_PATH.read_text(encoding="utf-8")
        assert len(text.encode()) < 40_000 and text.endswith("}\n")
        first_entry = next(line for line in text.splitlines() if line.startswith('    "anthropic/claude-fable-5": '))
        assert ": {" in first_entry and first_entry.rstrip(",").endswith("}")  # one compact line per slug

    def test_no_test_or_source_file_references_the_scratchpad(self):
        for folder in ("tests", "providers", "conf", "scripts"):
            for path in (REPO / folder).rglob("*"):
                if path.suffix in (".py", ".json", ".md") and path.name != "test_batch_dynamic_resolution.py":
                    assert "scratchpad" not in path.read_text(encoding="utf-8", errors="ignore"), path

    def test_the_loaded_snapshot_carries_a_view_for_every_slug(self):
        snapshot = load_snapshot()
        assert snapshot is not None and len(snapshot.views) == 92
        view = snapshot.view_for(OPUS_55)
        assert view is not None and view["context_length"] == 1_000_000
        assert snapshot.view_for("not/listed") is None

    def test_a_format_1_snapshot_loads_with_every_field_missing(self, tmp_path):
        old = tmp_path / "v1.json"
        old.write_text(json.dumps({"fetched_at": "2026-11-21", "batch_models": [OPUS_55, GEMINI_FLASH]}))

        loaded = load_snapshot(old)

        assert loaded is not None and loaded.slugs == frozenset({OPUS_55, GEMINI_FLASH})
        assert loaded.views == {} and loaded.view_for(OPUS_55) is None

    def test_a_hand_edited_value_that_is_out_of_range_falls_back_per_field(self, tmp_path):
        document = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        document["model_data"][OPUS_55]["context_length"] = 10**12  # absurd
        document["model_data"][OPUS_55]["supported_parameters"] = "temperature"  # not a list
        document["model_data"][SONNET_55] = "garbage"  # not even a mapping
        edited = tmp_path / "edited.json"
        edited.write_text(json.dumps(document))

        snapshot = load_snapshot(edited)

        assert snapshot is not None
        view = snapshot.view_for(OPUS_55)
        assert view is not None
        assert view["context_length"] is None and view["supported_parameters"] is None  # rejected, not clamped
        assert view["max_completion_tokens"] == 128_000  # the rest of the entry is still believed
        assert snapshot.view_for(SONNET_55) == validate_view(None)
        # and the provider answers the rejected fields from the next source instead of inventing them
        bc.set_batch_catalog_source(
            BatchCatalogSource(fetcher=RecordingFetcher(error=httpx.ConnectError("x")), snapshot=lambda: snapshot)
        )
        resolution = OpenRouterProvider(api_key="k")._resolve_batch_model("opus")
        assert resolution.capabilities.context_window == 200_000  # the curated entry, per field
        assert resolution.capabilities.max_output_tokens == 128_000  # still from the snapshot
        assert resolution.sources["context_length"] == "curated"
        assert resolution.sources["max_completion_tokens"] == "snapshot"

    def test_a_parsed_body_and_the_snapshot_carry_equal_views_for_every_slug(self):
        live = parse_catalog_payload(catalog_fixture_payload(), origin="live")
        snapshot = load_snapshot()

        assert snapshot is not None and live.slugs == snapshot.slugs
        assert {slug: dict(view) for slug, view in live.views.items()} == {
            slug: dict(view) for slug, view in snapshot.views.items()
        }
