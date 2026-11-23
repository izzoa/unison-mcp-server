"""The synchronous and streaming paths check every model against what is published (D13, D16).

Three things are proved here:

* the tools accept a temperature up to 2 and the model's rule clamps it, with the existing correction text and log
  level (and a value from 0 to 1 produces the identical request on every provider type);
* the output cap of a capability is the smallest of the curated and the published figures, on every provider;
* the check lives at ONE choke point, so simple tools, workflow tools, consensus and the providers' own validation
  all see the same answer.

Request construction is compared BEFORE and AFTER the change by running each provider against a recording client twice:
once with the choke point disabled (the behaviour of the build without the check) and once enabled.
"""

from __future__ import annotations

import logging
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from providers.anthropic import AnthropicModelProvider
from providers.azure_openai import AzureOpenAIProvider
from providers.custom import CustomProvider
from providers.dial import DIALModelProvider
from providers.gemini import GeminiModelProvider
from providers.openai import OpenAIModelProvider
from providers.openrouter import OpenRouterProvider
from providers.shared import ProviderType
from providers.xai import XAIModelProvider
from tools.chat import ChatTool
from utils.model_context import ModelContext

WITHIN_EVERY_RANGE = (0.0, 0.3, 0.7, 1.0)


# ----------------------------------------------------------------------
# A recording client for every provider type
# ----------------------------------------------------------------------


class _Response:
    def __init__(self) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="hello"), finish_reason="stop")]
        self.model = "m"
        self.id = "resp"
        self.created = 0
        self.usage = types.SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        self.output_text = "hello"
        self.output = []


class _Recorder:
    """An OpenAI-style client that remembers the kwargs of the one request it received."""

    def __init__(self) -> None:
        self.sent: dict[str, Any] = {}
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._chat))
        self.responses = types.SimpleNamespace(create=self._responses)

    def _chat(self, **kwargs: Any) -> _Response:
        self.sent = {"api": "chat", **kwargs}
        return _Response()

    def _responses(self, **kwargs: Any) -> _Response:
        self.sent = {"api": "responses", **kwargs}
        return _Response()


def _anthropic_sent(provider: AnthropicModelProvider, model: str, temperature: float, **extra: Any) -> dict[str, Any]:
    stream = MagicMock()
    stream.text_stream = iter(())
    stream.get_final_message.return_value = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text="hello")],
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    manager = MagicMock()
    manager.__enter__.return_value = stream
    manager.__exit__.return_value = False
    client = MagicMock()
    client.messages.stream.return_value = manager
    provider._client = client
    provider.generate_content("hi", model, temperature=temperature, **extra)
    return dict(client.messages.stream.call_args.kwargs)


def _gemini_sent(provider: GeminiModelProvider, model: str, temperature: float, **extra: Any) -> dict[str, Any]:
    client = MagicMock()
    answer = MagicMock()
    answer.text = "hello"
    answer.candidates = []
    answer.usage_metadata = None
    client.models.generate_content.return_value = answer
    provider._client = client
    provider.generate_content("hi", model, temperature=temperature, **extra)
    kwargs = dict(client.models.generate_content.call_args.kwargs)
    config = kwargs.get("config")
    kwargs["config"] = config.model_dump(exclude_none=True) if hasattr(config, "model_dump") else config
    return kwargs


def _compatible_sent(provider: Any, model: str, temperature: float, **extra: Any) -> dict[str, Any]:
    recorder = _Recorder()
    provider._client = recorder
    provider.generate_content("hi", model, temperature=temperature, **extra)
    return recorder.sent


def _dial_sent(provider: DIALModelProvider, model: str, temperature: float, **extra: Any) -> dict[str, Any]:
    recorder = _Recorder()
    provider._get_deployment_client = lambda name: recorder  # type: ignore[method-assign]
    provider.generate_content("hi", model, temperature=temperature, **extra)
    return recorder.sent


def _azure(monkeypatch: pytest.MonkeyPatch) -> AzureOpenAIProvider:
    recorder = _Recorder()
    monkeypatch.setattr("providers.azure_openai.AzureOpenAI", lambda **kwargs: recorder)
    monkeypatch.delenv("AZURE_OPENAI_ALLOWED_MODELS", raising=False)
    provider = AzureOpenAIProvider(
        api_key="key", azure_endpoint="https://example.openai.azure.com/", deployments={"gpt-4o": "prod-gpt4o"}
    )
    provider._recorder = recorder  # type: ignore[attr-defined]
    return provider


def _azure_sent(provider: AzureOpenAIProvider, model: str, temperature: float, **extra: Any) -> dict[str, Any]:
    provider.generate_content("hi", model, temperature=temperature, **extra)
    return provider._recorder.sent  # type: ignore[attr-defined,no-any-return]


#: (id, provider factory, sender, model): one row per provider type, including the OpenAI-compatible ones.
CASES = [
    ("anthropic", lambda mp: AnthropicModelProvider(api_key="sk-ant-test"), _anthropic_sent, "claude-sonnet-4-6"),
    ("anthropic-haiku", lambda mp: AnthropicModelProvider(api_key="sk-ant-test"), _anthropic_sent, "claude-haiku-4-5"),
    ("gemini", lambda mp: GeminiModelProvider("test-key"), _gemini_sent, "gemini-2.5-pro"),
    ("openai", lambda mp: OpenAIModelProvider("test-key"), _compatible_sent, "gpt-4.1"),
    ("xai", lambda mp: XAIModelProvider("test-key"), _compatible_sent, "grok-4"),
    (
        "openrouter-claude",
        lambda mp: OpenRouterProvider(api_key="test-key"),
        _compatible_sent,
        "anthropic/claude-opus-4.5",
    ),
    ("openrouter-gemini", lambda mp: OpenRouterProvider(api_key="test-key"), _compatible_sent, "google/gemini-2.5-pro"),
    (
        "custom",
        lambda mp: CustomProvider(api_key="", base_url="http://localhost:11434/v1"),
        _compatible_sent,
        "llama3.2",
    ),
    ("dial", lambda mp: DIALModelProvider("test-key"), _dial_sent, "anthropic.claude-sonnet-4.1-20250805-v1:0"),
    ("azure", _azure, _azure_sent, "gpt-4o"),
]
IDS = [case[0] for case in CASES]


def _disable_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.base as base

    monkeypatch.setattr(base, "apply_published_limits", lambda provider, identifier, capabilities: capabilities)


class TestARequestWithinEveryLimitIsByteIdentical:
    @pytest.mark.parametrize(("name", "factory", "sender", "model"), CASES, ids=IDS)
    def test_the_request_is_the_one_built_without_the_check(self, monkeypatch, name, factory, sender, model):
        after: dict[float, Any] = {}
        for temperature in WITHIN_EVERY_RANGE:
            after[temperature] = sender(factory(monkeypatch), model, temperature)

        _disable_the_check(monkeypatch)
        for temperature in WITHIN_EVERY_RANGE:
            before = sender(factory(monkeypatch), model, temperature)
            assert after[temperature] == before, f"{name} at {temperature}"


# ----------------------------------------------------------------------
# A temperature above a model's maximum is corrected by the existing machinery
# ----------------------------------------------------------------------


def _validate(provider: Any, model: str, temperature: float) -> tuple[float, list[str]]:
    capabilities = provider.get_capabilities(model)
    return ChatTool().validate_and_correct_temperature(temperature, ModelContext(model, provider, capabilities))


class TestAboveAModelsMaximumIsCorrected:
    @pytest.mark.parametrize(
        ("factory", "model"),
        [
            (lambda: AnthropicModelProvider(api_key="sk-ant-test"), "claude-sonnet-4-6"),
            (lambda: AnthropicModelProvider(api_key="sk-ant-test"), "claude-haiku-4-5"),
            (lambda: OpenRouterProvider(api_key="test-key"), "anthropic/claude-opus-4.5"),
            (lambda: OpenRouterProvider(api_key="test-key"), "anthropic/claude-sonnet-4.5"),
            (lambda: OpenRouterProvider(api_key="test-key"), "anthropic/claude-opus-4.1"),
            (lambda: DIALModelProvider("test-key"), "anthropic.claude-opus-4.1-20250805-v1:0"),
            (lambda: DIALModelProvider("test-key"), "anthropic.claude-sonnet-4.1-20250805-v1:0-with-thinking"),
        ],
    )
    def test_a_model_whose_range_ends_at_one_corrects_with_the_existing_text(self, factory, model, caplog):
        corrected, warnings = _validate(factory(), model, 1.5)

        assert corrected == 1.0
        assert len(warnings) == 1
        assert warnings[0] == (
            f"Temperature 1.5 invalid for {model}. Supports temperature range [0.0, 1.0]. Using 1.0 instead."
        )

    @pytest.mark.parametrize(
        ("factory", "model"),
        [
            (lambda: OpenAIModelProvider("test-key"), "gpt-4.1"),
            (lambda: OpenAIModelProvider("test-key"), "gpt-5-codex"),
            (lambda: GeminiModelProvider("test-key"), "gemini-2.5-pro"),
            (lambda: GeminiModelProvider("test-key"), "gemini-3-pro-preview"),
            (lambda: XAIModelProvider("test-key"), "grok-4"),
            (lambda: OpenRouterProvider(api_key="test-key"), "google/gemini-2.5-pro"),
            (lambda: OpenRouterProvider(api_key="test-key"), "openai/gpt-5"),
            (lambda: CustomProvider(api_key="", base_url="http://localhost:11434/v1"), "llama3.2"),
            # a vendor outside the table: its guessed "0..1" row never narrows a range somebody reviewed (D16 order)
            (lambda: OpenRouterProvider(api_key="test-key"), "mistralai/mistral-large-2411"),
            (lambda: OpenRouterProvider(api_key="test-key"), "meta-llama/llama-3-70b"),
            (lambda: DIALModelProvider("test-key"), "gemini-2.5-pro-preview-05-06"),
        ],
    )
    def test_a_model_whose_range_reaches_two_passes_it_through(self, factory, model):
        assert _validate(factory(), model, 1.5) == (1.5, [])
        assert _validate(factory(), model, 2.0) == (2.0, [])

    def test_the_corrected_value_is_what_the_request_carries(self, monkeypatch):
        provider = OpenRouterProvider(api_key="test-key")
        corrected, _ = _validate(provider, "anthropic/claude-opus-4.5", 1.5)
        assert _compatible_sent(provider, "anthropic/claude-opus-4.5", corrected)["temperature"] == 1.0

    def test_the_provider_itself_refuses_what_the_model_cannot_take(self):
        """A direct caller that skips the tools' correction now gets the provider's own ``ValueError`` for a
        Claude model above 1.0 (before, the reviewed 0..2 range let it through to an HTTP 400)."""

        provider = AnthropicModelProvider(api_key="sk-ant-test")
        with pytest.raises(ValueError, match="[Tt]emperature"):
            provider.validate_parameters("claude-sonnet-4-6", 1.5)
        provider.validate_parameters("claude-sonnet-4-6", 1.0)

    def test_the_correction_is_logged_at_warning_by_the_tool_layer(self, caplog):
        """Existing behaviour, unchanged: the tool layer logs the correction text it receives at WARNING."""

        provider = OpenRouterProvider(api_key="test-key")
        _, warnings = _validate(provider, "anthropic/claude-opus-4.5", 1.5)
        with caplog.at_level(logging.WARNING):
            logging.getLogger("tools.simple.base").warning(warnings[0])
        assert any("Using 1.0 instead" in r.getMessage() for r in caplog.records)


class TestTheToolsAcceptTheWidestRange:
    def test_the_schemas_say_two_from_one_constant(self):
        from tools.analyze import AnalyzeTool
        from tools.chat import ChatTool as Chat
        from tools.shared.base_models import COMMON_FIELD_DESCRIPTIONS, MAX_TEMPERATURE

        assert MAX_TEMPERATURE == 2.0
        chat = Chat().get_input_schema()["properties"]["temperature"]
        analyze = AnalyzeTool().get_input_schema()["properties"]["temperature"]
        for schema in (chat, analyze):
            assert (schema["minimum"], schema["maximum"]) == (0, 2.0)
            assert schema["description"] == COMMON_FIELD_DESCRIPTIONS["temperature"]
        assert "published" in COMMON_FIELD_DESCRIPTIONS["temperature"]

    def test_every_request_model_agrees(self):
        from pydantic import ValidationError

        from tools.shared.base_models import ToolRequest, WorkflowRequest
        from tools.thinkdeep import ThinkDeepWorkflowRequest

        for model in (ToolRequest, WorkflowRequest):
            assert (
                model(
                    **(
                        {
                            "temperature": 2.0,
                            "step": "s",
                            "step_number": 1,
                            "total_steps": 1,
                            "next_step_required": False,
                            "findings": "f",
                        }
                        if model is WorkflowRequest
                        else {"temperature": 2.0}
                    )
                ).temperature
                == 2.0
            )
        assert (
            ThinkDeepWorkflowRequest(
                step="s", step_number=1, total_steps=1, next_step_required=False, findings="f", temperature=2.0
            ).temperature
            == 2.0
        )
        with pytest.raises(ValidationError):
            ToolRequest(temperature=2.5)


# ----------------------------------------------------------------------
# The output cap, on every provider
# ----------------------------------------------------------------------


class TestTheOutputCapOnEveryProvider:
    def test_a_capability_is_lowered_to_what_is_published_and_the_registry_stays_curated(self):
        provider = OpenAIModelProvider("test-key")
        capabilities = provider.get_capabilities("gpt-5.2-pro")
        assert capabilities.max_output_tokens == 128_000
        assert provider.get_all_model_capabilities()["gpt-5.2-pro"].max_output_tokens == 272_000
        assert provider.get_capabilities("gpt-5.2-pro") is capabilities

    def test_gemini_takes_the_smaller_of_the_two_native_keys(self):
        # gemini/ is absent and vertex_ai/ says 65,535 against the reviewed 65,536
        assert GeminiModelProvider("test-key").get_capabilities("gemini-3-pro-preview").max_output_tokens == 65_535

    def test_a_published_figure_above_the_reviewed_cap_changes_nothing(self):
        provider = AnthropicModelProvider(api_key="sk-ant-test")
        registry_entry = provider.get_all_model_capabilities()["claude-sonnet-4-6"]
        # the pinned extract publishes 128,000 for the model; the reviewed cap is 64,000
        assert provider.get_capabilities("claude-sonnet-4-6").max_output_tokens == 64_000
        assert registry_entry.max_output_tokens == 64_000

    def test_anthropic_sends_the_bounded_default_cap(self, monkeypatch):
        provider = AnthropicModelProvider(api_key="sk-ant-test")
        assert _anthropic_sent(provider, "claude-sonnet-4-6", 1.0)["max_tokens"] == 64_000

    @pytest.mark.parametrize(
        ("factory", "sender", "model", "key", "bound"),
        [
            (
                lambda: AnthropicModelProvider(api_key="sk-ant-test"),
                _anthropic_sent,
                "claude-sonnet-4-6",
                "max_tokens",
                64_000,
            ),
            (lambda: OpenAIModelProvider("test-key"), _compatible_sent, "gpt-4.1", "max_tokens", 32_768),
            (
                lambda: OpenRouterProvider(api_key="test-key"),
                _compatible_sent,
                "anthropic/claude-opus-4.1",
                "max_tokens",
                64_000,  # the reviewed cap: the snapshot's 32,000 does not bound a synchronous request (D17)
            ),
            (
                lambda: OpenRouterProvider(api_key="test-key"),
                _compatible_sent,
                "anthropic/claude-opus-4.5",
                "max_tokens",
                64_000,
            ),
            (lambda: XAIModelProvider("test-key"), _compatible_sent, "grok-4", "max_tokens", 256_000),
        ],
    )
    def test_an_explicit_request_above_the_effective_cap_is_bounded_and_one_at_or_below_is_not(
        self, factory, sender, model, key, bound, caplog
    ):
        with caplog.at_level(logging.WARNING):
            sent = sender(factory(), model, 0.5, max_output_tokens=50_000_000)
            sender(factory(), model, 0.5, max_output_tokens=50_000_000)
        assert sent[key] == bound
        assert len([r for r in caplog.records if "is reduced to it" in r.getMessage()]) == 1  # once per model and cap
        assert sender(factory(), model, 0.5, max_output_tokens=1_000)[key] == 1_000

    def test_gemini_bounds_an_explicit_request_to_the_smaller_published_figure(self):
        sent = _gemini_sent(GeminiModelProvider("test-key"), "gemini-3-pro-preview", 0.5, max_output_tokens=500_000)
        assert sent["config"]["max_output_tokens"] == 65_535

    def test_a_request_that_names_no_cap_sends_none_where_the_provider_sent_none(self):
        assert "max_tokens" not in _compatible_sent(OpenAIModelProvider("test-key"), "gpt-4.1", 0.5)

    def test_dial_and_custom_keep_their_curated_caps_and_only_a_named_anthropic_vendor_is_narrowed(self):
        dial = DIALModelProvider("test-key")
        for name, capabilities in dial.get_all_model_capabilities().items():
            got = dial.get_capabilities(name)
            assert got.max_output_tokens == capabilities.max_output_tokens
            if name.startswith("anthropic."):
                assert (got.temperature_constraint.min_temp, got.temperature_constraint.max_temp) == (0.0, 1.0)
            else:
                assert got is capabilities
        custom = CustomProvider(api_key="", base_url="http://localhost:11434/v1")
        assert (
            custom.get_capabilities("llama3.2").max_output_tokens
            == custom.get_all_model_capabilities()["llama3.2"].max_output_tokens
        )


class TestTheSnapshotDoesNotBoundASynchronousRequest:
    """D17: on the synchronous and streaming paths the published output cap is LiteLLM's figure alone.

    The committed OpenRouter snapshot stores the ``:batch`` listing's ``max_completion_tokens``, which can sit far
    below what the plain listing accepts (``moonshotai/kimi-k3``: 16,384 against 943,718). It still bounds a BATCH
    request (``tests/test_published_limits_batch.py::TestTheSnapshotStillBoundsABatchRequest``).
    """

    @staticmethod
    def _snapshot_figure(slug: str) -> int:
        import json
        from pathlib import Path

        snapshot = json.loads((Path(__file__).parent.parent / "conf" / "openrouter_batch_catalog.json").read_text())
        return snapshot["model_data"][slug]["max_completion_tokens"]

    @staticmethod
    def _litellm_publishes(slug: str, figure: int) -> None:
        from providers.shared import published_limits as pl

        pl.set_published_index(
            pl.PublishedIndex.from_document(
                {"entries": {f"openrouter/{slug}": {"provider": "openrouter", "max_output_tokens": figure}}}
            )
        )

    def test_a_slug_whose_snapshot_cap_is_below_litellms_is_not_cut_to_the_snapshot(self):
        slug = "moonshotai/kimi-k3"
        assert self._snapshot_figure(slug) == 16_384  # the premise: the committed figure is the low one
        provider = OpenRouterProvider(api_key="test-key")
        # the pinned extract publishes 1,048,576 for it (LiteLLM's vendor key ``moonshot/kimi-k3``)
        capabilities = provider.get_capabilities(slug)
        assert getattr(capabilities, "_published_output_cap", None) == 1_048_576  # LiteLLM's figure, not 16,384
        assert _compatible_sent(provider, slug, 0.5, max_output_tokens=100_000)["max_tokens"] == 100_000
        # and a request above LiteLLM's own figure is still bounded, by LiteLLM
        assert _compatible_sent(provider, slug, 0.5, max_output_tokens=50_000_000)["max_tokens"] == 1_048_576

    def test_a_slug_the_snapshot_cuts_but_litellm_does_not_publish_is_not_cut_at_all(self):
        from providers.shared import published_limits as pl

        slug = "moonshotai/kimi-k3"
        assert self._snapshot_figure(slug) == 16_384
        pl.set_published_index(pl.PublishedIndex.from_document({"entries": {}}))  # LiteLLM says nothing
        provider = OpenRouterProvider(api_key="test-key")
        assert getattr(provider.get_capabilities(slug), "_published_output_cap", None) is None
        assert _compatible_sent(provider, slug, 0.5, max_output_tokens=100_000)["max_tokens"] == 100_000

    def test_a_reviewed_cap_above_the_snapshot_figure_is_kept(self):
        """claude-opus-4.1: reviewed 64,000, snapshot 32,000, LiteLLM has no bare key. Recorded in D17 as lost."""

        slug = "anthropic/claude-opus-4.1"
        assert self._snapshot_figure(slug) == 32_000
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.get_capabilities(slug).max_output_tokens == 64_000
        assert _compatible_sent(provider, slug, 0.5, max_output_tokens=50_000)["max_tokens"] == 50_000

    def test_litellm_still_lowers_a_curated_openrouter_cap(self):
        """The one source that remains still bounds: LiteLLM's openrouter key under the reviewed 272,000 (the curated OpenRouter cap of the model)."""

        provider = OpenRouterProvider(api_key="test-key")
        self._litellm_publishes("openai/gpt-5.2-pro", 100_000)
        assert provider.get_capabilities("openai/gpt-5.2-pro").max_output_tokens == 100_000

    def test_the_only_source_on_the_synchronous_path_is_litellm(self):
        from providers.shared import published_limits as pl

        slug = "moonshotai/kimi-k3"
        pl.set_published_index(pl.PublishedIndex.from_document({"entries": {}}))
        assert pl.published_output_caps(ProviderType.OPENROUTER, slug) == []
        self._litellm_publishes(slug, 500_000)
        assert pl.published_output_caps(ProviderType.OPENROUTER, slug) == [("litellm", 500_000)]


class TestWhichReviewedEntriesTheSynchronousPathLowers:
    """The set quoted in D16, D17, docs/tools/batch.md and the CHANGELOG, derived rather than asserted by hand.

    Every reviewed entry is lowered by LiteLLM's figure alone (D17). Entries a provider auto-discovers from LiteLLM
    itself (``Auto-discovered via LiteLLM``, for instance the Gemini ``deep-research-pro-preview-12-2025`` and
    ``gemini-exp-1206``, whose two LiteLLM keys disagree) are not reviewed entries and are left out of the count. If
    this set changes (a registry edit, a refreshed pinned extract), those four places change with it.
    """

    EXPECTED = {
        (ProviderType.OPENAI, "gpt-5.2-pro", 272_000, 128_000),
        (ProviderType.OPENROUTER, "openai/gpt-5.2-pro", 272_000, 128_000),
        (ProviderType.GOOGLE, "gemini-3-pro-preview", 65_536, 65_535),
        (ProviderType.OPENROUTER, "google/gemini-3-pro-preview", 65_536, 65_535),
    }

    def test_exactly_these_reviewed_entries_are_lowered_and_none_by_the_snapshot(self):
        providers = [
            OpenRouterProvider(api_key="test-key"),
            OpenAIModelProvider("test-key"),
            GeminiModelProvider("test-key"),
            XAIModelProvider("test-key"),
            AnthropicModelProvider(api_key="sk-ant-test"),
            DIALModelProvider("test-key"),
        ]
        lowered = set()
        for provider in providers:
            for name, reviewed in provider.get_all_model_capabilities().items():
                if reviewed.description.startswith("Auto-discovered"):
                    continue
                effective = provider.get_capabilities(name).max_output_tokens
                if effective != reviewed.max_output_tokens:
                    lowered.add((provider.get_provider_type(), name, reviewed.max_output_tokens, effective))
        assert lowered == self.EXPECTED
        # the one entry the snapshot used to lower on this path (D17) keeps its reviewed cap
        assert (ProviderType.OPENROUTER, "anthropic/claude-opus-4.1", 64_000, 32_000) not in lowered


class TestAStandInCapIsNeverEnforced:
    """OpenRouter's generic capability for an uncurated slug carries a made-up 32,768 output cap."""

    def test_a_request_above_the_stand_in_goes_through_as_sent(self):
        provider = OpenRouterProvider(api_key="test-key")
        sent = _compatible_sent(provider, "some-vendor/unlisted-model", 0.7, max_output_tokens=100_000)
        assert sent["max_tokens"] == 100_000

    def test_the_stand_in_is_still_a_stand_in_after_the_check(self):
        capabilities = OpenRouterProvider(api_key="test-key").get_capabilities("some-vendor/unlisted-model")
        assert capabilities._is_generic and getattr(capabilities, "_published_output_cap", None) is None

    def test_a_published_figure_for_an_uncurated_slug_does_bound_the_request(self, monkeypatch):
        from providers.shared import published_limits as pl

        pl.set_published_index(
            pl.PublishedIndex.from_document(
                {
                    "entries": {
                        "openrouter/some-vendor/unlisted-model": {"provider": "openrouter", "max_output_tokens": 8192}
                    }
                }
            )
        )
        provider = OpenRouterProvider(api_key="test-key")
        assert (
            _compatible_sent(provider, "some-vendor/unlisted-model", 0.7, max_output_tokens=100_000)["max_tokens"]
            == 8192
        )
        assert (
            _compatible_sent(provider, "some-vendor/unlisted-model", 0.7, max_output_tokens=4096)["max_tokens"] == 4096
        )

    def test_an_uncurated_vendor_outside_the_table_gets_the_fallback_range_for_its_stand_in(self):
        capabilities = OpenRouterProvider(api_key="test-key").get_capabilities("qwen/qwen-not-in-registry-9")
        assert capabilities._is_generic
        assert (capabilities.temperature_constraint.min_temp, capabilities.temperature_constraint.max_temp) == (
            0.0,
            1.0,
        )


class TestOneChokePoint:
    def test_every_native_provider_inherits_the_check_from_get_capabilities(self):
        import inspect

        from providers.base import ModelProvider

        source = inspect.getsource(ModelProvider.get_capabilities)
        assert "apply_published_limits" in source
        for cls in (
            AnthropicModelProvider,
            AzureOpenAIProvider,
            CustomProvider,
            DIALModelProvider,
            GeminiModelProvider,
            OpenAIModelProvider,
            OpenRouterProvider,
            XAIModelProvider,
        ):
            owner = next(c for c in cls.__mro__ if "get_capabilities" in vars(c))
            assert owner in (ModelProvider, AzureOpenAIProvider), cls  # Azure delegates to super().get_capabilities

    def test_the_tools_see_the_same_capability_as_the_provider(self):
        provider = AnthropicModelProvider(api_key="sk-ant-test")
        context = ModelContext("claude-sonnet-4-6", provider, provider.get_capabilities("claude-sonnet-4-6"))
        assert context.capabilities is provider.get_capabilities("claude-sonnet-4-6")
        assert context.capabilities.temperature_constraint.max_temp == 1.0

    def test_workflow_and_consensus_tools_validate_against_the_same_rule(self):
        from tools.analyze import AnalyzeTool
        from tools.consensus import ConsensusTool

        provider = OpenRouterProvider(api_key="test-key")
        context = ModelContext(
            "anthropic/claude-opus-4.5", provider, provider.get_capabilities("anthropic/claude-opus-4.5")
        )
        for tool in (AnalyzeTool(), ConsensusTool(), ChatTool()):
            corrected, warnings = tool.validate_and_correct_temperature(1.8, context)
            assert corrected == 1.0 and "Using 1.0 instead" in warnings[0]


class TestTemperatureOfALiteLLMFixedModel:
    def test_the_sampling_flag_makes_an_anthropic_model_take_temperature_one_only_end_to_end(self):
        from providers.shared import published_limits as pl

        provider = AnthropicModelProvider(api_key="sk-ant-test")
        assert provider.get_capabilities("claude-sonnet-4-6").supports_temperature  # sampling is allowed today

        pl.set_published_index(
            pl.PublishedIndex.from_document(
                {
                    "entries": {
                        "claude-sonnet-4-6": {
                            "provider": "anthropic",
                            "max_output_tokens": 64_000,
                            "supports_sampling_params": False,
                        }
                    }
                }
            )
        )
        flagged = AnthropicModelProvider(api_key="sk-ant-test")
        capabilities = flagged.get_capabilities("claude-sonnet-4-6")
        assert not capabilities.supports_temperature and capabilities.get_effective_temperature(0.2) is None
        sent = _anthropic_sent(flagged, "claude-sonnet-4-6", 0.4, thinking_mode="minimal")
        assert "temperature" not in sent  # nothing is sent
        corrected, warnings = _validate(flagged, "claude-sonnet-4-6", 0.4)
        assert corrected == 1.0 and "Only supports temperature=1.0" in warnings[0]


def test_the_provider_types_this_suite_covers_are_all_of_them():
    assert {
        ProviderType.ANTHROPIC,
        ProviderType.GOOGLE,
        ProviderType.OPENAI,
        ProviderType.XAI,
        ProviderType.AZURE,
        ProviderType.DIAL,
        ProviderType.CUSTOM,
        ProviderType.OPENROUTER,
    } == set(ProviderType)
