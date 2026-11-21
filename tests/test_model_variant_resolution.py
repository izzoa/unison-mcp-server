"""Provider-side resolution of variant identifiers (openspec: fix-openrouter-model-variants)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.custom import CustomProvider
from providers.openrouter import OpenRouterProvider
from providers.shared import ModelCapabilities, ProviderType
from providers.shared.model_response import ModelResponse
from tools.chat import ChatTool
from tools.consensus import ConsensusTool

GENERIC_CONTEXT_WINDOW = 32_768  # OpenRouter's fallback for an uncatalogued provider/model


@pytest.fixture
def openrouter():
    return OpenRouterProvider(api_key="test-key")


@pytest.fixture
def custom():
    return CustomProvider(api_key="", base_url="http://localhost:11434/v1")


class TestOpenRouterVariantInheritance:
    BASE = "anthropic/claude-opus-4.1"

    def test_base_is_catalogued_with_a_non_generic_context_window(self, openrouter):
        base = openrouter.get_capabilities(self.BASE)
        assert not hasattr(base, "_is_generic")
        assert base.context_window != GENERIC_CONTEXT_WINDOW  # otherwise the test proves nothing

    @pytest.mark.parametrize("variant", ["batch", "free", "extended", "exacto", "thinking", "online", "nitro", "floor"])
    def test_unlisted_variant_inherits_base_capabilities(self, openrouter, variant):
        base = openrouter.get_capabilities(self.BASE)
        caps = openrouter.get_capabilities(f"{self.BASE}:{variant}")

        assert caps.context_window == base.context_window
        assert caps.max_output_tokens == base.max_output_tokens
        assert caps.supports_extended_thinking == base.supports_extended_thinking
        assert not hasattr(caps, "_is_generic")
        # Re-keyed to the identifier asked for, so the variant is what gets sent.
        assert caps.model_name == f"{self.BASE}:{variant}"

    def test_context_window_is_not_silently_reduced(self, openrouter):
        caps = openrouter.get_capabilities(f"{self.BASE}:batch")
        assert caps.context_window > GENERIC_CONTEXT_WINDOW

    def test_inheritance_does_not_mutate_the_catalog_entry(self, openrouter):
        openrouter.get_capabilities(f"{self.BASE}:batch")
        assert openrouter.get_capabilities(self.BASE).model_name == self.BASE

    def test_catalogued_variant_entry_takes_precedence(self, openrouter):
        listed = ModelCapabilities(
            provider=ProviderType.OPENROUTER,
            model_name=f"{self.BASE}:batch",
            friendly_name="OpenRouter (variant)",
            context_window=123_456,
            max_output_tokens=7_890,
        )
        base = openrouter.get_capabilities(self.BASE)
        registry = MagicMock()
        registry.get_capabilities.side_effect = lambda name: {listed.model_name: listed, self.BASE: base}.get(name)
        registry.resolve.side_effect = registry.get_capabilities.side_effect
        openrouter._registry = registry  # instance attribute shadows the class-level registry

        caps = openrouter.get_capabilities(f"{self.BASE}:batch")
        assert caps.context_window == 123_456

    def test_alias_with_variant_is_not_given_capabilities(self, openrouter):
        """``opus:batch`` is not an identifier OpenRouter serves; inheriting through an alias would hide that."""
        assert openrouter.get_capabilities("opus").model_name == "anthropic/claude-opus-4.5"
        with pytest.raises(ValueError, match="Unsupported model"):
            openrouter.get_capabilities("opus:batch")

    def test_uncatalogued_base_does_not_inherit_from_an_unrelated_model(self, openrouter):
        caps = openrouter.get_capabilities("some-vendor/unlisted-model:batch")
        assert getattr(caps, "_is_generic", False)  # unchanged generic path, not another model's limits
        assert caps.context_window == GENERIC_CONTEXT_WINDOW


class TestDiscountedVariantIsNeverResolvedToItsBase:
    """The case that motivated the change: ``:batch`` is priced at 50% of the base model."""

    BASE = "anthropic/claude-opus-4.1"

    def test_openrouter_forwards_the_batch_identifier_whole(self, openrouter):
        assert openrouter._resolve_model_name(f"{self.BASE}:batch") == f"{self.BASE}:batch"

    def test_openrouter_capabilities_are_keyed_to_the_batch_identifier(self, openrouter):
        assert openrouter.get_capabilities(f"{self.BASE}:batch").model_name == f"{self.BASE}:batch"

    @pytest.mark.parametrize("variant", ["batch", "floor", "online", "thinking", "some-future-variant"])
    def test_no_variant_is_dropped_by_resolution(self, openrouter, variant):
        identifier = f"{self.BASE}:{variant}"
        assert openrouter._resolve_model_name(identifier) == identifier

    @pytest.mark.asyncio
    async def test_the_provider_request_carries_the_batch_identifier(self, openrouter):
        """End to end through the real request path: the model sent to the API keeps ``:batch``."""
        sent = {}

        def fake_create(**kwargs):
            sent.update(kwargs)
            completion = MagicMock()
            completion.choices = [MagicMock(message=MagicMock(content="ok"), finish_reason="stop")]
            completion.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            completion.model = kwargs["model"]
            return completion

        client = MagicMock()
        client.chat.completions.create.side_effect = fake_create
        openrouter._client = client

        await asyncio.to_thread(
            openrouter.generate_content,
            prompt="hi",
            model_name=f"{self.BASE}:batch",
            temperature=0.5,
        )

        assert sent["model"] == f"{self.BASE}:batch"


class TestCustomTaggedIdentifiers:
    def test_tagged_identifier_is_forwarded_whole(self, custom):
        assert custom._resolve_model_name("llama3.2:3b") == "llama3.2:3b"
        assert custom._resolve_model_name("llama3.2:latest") == "llama3.2:latest"

    def test_tagged_identifier_inherits_the_declared_base_capabilities(self, custom):
        base = custom.get_capabilities("llama3.2")
        caps = custom.get_capabilities("llama3.2:3b")
        assert caps.model_name == "llama3.2:3b"
        assert caps.context_window == base.context_window
        assert caps.provider == ProviderType.CUSTOM

    def test_tagged_identifier_validates(self, custom):
        assert custom.validate_model_name("llama3.2:3b")

    def test_alias_in_front_of_a_tag_resolves_and_keeps_the_tag(self, custom):
        assert custom._resolve_model_name("local-llama:3b") == "llama3.2:3b"
        assert custom.validate_model_name("local-llama:3b")

    def test_undeclared_tagged_model_is_forwarded_whole_not_substituted(self, custom):
        assert custom._resolve_model_name("mystery-model:7b") == "mystery-model:7b"
        assert not custom.validate_model_name("mystery-model:7b")  # fails visibly; no other model substituted

    def test_untagged_resolution_is_unchanged(self, custom):
        assert custom._resolve_model_name("local-llama") == "llama3.2"
        assert custom._resolve_model_name("llama3.2") == "llama3.2"

    def test_openrouter_variant_is_not_claimed_by_the_custom_provider(self, custom):
        assert not custom.validate_model_name("anthropic/claude-opus-4.1:batch")


class TestResolutionDifferenceIsReported:
    def _model_info(self, requested, resolved):
        provider = MagicMock()
        provider.get_provider_type.return_value = ProviderType.GOOGLE
        return {
            "provider": provider,
            "model_name": requested,
            "model_response": ModelResponse(content="answer", model_name=resolved, provider=ProviderType.GOOGLE),
        }

    def test_simple_tool_metadata_names_both_when_they_differ(self):
        tool = ChatTool()
        request = MagicMock()
        with (
            patch.object(tool, "_create_continuation_offer", return_value=None),
            patch.object(tool, "get_request_continuation_id", return_value=None),
            patch.object(tool, "format_response", return_value="answer"),
        ):
            output = tool._parse_response("answer", request, self._model_info("flash", "gemini-2.5-flash"))

        assert output.metadata["model_used"] == "flash"
        assert output.metadata["model_requested"] == "flash"
        assert output.metadata["model_resolved"] == "gemini-2.5-flash"

    def test_simple_tool_metadata_is_silent_when_nothing_was_resolved(self):
        tool = ChatTool()
        request = MagicMock()
        with (
            patch.object(tool, "_create_continuation_offer", return_value=None),
            patch.object(tool, "get_request_continuation_id", return_value=None),
            patch.object(tool, "format_response", return_value="answer"),
        ):
            output = tool._parse_response("answer", request, self._model_info("gemini-2.5-flash", "gemini-2.5-flash"))

        assert "model_requested" not in output.metadata
        assert "model_resolved" not in output.metadata

    def test_workflow_metadata_names_both_when_they_differ(self):
        from tools.debug import DebugIssueTool
        from utils.tool_execution_context import ToolExecutionContext

        provider = MagicMock()
        provider.get_provider_type.return_value = ProviderType.GOOGLE
        model_context = MagicMock()
        model_context.provider = provider
        model_context.capabilities.model_name = "gemini-2.5-flash"
        arguments = {"_context": ToolExecutionContext(model_context=model_context, resolved_model_name="flash")}

        response = {}
        DebugIssueTool()._add_workflow_metadata(response, arguments)

        assert response["metadata"]["model_requested"] == "flash"
        assert response["metadata"]["model_resolved"] == "gemini-2.5-flash"

    def test_workflow_metadata_is_silent_when_nothing_was_resolved(self):
        from tools.debug import DebugIssueTool
        from utils.tool_execution_context import ToolExecutionContext

        provider = MagicMock()
        provider.get_provider_type.return_value = ProviderType.GOOGLE
        model_context = MagicMock()
        model_context.provider = provider
        model_context.capabilities.model_name = "gemini-2.5-flash"
        arguments = {
            "_context": ToolExecutionContext(model_context=model_context, resolved_model_name="gemini-2.5-flash")
        }

        response = {}
        DebugIssueTool()._add_workflow_metadata(response, arguments)

        assert "model_requested" not in response["metadata"]

    @pytest.mark.asyncio
    async def test_consensus_entry_metadata_names_both_when_they_differ(self):
        tool = ConsensusTool()
        provider = MagicMock()
        provider.get_provider_type.return_value = ProviderType.GOOGLE
        response = ModelResponse(content="verdict", model_name="gemini-2.5-flash", provider=ProviderType.GOOGLE)
        request = MagicMock(relevant_files=None, images=None)

        with (
            patch.object(tool, "get_model_provider", return_value=provider),
            patch.object(tool, "validate_and_correct_temperature", return_value=(0.2, [])),
            patch("utils.observability.instrumented_async_generate", new=AsyncMock(return_value=response)),
        ):
            result = await tool._consult_model({"model": "flash", "stance": "for"}, request)

        assert result["status"] == "success"
        assert result["metadata"]["model_requested"] == "flash"
        assert result["metadata"]["model_resolved"] == "gemini-2.5-flash"


class TestConsensusStanceComesOnlyFromItsField:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "identifier", ["o3:for", "gemini-2.5-pro:against", "vendor/model:neutral", "flash:against"]
    )
    async def test_stance_like_identifier_does_not_acquire_that_stance(self, identifier):
        tool = ConsensusTool()
        provider = MagicMock()
        provider.get_provider_type.return_value = ProviderType.GOOGLE
        response = ModelResponse(content="verdict", model_name=identifier, provider=ProviderType.GOOGLE)
        request = MagicMock(relevant_files=None, images=None)
        seen = {}

        def capture_prompt(stance, custom=None):
            seen["stance"] = stance
            return "system prompt"

        with (
            patch.object(tool, "get_model_provider", return_value=provider) as get_provider,
            patch.object(tool, "validate_and_correct_temperature", return_value=(0.2, [])),
            patch.object(tool, "_get_stance_enhanced_prompt", side_effect=capture_prompt),
            patch("utils.observability.instrumented_async_generate", new=AsyncMock(return_value=response)),
        ):
            result = await tool._consult_model({"model": identifier}, request)  # no stance field supplied

        assert seen["stance"] == "neutral"  # the default, not the identifier's trailing segment
        assert result["stance"] == "neutral"
        assert result["model"] == identifier
        get_provider.assert_called_once_with(identifier)

    @pytest.mark.asyncio
    async def test_explicit_stance_field_is_still_honoured(self):
        tool = ConsensusTool()
        provider = MagicMock()
        provider.get_provider_type.return_value = ProviderType.GOOGLE
        response = ModelResponse(content="verdict", model_name="flash", provider=ProviderType.GOOGLE)
        request = MagicMock(relevant_files=None, images=None)
        seen = {}

        def capture_prompt(stance, custom=None):
            seen["stance"] = stance
            return "system prompt"

        with (
            patch.object(tool, "get_model_provider", return_value=provider),
            patch.object(tool, "validate_and_correct_temperature", return_value=(0.2, [])),
            patch.object(tool, "_get_stance_enhanced_prompt", side_effect=capture_prompt),
            patch("utils.observability.instrumented_async_generate", new=AsyncMock(return_value=response)),
        ):
            result = await tool._consult_model({"model": "flash", "stance": "against"}, request)

        assert seen["stance"] == "against"
        assert result["stance"] == "against"

    def test_duplicate_detection_still_keys_on_model_and_stance_fields(self):
        from tools.consensus import ConsensusRequest

        base = {"step": "s", "step_number": 1, "total_steps": 2, "next_step_required": True, "findings": "f"}
        ConsensusRequest(**base, models=[{"model": "flash", "stance": "for"}, {"model": "flash", "stance": "against"}])
        with pytest.raises(ValueError, match="Duplicate model \\+ stance"):
            ConsensusRequest(**base, models=[{"model": "flash", "stance": "for"}, {"model": "flash", "stance": "for"}])
