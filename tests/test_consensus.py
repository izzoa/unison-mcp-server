"""
Tests for the Consensus tool using WorkflowTool architecture.
"""

from unittest.mock import Mock

import pytest

from tools.consensus import ConsensusRequest, ConsensusTool
from tools.models import ToolModelCategory


class TestConsensusTool:
    """Test suite for ConsensusTool using WorkflowTool architecture."""

    def test_tool_metadata(self):
        """Test basic tool metadata and configuration."""
        tool = ConsensusTool()

        assert tool.get_name() == "consensus"
        assert "consensus" in tool.get_description()
        assert tool.get_default_temperature() == 1.0  # TEMPERATURE_ANALYTICAL
        assert tool.get_model_category() == ToolModelCategory.EXTENDED_REASONING
        assert tool.requires_model() is False  # Consensus manages its own models

    def test_request_validation_step1(self):
        """Test Pydantic request model validation for step 1."""
        # Valid step 1 request with models
        step1_request = ConsensusRequest(
            step="Analyzing the real-time collaboration proposal",
            step_number=1,
            total_steps=1,  # all models are consulted concurrently inside step 1
            next_step_required=False,
            findings="Initial assessment shows strong value but technical complexity",
            confidence="medium",
            models=[{"model": "flash", "stance": "neutral"}, {"model": "o3-mini", "stance": "for"}],
            relevant_files=["/proposal.md"],
        )

        assert step1_request.step_number == 1
        assert step1_request.confidence == "medium"
        assert len(step1_request.models) == 2
        assert step1_request.models[0]["model"] == "flash"

    def test_request_validation_missing_models_step1(self):
        """Test that step 1 requires models field."""
        with pytest.raises(ValueError, match="Step 1 requires 'models' field"):
            ConsensusRequest(
                step="Test step",
                step_number=1,
                total_steps=1,
                next_step_required=False,
                findings="Test findings",
                # Missing models field
            )

    def test_request_validation_later_steps(self):
        """Test request validation for steps 2+ (accepted for compatibility; they consult no model)."""
        # Step 2+ doesn't require models field
        step2_request = ConsensusRequest(
            step="A later step, which consults no model",
            step_number=2,
            total_steps=4,
            next_step_required=True,
            findings="Model provided supportive perspective",
            confidence="medium",
            continuation_id="test-id",
        )

        assert step2_request.step_number == 2
        assert step2_request.models is None  # Not required after step 1

    def test_retired_legacy_fields_are_not_part_of_the_request_model(self):
        """current_model_index and model_responses are gone; a client that still sends them is ignored, not stored."""
        assert "current_model_index" not in ConsensusRequest.model_fields
        assert "model_responses" not in ConsensusRequest.model_fields

        request = ConsensusRequest(
            step="A later step",
            step_number=2,
            total_steps=2,
            next_step_required=True,
            findings="Notes",
            current_model_index=1,
            model_responses=[{"model": "flash"}],
        )
        assert not hasattr(request, "current_model_index")
        assert not hasattr(request, "model_responses")

    def test_request_validation_duplicate_model_stance(self):
        """Test that duplicate model+stance combinations are rejected."""
        # Valid: same model with different stances
        valid_request = ConsensusRequest(
            step="Analyze this proposal",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Initial analysis",
            models=[
                {"model": "o3", "stance": "for"},
                {"model": "o3", "stance": "against"},
                {"model": "flash", "stance": "neutral"},
            ],
            continuation_id="test-id",
        )
        assert len(valid_request.models) == 3

        # Invalid: duplicate model+stance combination
        with pytest.raises(ValueError, match="Duplicate model \\+ stance combination"):
            ConsensusRequest(
                step="Analyze this proposal",
                step_number=1,
                total_steps=1,
                next_step_required=False,
                findings="Initial analysis",
                models=[
                    {"model": "o3", "stance": "for"},
                    {"model": "flash", "stance": "neutral"},
                    {"model": "o3", "stance": "for"},  # Duplicate!
                ],
                continuation_id="test-id",
            )

    def test_input_schema_generation(self):
        """Test that input schema is generated correctly."""
        tool = ConsensusTool()
        schema = tool.get_input_schema()

        # Verify consensus workflow fields are present
        assert "step" in schema["properties"]
        assert "step_number" in schema["properties"]
        assert "total_steps" in schema["properties"]
        assert "next_step_required" in schema["properties"]
        assert "findings" in schema["properties"]
        # confidence field should be excluded
        assert "confidence" not in schema["properties"]
        assert "models" in schema["properties"]
        # relevant_files should be present as it's used by consensus
        assert "relevant_files" in schema["properties"]

        # model field should NOT be present as consensus uses 'models' field instead
        assert "model" not in schema["properties"]

        # The retired sequential-flow fields are no longer published
        assert "current_model_index" not in schema["properties"]
        assert "model_responses" not in schema["properties"]

        # Verify workflow fields that should NOT be present
        assert "files_checked" not in schema["properties"]
        assert "hypothesis" not in schema["properties"]
        assert "issues_found" not in schema["properties"]
        assert "temperature" not in schema["properties"]
        assert "thinking_mode" not in schema["properties"]

        # Images should be present now
        assert "images" in schema["properties"]
        assert schema["properties"]["images"]["type"] == "array"
        assert schema["properties"]["images"]["items"]["type"] == "string"

        # Verify field types
        assert schema["properties"]["step"]["type"] == "string"
        assert schema["properties"]["step_number"]["type"] == "integer"
        assert schema["properties"]["models"]["type"] == "array"

        # Verify models array structure
        models_items = schema["properties"]["models"]["items"]
        assert models_items["type"] == "object"
        assert "model" in models_items["properties"]
        assert "stance" in models_items["properties"]
        assert "stance_prompt" in models_items["properties"]

    def test_get_required_actions(self):
        """Required actions are the same for every step: all models are consulted in step 1, so only synthesis remains."""
        tool = ConsensusTool()

        for step_number, total_steps in ((1, 1), (2, 4), (4, 4)):
            actions = tool.get_required_actions(step_number, "exploring", "Findings", total_steps)
            assert any("All models have been consulted" in action for action in actions)
            assert any("Synthesize all perspectives" in action for action in actions)
            for action in actions:
                assert "Wait for the next" not in action
                assert "consult other models" not in action

    def test_prepare_step_data(self):
        """Test step data preparation for consensus workflow."""
        tool = ConsensusTool()
        request = ConsensusRequest(
            step="Test step",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="Test findings",
            confidence="medium",
            models=[{"model": "test"}],
            relevant_files=["/test.py"],
        )

        step_data = tool.prepare_step_data(request)

        # Verify consensus-specific fields
        assert step_data["step"] == "Test step"
        assert step_data["findings"] == "Test findings"
        assert step_data["relevant_files"] == ["/test.py"]

        # Verify unused workflow fields are empty
        assert step_data["files_checked"] == []
        assert step_data["relevant_context"] == []
        assert step_data["issues_found"] == []
        assert step_data["hypothesis"] is None

    def test_stance_enhanced_prompt_generation(self):
        """Test stance-enhanced prompt generation."""
        tool = ConsensusTool()

        # Test different stances
        for_prompt = tool._get_stance_enhanced_prompt("for")
        assert "SUPPORTIVE PERSPECTIVE" in for_prompt

        against_prompt = tool._get_stance_enhanced_prompt("against")
        assert "CRITICAL PERSPECTIVE" in against_prompt

        neutral_prompt = tool._get_stance_enhanced_prompt("neutral")
        assert "BALANCED PERSPECTIVE" in neutral_prompt

        # Test custom stance prompt
        custom = "Focus on specific aspects"
        custom_prompt = tool._get_stance_enhanced_prompt("for", custom)
        assert custom in custom_prompt
        assert "SUPPORTIVE PERSPECTIVE" not in custom_prompt

    def test_should_call_expert_analysis(self):
        """Test that consensus workflow doesn't use expert analysis."""
        tool = ConsensusTool()
        assert tool.should_call_expert_analysis({}) is False
        assert tool.requires_expert_analysis() is False

    def test_execute_workflow_step1_basic(self):
        """Test basic workflow validation for step 1."""
        tool = ConsensusTool()

        # Test that step 1 sets up the workflow correctly
        arguments = {
            "step": "Initial analysis of proposal",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "Found pros and cons",
            "models": [{"model": "flash", "stance": "neutral"}, {"model": "o3-mini", "stance": "for"}],
        }

        # Verify models_to_consult is set correctly from step 1
        request = tool.get_workflow_request_model()(**arguments)
        assert len(request.models) == 2
        assert request.models[0]["model"] == "flash"
        assert request.models[1]["model"] == "o3-mini"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sent_total_steps, sent_next_step_required", [(4, True), (1, False), (1, True)])
    async def test_step_one_ignores_total_steps_and_next_step_required(self, sent_total_steps, sent_next_step_required):
        """Step 1 consults every model, reports total_steps 1 and is final whatever the caller sent."""
        import json
        from unittest.mock import AsyncMock, patch

        tool = ConsensusTool()
        arguments = {
            "step": "Evaluate the proposal",
            "step_number": 1,
            "total_steps": sent_total_steps,
            "next_step_required": sent_next_step_required,
            "findings": "Analysis complete",
            "models": [{"model": "flash", "stance": "neutral"}, {"model": "o3-mini", "stance": "for"}],
        }

        async def consult(config, request):  # noqa: ARG001 - stands in for the provider call
            return {"model": config["model"], "stance": config["stance"], "status": "success", "verdict": "ok"}

        with patch.object(tool, "_consult_model", new=AsyncMock(side_effect=consult)) as consulted:
            blocks = await tool.execute_workflow(arguments)

        payload = json.loads(blocks[0].text)
        assert consulted.await_count == 2  # every model, in this one call
        assert payload["status"] == "consensus_workflow_complete"
        assert payload["step_number"] == 1
        assert payload["total_steps"] == 1
        assert payload["consensus_complete"] is True
        assert payload["consensus_workflow_status"] == "consensus_workflow_complete"  # agrees with status
        assert payload["metadata"]["total_models"] == 2
        assert payload["metadata"]["consensus_complete"] is True
        assert "consultation_step" not in payload["metadata"]
        assert "next_step_required" not in payload
        assert [r["model"] for r in payload["accumulated_responses"]] == ["flash", "o3-mini"]  # roster order

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sent_total_steps", [2, 3, 9])
    async def test_step_two_with_next_step_required_never_promises_another_model(self, sent_total_steps):
        """A non-conforming step 2+ call (next_step_required true) must not say a model is still to be consulted."""
        import json
        from unittest.mock import AsyncMock, patch

        tool = ConsensusTool()
        models = [{"model": "flash", "stance": "neutral"}, {"model": "o3-mini", "stance": "for"}]
        step_one = {
            "step": "Evaluate the proposal",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "Analysis complete",
            "models": models,
        }

        async def consult(config, request):  # noqa: ARG001 - stands in for the provider call
            return {"model": config["model"], "stance": config["stance"], "status": "success", "verdict": "ok"}

        with patch.object(tool, "_consult_model", new=AsyncMock(side_effect=consult)) as consulted:
            await tool.execute_workflow(step_one)
            assert consulted.await_count == 2
            later = dict(step_one, step_number=2, total_steps=sent_total_steps, next_step_required=True)
            blocks = await tool.execute_workflow(later)
            assert consulted.await_count == 2  # step 2 consults nobody

        payload = json.loads(blocks[0].text)
        assert payload["status"] == "ready_for_synthesis"
        assert payload["consensus_workflow_status"] == "ready_for_synthesis"
        # Whatever total_steps was sent, the metadata reports a finished consultation, never a consultation step
        assert payload["metadata"]["consensus_complete"] is True
        assert payload["metadata"]["models_consulted"] == ["flash:neutral", "o3-mini:for"]
        for retired in ("models_to_consult", "consultation_step", "total_consultation_steps"):
            assert retired not in payload["metadata"]
        assert "next_model" not in payload
        assert "models_remaining" not in payload
        next_steps = payload["next_steps"].lower()
        assert "no further models" in next_steps
        assert "in progress" not in next_steps and "next:" not in next_steps

    @pytest.mark.asyncio
    @pytest.mark.parametrize("step_number", [2, 3])
    @pytest.mark.parametrize(
        "next_step_required, expected",
        [(True, "ready_for_synthesis"), (False, "consensus_workflow_complete")],
    )
    async def test_step_two_plus_workflow_status_always_agrees_with_status(
        self, step_number, next_step_required, expected
    ):
        import json
        from unittest.mock import AsyncMock, patch

        tool = ConsensusTool()
        step_one = {
            "step": "Evaluate the proposal",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "Analysis complete",
            "models": [{"model": "flash", "stance": "neutral"}],
        }

        async def consult(config, request):  # noqa: ARG001 - stands in for the provider call
            return {"model": config["model"], "stance": config["stance"], "status": "success", "verdict": "ok"}

        with patch.object(tool, "_consult_model", new=AsyncMock(side_effect=consult)):
            await tool.execute_workflow(step_one)
            later = dict(
                step_one, step_number=step_number, total_steps=step_number, next_step_required=next_step_required
            )
            payload = json.loads((await tool.execute_workflow(later))[0].text)

        assert payload["status"] == expected
        assert payload["consensus_workflow_status"] == payload["status"]

    @pytest.mark.asyncio
    async def test_step_two_plus_repeats_verdicts_only_in_the_process_that_ran_step_one(self):
        """docs/tools/consensus.md: a step 2+ reply repeats accumulated_responses only where step 1 ran."""
        import json
        from unittest.mock import AsyncMock, patch

        step_one = {
            "step": "Evaluate the proposal",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "Analysis complete",
            "models": [{"model": "flash", "stance": "neutral"}],
        }
        later = dict(step_one, step_number=2, total_steps=2, next_step_required=True)

        async def consult(config, request):  # noqa: ARG001 - stands in for the provider call
            return {"model": config["model"], "stance": config["stance"], "status": "success", "verdict": "ok"}

        ran_step_one = ConsensusTool()
        with patch.object(ran_step_one, "_consult_model", new=AsyncMock(side_effect=consult)):
            await ran_step_one.execute_workflow(step_one)
            same_process = json.loads((await ran_step_one.execute_workflow(later))[0].text)
        fresh_process = json.loads((await ConsensusTool().execute_workflow(later))[0].text)

        assert [r["model"] for r in same_process["accumulated_responses"]] == ["flash"]
        assert same_process["metadata"]["models_consulted"] == ["flash:neutral"]
        assert "accumulated_responses" not in fresh_process
        assert fresh_process["metadata"]["models_consulted"] == [] and fresh_process["metadata"]["total_models"] == 0

    def test_handle_work_continuation_says_no_further_model_is_consulted(self):
        tool = ConsensusTool()
        tool.models_to_consult = [{"model": "flash", "stance": "neutral"}, {"model": "o3-mini", "stance": "for"}]
        for step_number, total_steps in ((2, 2), (2, 4), (3, 9)):
            request = Mock(step_number=step_number, total_steps=total_steps)
            result = tool.handle_work_continuation({}, request)
            assert result["status"] == "ready_for_synthesis"
            assert "next_model" not in result and "models_remaining" not in result
            assert "No further models" in result["next_steps"]

    def test_consult_model_basic_structure(self):
        """Test basic model consultation structure."""
        tool = ConsensusTool()

        # Test that _get_stance_enhanced_prompt works
        for_prompt = tool._get_stance_enhanced_prompt("for")
        against_prompt = tool._get_stance_enhanced_prompt("against")
        neutral_prompt = tool._get_stance_enhanced_prompt("neutral")

        assert "SUPPORTIVE PERSPECTIVE" in for_prompt
        assert "CRITICAL PERSPECTIVE" in against_prompt
        assert "BALANCED PERSPECTIVE" in neutral_prompt

    def test_model_configuration_validation(self):
        """Test model configuration validation."""
        tool = ConsensusTool()

        # Test single model config
        models = [{"model": "flash", "stance": "neutral"}]
        arguments = {
            "step": "Test",
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "findings": "Test findings",
            "models": models,
        }

        request = tool.get_workflow_request_model()(**arguments)
        assert len(request.models) == 1
        assert request.models[0]["model"] == "flash"
        assert request.models[0]["stance"] == "neutral"

    def test_customize_workflow_response(self):
        """Step 1 repeats the response's own status; a later step can only report that synthesis is next."""
        tool = ConsensusTool()
        tool.accumulated_responses = [{"model": "test", "response": "data"}]

        # Step 1 is the whole consultation, so its workflow status must not contradict its status
        request = Mock(step_number=1)
        result = tool.customize_workflow_response({"status": "consensus_workflow_complete"}, request)
        assert result["consensus_workflow_status"] == "consensus_workflow_complete"
        result = tool.customize_workflow_response({"status": "all_providers_unavailable"}, request)
        assert result["consensus_workflow_status"] == "all_providers_unavailable"

        # A later step mirrors its own status too, whichever of the two step 2+ variants it is
        request = Mock(step_number=2, total_steps=2)
        for own_status in ("ready_for_synthesis", "consensus_workflow_complete"):
            result = tool.customize_workflow_response({"status": own_status}, request)
            assert result["consensus_workflow_status"] == own_status
        # Only a hand-built response with no status at all gets the fallback
        assert tool.customize_workflow_response({}, request)["consensus_workflow_status"] == "ready_for_synthesis"

    @pytest.mark.parametrize("step_number, total_steps", [(1, 1), (2, 2), (2, 4), (3, 9)])
    def test_consensus_metadata_is_always_the_finished_shape(self, step_number, total_steps):
        """No step reports an in-progress consultation: the models_to_consult / consultation_step shape is gone."""
        tool = ConsensusTool()
        tool.models_to_consult = [{"model": "flash", "stance": "neutral"}, {"model": "o3-mini", "stance": "for"}]

        result = tool.customize_workflow_response({}, Mock(step_number=step_number, total_steps=total_steps))

        metadata = result["metadata"]
        assert metadata["workflow_type"] == "multi_model_consensus"
        assert metadata["models_consulted"] == ["flash:neutral", "o3-mini:for"]
        assert metadata["consensus_complete"] is True
        assert metadata["total_models"] == 2
        for retired in ("models_to_consult", "consultation_step", "total_consultation_steps"):
            assert retired not in metadata

    def test_consensus_is_not_published_read_only(self):
        """Batch mode bills a provider batch and persists to the batch store, so the base read-only hint is overridden."""
        assert ConsensusTool().get_annotations() == {"readOnlyHint": False}

    @pytest.mark.asyncio
    async def test_consensus_with_relevant_files_model_context_fix(self):
        """Test that consensus tool properly handles relevant_files without RuntimeError.

        This is a regression test for the bug where _prepare_file_content_for_prompt
        was called without model_context parameter, causing RuntimeError:
        'Model context not provided for file preparation'

        Bug details:
        - Occurred when consensus tool processed requests with relevant_files
        - _consult_model method called _prepare_file_content_for_prompt without model_context
        - Method expected model_context parameter but got None (default value)
        - Runtime validation in base_tool.py threw RuntimeError
        """
        from unittest.mock import AsyncMock, Mock, patch

        from utils.model_context import ModelContext

        tool = ConsensusTool()

        # Create a mock request with relevant_files (the trigger condition)
        mock_request = Mock()
        mock_request.relevant_files = ["/test/file1.py", "/test/file2.js"]
        mock_request.continuation_id = None

        # Mock model configuration
        model_config = {"model": "flash", "stance": "neutral"}

        # Mock the provider and model name resolution
        with (
            patch.object(tool, "get_model_provider") as mock_get_provider,
            patch.object(tool, "_prepare_file_content_for_prompt") as mock_prepare_files,
            patch.object(tool, "_get_stance_enhanced_prompt") as mock_get_prompt,
            patch.object(tool, "get_name", return_value="consensus"),
        ):

            # Setup mocks
            mock_provider = Mock()
            mock_provider.generate_content = AsyncMock(return_value={"response": "test response"})
            mock_get_provider.return_value = mock_provider
            mock_prepare_files.return_value = ("file content", [])
            mock_get_prompt.return_value = "system prompt"

            # Set up the tool's attributes that would be set during normal execution
            tool.original_proposal = "Test proposal"

            try:
                # This should not raise RuntimeError after the fix
                # The method should create ModelContext and pass it to _prepare_file_content_for_prompt
                await tool._consult_model(model_config, mock_request)

                # Verify that _prepare_file_content_for_prompt was called with model_context
                mock_prepare_files.assert_called_once()
                call_args = mock_prepare_files.call_args

                # Check that model_context was passed as keyword argument
                assert "model_context" in call_args.kwargs, "model_context should be passed as keyword argument"

                # Verify the model_context is a proper ModelContext instance
                model_context = call_args.kwargs["model_context"]
                assert isinstance(model_context, ModelContext), "model_context should be ModelContext instance"

                # Verify model_context properties are correct
                assert model_context.model_name == "flash"
                # Note: provider is accessed lazily, conversation_history and tool_name
                # are not part of ModelContext constructor in current implementation

            except RuntimeError as e:
                if "Model context not provided" in str(e):
                    pytest.fail("The model_context fix is not working. RuntimeError still occurs: " + str(e))
                else:
                    # Re-raise if it's a different RuntimeError
                    raise


if __name__ == "__main__":
    import unittest

    unittest.main()
