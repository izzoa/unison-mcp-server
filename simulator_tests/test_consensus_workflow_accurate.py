"""
Accurate Consensus Workflow Test

Validates the consensus workflow as it behaves today: every model is consulted
concurrently inside step 1 (no per-model steps), so a two-model request finishes
in a single step with the responses accumulated in the order the models were
listed:

1. Step 1: the calling agent's analysis is recorded and all models are consulted
   concurrently.
2. The response is already final (`consensus_workflow_complete`) and carries every
   model response, ready for the agent to synthesize.

The earlier sequential flow (step 1 = analysis + first model, step 2 = second
model + synthesis) was replaced by concurrent dispatch; this test pins the
current contract.
"""

import json

from .conversation_base_test import ConversationBaseTest


class TestConsensusWorkflowAccurate(ConversationBaseTest):
    """Test the consensus workflow with concurrent model dispatch"""

    @property
    def test_name(self) -> str:
        return "consensus_workflow_accurate"

    @property
    def test_description(self) -> str:
        return "Test consensus workflow: all models consulted concurrently in step 1, response final"

    def run_test(self) -> bool:
        """Run the consensus workflow test"""
        # Set up the test environment
        self.setUp()

        try:
            self.logger.info("Testing consensus workflow with concurrent model dispatch")
            self.logger.info("Expected flow: Step 1 (analysis + flash:for + flash:against) -> final response")

            self.logger.info("=== STEP 1: analysis + concurrent consultation of flash:for and flash:against ===")

            step1_response, _ = self.call_mcp_tool_direct(
                "consensus",
                {
                    "step": "Should we add a new AI-powered search feature to our application? Please analyze the technical feasibility, user value, and implementation complexity.",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Initial assessment of AI search feature proposal considering user needs, technical constraints, and business value.",
                    "models": [
                        {
                            "model": "flash",
                            "stance": "for",
                            "stance_prompt": "Focus on innovation benefits and competitive advantages.",
                        },
                        {
                            "model": "flash",
                            "stance": "against",
                            "stance_prompt": "Focus on implementation complexity and resource requirements.",
                        },
                    ],
                    "model": "flash",  # The calling agent's execution model
                },
            )

            if not step1_response:
                self.logger.error("Step 1 failed - no response")
                return False

            step1_data = json.loads(step1_response)
            self.logger.info(f"Step 1 status: {step1_data.get('status')}")

            # Concurrent dispatch finishes the whole consultation inside step 1
            if step1_data.get("status") != "consensus_workflow_complete":
                self.logger.error(f"Expected status 'consensus_workflow_complete', got: {step1_data.get('status')}")
                return False

            if step1_data.get("step_number") != 1:
                self.logger.error(f"Expected step_number 1, got: {step1_data.get('step_number')}")
                return False

            if step1_data.get("total_steps") != 1:
                self.logger.error(f"Expected total_steps 1 (concurrent dispatch), got: {step1_data.get('total_steps')}")
                return False

            if not step1_data.get("consensus_complete"):
                self.logger.error("Expected consensus_complete=True after step 1")
                return False

            # The agent's own analysis is carried through
            if "agent_analysis" not in step1_data:
                self.logger.error("Expected agent_analysis in step 1 response")
                return False

            # Every model response is present, in the order the models were listed
            accumulated = step1_data.get("accumulated_responses")
            if not isinstance(accumulated, list) or len(accumulated) != 2:
                self.logger.error(f"Expected 2 accumulated responses, got: {accumulated}")
                return False

            for index, (expected_model, expected_stance) in enumerate([("flash", "for"), ("flash", "against")]):
                entry = accumulated[index]
                if entry.get("model") != expected_model or entry.get("stance") != expected_stance:
                    self.logger.error(
                        f"Response {index + 1} should be {expected_model}:{expected_stance}, "
                        f"got: {entry.get('model')}:{entry.get('stance')}"
                    )
                    return False
                if entry.get("status") != "success":
                    self.logger.error(f"Response {index + 1} did not succeed: {entry.get('status')}")
                    return False
                if not entry.get("verdict"):
                    self.logger.error(f"Response {index + 1} has no verdict")
                    return False

            self.logger.info("✓ Step 1 completed - both models consulted, responses accumulated in order")
            self.logger.info(f"flash:for verdict preview: {accumulated[0].get('verdict', '')[:100]}...")
            self.logger.info(f"flash:against verdict preview: {accumulated[1].get('verdict', '')[:100]}...")

            # Verify consensus completion data
            complete_consensus = step1_data.get("complete_consensus")
            if not complete_consensus:
                self.logger.error("Expected complete_consensus data in step 1 response")
                return False

            if complete_consensus.get("total_responses") != 2:
                self.logger.error(f"Expected 2 model responses, got: {complete_consensus.get('total_responses')}")
                return False

            models_consulted = complete_consensus.get("models_consulted", [])
            expected_models = ["flash:for", "flash:against"]
            if models_consulted != expected_models:
                self.logger.error(f"Expected models {expected_models}, got: {models_consulted}")
                return False

            # The response must tell the agent to synthesize, and offer a way to continue
            if "next_steps" not in step1_data:
                self.logger.error("Expected next_steps (synthesis instructions) in step 1 response")
                return False

            offer = step1_data.get("continuation_offer") or {}
            if not offer.get("continuation_id"):
                self.logger.error("Expected a continuation_offer with a continuation_id")
                return False

            self.logger.info("🎉 CONSENSUS WORKFLOW TEST PASSED")
            self.logger.info("✓ Step 1: both models consulted concurrently and the response is final")
            self.logger.info("✓ Responses accumulated in the order the models were listed")
            self.logger.info("✓ Completion data, synthesis instructions and continuation offer present")

            return True

        except Exception as e:
            self.logger.error(f"Consensus workflow test failed with exception: {str(e)}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
