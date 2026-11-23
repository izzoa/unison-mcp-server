"""
Test consensus tool with three models consulted concurrently in a single step
"""

import json

from .base_test import BaseSimulatorTest


class TestConsensusThreeModels(BaseSimulatorTest):
    """Test consensus tool functionality with three models (concurrent dispatch in step 1)"""

    @property
    def test_name(self) -> str:
        return "consensus_three_models"

    @property
    def test_description(self) -> str:
        return "Test consensus tool with three models using flash:against, flash:for, flash:neutral"

    def run_test(self) -> bool:
        """Run three-model consensus test"""
        try:
            self.logger.info("Testing consensus tool with three models: flash:against, flash:for, flash:neutral")

            # Send request with three objects using the workflow parameters
            response, continuation_id = self.call_mcp_tool(
                "consensus",
                {
                    "step": "Is a sync manager class a good idea for my CoolTodos app?",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Initial analysis needed on sync manager class architecture decision for CoolTodos app",
                    "models": [
                        {
                            "model": "flash",
                            "stance": "against",
                            "stance_prompt": "You are a software architecture critic. Focus on the potential downsides of adding a sync manager class: complexity overhead, maintenance burden, potential for over-engineering, and whether simpler alternatives exist. Consider if this adds unnecessary abstraction layers.",
                        },
                        {
                            "model": "flash",
                            "stance": "for",
                            "stance_prompt": "You are a software architecture advocate. Focus on the benefits of a sync manager class: separation of concerns, testability, maintainability, and how it can improve the overall architecture. Consider scalability and code organization advantages.",
                        },
                        {
                            "model": "flash",
                            "stance": "neutral",
                            "stance_prompt": "You are a pragmatic software engineer. Provide a balanced analysis considering both the benefits and drawbacks. Focus on the specific context of a CoolTodos app and what factors would determine if this is the right choice.",
                        },
                    ],
                },
            )

            # Validate response
            if not response:
                self.logger.error("Failed to get response from three-model consensus tool")
                return False

            self.logger.info(f"Three-model consensus response preview: {response[:500]}...")

            # Parse the JSON response
            try:
                consensus_data = json.loads(response)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse three-model consensus response as JSON: {response}")
                return False

            # Validate consensus structure
            if "status" not in consensus_data:
                self.logger.error("Missing 'status' field in three-model consensus response")
                return False

            # Concurrent dispatch consults all three models inside step 1
            expected_status = "consensus_workflow_complete"
            if consensus_data["status"] != expected_status:
                self.logger.error(
                    f"Three-model consensus failed with status: {consensus_data['status']}, expected: {expected_status}"
                )

                # Log additional error details for debugging
                if "error" in consensus_data:
                    self.logger.error(f"Error message: {consensus_data['error']}")
                if "next_steps" in consensus_data:
                    self.logger.error(f"Suggested next steps: {consensus_data['next_steps']}")

                return False

            # Check step information
            if consensus_data.get("step_number") != 1:
                self.logger.error(f"Expected step_number 1, got: {consensus_data.get('step_number')}")
                return False

            if not consensus_data.get("consensus_complete"):
                self.logger.error("Expected consensus_complete=True after step 1")
                return False

            # Check that all three model responses are present, in the order listed
            accumulated = consensus_data.get("accumulated_responses")
            if not isinstance(accumulated, list) or len(accumulated) != 3:
                self.logger.error(f"Expected 3 accumulated responses, got: {accumulated}")
                return False

            expected_stances = ["against", "for", "neutral"]
            for index, expected_stance in enumerate(expected_stances):
                entry = accumulated[index]
                if entry.get("model") != "flash" or entry.get("stance") != expected_stance:
                    self.logger.error(
                        f"Response {index + 1} should be flash:{expected_stance}, "
                        f"got: {entry.get('model')}:{entry.get('stance')}"
                    )
                    return False
                if entry.get("status") != "success" or not entry.get("verdict"):
                    self.logger.error(f"Response {index + 1} missing a successful verdict: {entry.get('status')}")
                    return False
                self.logger.info(f"Consulted flash:{expected_stance} - status {entry.get('status')}")

            # Check the completion summary
            complete_consensus = consensus_data.get("complete_consensus") or {}
            if complete_consensus.get("total_responses") != 3:
                self.logger.error(f"Expected 3 total responses, got: {complete_consensus.get('total_responses')}")
                return False

            expected_models = ["flash:against", "flash:for", "flash:neutral"]
            if complete_consensus.get("models_consulted") != expected_models:
                self.logger.error(
                    f"Expected models {expected_models}, got: {complete_consensus.get('models_consulted')}"
                )
                return False

            # Check metadata reports the three models
            metadata = consensus_data.get("metadata", {})
            if metadata.get("total_models") != 3:
                self.logger.error(f"Expected metadata.total_models 3, got: {metadata.get('total_models')}")
                return False

            self.logger.info(f"Models in metadata: {metadata.get('models_consulted')}")

            # Verify we have the calling agent's analysis
            agent_analysis = consensus_data.get("agent_analysis")
            if not agent_analysis:
                self.logger.error("Missing the agent's analysis in step 1")
                return False

            analysis_text = agent_analysis.get("initial_analysis", "")
            self.logger.info(f"Agent analysis length: {len(analysis_text)} characters")

            self.logger.info("✓ Three-model consensus tool test completed successfully")
            self.logger.info("✓ All three models consulted concurrently in step 1, in the order listed")
            self.logger.info(f"✓ Analysis provided: {len(analysis_text)} characters")
            self.logger.info("✓ Response is final and ready for synthesis")

            return True

        except Exception as e:
            self.logger.error(f"Three-model consensus test failed with exception: {str(e)}")
            return False
