"""
Test to validate line number handling across different tools
"""

import json
import os
import re

from .base_test import BaseSimulatorTest


class LineNumberValidationTest(BaseSimulatorTest):
    """Test that validates correct line number handling in chat, analyze, and refactor tools"""

    @property
    def test_name(self) -> str:
        return "line_number_validation"

    @property
    def test_description(self) -> str:
        return "Line number handling validation across tools"

    # Real statuses of a single-step workflow call: the expert model ran, or the tool completed locally.
    ANALYZE_STATUSES = ("calling_expert_analysis", "analysis_complete_ready_for_implementation")
    REFACTOR_STATUSES = ("calling_expert_analysis", "refactoring_analysis_complete_ready_for_implementation")
    # The refactor expert may answer in its own structured format (RefactorAnalysisComplete), which the
    # tool passes through unchanged instead of wrapping it in the workflow envelope.
    REFACTOR_STRUCTURED_STATUS = "refactor_analysis_complete"
    LINE_REFERENCE = re.compile(r"\blines?\s*\d+|\bL\d+\b|\bstart_line\b", re.IGNORECASE)

    def _expert_analysis_text(self, content: str, allowed_statuses: tuple[str, ...], label: str) -> str | None:
        """Parse a workflow response, check its status, and return the expert analysis text.

        Returns None (after logging the reason) when the response is not valid JSON, carries an
        unexpected status, or has no expert analysis text to inspect.
        """
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            self.logger.error(f"  ❌ {label} tool response is not valid JSON: {content[:200]}")
            return None

        status = result.get("status")
        if status not in allowed_statuses:
            self.logger.error(f"  ❌ {label} tool returned status {status!r}, expected one of {allowed_statuses}")
            return None

        expert = result.get("expert_analysis")
        if not isinstance(expert, dict):
            self.logger.error(f"  ❌ {label} tool response has no expert_analysis")
            return None

        text = expert.get("raw_analysis") or expert.get("analysis") or expert.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            self.logger.error(f"  ❌ {label} tool expert analysis is empty: {expert}")
            return None
        return text

    def _structured_refactor_payload(self, result: dict) -> dict | None:
        """Return the expert's structured RefactorAnalysisComplete payload, wherever the server put it.

        The real server answers with the workflow envelope (status "calling_expert_analysis") and nests
        the expert's own JSON under "expert_analysis"; the payload can also be the whole response.
        """
        if result.get("status") == self.REFACTOR_STRUCTURED_STATUS:
            return result
        expert = result.get("expert_analysis")
        if isinstance(expert, dict) and (
            expert.get("status") == self.REFACTOR_STRUCTURED_STATUS
            or isinstance(expert.get("refactor_opportunities"), list)
        ):
            return expert
        return None

    def _check_refactor_line_numbers(self, content: str, line_count: int) -> bool:
        """Validate the refactor response in either of its two real shapes."""
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            self.logger.error(f"  ❌ Refactor tool response is not valid JSON: {content[:200]}")
            return False

        structured = self._structured_refactor_payload(result)
        if structured is not None:
            opportunities = structured.get("refactor_opportunities")
            if not isinstance(opportunities, list):
                self.logger.error("  ❌ Refactor tool response has no refactor_opportunities list")
                return False
            if not opportunities:
                # A legitimate expert verdict on a tiny file, but nothing to verify line numbers against.
                self.logger.info("  ℹ️ Refactor expert reported no opportunities; line numbers not exercised")
                return True
            for opportunity in opportunities:
                start, end = opportunity.get("start_line"), opportunity.get("end_line")
                if not (isinstance(start, int) and isinstance(end, int) and 1 <= start <= end <= line_count):
                    self.logger.error(
                        f"  ❌ Refactor opportunity has an invalid line range {start}-{end} (file has {line_count} lines)"
                    )
                    return False
            self.logger.info(f"  ✅ Refactor tool returned {len(opportunities)} opportunities with valid line ranges")
            return True

        text = self._expert_analysis_text(content, self.REFACTOR_STATUSES, "Refactor")
        if text is None:
            return False
        if not self.LINE_REFERENCE.search(text):
            self.logger.error(f"  ❌ Refactor tool response contained no line number references: {text[:200]}")
            return False
        self.logger.info("  ✅ Refactor tool provided line number references")
        return True

    def run_test(self) -> bool:
        """Test line number handling in different tools"""
        try:
            self.logger.info("Test: Line number handling validation")

            # Setup test files
            self.setup_test_files()

            # Create a test file with known content
            test_file_content = '''# Example code with specific elements
def calculate_total(items):
    """Calculate total with tax"""
    subtotal = 0
    tax_rate = 0.08  # Line 5 - tax_rate defined

    for item in items:  # Line 7 - loop starts
        if item.price > 0:
            subtotal += item.price

    tax_amount = subtotal * tax_rate  # Line 11
    return subtotal + tax_amount

def validate_data(data):
    """Validate input data"""  # Line 15
    required_fields = ["name", "email", "age"]  # Line 16

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    return True  # Line 22
'''

            line_count = len(test_file_content.splitlines())
            test_file_path = os.path.join(self.test_dir, "line_test.py")
            with open(test_file_path, "w") as f:
                f.write(test_file_content)

            self.logger.info(f"Created test file: {test_file_path}")

            # Test 1: Chat tool asking about specific line
            self.logger.info("  1.1: Testing chat tool with line number question")
            content, continuation_id = self.call_mcp_tool(
                "chat",
                {
                    "prompt": "Where is tax_rate defined in this file? Please tell me the exact line number.",
                    "absolute_file_paths": [test_file_path],
                    "model": "flash",
                },
            )

            if content:
                # Check if the response mentions line 5
                if "line 5" in content.lower() or "line 5" in content:
                    self.logger.info("  ✅ Chat tool correctly identified tax_rate at line 5")
                else:
                    self.logger.warning(f"  ⚠️ Chat tool response didn't mention line 5: {content[:200]}...")
            else:
                self.logger.error("  ❌ Chat tool request failed")
                return False

            # Test 2: Analyze tool with line number reference (single-step workflow call)
            self.logger.info("  1.2: Testing analyze tool with line number analysis")
            content, continuation_id = self.call_mcp_tool(
                "analyze",
                {
                    "step": "What happens between lines 7-11 in this code? Focus on the loop logic.",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "Starting analysis of the loop between lines 7 and 11",
                    "relevant_files": [test_file_path],
                    "model": "flash",
                },
            )

            if not content:
                self.logger.error("  ❌ Analyze tool request failed")
                return False

            analysis_text = self._expert_analysis_text(content, self.ANALYZE_STATUSES, "Analyze")
            if analysis_text is None:
                return False
            if not (self.LINE_REFERENCE.search(analysis_text) or "loop" in analysis_text.lower()):
                self.logger.error(
                    f"  ❌ Analyze tool response did not reference lines or the loop: {analysis_text[:200]}"
                )
                return False
            self.logger.info("  ✅ Analyze tool correctly analyzed the specified line range")

            # Test 3: Refactor tool with line number precision (single-step workflow call)
            self.logger.info("  1.3: Testing refactor tool line number precision")
            content, continuation_id = self.call_mcp_tool(
                "refactor",
                {
                    "step": "Analyze this code for refactoring opportunities and cite the line numbers involved",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": (
                        "Starting refactoring analysis of line_test.py. Candidate smells: the hard-coded tax_rate "
                        "magic number on line 5, the missing type hints, and the manual accumulation loop on lines 7-9."
                    ),
                    "relevant_files": [test_file_path],
                    "refactor_type": "codesmells",
                    "model": "flash",
                },
            )

            if not content:
                self.logger.error("  ❌ Refactor tool request failed")
                return False

            if not self._check_refactor_line_numbers(content, line_count):
                return False

            # Test 4: Validate log patterns
            self.logger.info("  1.4: Validating line number processing in logs")

            # Get logs from server
            try:
                log_file_path = "logs/mcp_server.log"
                with open(log_file_path) as f:
                    lines = f.readlines()
                    logs = "".join(lines[-500:])
            except Exception as e:
                self.logger.error(f"Failed to read server logs: {e}")
                logs = ""
                pass

            # Check for line number formatting patterns
            line_number_patterns = ["Line numbers for", "enabled", "│", "line number"]  # The line number separator

            found_patterns = 0
            for pattern in line_number_patterns:
                if pattern in logs:
                    found_patterns += 1

            self.logger.info(f"    Found {found_patterns}/{len(line_number_patterns)} line number patterns in logs")

            if found_patterns >= 2:
                self.logger.info("  ✅ Line number processing confirmed in logs")
            else:
                self.logger.warning("  ⚠️ Limited line number processing evidence in logs")

            self.logger.info("  ✅ Line number validation test completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Line number validation test failed: {type(e).__name__}: {e}")
            return False
