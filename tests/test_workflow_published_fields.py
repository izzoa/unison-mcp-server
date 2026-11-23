"""Published-schema and runtime fixes for the workflow tools.

Covers the dead published fields that were removed from the input schemas, the tools that never call a
model no longer advertising ``model`` / ``use_assistant_model``, the precommit step guidance (a single step is
accepted and the three-step flow is guidance only), the tracer's published text, the tool-specific
descriptions that now reach the published schema, and the two thinkdeep runtime fixes.

No network, no API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pytest

from tools import (
    AnalyzeTool,
    CodeReviewTool,
    DebugIssueTool,
    DocgenTool,
    PlannerTool,
    PrecommitTool,
    RefactorTool,
    TestGenTool,
    ThinkDeepTool,
)
from tools.analyze import ANALYZE_WORKFLOW_FIELD_DESCRIPTIONS
from tools.codereview import CODEREVIEW_WORKFLOW_FIELD_DESCRIPTIONS, CodeReviewRequest
from tools.debug import DEBUG_INVESTIGATION_FIELD_DESCRIPTIONS
from tools.docgen import DocgenRequest
from tools.precommit import PRECOMMIT_WORKFLOW_FIELD_DESCRIPTIONS, PrecommitRequest
from tools.refactor import REFACTOR_FIELD_DESCRIPTIONS
from tools.secaudit import SECAUDIT_WORKFLOW_FIELD_DESCRIPTIONS, SecauditTool
from tools.shared.base_models import WORKFLOW_FIELD_DESCRIPTIONS
from tools.testgen import TESTGEN_WORKFLOW_FIELD_DESCRIPTIONS
from tools.thinkdeep import ThinkDeepWorkflowRequest
from tools.tracer import TRACER_WORKFLOW_FIELD_DESCRIPTIONS, TracerTool

SNAPSHOT = Path(__file__).parent / "fixtures" / "tool_schemas_pre_batch.json"

# The explicit list of properties removed from the published schemas on purpose (the fixture
# tool_schemas_pre_batch.json was updated for exactly these, and nothing else in the same commit):
# every one is a field the tool never reads. The request models keep the inherited fields.
REMOVED_PUBLISHED_FIELDS: dict[str, tuple[str, ...]] = {
    "analyze": ("confidence",),
    "codereview": ("confidence", "hypothesis"),
    "precommit": ("confidence", "hypothesis", "focus_on"),
    "refactor": ("hypothesis",),
    "secaudit": ("hypothesis",),
    "debug": ("issues_found",),
    "testgen": ("issues_found", "hypothesis"),
    "docgen": ("issues_found", "use_assistant_model"),
    "planner": ("model", "use_assistant_model"),
    "tracer": ("model", "use_assistant_model"),
}

TOOL_CLASSES = {
    "analyze": AnalyzeTool,
    "codereview": CodeReviewTool,
    "precommit": PrecommitTool,
    "refactor": RefactorTool,
    "secaudit": SecauditTool,
    "debug": DebugIssueTool,
    "testgen": TestGenTool,
    "docgen": DocgenTool,
    "thinkdeep": ThinkDeepTool,
    "planner": PlannerTool,
    "tracer": TracerTool,
}


def _schema(name: str) -> dict:
    return TOOL_CLASSES[name]().get_input_schema()


def _auto_mode_schema(name: str) -> dict:
    with patch("config.DEFAULT_MODEL", "auto"), patch("config.IS_AUTO_MODE", True):
        tool = TOOL_CLASSES[name]()
        return tool.get_input_schema()


# ----------------------------------------------------------------------
# Decision 2: dead published fields
# ----------------------------------------------------------------------


class TestRemovedPublishedFields:
    @pytest.mark.parametrize("name", sorted(REMOVED_PUBLISHED_FIELDS))
    def test_removed_fields_are_in_no_property_list_and_no_required_list(self, name):
        for schema in (_schema(name), _auto_mode_schema(name)):
            for field in REMOVED_PUBLISHED_FIELDS[name]:
                assert field not in schema["properties"], f"{name} still publishes {field}"
                assert field not in schema["required"], f"{name} still requires {field}"

    @pytest.mark.parametrize("name", sorted(set(REMOVED_PUBLISHED_FIELDS) - {"secaudit", "tracer"}))
    def test_the_schema_fixture_matches_the_explicit_removals(self, name):
        # secaudit and tracer are not part of the fixture
        snapshot = json.loads(SNAPSHOT.read_text())
        for field in REMOVED_PUBLISHED_FIELDS[name]:
            assert field not in snapshot[name]["properties"]
            assert field not in snapshot[name]["required"]

    @pytest.mark.parametrize("name", sorted(REMOVED_PUBLISHED_FIELDS))
    def test_a_client_that_still_sends_a_removed_field_is_rejected_at_the_boundary(self, name):
        schema = _schema(name)
        sample = {
            "confidence": "high",
            "hypothesis": "h",
            "focus_on": "x",
            "issues_found": [],
            "use_assistant_model": False,
            "model": "flash",
        }
        for field in REMOVED_PUBLISHED_FIELDS[name]:
            with pytest.raises(jsonschema.ValidationError, match=f"'{field}' was unexpected"):
                jsonschema.validate(instance={**_valid_call(name), field: sample[field]}, schema=schema)

    @pytest.mark.parametrize("name", sorted(REMOVED_PUBLISHED_FIELDS))
    def test_a_call_without_the_removed_fields_still_validates(self, name):
        jsonschema.validate(instance=_valid_call(name), schema=_schema(name))

    def test_request_models_keep_the_inherited_fields_that_shared_code_still_touches(self):
        # The shared consolidation code reads these off every request, so the models keep them.
        for field in ("confidence", "hypothesis", "issues_found", "use_assistant_model"):
            assert field in CodeReviewRequest.model_fields
        assert "focus_on" not in PrecommitRequest.model_fields  # never read anywhere, so gone entirely

    def test_analyze_still_reports_a_fixed_confidence_in_its_status(self):
        tool = AnalyzeTool()
        request = type("R", (), {})()  # no confidence attribute at all
        assert tool.get_request_confidence(request) == "medium" == tool.get_confidence_level(request)


def _valid_call(name: str) -> dict:
    call: dict = {"step": "s", "step_number": 1, "total_steps": 1, "next_step_required": False, "findings": "f"}
    extra = {
        "analyze": {"relevant_files": ["/a.py"]},
        "codereview": {"relevant_files": ["/a.py"]},
        "precommit": {"path": "/repo"},
        "refactor": {"relevant_files": ["/a.py"]},
        "secaudit": {"relevant_files": ["/a.py"]},
        "debug": {},
        "testgen": {"relevant_files": ["/a.py"]},
        "docgen": {
            "document_complexity": True,
            "document_flow": True,
            "update_existing": True,
            "comments_on_complex_logic": True,
            "num_files_documented": 0,
            "total_files_to_document": 0,
        },
        "thinkdeep": {},
        "planner": {},
        "tracer": {"target_description": "t", "trace_mode": "precision"},
    }
    call.update(extra[name])
    if name == "planner":  # planner steps carry no findings
        del call["findings"]
    return call


# ----------------------------------------------------------------------
# Decision 5b: planner, tracer and docgen never call a model
# ----------------------------------------------------------------------


class TestToolsThatNeverCallAModel:
    @pytest.mark.parametrize("name", ["planner", "tracer", "docgen"])
    def test_model_and_use_assistant_model_are_not_published(self, name):
        for schema in (_schema(name), _auto_mode_schema(name)):
            assert "model" not in schema["properties"]
            assert "use_assistant_model" not in schema["properties"]

    @pytest.mark.parametrize("name", ["planner", "tracer", "docgen"])
    def test_model_is_not_required_in_auto_mode(self, name):
        with patch("config.DEFAULT_MODEL", "auto"), patch("config.IS_AUTO_MODE", True):
            tool = TOOL_CLASSES[name]()
            assert tool.is_effective_auto_mode()  # the default install runs in auto mode
            schema = tool.get_input_schema()
        assert "model" not in schema["required"]
        jsonschema.validate(instance=_valid_call(name), schema=schema)

    def test_docgen_request_defaults_use_assistant_model_to_false(self):
        request = DocgenRequest(step="s", step_number=1, total_steps=2, next_step_required=True, findings="f")
        assert request.use_assistant_model is False
        assert "use_assistant_model" not in request.model_dump()

    def test_tools_that_do_call_a_model_still_publish_model_in_auto_mode(self):
        for name in ("analyze", "codereview", "debug", "thinkdeep"):
            schema = _auto_mode_schema(name)
            assert "model" in schema["properties"]
            assert "model" in schema["required"]


# ----------------------------------------------------------------------
# Decision 3: precommit step guidance
# ----------------------------------------------------------------------


def _precommit_request(*, total_steps: int, precommit_type: str, step_number: int = 1, next_step_required=False):
    return PrecommitRequest(
        step="validate",
        step_number=step_number,
        total_steps=total_steps,
        next_step_required=next_step_required,
        findings="f",
        path="/repo",
        precommit_type=precommit_type,
    )


class TestFixedCurrentConfidence:
    """analyze, codereview and precommit publish no confidence input, so their status blocks agree on one value."""

    STATUS_BLOCK = {"analyze": "analysis_status", "codereview": "code_review_status", "precommit": "validation_status"}

    def _status_block(self, name: str) -> dict:
        tool = TOOL_CLASSES[name]()
        arguments = _valid_call(name)
        request = tool.get_workflow_request_model()(**{**arguments, "next_step_required": True})
        response = tool.build_base_response(request)
        response = tool.customize_workflow_response(response, request)
        return response[self.STATUS_BLOCK[name]]

    @pytest.mark.parametrize("name", sorted(STATUS_BLOCK))
    def test_current_confidence_is_the_fixed_medium(self, name):
        assert self._status_block(name)["current_confidence"] == "medium"

    def test_the_three_tools_report_the_same_value(self):
        assert len({self._status_block(name)["current_confidence"] for name in self.STATUS_BLOCK}) == 1

    @pytest.mark.parametrize("name", sorted(STATUS_BLOCK))
    def test_a_request_model_default_or_a_sent_confidence_does_not_leak_into_the_status(self, name):
        tool = TOOL_CLASSES[name]()
        for sent in (None, "low", "certain"):
            request = type("R", (), {"confidence": sent})()
            assert tool.get_request_confidence(request) == "medium"
        assert tool.get_request_confidence(type("R", (), {})()) == "medium"  # no attribute at all


class TestPrecommitStepGuidance:
    def test_published_total_steps_minimum_is_one(self):
        assert _schema("precommit")["properties"]["total_steps"]["minimum"] == 1

    def test_a_single_internal_step_validates_at_the_boundary_and_in_the_model(self):
        schema = _schema("precommit")
        call = {**_valid_call("precommit"), "precommit_type": "internal", "total_steps": 1}
        jsonschema.validate(instance=call, schema=schema)
        request = _precommit_request(total_steps=1, precommit_type="internal")
        assert request.total_steps == 1
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance={**call, "total_steps": 0}, schema=schema)

    def test_step_one_of_an_internal_single_step_is_not_told_to_continue_to_step_three(self):
        tool = PrecommitTool()
        request = _precommit_request(total_steps=1, precommit_type="internal")
        actions = tool.get_required_actions(1, "internal", "f", 1, request)
        assert not any("three steps" in action or "step 3" in action for action in actions)
        assert not any("MUST set next_step_required" in action for action in actions)

    @pytest.mark.parametrize(
        ("total_steps", "precommit_type"), [(1, "external"), (3, "external"), (3, "internal"), (5, "internal")]
    )
    def test_step_one_is_guided_towards_three_steps_for_external_or_three_planned_steps(
        self, total_steps, precommit_type
    ):
        tool = PrecommitTool()
        request = _precommit_request(total_steps=total_steps, precommit_type=precommit_type)
        actions = tool.get_required_actions(1, precommit_type, "f", total_steps, request)
        guided = [action for action in actions if "three steps" in action]
        assert len(guided) == 1
        assert guided[0].startswith("GUIDANCE")

    def test_step_one_without_a_request_keeps_the_external_default_guidance(self):
        actions = PrecommitTool().get_required_actions(1, "external", "", 3)
        assert any("three steps" in action for action in actions)

    def test_no_published_or_guidance_text_claims_the_step_count_is_enforced(self):
        tool = PrecommitTool()
        texts = [
            PRECOMMIT_WORKFLOW_FIELD_DESCRIPTIONS["total_steps"],
            PRECOMMIT_WORKFLOW_FIELD_DESCRIPTIONS["next_step_required"],
            json.dumps(_schema("precommit")),
        ]
        for step_number in (1, 2, 3):
            request = _precommit_request(
                total_steps=3, precommit_type="external", step_number=step_number, next_step_required=False
            )
            texts.extend(tool.get_required_actions(step_number, "external", "f\ng\nh", 3, request))
            texts.append(tool.get_precommit_step_guidance(step_number, request)["next_steps"])
        for text in texts:
            lowered = text.lower()
            assert "minimum step" not in lowered
            assert "minimum required steps" not in lowered
            assert "minimum steps" not in lowered
            assert "step 3 minimum" not in lowered
            assert "enforced" not in lowered

    def test_finishing_early_gets_corrective_guidance_not_a_rejection(self):
        tool = PrecommitTool()
        request = _precommit_request(total_steps=3, precommit_type="external", step_number=2)
        guidance = tool.get_precommit_step_guidance(2, request)["next_steps"]
        assert guidance.startswith("ERROR")
        assert "next_step_required=True" in guidance


# ----------------------------------------------------------------------
# Decision 8: tracer trace_mode and confidence text
# ----------------------------------------------------------------------


class TestTracerPublishedText:
    def test_trace_mode_text_says_it_is_required_on_every_call(self):
        schema = _schema("tracer")
        description = schema["properties"]["trace_mode"]["description"]
        assert "Required on every call" in description
        assert "'ask' (step 1 pauses" in description
        assert "default" not in description.lower()
        assert "trace_mode" in schema["required"]
        assert schema["properties"]["trace_mode"]["enum"] == ["precision", "dependencies", "ask"]

    def test_confidence_has_a_tracer_specific_description_and_no_validation_claim(self):
        schema = _schema("tracer")
        description = schema["properties"]["confidence"]["description"]
        assert description == TRACER_WORKFLOW_FIELD_DESCRIPTIONS["confidence"]
        assert "never calls an external model" in description
        assert "PREVENTS" not in description
        assert "external validation" not in description
        assert schema["properties"]["confidence"]["enum"][-1] == "certain"

    def test_no_tracer_text_claims_certain_prevents_validation(self):
        for text in TRACER_WORKFLOW_FIELD_DESCRIPTIONS.values():
            assert "PREVENTS" not in text


# ----------------------------------------------------------------------
# Decision 9: tool-specific descriptions reach the published schema
# ----------------------------------------------------------------------

RELEVANT_CONTEXT_TEXT = {
    "analyze": ANALYZE_WORKFLOW_FIELD_DESCRIPTIONS["relevant_context"],
    "codereview": CODEREVIEW_WORKFLOW_FIELD_DESCRIPTIONS["relevant_context"],
    "debug": DEBUG_INVESTIGATION_FIELD_DESCRIPTIONS["relevant_context"],
    "precommit": PRECOMMIT_WORKFLOW_FIELD_DESCRIPTIONS["relevant_context"],
    "refactor": REFACTOR_FIELD_DESCRIPTIONS["relevant_context"],
    "secaudit": SECAUDIT_WORKFLOW_FIELD_DESCRIPTIONS["relevant_context"],
    "testgen": TESTGEN_WORKFLOW_FIELD_DESCRIPTIONS["relevant_context"],
    "tracer": TRACER_WORKFLOW_FIELD_DESCRIPTIONS["relevant_context"],
}


class TestToolSpecificDescriptionsArePublished:
    @pytest.mark.parametrize("name", sorted(RELEVANT_CONTEXT_TEXT))
    def test_relevant_context_carries_the_tool_specific_text(self, name):
        entry = _schema(name)["properties"]["relevant_context"]
        assert entry["description"] == RELEVANT_CONTEXT_TEXT[name]
        assert entry["description"] != WORKFLOW_FIELD_DESCRIPTIONS["relevant_context"]
        assert entry["type"] == "array" and entry["items"] == {"type": "string"}

    def test_tracer_publishes_its_own_text_for_every_shared_workflow_field(self):
        properties = _schema("tracer")["properties"]
        for field in (
            "step",
            "step_number",
            "total_steps",
            "next_step_required",
            "findings",
            "files_checked",
            "relevant_files",
            "relevant_context",
            "confidence",
        ):
            assert properties[field]["description"] == TRACER_WORKFLOW_FIELD_DESCRIPTIONS[field], field
            assert properties[field]["description"] != WORKFLOW_FIELD_DESCRIPTIONS[field], field
        assert properties["step_number"]["minimum"] == 1 and properties["total_steps"]["minimum"] == 1


# ----------------------------------------------------------------------
# Decision 5a: thinkdeep runtime fixes
# ----------------------------------------------------------------------


def _thinkdeep_request(**overrides) -> ThinkDeepWorkflowRequest:
    values = {"step": "s", "step_number": 1, "total_steps": 3, "next_step_required": True, "findings": "f"}
    values.update(overrides)
    return ThinkDeepWorkflowRequest(**values)


class TestThinkDeepExploringStep:
    def test_an_exploring_step_is_not_told_the_analysis_is_complete(self):
        actions = ThinkDeepTool().get_required_actions(2, "exploring", "f", 3)
        assert not any("Analysis complete" in action for action in actions)
        assert any("Continue gathering evidence" in action for action in actions)

    def test_exploring_gets_the_same_actions_as_low(self):
        tool = ThinkDeepTool()
        assert tool.get_required_actions(2, "exploring", "f", 3) == tool.get_required_actions(2, "low", "f", 3)

    def test_only_certain_is_told_the_analysis_is_complete(self):
        tool = ThinkDeepTool()
        assert tool.get_required_actions(2, "certain", "f", 3) == ["Analysis complete - ready for implementation"]


class TestThinkDeepUseAssistantModel:
    """thinkdeep publishes use_assistant_model and honors it, like the other workflow tools."""

    def _consolidated(self):
        from tools.shared.base_models import ConsolidatedFindings

        findings = ConsolidatedFindings()
        findings.findings.extend(["one", "two"])  # enough for the fallback rule too
        return findings

    def _final(self, **overrides):
        return _thinkdeep_request(next_step_required=False, confidence="high", **overrides)

    def test_the_published_schema_carries_the_field_with_the_workflow_description(self):
        for schema in (_schema("thinkdeep"), _auto_mode_schema("thinkdeep")):
            prop = schema["properties"]["use_assistant_model"]
            assert prop["type"] == "boolean" and prop["default"] is True
            assert prop["description"] == WORKFLOW_FIELD_DESCRIPTIONS["use_assistant_model"]
            assert "use_assistant_model" not in schema["required"]
            jsonschema.validate(instance={**_valid_call("thinkdeep"), "use_assistant_model": False}, schema=schema)

    def test_false_skips_expert_analysis(self):
        tool = ThinkDeepTool()
        request = self._final(use_assistant_model=False)
        assert tool.should_call_expert_analysis(self._consolidated(), request) is False

    def test_true_and_the_default_keep_expert_analysis(self):
        tool = ThinkDeepTool()
        assert tool.should_call_expert_analysis(self._consolidated(), self._final(use_assistant_model=True)) is True
        default = self._final()
        assert default.use_assistant_model is True
        assert tool.should_call_expert_analysis(self._consolidated(), default) is True

    def test_false_on_a_final_step_ends_in_local_work_complete_without_calling_a_model(self):
        import asyncio
        from unittest.mock import AsyncMock

        tool = ThinkDeepTool()
        tool.consolidated_findings = self._consolidated()
        tool._call_expert_analysis = AsyncMock(return_value={"status": "analysis_complete"})
        for flag, expected_calls in ((False, 0), (True, 1)):
            tool._call_expert_analysis.reset_mock()
            request = self._final(use_assistant_model=flag)
            response = asyncio.run(tool.handle_work_completion({}, request, {}))
            assert tool._call_expert_analysis.await_count == expected_calls
            assert response["status"] == ("calling_expert_analysis" if flag else "local_work_complete")

    def test_certain_still_skips_whatever_the_flag_says(self):
        tool = ThinkDeepTool()
        request = _thinkdeep_request(next_step_required=False, confidence="certain", use_assistant_model=True)
        assert tool.should_call_expert_analysis(self._consolidated(), request) is False


class TestThinkDeepUsesTheCurrentRequest:
    def test_the_final_step_temperature_wins_over_an_earlier_call(self):
        tool = ThinkDeepTool()
        first = _thinkdeep_request(temperature=0.9, thinking_mode="low")
        tool.customize_workflow_response({}, first)  # what the cached tool instance saw earlier
        final = _thinkdeep_request(step_number=3, temperature=0.3, thinking_mode="max", next_step_required=True)
        assert tool.get_request_temperature(final) == 0.3
        assert tool.get_request_thinking_mode(final) == "max"

    def test_a_final_step_that_sets_nothing_gets_the_defaults_not_the_earlier_values(self):
        tool = ThinkDeepTool()
        tool.customize_workflow_response({}, _thinkdeep_request(temperature=0.9, thinking_mode="low"))
        final = _thinkdeep_request(step_number=3)
        assert tool.get_request_temperature(final) == tool.get_default_temperature() == 1.0
        with patch("config.DEFAULT_THINKING_MODE_THINKDEEP", "medium"):
            assert tool.get_request_thinking_mode(final) == "medium"

    def test_the_tool_keeps_no_per_instance_request_parameters(self):
        tool = ThinkDeepTool()
        tool.customize_workflow_response({}, _thinkdeep_request(temperature=0.9))
        assert not hasattr(tool, "stored_request_params")

    def test_the_default_thinking_mode_is_the_configured_thinkdeep_default(self):
        from config import DEFAULT_THINKING_MODE_THINKDEEP

        tool = ThinkDeepTool()
        assert tool.get_expert_thinking_mode() == tool.get_default_thinking_mode() == DEFAULT_THINKING_MODE_THINKDEEP
