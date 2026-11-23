"""Schema-related tests for ConsensusTool."""

from types import MethodType

from tools.consensus import ConsensusTool


def test_consensus_models_field_includes_available_models(monkeypatch):
    """Consensus schema should surface available model guidance like single-model tools."""

    tool = ConsensusTool()

    monkeypatch.setattr(
        tool,
        "_get_ranked_model_summaries",
        MethodType(lambda self, limit=5: (["gemini-2.5-pro (score 100, 1.0M ctx, thinking)"], 1, False), tool),
    )
    monkeypatch.setattr(tool, "_get_restriction_note", MethodType(lambda self: None, tool))

    schema = tool.get_input_schema()
    models_field_description = schema["properties"]["models"]["description"]

    assert "listmodels" in models_field_description
    assert "Top models" in models_field_description


# ---------------------------------------------------------------------------
# The published step contract: one call, all models concurrently, finished at step 1
# ---------------------------------------------------------------------------

_STEP_FIELDS = (
    "step",
    "step_number",
    "total_steps",
    "next_step_required",
    "findings",
)

# Wording that described the removed sequential flow (one step per model, then a synthesis step).
_REMOVED_FLOW_PHRASES = (
    "final synthesis",  # the removed flow had a final synthesis step; the description may only say there is none
    "plus the final",
    "consultations remain",
    "each model response",
    "steps 2+",
    "latest model response",
    "newest model response",
    "next model to consult",
    "gathered so far",
    "structured debate",
)


def _published() -> dict:
    return ConsensusTool().get_input_schema()["properties"]


def test_step_field_descriptions_describe_the_single_step_contract():
    properties = _published()
    step_number = properties["step_number"]["description"].lower()
    total_steps = properties["total_steps"]["description"].lower()
    next_step_required = properties["next_step_required"]["description"].lower()

    assert "send 1" in step_number and "concurrently" in step_number and "no per-model steps" in step_number
    assert (
        "send 1" in total_steps
        and "no per-model steps" in total_steps
        and "no server-side synthesis step" in total_steps
    )
    assert "whatever value you send" in total_steps
    assert "send false" in next_step_required and "ignores this value" in next_step_required
    assert "carries no next_step_required field" in next_step_required
    assert "consensus_complete true" in next_step_required


def test_step_descriptions_name_the_step_one_responses_that_are_not_consensus_workflow_complete():
    """Step 1 can also answer all_providers_unavailable, or status running with a batch_handle under batch."""
    properties = _published()
    next_step_required = properties["next_step_required"]["description"]
    assert "all_providers_unavailable" in next_step_required
    assert "batch_handle" in next_step_required and "status running" in next_step_required
    step_number = properties["step_number"]["description"]
    assert "batch_handle" in step_number and "running" in step_number


def test_no_published_step_description_mentions_the_removed_sequential_flow():
    properties = _published()
    for name in _STEP_FIELDS:
        text = properties[name]["description"].lower()
        for phrase in _REMOVED_FLOW_PHRASES:
            assert phrase not in text, f"{name}: {phrase!r}"
    description = ConsensusTool().get_description().lower()
    for phrase in _REMOVED_FLOW_PHRASES:
        assert phrase not in description, phrase


def test_request_model_and_published_schema_carry_the_same_step_descriptions():
    from tools.consensus import CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS, ConsensusRequest

    properties = _published()
    for name in _STEP_FIELDS:
        assert properties[name]["description"] == CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS[name], name
        assert ConsensusRequest.model_fields[name].description == CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS[name], name


def test_step_and_findings_descriptions_say_what_is_sent_to_the_models():
    properties = _published()
    step = properties["step"]["description"]
    findings = properties["findings"]["description"].lower()
    assert step.startswith("Consensus prompt")
    assert "every consulted model" in step and "agent_analysis.initial_analysis" in step
    assert "never sent to the consulted models" in findings and "agent_analysis.findings" in findings


def test_models_description_is_stated_once_and_names_concurrency_and_order():
    description = _published()["models"]["description"]
    assert description.count("provide at least two entries") == 1
    assert description.startswith("Models to consult")
    assert "concurrently" in description and "order listed" in description


def test_tool_description_states_the_one_call_contract():
    description = ConsensusTool().get_description()
    assert "step_number 1, total_steps 1, next_step_required false" in description
    assert "concurrently" in description and "accumulated_responses" in description
    assert "does not synthesize for you" in description


def test_prompt_and_registry_descriptions_do_not_describe_a_step_by_step_single_model_flow():
    from conf.prompt_templates import PROMPT_TEMPLATES
    from tools.registry import TOOL_DEFINITIONS

    entry = PROMPT_TEMPLATES["consensus"]
    registry_description = TOOL_DEFINITIONS["consensus"][2]
    for text in (entry["description"], registry_description):
        assert "step-by-step" not in text.lower()
        assert "concurrently" in text
    # consensus takes a `models` roster, not a single model, so the prompt must not name one
    assert "{model}" not in entry["template"] and "`models`" in entry["template"]


def test_retired_legacy_fields_are_not_published():
    """current_model_index and model_responses described a sequential flow that no longer exists."""
    from tools.consensus import CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS, ConsensusRequest

    schema = ConsensusTool().get_input_schema()
    for name in ("current_model_index", "model_responses"):
        assert name not in schema["properties"], name
        assert name not in schema.get("required", []), name
        assert name not in CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS, name
        assert name not in ConsensusRequest.model_fields, name
