"""Tests for clink/parsers/opencode.py."""

from __future__ import annotations

import json

import pytest

from clink.parsers.base import ParserError
from clink.parsers.opencode import OpencodeJSONLParser


def _jsonl(*events: dict) -> str:
    return "\n".join(json.dumps(e) for e in events)


def test_joins_assistant_text_chunks_with_double_newline():
    parser = OpencodeJSONLParser()
    stdout = _jsonl(
        {"type": "step_start"},
        {"type": "text", "text": "First chunk"},
        {"type": "tool_use", "name": "read_file"},
        {"type": "text", "text": "Second chunk"},
    )
    parsed = parser.parse(stdout=stdout, stderr="")
    assert parsed.content == "First chunk\n\nSecond chunk"


def test_extracts_usage_and_model_used_from_summary_event():
    parser = OpencodeJSONLParser()
    stdout = _jsonl(
        {"type": "text", "text": "Hello"},
        {
            "type": "step_end",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model_used": "anthropic/claude-sonnet-4-5",
        },
    )
    parsed = parser.parse(stdout=stdout, stderr="")
    assert parsed.metadata["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert parsed.metadata["model_used"] == "anthropic/claude-sonnet-4-5"


def test_preserves_non_text_events_under_metadata_events():
    parser = OpencodeJSONLParser()
    stdout = _jsonl(
        {"type": "step_start"},
        {"type": "tool_use", "name": "grep"},
        {"type": "text", "text": "result"},
    )
    parsed = parser.parse(stdout=stdout, stderr="")
    types = [e.get("type") for e in parsed.metadata["events"]]
    assert types == ["step_start", "tool_use", "text"]
    assert "step_start" not in parsed.content
    assert "tool_use" not in parsed.content


def test_defensive_parsing_tolerates_missing_optional_fields():
    parser = OpencodeJSONLParser()
    stdout = _jsonl(
        {"type": "text", "text": "minimal"},
        {"type": "step_end"},  # no usage, no model_used
    )
    parsed = parser.parse(stdout=stdout, stderr="")
    assert parsed.content == "minimal"
    assert "usage" not in parsed.metadata
    assert "model_used" not in parsed.metadata


def test_skips_non_json_lines_without_failing():
    parser = OpencodeJSONLParser()
    stdout = "Garbage prelude\n" + _jsonl({"type": "text", "text": "after garbage"})
    parsed = parser.parse(stdout=stdout, stderr="")
    assert parsed.content == "after garbage"


def test_empty_stdout_raises_parser_error():
    parser = OpencodeJSONLParser()
    with pytest.raises(ParserError):
        parser.parse(stdout="", stderr="")


def test_unparseable_stdout_raises_parser_error():
    parser = OpencodeJSONLParser()
    with pytest.raises(ParserError):
        parser.parse(stdout="not json at all\nstill not json", stderr="")


def test_events_with_no_text_raises_parser_error_unless_errors_present():
    parser = OpencodeJSONLParser()
    stdout = _jsonl({"type": "step_start"}, {"type": "tool_use"})
    with pytest.raises(ParserError):
        parser.parse(stdout=stdout, stderr="")


def test_error_event_used_as_content_when_no_text():
    parser = OpencodeJSONLParser()
    stdout = _jsonl({"type": "error", "message": "rate limit"})
    parsed = parser.parse(stdout=stdout, stderr="")
    assert "rate limit" in parsed.content
    assert parsed.metadata["errors"] == ["rate limit"]


def test_stderr_surfaced_in_metadata_when_non_empty():
    parser = OpencodeJSONLParser()
    stdout = _jsonl({"type": "text", "text": "ok"})
    parsed = parser.parse(stdout=stdout, stderr=" warning: something \n")
    assert parsed.metadata["stderr"] == "warning: something"


def test_handles_nested_content_dict_in_text_event():
    parser = OpencodeJSONLParser()
    stdout = _jsonl({"type": "text", "content": {"text": "nested"}})
    parsed = parser.parse(stdout=stdout, stderr="")
    assert parsed.content == "nested"


def test_real_opencode_event_shape_with_part_wrapper():
    """Regression: opencode wraps text/tokens under event['part'] in real output.

    Captured from `echo 'hi' | opencode run --format json -m anthropic/claude-sonnet-4-5`
    against opencode v0.18.x. The parser must handle both flat and part-wrapped
    shapes so future opencode versions that flatten don't break either path.
    """
    parser = OpencodeJSONLParser()
    stdout = _jsonl(
        {
            "type": "step_start",
            "part": {"id": "prt_x", "type": "step-start"},
        },
        {
            "type": "text",
            "part": {"type": "text", "text": "ok!", "time": {"start": 1, "end": 2}},
        },
        {
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "reason": "stop",
                "tokens": {"total": 100, "input": 2, "output": 5},
                "cost": 0.05,
            },
        },
    )
    parsed = parser.parse(stdout=stdout, stderr="")
    assert parsed.content == "ok!"
    assert parsed.metadata["usage"] == {"total": 100, "input": 2, "output": 5}
