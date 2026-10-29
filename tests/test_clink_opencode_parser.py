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
    # The flat ``{"message": ...}`` shape is kept as a fallback; no released
    # opencode emits it (v2 nests {type, message}, v1 nests {name, data}).
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


# ---------------------------------------------------------------------------
# Real opencode output, captured with isolated XDG dirs against a local mock
# provider (temp paths rewritten to /tmp/project). The flat error shape used by
# the older synthetic tests above is not what any released version emits.
# ---------------------------------------------------------------------------

# opencode v2.2.0, one text step: no step events and no token summary at all
V2_TEXT_ONLY = "\n".join(
    [
        r'{"type":"text","timestamp":1793218794349,"sessionID":"ses_e7bd4ae9bffee9ewqVPWXedPL5","part":{"id":"prt_1842b533a001DLTvvUYzxIppre_text-0","sessionID":"ses_e7bd4ae9bffee9ewqVPWXedPL5","messageID":"msg_1842b533a001DLTvvUYzxIppre","type":"text","text":"PONG","time":{"start":1793218794302,"end":1793218794349}}}',
    ]
)

# opencode v2.2.0, a shell tool step then the final text step; the final step has neither step_start nor step_finish (upstream #59652)
V2_MULTI_STEP_DROPPED_FINAL_SUMMARY = "\n".join(
    [
        r'{"type":"step_start","timestamp":1793218802854,"sessionID":"ses_e7bd498b2ffeaoKcNmKEXXEr6i","part":{"id":"prt_1842b74a600142hTJkR4oiRF0b","sessionID":"ses_e7bd498b2ffeaoKcNmKEXXEr6i","messageID":"msg_1842b7484001bbrRgtz5QWGhXV","type":"step-start"}}',
        r'{"type":"tool_use","timestamp":1793218803240,"sessionID":"ses_e7bd498b2ffeaoKcNmKEXXEr6i","part":{"partID":"prt_1842b74ad001ddzKPmFAo1qfLB","sessionID":"ses_e7bd498b2ffeaoKcNmKEXXEr6i","messageID":"msg_1842b7484001bbrRgtz5QWGhXV","type":"tool","id":"call_mock_1","tool":"shell","state":{"status":"completed","input":{"command":"echo MOCK_TOOL_RAN"},"output":"MOCK_TOOL_RAN\n","title":"shell","metadata":{"metadata":{"status":"completed","truncated":false,"exit":0},"content":[{"type":"text","text":"MOCK_TOOL_RAN\n"}],"providerCall":{"executed":false},"providerResult":{"executed":false},"rawInput":"{\"command\": \"echo MOCK_TOOL_RAN\"}"},"time":{"start":1793218802861,"end":1793218803240}}}}',
        r'{"type":"step_finish","timestamp":1793218803242,"sessionID":"ses_e7bd498b2ffeaoKcNmKEXXEr6i","part":{"id":"prt_1842b762a001o1h38WfhrYeRGf","sessionID":"ses_e7bd498b2ffeaoKcNmKEXXEr6i","messageID":"msg_1842b7484001bbrRgtz5QWGhXV","type":"step-finish","reason":"tool-calls","cost":0,"tokens":{"input":10108,"output":20,"reasoning":0,"cache":{"read":0,"write":0}}}}',
        r'{"type":"text","timestamp":1793218803280,"sessionID":"ses_e7bd498b2ffeaoKcNmKEXXEr6i","part":{"id":"prt_1842b76310015koyU5XmXmFFIb_text-0","sessionID":"ses_e7bd498b2ffeaoKcNmKEXXEr6i","messageID":"msg_1842b76310015koyU5XmXmFFIb","type":"text","text":"TOOL_DONE","time":{"start":1793218803252,"end":1793218803280}}}',
    ]
)

# opencode v2.2.0, three steps (two tool calls then text) with every step_finish reported
V2_MULTI_STEP_COMPLETE = "\n".join(
    [
        r'{"type":"step_start","timestamp":1793218898316,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"id":"prt_1842ce98c001yod0eSaybSHCaI","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842ce78800126ruH1jbv7jcPp","type":"step-start"}}',
        r'{"type":"tool_use","timestamp":1793218901546,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"partID":"prt_1842ce9b50017qAHtIjPe3JFeE","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842ce78800126ruH1jbv7jcPp","type":"tool","id":"call_mock_1","tool":"shell","state":{"status":"completed","input":{"command":"sleep 2"},"output":"(no output)","title":"shell","metadata":{"metadata":{"status":"completed","truncated":false,"exit":0},"content":[{"type":"text","text":"(no output)"}],"providerCall":{"executed":false},"providerResult":{"executed":false},"rawInput":"{\"command\": \"sleep 2\"}"},"time":{"start":1793218898357,"end":1793218901546}}}}',
        r'{"type":"step_finish","timestamp":1793218902013,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"id":"prt_1842cf7fd001PL7WqXSEEdpkUK","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842ce78800126ruH1jbv7jcPp","type":"step-finish","reason":"tool-calls","cost":0,"tokens":{"input":10108,"output":20,"reasoning":0,"cache":{"read":0,"write":0}}}}',
        r'{"type":"step_start","timestamp":1793218903625,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"id":"prt_1842cfe49001QoO3RKQHlHI9E3","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842cfd30001h60gmSBlXf6AUV","type":"step-start"}}',
        '{"type":"tool_use","timestamp":1793218904514,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"partID":"prt_1842cfee4001l7wzEBlNWbJK7n","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842cfd30001h60gmSBlXf6AUV","type":"tool","id":"call_mock_2","tool":"execute","state":{"status":"completed","input":{"code":"return await tools.fake.clink_probe_tool({ message: \'hello-mcp\' })"},"output":"PROBE_OK hello-mcp","title":"execute","metadata":{"metadata":{"toolCalls":[{"tool":"fake.clink_probe_tool","status":"completed","input":{"message":"hello-mcp"}}],"truncated":false},"content":[{"type":"text","text":"PROBE_OK hello-mcp"}],"providerCall":{"executed":false},"providerResult":{"executed":false},"rawInput":"{\\"code\\": \\"return await tools.fake.clink_probe_tool({ message: \'hello-mcp\' })\\"}"},"time":{"start":1793218903780,"end":1793218904514}}}}',
        r'{"type":"step_finish","timestamp":1793218904522,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"id":"prt_1842d01ca001PVNt7iCklxIg20","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842cfd30001h60gmSBlXf6AUV","type":"step-finish","reason":"tool-calls","cost":0,"tokens":{"input":10261,"output":20,"reasoning":0,"cache":{"read":0,"write":0}}}}',
        r'{"type":"step_start","timestamp":1793218905294,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"id":"prt_1842d04ce001w19SuRJaA1HJ2K","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842d03e7001jlkhep3WYZuoYG","type":"step-start"}}',
        r'{"type":"text","timestamp":1793218905500,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"id":"prt_1842d04df001nm2P1mhbTMsK6R","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842d03e7001jlkhep3WYZuoYG","type":"text","text":"MCP_DONE","time":{"start":1793218905311,"end":1793218905500}}}',
        r'{"type":"step_finish","timestamp":1793218905543,"sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","part":{"id":"prt_1842d05c7001nt3tgvXtSB9cz1","sessionID":"ses_e7bd3777cffel4CWzEpVRtSEsd","messageID":"msg_1842d03e7001jlkhep3WYZuoYG","type":"step-finish","reason":"stop","cost":0,"tokens":{"input":10338,"output":1,"reasoning":0,"cache":{"read":0,"write":0}}}}',
    ]
)

# opencode v2.2.0 -m nosuch/model: exit 1, empty stderr, reason only on stdout
V2_ERROR_NO_ROUTE = "\n".join(
    [
        r'{"type":"error","timestamp":1793218807856,"sessionID":"ses_e7bd47aeeffe01qd9F5H1UOa74","error":{"type":"provider.no-route","message":"Model unavailable: nosuch/model"}}',
    ]
)

# opencode v2.2.0, provider answered 404 model_not_found after a step_start
V2_ERROR_PROVIDER_REJECTS = "\n".join(
    [
        r'{"type":"step_start","timestamp":1793218805759,"sessionID":"ses_e7bd4857effeDg1IIG3dJtMFMy","part":{"id":"prt_1842b7fff001aCUAOEa55fm7JN","sessionID":"ses_e7bd4857effeDg1IIG3dJtMFMy","messageID":"msg_1842b7fc0001TtP2CWxTdfRXP7","type":"step-start"}}',
        r'{"type":"error","timestamp":1793218805761,"sessionID":"ses_e7bd4857effeDg1IIG3dJtMFMy","error":{"type":"provider.invalid-request","message":"The model `err` does not exist or you do not have access to it.","status":404,"response":{"body":"{\"error\": {\"message\": \"The model `err` does not exist or you do not have access to it.\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": \"model_not_found\"}}"}}}',
    ]
)

# opencode 1.18.35 session error: {name, data:{message}} shape
V1_ERROR = "\n".join(
    [
        r'{"type":"error","timestamp":1793218818566,"sessionID":"ses_e7bd45324ffeOu3Gcvr5EUC2Y8","error":{"name":"UnknownError","data":{"message":"Unexpected server error. Check server logs for details.","ref":"err_6d1e8a71"}}}',
    ]
)


def _parse(stdout: str):
    return OpencodeJSONLParser().parse(stdout=stdout, stderr="")


def test_v2_text_only_stream_has_no_usage_and_is_flagged_incomplete():
    parsed = _parse(V2_TEXT_ONLY)

    assert parsed.content == "PONG"
    assert "usage" not in parsed.metadata
    assert parsed.metadata["usage_complete"] is False
    assert parsed.metadata["session_id"].startswith("ses_")
    assert "errors" not in parsed.metadata
    assert "error_only" not in parsed.metadata


def test_v2_usage_is_summed_across_every_step_including_nested_cache_counts():
    parsed = _parse(V2_MULTI_STEP_COMPLETE)

    assert parsed.content == "MCP_DONE"
    assert parsed.metadata["usage"] == {
        "input": 10108 + 10261 + 10338,
        "output": 20 + 20 + 1,
        "reasoning": 0,
        "cache": {"read": 0, "write": 0},
    }
    assert parsed.metadata["usage_complete"] is True


def test_v2_dropped_final_summary_is_flagged_even_without_a_matching_step_start():
    # The final step produced the text but has neither step_start nor step_finish,
    # so only the first step's tokens are reported.
    parsed = _parse(V2_MULTI_STEP_DROPPED_FINAL_SUMMARY)

    assert parsed.content == "TOOL_DONE"
    assert parsed.metadata["usage"]["input"] == 10108
    assert parsed.metadata["usage_complete"] is False


def test_v2_session_id_is_recorded_from_the_first_event_that_has_one():
    parsed = _parse(V2_MULTI_STEP_COMPLETE)

    assert parsed.metadata["session_id"] == json.loads(V2_MULTI_STEP_COMPLETE.splitlines()[0])["sessionID"]


def test_v2_nested_error_is_surfaced_as_type_and_message():
    parsed = _parse(V2_ERROR_NO_ROUTE)

    assert parsed.content == "provider.no-route: Model unavailable: nosuch/model"
    assert parsed.metadata["errors"] == ["provider.no-route: Model unavailable: nosuch/model"]
    assert parsed.metadata["error_only"] is True
    assert parsed.metadata["usage_complete"] is False


def test_v2_provider_rejection_after_a_step_start_is_error_only():
    parsed = _parse(V2_ERROR_PROVIDER_REJECTS)

    assert parsed.metadata["error_only"] is True
    assert parsed.content.startswith("provider.invalid-request: The model `err` does not exist")
    assert parsed.metadata["usage_complete"] is False


def test_v1_nested_error_uses_name_and_data_message():
    parsed = _parse(V1_ERROR)

    assert parsed.content == "UnknownError: Unexpected server error. Check server logs for details."
    assert parsed.metadata["error_only"] is True


@pytest.mark.parametrize(
    "error,expected",
    [
        ({"message": "boom"}, "boom"),
        ({"type": "provider.timeout"}, "provider.timeout"),
        ({"name": "AbortedError", "data": {"message": "stopped"}}, "AbortedError: stopped"),
        ({"type": "t", "name": "n", "message": "m"}, "t: m"),
    ],
)
def test_error_shapes_without_both_type_and_message(error, expected):
    parsed = _parse(_jsonl({"type": "error", "error": error}))

    assert parsed.metadata["errors"] == [expected]


def test_text_plus_error_keeps_the_text_and_is_not_error_only():
    parsed = _parse(
        _jsonl(
            {"type": "text", "part": {"text": "partial answer"}},
            {"type": "error", "error": {"type": "provider.timeout", "message": "stream stalled"}},
        )
    )

    assert parsed.content == "partial answer"
    assert parsed.metadata["errors"] == ["provider.timeout: stream stalled"]
    assert "error_only" not in parsed.metadata


def test_usage_sums_numbers_recurses_into_nested_counts_and_keeps_the_first_non_numeric_value():
    parsed = _parse(
        _jsonl(
            {"type": "step_finish", "part": {"tokens": {"input": 5, "cache": {"read": 2}, "tier": "a"}}},
            {"type": "step_finish", "part": {"tokens": {"input": 7, "cache": {"read": 3, "write": 1}, "tier": "b"}}},
            {"type": "text", "part": {"text": "x"}},
        )
    )

    assert parsed.metadata["usage"] == {"input": 12, "cache": {"read": 5, "write": 1}, "tier": "a"}


def test_legacy_summary_without_step_events_counts_as_complete():
    parsed = _parse(
        _jsonl(
            {"type": "text", "text": "Hello"},
            {"type": "step_end", "usage": {"input_tokens": 10, "output_tokens": 5}},
        )
    )

    assert parsed.metadata["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert parsed.metadata["usage_complete"] is True


def test_more_step_starts_than_summaries_is_incomplete_even_without_message_ids():
    parsed = _parse(
        _jsonl(
            {"type": "step_start"},
            {"type": "step_finish", "part": {"tokens": {"input": 1}}},
            {"type": "step_start"},
            {"type": "text", "part": {"text": "x"}},
        )
    )

    assert parsed.metadata["usage"] == {"input": 1}
    assert parsed.metadata["usage_complete"] is False


def test_an_unusable_error_object_falls_back_to_the_top_level_message():
    parsed = _parse(_jsonl({"type": "error", "message": "boom", "error": {}}))

    assert parsed.metadata["errors"] == ["boom"]
    assert parsed.content == "boom"
