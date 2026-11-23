"""Offline checks of the refactor response parser in the line_number_validation simulator.

The real server wraps the refactor expert's structured JSON in the workflow envelope
(status "calling_expert_analysis", payload under "expert_analysis"), so the parser has to
look there as well as at the top level. No network, no server.
"""

import json

import pytest

from simulator_tests.test_line_number_validation import LineNumberValidationTest

LINE_COUNT = 22


def _check(payload: dict) -> bool:
    return LineNumberValidationTest(verbose=False)._check_refactor_line_numbers(json.dumps(payload), LINE_COUNT)


def _nested(opportunities) -> dict:
    return {
        "status": "calling_expert_analysis",
        "expert_analysis": {"status": "refactor_analysis_complete", "refactor_opportunities": opportunities},
    }


def test_nested_structured_payload_with_valid_ranges_passes():
    assert _check(_nested([{"start_line": 5, "end_line": 11}]))


def test_nested_structured_payload_with_out_of_range_lines_fails():
    assert not _check(_nested([{"start_line": 5, "end_line": LINE_COUNT + 5}]))


def test_nested_structured_payload_without_line_numbers_fails():
    assert not _check(_nested([{"description": "no lines"}]))


def test_nested_structured_payload_with_no_opportunities_passes():
    assert _check(_nested([]))


def test_top_level_structured_payload_still_passes():
    assert _check(
        {"status": "refactor_analysis_complete", "refactor_opportunities": [{"start_line": 1, "end_line": 2}]}
    )


@pytest.mark.parametrize("text,expected", [("Duplicate logic on line 7", True), ("Looks fine overall", False)])
def test_envelope_text_shape(text, expected):
    payload = {"status": "calling_expert_analysis", "expert_analysis": {"raw_analysis": text}}
    assert _check(payload) is expected
