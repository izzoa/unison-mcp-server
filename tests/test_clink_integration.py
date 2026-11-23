import json
import os
import shutil
import time

import pytest

from tools.clink import CLinkTool

# clink waits a bounded 45 s per call and, when the CLI needs longer (a cold CLI start under load easily
# does), returns a "running" job handle instead of the answer. Callers are told to call clink again with only
# the job_id; these tests do the same, for up to this long in total.
_MAX_TOTAL_WAIT_SECONDS = 300


def _running_job_id(payload: dict) -> str | None:
    """The job_id of a still-running clink job, whether reported at the top level or as the content."""

    if payload.get("status") == "running" and payload.get("job_id"):
        return payload["job_id"]
    content = payload.get("content")
    if isinstance(content, str) and content.lstrip().startswith("{"):
        try:
            inner = json.loads(content)
        except ValueError:
            return None
        if isinstance(inner, dict) and inner.get("status") == "running" and inner.get("job_id"):
            return inner["job_id"]
    return None


async def _final_payload(tool: CLinkTool, results) -> dict:
    """Follow clink's documented job protocol until the CLI's final result arrives."""

    deadline = time.monotonic() + _MAX_TOTAL_WAIT_SECONDS
    payload = json.loads(results[0].text)
    while (job_id := _running_job_id(payload)) is not None:
        assert time.monotonic() < deadline, f"clink job {job_id} still running after {_MAX_TOTAL_WAIT_SECONDS} s"
        results = await tool.execute({"job_id": job_id})
        assert results, "clink tool returned no outputs while polling a running job"
        payload = json.loads(results[0].text)
    return payload


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clink_gemini_single_digit_sum():
    if shutil.which("gemini") is None:
        pytest.skip("gemini CLI is not installed or on PATH")

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        pytest.skip("Gemini API key is not configured")

    tool = CLinkTool()
    prompt = "Respond with a single digit equal to the sum of 2 + 2. Output only that digit."

    results = await tool.execute(
        {
            "prompt": prompt,
            "cli_name": "gemini",
            "role": "default",
            "absolute_file_paths": [],
            "images": [],
        }
    )

    assert results, "clink tool returned no outputs"
    payload = await _final_payload(tool, results)
    status = payload["status"]
    assert status in {"success", "continuation_available"}

    content = payload.get("content", "").strip()
    # CLI may include additional metadata like <SUMMARY> tags; check first line or that "4" is present
    first_line = content.split("\n")[0].strip()
    assert first_line == "4" or "4" in content, f"Expected '4' in response, got: {content[:100]}"

    if status == "continuation_available":
        offer = payload.get("continuation_offer") or {}
        assert offer.get("continuation_id"), "Expected continuation metadata when status indicates availability"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clink_claude_single_digit_sum():
    if shutil.which("claude") is None:
        pytest.skip("claude CLI is not installed or on PATH")

    tool = CLinkTool()
    prompt = "Respond with a single digit equal to the sum of 2 + 2. Output only that digit."

    results = await tool.execute(
        {
            "prompt": prompt,
            "cli_name": "claude",
            "role": "default",
            "absolute_file_paths": [],
            "images": [],
        }
    )

    assert results, "clink tool returned no outputs"
    payload = await _final_payload(tool, results)
    status = payload["status"]

    if status == "error":
        metadata = payload.get("metadata") or {}
        reason = payload.get("content") or metadata.get("message") or "Claude CLI reported an error"
        pytest.skip(f"Skipping Claude integration test: {reason}")

    assert status in {"success", "continuation_available"}

    content = payload.get("content", "").strip()
    if content.startswith("API Error:"):
        # The CLI itself failed for an account or environment reason (for example a model that needs usage
        # credits) and clink relays its message as the content. Same treatment as the error status above.
        pytest.skip(f"Skipping Claude integration test: the claude CLI reported: {content[:200]}")
    assert content == "4"

    if status == "continuation_available":
        offer = payload.get("continuation_offer") or {}
        assert offer.get("continuation_id"), "Expected continuation metadata when status indicates availability"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clink_codex_single_digit_sum():
    if shutil.which("codex") is None:
        pytest.skip("codex CLI is not installed or on PATH")

    tool = CLinkTool()
    prompt = "Respond with a single digit equal to the sum of 2 + 2. Output only that digit. Do not use any tools."

    results = await tool.execute(
        {
            "prompt": prompt,
            "cli_name": "codex",
            "role": "default",
            "absolute_file_paths": [],
            "images": [],
        }
    )

    assert results, "clink tool returned no outputs"
    payload = await _final_payload(tool, results)
    status = payload["status"]

    if status == "error":
        metadata = payload.get("metadata") or {}
        reason = payload.get("content") or metadata.get("message") or "Codex CLI reported an error"
        pytest.skip(f"Skipping Codex integration test: {reason}")

    assert status in {"success", "continuation_available"}

    content = payload.get("content", "").strip()
    first_line = content.split("\n")[0].strip()
    assert first_line == "4" or "4" in content, f"Expected '4' in response, got: {content[:200]}"

    if status == "continuation_available":
        offer = payload.get("continuation_offer") or {}
        assert offer.get("continuation_id"), "Expected continuation metadata when status indicates availability"
