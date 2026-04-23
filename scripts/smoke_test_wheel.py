#!/usr/bin/env python3
"""Smoke test: verify an installed unison-mcp-server wheel can complete the MCP initialize handshake.

Spawns the installed ``unison-mcp-server`` entry point as a subprocess, sends a JSON-RPC
``initialize`` request on stdin, reads the response from stdout, and validates it. Any
import-time failure (e.g. a missing subpackage in the wheel) exits the subprocess before
it writes a response, which this script then reports as a failure with the captured
stderr so the root cause is visible in CI logs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Final

TIMEOUT_SECONDS: Final[int] = 30

INITIALIZE_REQUEST: Final[dict[str, object]] = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "wheel-smoke-test", "version": "1.0"},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry-point",
        default="unison-mcp-server",
        help="Path to the installed unison-mcp-server executable (default: unison-mcp-server on PATH).",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("GEMINI_API_KEY", "dummy")

    request_line = json.dumps(INITIALIZE_REQUEST) + "\n"

    try:
        proc = subprocess.run(
            [args.entry_point],
            input=request_line,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr_tail = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
        sys.stderr.write(f"SMOKE TEST FAILED: entry point hung for >{TIMEOUT_SECONDS}s without responding.\n")
        if stderr_tail:
            sys.stderr.write("--- subprocess stderr (tail) ---\n")
            sys.stderr.write(stderr_tail)
            sys.stderr.write("\n")
        return 2
    except FileNotFoundError:
        sys.stderr.write(f"SMOKE TEST FAILED: entry point '{args.entry_point}' not found on PATH.\n")
        return 3

    stdout = proc.stdout or ""
    stderr_tail = (proc.stderr or "")[-4000:]

    first_line = stdout.strip().splitlines()[0] if stdout.strip() else ""
    if not first_line:
        sys.stderr.write("SMOKE TEST FAILED: subprocess produced no stdout (likely crashed during import).\n")
        sys.stderr.write(f"Subprocess exit code: {proc.returncode}\n")
        if stderr_tail:
            sys.stderr.write("--- subprocess stderr (tail) ---\n")
            sys.stderr.write(stderr_tail)
            sys.stderr.write("\n")
        return 4

    try:
        response = json.loads(first_line)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"SMOKE TEST FAILED: first stdout line is not valid JSON: {exc}\n")
        sys.stderr.write(f"Raw line: {first_line!r}\n")
        if stderr_tail:
            sys.stderr.write("--- subprocess stderr (tail) ---\n")
            sys.stderr.write(stderr_tail)
            sys.stderr.write("\n")
        return 5

    if response.get("jsonrpc") != "2.0":
        sys.stderr.write(f"SMOKE TEST FAILED: response jsonrpc field is {response.get('jsonrpc')!r}, expected '2.0'.\n")
        return 6

    if response.get("id") != INITIALIZE_REQUEST["id"]:
        sys.stderr.write(
            f"SMOKE TEST FAILED: response id is {response.get('id')!r}, " f"expected {INITIALIZE_REQUEST['id']!r}.\n"
        )
        return 7

    if "error" in response:
        sys.stderr.write(f"SMOKE TEST FAILED: initialize returned JSON-RPC error: {response['error']!r}\n")
        return 8

    sys.stdout.write("SMOKE TEST PASSED: initialize handshake completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
