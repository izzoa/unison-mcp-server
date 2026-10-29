#!/usr/bin/env python3
"""Minimal stdio MCP server for testing what environment/cwd opencode gives MCP children.

Stdlib only. Newline-delimited JSON-RPC 2.0 over stdin/stdout (MCP stdio transport).
Implements: initialize (protocol version negotiation), notifications/initialized, ping,
tools/list (one tool: clink_probe_tool), tools/call (returns text), resources/list,
prompts/list (empty). Anything else -> JSON-RPC error -32601.

On startup it writes JSON to the file named by env FAKE_MCP_ENV_DUMP (default ./fake_mcp_env.json):
    {"env": {<full os.environ>}, "cwd": "<os.getcwd()>", "argv": [...], "pid": N, "ppid": N, "started": <ts>}
If env FAKE_MCP_LOG is set, every inbound JSON-RPC message is also appended there as one JSON line.
Diagnostics go to stderr only (stdout is reserved for JSON-RPC).
"""

import json
import os
import sys
import time

SUPPORTED_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
TOOL = {
    "name": "clink_probe_tool",
    "description": "Trivial probe tool used to verify MCP wiring. Echoes the optional 'message' argument.",
    "inputSchema": {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "Text to echo back"}},
        "additionalProperties": False,
    },
}


def dump_env():
    path = os.environ.get("FAKE_MCP_ENV_DUMP", "./fake_mcp_env.json")
    payload = {
        "env": dict(os.environ),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "started": time.time(),
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError as exc:
        print(f"fake_mcp: cannot write env dump {path}: {exc}", file=sys.stderr, flush=True)


def log_msg(msg):
    path = os.environ.get("FAKE_MCP_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "msg": msg}) + "\n")
    except OSError:
        pass


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def reply(req_id, result):
    send({"jsonrpc": "2.0", "id": req_id, "result": result})


def error(req_id, code, message):
    send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def handle(msg):
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}
    is_request = "id" in msg and method is not None
    if method is None:  # a response to something we never sent; ignore
        return
    if not is_request:  # notification (e.g. notifications/initialized, notifications/cancelled)
        return
    if method == "initialize":
        want = params.get("protocolVersion")
        version = want if want in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        reply(
            req_id,
            {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "fake-mcp", "version": "0.1.0"},
            },
        )
    elif method == "ping":
        reply(req_id, {})
    elif method == "tools/list":
        reply(req_id, {"tools": [TOOL]})
    elif method == "tools/call":
        name = params.get("name")
        if name != TOOL["name"]:
            return error(req_id, -32602, f"Unknown tool: {name}")
        args = params.get("arguments") or {}
        reply(
            req_id,
            {"content": [{"type": "text", "text": "PROBE_OK {}".format(args.get("message", ""))}], "isError": False},
        )
    elif method == "resources/list":
        reply(req_id, {"resources": []})
    elif method == "resources/templates/list":
        reply(req_id, {"resourceTemplates": []})
    elif method == "prompts/list":
        reply(req_id, {"prompts": []})
    else:
        error(req_id, -32601, f"Method not found: {method}")


def main():
    dump_env()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            error(None, -32700, "Parse error")
            continue
        log_msg(msg)
        if isinstance(msg, list):  # JSON-RPC batch
            for m in msg:
                handle(m)
        else:
            handle(msg)


if __name__ == "__main__":
    main()
