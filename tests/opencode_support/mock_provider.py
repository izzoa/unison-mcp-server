#!/usr/bin/env python3
"""Mock OpenAI-compatible chat-completions server for testing opencode (stdlib only).

    python mock_provider.py [--host 127.0.0.1] [--port N] [--port-file FILE] [--log FILE] [--tool-command CMD]

--port 0 (default) picks a random free port, written to --port-file and printed as
"MOCK_PROVIDER_LISTENING http://HOST:PORT". --log appends one JSON line per request
({"event":"request", "path", "authorization", "model", "tool_names", "body", ...}), plus
{"event":"client_disconnected"} when the client drops mid-stream and {"event":"response_complete"}
when a streamed reply is fully written.

The reply is chosen by the request's "model":
    text -> streamed "PONG"
    tool -> first request: a streamed tool call to the shell tool running --tool-command
            (default "echo MOCK_TOOL_RAN"); once a tool result is present: "TOOL_DONE"
    slow -> a long numbered-line reply (450 one-token chunks, 0.1 s apart, ~45 s) that a test can
            cancel mid-stream
    err  -> HTTP 404 model_not_found
    else -> like text

Note opencode v2 sends a second, tool-less title-generation request per run unless given --title.
"""

import argparse
import json
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_IDS = ["text", "tool", "slow", "err"]
TOOL_COMMAND = "echo MOCK_TOOL_RAN"
SHELL_TOOL_PREFS = ["bash", "shell", "run_command", "exec", "execute", "terminal", "sh"]
SLOW_LINES = 90
SLOW_CHUNKS_PER_LINE = 5
SLOW_DELAY = 0.1

_log_lock = threading.Lock()
LOG_PATH = None


def log_event(obj):
    if not LOG_PATH:
        return
    line = json.dumps(obj, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class ClientGone(Exception):
    pass


def pick_shell_tool(tools):
    """Return (name, parameters_schema) of the shell-ish tool offered by the request, or None."""
    found = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            found.append((fn["name"], fn.get("parameters") or {}))
    by_name = {n.lower(): (n, p) for n, p in found}
    for pref in SHELL_TOOL_PREFS:
        if pref in by_name:
            return by_name[pref]
    for n, p in found:  # fuzzy
        low = n.lower()
        if "bash" in low or "shell" in low:
            return n, p
    return None


def build_tool_args(schema, command):
    props = (schema or {}).get("properties") or {}
    args = {}
    for key, spec in props.items():
        low = key.lower()
        typ = spec.get("type") if isinstance(spec, dict) else None
        if low in ("command", "cmd", "script", "input"):
            args[key] = command
        elif low == "description":
            args[key] = "Print a marker"
        elif low in ("workdir", "cwd", "directory"):
            continue
        elif key in (schema.get("required") or []):
            if typ == "number" or typ == "integer":
                args[key] = 1
            elif typ == "boolean":
                args[key] = False
            else:
                args[key] = "x"
    if not args:
        args["command"] = command
    return args


class Handler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/1.0"
    protocol_version = "HTTP/1.0"  # close-delimited bodies: simplest correct SSE

    def log_message(self, fmt, *args):  # silence default stderr access log
        pass

    # -- helpers ---------------------------------------------------------------
    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, obj):
        payload = obj if isinstance(obj, str) else json.dumps(obj)
        try:
            self.wfile.write(("data: " + payload + "\n\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            raise ClientGone()

    # -- routing ---------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        log_event(
            {
                "event": "request",
                "ts": time.time(),
                "method": "GET",
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": None,
            }
        )
        if path in ("/v1/models", "/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": m, "object": "model", "created": 0, "owned_by": "mock"} for m in MODEL_IDS],
                },
            )
        else:
            self._send_json(
                404,
                {
                    "error": {
                        "message": "not found: " + path,
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "not_found",
                    }
                },
            )

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except Exception:
            body = raw.decode("utf-8", "replace")
        model = body.get("model") if isinstance(body, dict) else None
        stream = bool(body.get("stream")) if isinstance(body, dict) else False
        tools = body.get("tools") if isinstance(body, dict) else None
        tool_names = [((t or {}).get("function") or {}).get("name") for t in (tools or [])]
        log_event(
            {
                "event": "request",
                "ts": time.time(),
                "method": "POST",
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "headers": dict(self.headers.items()),
                "model": model,
                "stream": stream,
                "tool_names": tool_names,
                "body": body,
            }
        )
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/chat/completions", "/chat/completions"):
            return self._send_json(
                404,
                {
                    "error": {
                        "message": "unsupported path " + path,
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "not_found",
                    }
                },
            )
        if not isinstance(body, dict):
            return self._send_json(
                400,
                {
                    "error": {
                        "message": "invalid JSON body",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_json",
                    }
                },
            )
        if model == "err":
            return self._send_json(
                404,
                {
                    "error": {
                        "message": "The model `err` does not exist or you do not have access to it.",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "model_not_found",
                    }
                },
            )
        try:
            self._respond(body, model or "text", stream)
        except ClientGone:
            log_event({"event": "client_disconnected", "ts": time.time(), "model": model, "path": self.path})
        except Exception as exc:  # never crash the thread noisily
            log_event({"event": "handler_error", "ts": time.time(), "model": model, "error": repr(exc)})

    # -- response builders -----------------------------------------------------
    def _respond(self, body, model, stream):
        messages = body.get("messages") or []
        has_tool_result = any(isinstance(m, dict) and m.get("role") == "tool" for m in messages)
        cid = f"chatcmpl-mock-{int(time.time() * 1000)}"
        created = int(time.time())
        prompt_tokens = 10 + sum(len(json.dumps(m)) // 4 for m in messages)

        def chunk(delta, finish=None, **extra):
            d = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            d.update(extra)
            return d

        def usage_obj(completion):
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion,
                "total_tokens": prompt_tokens + completion,
            }

        # Unique tool-call id per step (multi-step runs must not reuse ids).
        call_id = f"call_mock_{sum(1 for m in messages if isinstance(m, dict) and m.get('role') == 'tool') + 1}"

        # Decide the plan: list of text pieces and/or a tool call.
        tool_call = None
        pieces = None
        delay = 0.0
        if model == "tool" and not has_tool_result:
            shell = pick_shell_tool(body.get("tools"))
            if shell:
                tool_call = (shell[0], build_tool_args(shell[1], command=TOOL_COMMAND))
            else:
                pieces = ["NO_SHELL_TOOL_OFFERED"]
        elif model == "tool":
            pieces = ["TOOL_DONE"]
        elif model == "slow":
            # numbered lines, each emitted as several one-token chunks: "1", ".", " mock", " line", "\n"
            pieces = []
            for i in range(SLOW_LINES):
                pieces += [str(i + 1), ".", " mock", " line", "\n"][:SLOW_CHUNKS_PER_LINE]
            delay = SLOW_DELAY
        else:
            pieces = ["PONG"]

        if not stream:
            if tool_call:
                msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_call[0], "arguments": json.dumps(tool_call[1])},
                        }
                    ],
                }
                finish, completion = "tool_calls", 20
            else:
                msg = {"role": "assistant", "content": "".join(pieces)}
                finish, completion = "stop", max(1, len(pieces))
            return self._send_json(
                200,
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": usage_obj(completion),
                },
            )

        self._sse_start()
        self._sse(chunk({"role": "assistant", "content": ""}))
        if tool_call:
            args_json = json.dumps(tool_call[1])
            half = max(1, len(args_json) // 2)
            self._sse(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": tool_call[0], "arguments": ""},
                            }
                        ]
                    }
                )
            )
            self._sse(chunk({"tool_calls": [{"index": 0, "function": {"arguments": args_json[:half]}}]}))
            self._sse(chunk({"tool_calls": [{"index": 0, "function": {"arguments": args_json[half:]}}]}))
            self._sse(chunk({}, "tool_calls"))
            completion = 20
        else:
            for p in pieces:
                self._sse(chunk({"content": p}))
                if delay:
                    time.sleep(delay)
            self._sse(chunk({}, "stop"))
            completion = len(pieces)
        # Usage-only final chunk (OpenAI include_usage style), then [DONE].
        self._sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage_obj(completion),
            }
        )
        self._sse("[DONE]")
        log_event({"event": "response_complete", "ts": time.time(), "model": model})


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):  # swallow disconnect tracebacks
        pass


def main():
    global LOG_PATH, TOOL_COMMAND
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 = random free port")
    ap.add_argument("--port-file", help="write the chosen port number to this file")
    ap.add_argument("--log", help="append one JSON line per request to this file")
    ap.add_argument("--tool-command", default=TOOL_COMMAND, help="shell command the 'tool' model asks opencode to run")
    args = ap.parse_args()
    LOG_PATH = args.log
    TOOL_COMMAND = args.tool_command
    srv = QuietServer((args.host, args.port), Handler)
    port = srv.server_address[1]
    if args.port_file:
        tmp = args.port_file + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(str(port))
        import os

        os.replace(tmp, args.port_file)
    print(f"MOCK_PROVIDER_LISTENING http://{args.host}:{port}", flush=True)

    def _stop(*_):
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
