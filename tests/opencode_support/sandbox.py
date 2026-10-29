"""A throwaway opencode v2 environment for the opt-in integration tests.

Everything lives under one temp root: private ``XDG_*`` dirs (so the user's
config, MCP servers, plugins, ``opencode.db`` and background service are never
read, written, or attached to), a project directory whose ``opencode.json``
points at a local mock provider and a fake stdio MCP server. Nothing is spent
and nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SUPPORT = Path(__file__).parent
MOCK_MODELS = ("text", "tool", "slow", "err")


class Sandbox:
    def __init__(
        self, root: Path, *, api_key_var: str = "MOCK_API_KEY", tool_command: str = "echo MOCK_TOOL_RAN"
    ) -> None:
        self.root = root
        self.project = root / "project"
        self.api_key_var = api_key_var
        self.tool_command = tool_command
        self.log = root / "mock.log"
        self.port_file = root / "mock.port"
        self.mcp_dump = root / "mcp_env.json"
        self.xdg = {
            "XDG_CONFIG_HOME": root / "xdg" / "config",
            "XDG_DATA_HOME": root / "xdg" / "data",
            "XDG_STATE_HOME": root / "xdg" / "state",
            "XDG_CACHE_HOME": root / "xdg" / "cache",
        }
        self._mock: subprocess.Popen[bytes] | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> Sandbox:
        for path in (self.project, *self.xdg.values()):
            path.mkdir(parents=True, exist_ok=True)
        self._mock = subprocess.Popen(
            [
                sys.executable,
                str(SUPPORT / "mock_provider.py"),
                "--port",
                "0",
                "--port-file",
                str(self.port_file),
                "--log",
                str(self.log),
                "--tool-command",
                self.tool_command,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 15
        while not self.port_file.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("mock provider did not start")
            time.sleep(0.05)
        (self.project / "opencode.json").write_text(json.dumps(self._config(int(self.port_file.read_text()))))
        return self

    def stop(self) -> None:
        if self._mock is not None and self._mock.poll() is None:
            os.killpg(self._mock.pid, signal.SIGTERM)
            try:
                self._mock.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self._mock.pid, signal.SIGKILL)

    def _config(self, port: int) -> dict[str, Any]:
        return {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "provider": {
                "mock": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Mock",
                    "options": {"baseURL": f"http://127.0.0.1:{port}/v1", "apiKey": f"{{env:{self.api_key_var}}}"},
                    "models": {name: {"name": name} for name in MOCK_MODELS},
                }
            },
            "mcp": {
                "servers": {
                    "fake": {
                        "type": "local",
                        "command": [sys.executable, str(SUPPORT / "fake_mcp.py")],
                        "environment": {"FAKE_MCP_ENV_DUMP": str(self.mcp_dump)},
                    }
                }
            },
        }

    # -- environment -----------------------------------------------------------

    @property
    def env(self) -> dict[str, str]:
        """Variables that point opencode at this sandbox; apply with ``monkeypatch.setenv``."""
        env = {name: str(path) for name, path in self.xdg.items()}
        env.setdefault("MOCK_API_KEY", "mock-key-123")
        return env

    def cli(self, *args: str, timeout: float = 90) -> subprocess.CompletedProcess[str]:
        """Run an opencode subcommand against this sandbox (always on a private server)."""
        binary = shutil.which("opencode")
        assert binary is not None
        return subprocess.run(
            [binary, *args, "--standalone"],
            cwd=self.project,
            env={**os.environ, **self.env, "PWD": str(self.project)},
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )

    # -- observations ------------------------------------------------------------

    def mock_events(self) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def chat_requests(self) -> list[dict[str, Any]]:
        return [
            e
            for e in self.mock_events()
            if e.get("event") == "request" and str(e.get("path", "")).endswith("/chat/completions")
        ]

    def mcp_env(self) -> dict[str, str] | None:
        if not self.mcp_dump.exists():
            return None
        env: dict[str, str] = json.loads(self.mcp_dump.read_text())["env"]
        return env

    def sessions(self) -> list[tuple[str, int | None]]:
        """``(id, time_suspended)`` for every session in the sandbox's own database."""
        db = self.xdg["XDG_DATA_HOME"] / "opencode" / "opencode.db"
        if not db.exists():
            return []
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return [(row[0], row[1]) for row in connection.execute("select id, time_suspended from session_v2")]
        finally:
            connection.close()


def sessions_in_real_database(directory_prefix: Path) -> int:
    """How many sessions under ``directory_prefix`` leaked into the user's real opencode DB (must be 0)."""
    db = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    if not db.exists():
        return 0
    try:
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return 0
    try:
        row = connection.execute(
            "select count(*) from session_v2 where directory like ?", (f"{directory_prefix}%",)
        ).fetchone()
        return int(row[0])
    except sqlite3.Error:
        return 0
    finally:
        connection.close()
