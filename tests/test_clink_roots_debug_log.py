"""The handler logs a failed MCP roots request at debug level (clink-working-dir-derivation, "Roots never fail a call").

``tests/test_clink_roots_handler.py`` covers the outcomes; this file pins the spec's logging obligation: a roots
failure is logged at debug level and the call proceeds without roots, with a warning.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from mcp.types import CallToolRequestParams

from handlers.tool_handlers import resolve_clink_roots

LEGACY_VERSION = "2025-11-25"


class _FailingSession:
    protocol_version = LEGACY_VERSION
    can_send_request = True

    def check_client_capability(self, capability) -> bool:
        return True

    async def send_request(self, request, result_type, request_read_timeout_seconds=None):
        raise TimeoutError("host never answered")


@pytest.mark.asyncio
async def test_a_failed_roots_request_is_logged_at_debug_and_warned_about(monkeypatch, caplog):
    monkeypatch.setenv("CLINK_MCP_ROOTS", "legacy")
    ctx = SimpleNamespace(session=_FailingSession())
    params = CallToolRequestParams(name="clink", arguments={"prompt": "hi"})

    with caplog.at_level(logging.DEBUG, logger="handlers.tool_handlers"):
        outcome = await resolve_clink_roots(ctx, params)

    assert outcome == {"uris": [], "warnings": ["roots unavailable: the roots/list request failed (TimeoutError)"]}
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG and "roots/list request failed" in r.getMessage()]
    assert debug and "TimeoutError" in debug[0].getMessage()
