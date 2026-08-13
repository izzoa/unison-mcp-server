"""Request-scoped access to the active MCP request context.

mcp 2.x removed ``server.request_context``; the SDK now injects a
``ServerRequestContext`` into each handler instead. The handler adapters in
``handlers/`` bind that context here so code deeper in the stack (client-info
logging, streaming progress, workflow progress tokens) can reach the active
request without threading it through every internal signature.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

_current_request_context: ContextVar[Any | None] = ContextVar("mcp_request_context", default=None)


def set_current_request_context(ctx: Any | None) -> Token[Any | None]:
    """Bind the active request context; returns a token for reset."""
    return _current_request_context.set(ctx)


def reset_current_request_context(token: Token[Any | None]) -> None:
    """Restore the previous binding."""
    _current_request_context.reset(token)


def get_current_request_context() -> Any | None:
    """Return the active request context, or ``None`` outside a request."""
    return _current_request_context.get()
