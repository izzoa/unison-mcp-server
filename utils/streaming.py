"""MCP progress notification bridge for streaming provider responses.

This module provides :class:`StreamProgressNotifier`, a utility that relays
:class:`~providers.base.StreamChunk` objects from a streaming provider call to
the MCP client via ``notifications/progress`` messages.  It includes built-in
rate limiting to avoid flooding the client with excessive notifications.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Default rate-limiting thresholds
_MIN_NOTIFY_INTERVAL_SECONDS = 0.1  # 100 ms between notifications
_MIN_CHUNK_SIZE = 50  # buffer until at least 50 characters accumulated


class StreamProgressNotifier:
    """Bridge streaming chunks to MCP progress notifications.

    The notifier accumulates small/rapid chunks into a buffer and flushes
    them as a single MCP progress notification when either:

    * The buffer exceeds :pyattr:`min_chunk_size` characters, **or**
    * At least :pyattr:`min_interval` seconds have elapsed since the last
      notification, **or**
    * A chunk with ``is_final=True`` arrives (the buffer is always flushed
      immediately).

    If the MCP session or progress token is unavailable the notifier
    operates as a silent no-op — no exceptions are raised and the tool
    completes normally.

    Args:
        server: Optional MCP ``Server`` instance. The session is resolved from
            the ContextVar-bound request context first; a legacy
            ``server.request_context`` attribute is a fallback for test
            doubles.
        progress_token: Opaque token from the client's request metadata.
            ``None`` means no progress reporting was requested.
        min_interval: Minimum seconds between notifications (default 0.1).
        min_chunk_size: Minimum buffered characters before sending
            (default 50).
    """

    def __init__(
        self,
        server: object,
        progress_token: object | None = None,
        min_interval: float = _MIN_NOTIFY_INTERVAL_SECONDS,
        min_chunk_size: int = _MIN_CHUNK_SIZE,
    ) -> None:
        self._server = server
        self._progress_token = progress_token
        self._min_interval = min_interval
        self._min_chunk_size = min_chunk_size

        self._buffer: str = ""
        # Offset so the FIRST non-final chunk is not blocked by the interval floor
        # (first send passes, subsequent sends are throttled to min_interval).
        self._last_notify_time: float = time.monotonic() - self._min_interval
        self._chunks_sent: int = 0
        self._total_progress: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def notify_chunk(self, chunk) -> None:
        """Accept a :class:`StreamChunk` and send a progress notification if thresholds are met.

        For the final chunk (``is_final=True``) the buffer is always
        flushed regardless of size or time thresholds.

        Args:
            chunk: A :class:`~providers.base.StreamChunk` instance.
        """
        self._buffer += chunk.text

        if chunk.is_final:
            await self._flush(is_final=True)
            return

        now = time.monotonic()
        elapsed = now - self._last_notify_time
        # Send a non-final notification only when BOTH the wall-clock interval has
        # elapsed AND enough content has accumulated. Requiring the interval as a
        # hard floor (not an OR) prevents a fast/bursty stream from emitting a
        # flurry of back-to-back notifications even when the size threshold is met;
        # requiring the size avoids tiny sends. The final chunk (handled above)
        # always flushes regardless.
        if elapsed >= self._min_interval and len(self._buffer) >= self._min_chunk_size:
            await self._flush(is_final=False)

    async def notify_complete(self, total_text: str) -> None:
        """Send a final completion notification.

        Args:
            total_text: The fully assembled response text.
        """
        await self._send_notification(
            message=f"[complete] {len(total_text)} characters",
            is_final=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _flush(self, *, is_final: bool) -> None:
        """Flush the buffer as a progress notification."""
        if not self._buffer and not is_final:
            return

        text = self._buffer
        self._buffer = ""
        await self._send_notification(message=text, is_final=is_final)

    async def _send_notification(self, *, message: str, is_final: bool) -> None:
        """Send a single MCP progress notification, swallowing errors."""
        if self._progress_token is None:
            return

        try:
            session = self._get_session()
            if session is None:
                return

            self._chunks_sent += 1
            self._total_progress += 1.0

            await session.send_progress_notification(
                progress_token=self._progress_token,
                progress=self._total_progress,
                total=None,
                message=message,
            )
            self._last_notify_time = time.monotonic()

        except Exception:
            # Client may not support progress notifications — fail silently.
            logger.debug("Failed to send progress notification", exc_info=True)

    def _get_session(self):
        """Retrieve the current MCP session.

        Prefers the request context bound by the mcp 2.x handler adapters,
        falling back to the legacy ``server.request_context`` attribute (kept
        for 1.x-shaped test doubles). Returns ``None`` if unavailable (e.g.
        called outside a request handler).
        """
        try:
            from utils.mcp_context import get_current_request_context

            ctx = get_current_request_context()
            if ctx is None and self._server is not None:
                ctx = self._server.request_context
            return ctx.session if ctx is not None else None
        except (LookupError, AttributeError):
            return None
