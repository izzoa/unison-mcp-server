"""Circuit breaker for provider fault isolation.

Implements the three-state pattern (Closed / Open / Half-Open) to detect
sustained provider failures and short-circuit requests, avoiding the full
retry * timeout wait when a provider is hard-down.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ProviderUnavailable(Exception):
    """Raised when a provider's circuit breaker is open.

    Distinct from provider API errors so callers can handle circuit-open
    separately from transient failures.
    """

    def __init__(
        self,
        provider_name: str,
        circuit_state: str,
        message: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.circuit_state = circuit_state
        default_msg = f"Provider '{provider_name}' is unavailable " f"(circuit {circuit_state})"
        super().__init__(message or default_msg)


class CircuitBreaker:
    """Per-provider circuit breaker with configurable thresholds.

    Thread-safe: all state mutations are protected by a lock that is held
    only for the duration of in-memory counter/flag updates (microseconds),
    never across I/O.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_seconds: float = 60.0,
        half_open_max_calls: int = 1,
        provider_name: str = "",
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout_seconds = reset_timeout_seconds
        self._half_open_max_calls = half_open_max_calls
        self._provider_name = provider_name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._opened_at: float | None = None
        self._half_open_in_flight = 0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""
        with self._lock:
            return self._state

    def allow_request(self) -> bool:
        """Check whether a request should proceed.

        Returns ``True`` if the request is allowed, ``False`` to fail fast.
        Handles the OPEN -> HALF_OPEN transition when the cooldown expires.
        """
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True

            if self._state is CircuitState.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0)
                if elapsed >= self._reset_timeout_seconds:
                    self._transition(CircuitState.HALF_OPEN)
                    self._half_open_in_flight = 1
                    return True
                return False

            # HALF_OPEN
            if self._half_open_in_flight < self._half_open_max_calls:
                self._half_open_in_flight += 1
                return True
            return False

    def record_success(self) -> None:
        """Record a successful provider call."""
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._transition(CircuitState.CLOSED)
                self._failure_count = 0
                self._half_open_in_flight = 0
            elif self._state is CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed provider call (after retries exhausted)."""
        with self._lock:
            self._last_failure_time = time.monotonic()

            if self._state is CircuitState.HALF_OPEN:
                self._opened_at = time.monotonic()
                self._half_open_in_flight = 0
                self._transition(CircuitState.OPEN)
                return

            if self._state is CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._failure_threshold:
                    self._opened_at = time.monotonic()
                    self._transition(CircuitState.OPEN)

    def health_status(self) -> dict:
        """Return a diagnostic snapshot of current circuit state."""
        with self._lock:
            status: dict = {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self._failure_threshold,
                "reset_timeout_seconds": self._reset_timeout_seconds,
                "provider_name": self._provider_name,
                "last_failure_time": None,
            }

            if self._last_failure_time is not None:
                # Convert monotonic to wall-clock for reporting
                wall_offset = time.time() - time.monotonic()
                wall_time = self._last_failure_time + wall_offset
                status["last_failure_time"] = datetime.fromtimestamp(wall_time, tz=timezone.utc).isoformat()

            if self._state is CircuitState.OPEN and self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at
                remaining = max(0.0, self._reset_timeout_seconds - elapsed)
                status["time_until_half_open_seconds"] = round(remaining, 1)

            return status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: CircuitState) -> None:
        """Transition state and emit a log entry. Must be called under lock."""
        old_state = self._state
        self._state = new_state

        if new_state is CircuitState.OPEN:
            logger.warning(
                "Circuit breaker %s -> %s for provider '%s' " "(failure_count=%d, threshold=%d)",
                old_state.value,
                new_state.value,
                self._provider_name,
                self._failure_count,
                self._failure_threshold,
            )
        else:
            logger.info(
                "Circuit breaker %s -> %s for provider '%s'",
                old_state.value,
                new_state.value,
                self._provider_name,
            )
