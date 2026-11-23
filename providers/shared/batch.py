"""Provider-neutral types for asynchronous batch execution.

A batch is submitted once, processed by the provider over minutes to hours, and
collected later. These types describe that lifecycle without referring to any one
backend: callers build :class:`BatchRequest` objects the same way regardless of which
provider executes them, and read back :class:`BatchStatus`, :class:`BatchOutcome` and
:class:`CancelOutcome` values.

Correlation is by the caller-supplied ``custom_id`` only. Providers return results in
arbitrary order, so ordering must never be relied on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .model_response import ModelResponse

__all__ = [
    "BatchError",
    "BatchFailed",
    "BatchIncompleteResults",
    "BatchListEntry",
    "BatchListing",
    "BatchNotFound",
    "BatchNotReady",
    "BatchOutcome",
    "BatchProviderError",
    "BatchRateLimited",
    "BatchRequest",
    "BatchResultsExpired",
    "BatchState",
    "BatchStatus",
    "CancelOutcome",
    "CancelState",
    "UnsupportedBatchOperation",
    "validate_batch_requests",
]


class BatchState(str, Enum):
    """Processing state of a provider batch."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    #: The provider is stopping the batch (OpenRouter documents this status). Not terminal: the
    #: batch only becomes ``cancelled`` once the provider has finished stopping it.
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True once the provider will make no further progress on the batch."""

        return self in _TERMINAL_STATES


_TERMINAL_STATES = frozenset({BatchState.COMPLETED, BatchState.FAILED, BatchState.EXPIRED, BatchState.CANCELLED})


class CancelState(str, Enum):
    """Result of asking a provider to cancel a batch."""

    CANCELLED = "cancelled"
    CANCELLING = "cancelling"
    ALREADY_TERMINAL = "already_terminal"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class BatchRequest:
    """One request inside a batch.

    ``custom_id`` is chosen by the caller and must be unique within the batch. ``images``
    is carried so an adapter can refuse non-text content locally; it is never silently
    dropped.

    A batch carries exactly one model, chosen by the ``model_name`` given to
    ``submit_batch``. A caller that wants an adapter to verify a request is meant for that
    same model sets ``extra["model"]`` to the identifier the request was written for; an
    adapter rejects the submission when it resolves to a different model. Requests that
    do not set the key are taken to target the batch model.
    """

    custom_id: str
    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    images: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_non_text_content(self) -> bool:
        """True when the request carries image (or other attachment) content."""

        return bool(self.images)


@dataclass(frozen=True)
class BatchStatus:
    """Snapshot of a batch's processing state."""

    state: BatchState
    provider: str = ""
    total: int = 0
    completed: int = 0
    failed: int = 0
    error_message: str | None = None
    expires_at: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal


@dataclass(frozen=True)
class BatchError:
    """A failure of one request (or of a whole batch) inside a batch.

    ``retryable`` is True when resubmitting the same request unchanged could succeed
    (rate limits, transient server errors) and False when it cannot (malformed request,
    content rejected).
    """

    message: str
    code: str | None = None
    retryable: bool = False


@dataclass(frozen=True)
class BatchOutcome:
    """Result of one request, correlated by ``custom_id``."""

    custom_id: str
    success: bool
    response: ModelResponse | None = None
    error: BatchError | None = None

    @classmethod
    def succeeded(cls, custom_id: str, response: ModelResponse) -> BatchOutcome:
        return cls(custom_id=custom_id, success=True, response=response)

    @classmethod
    def failed(cls, custom_id: str, error: BatchError) -> BatchOutcome:
        return cls(custom_id=custom_id, success=False, error=error)


@dataclass(frozen=True)
class CancelOutcome:
    """Explicit answer to a cancellation request; never pretends to have cancelled."""

    state: CancelState
    detail: str = ""

    @property
    def batch_continues(self) -> bool:
        """True when the provider batch is still running (and still billed)."""

        return self.state in (CancelState.UNSUPPORTED, CancelState.CANCELLING)


@dataclass(frozen=True)
class BatchListEntry:
    """Metadata-only row from a provider's batch listing (never carries results).

    ``created_at`` is epoch seconds when the provider reported a parseable time.
    ``model`` and ``endpoint`` are exactly as the provider reported them.
    """

    batch_id: str
    state: BatchState | None = None
    model: str | None = None
    endpoint: str | None = None
    created_at: float | None = None
    total: int | None = None


@dataclass(frozen=True)
class BatchListing:
    """One page of a provider's batch listing.

    ``may_be_truncated`` is True when older batches could exist beyond ``entries`` (the
    page was full). A reconciler must not conclude "no such batch exists" from a page that
    may be truncated and does not reach back past the time it is looking for.
    """

    entries: tuple[BatchListEntry, ...] = ()
    may_be_truncated: bool = False


class UnsupportedBatchOperation(NotImplementedError):
    """Raised when a provider does not implement a batch operation."""

    def __init__(self, provider: str, operation: str) -> None:
        self.provider = provider
        self.operation = operation
        super().__init__(f"Provider '{provider}' does not support batch operation '{operation}'")


class BatchProviderError(RuntimeError):
    """A provider answered a batch call with an error (or could not be reached).

    ``retryable`` is True when repeating the same call later could succeed (rate limit,
    5xx, transport failure) so a wait loop can absorb it; False for errors that will
    recur (insufficient credits, malformed submission, unknown batch).
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        retryable: bool = False,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.code = code


class BatchRateLimited(BatchProviderError):
    """HTTP 429: over a rate limit or an in-flight cap. Always retryable."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("status_code", 429)
        kwargs["retryable"] = True
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class BatchNotFound(BatchProviderError):
    """The provider does not know the batch identifier (HTTP 404)."""


class BatchResultsExpired(BatchProviderError):
    """The batch's results passed the provider's retention window (HTTP 410)."""


class BatchNotReady(BatchProviderError):
    """Results were requested for a batch that has not reached a terminal state."""

    def __init__(self, message: str, *, state: BatchState, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.state = state


class BatchFailed(BatchProviderError):
    """The batch ended without results (failed, expired or provider-cancelled)."""

    def __init__(self, message: str, *, state: BatchState, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.state = state


class BatchIncompleteResults(BatchProviderError):
    """A terminal batch returned fewer usable results than requests, and the shortfall
    cannot be attributed to specific ids because the submitted ids were not supplied.

    ``outcomes`` carries the results that did arrive so a caller can still use them.
    """

    def __init__(self, message: str, *, outcomes: list[BatchOutcome], **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.outcomes = outcomes


def validate_batch_requests(requests: Sequence[BatchRequest]) -> None:
    """Reject an empty submission or duplicate ``custom_id`` values.

    Called before any network traffic so that a bad submission never creates a batch.

    Raises:
        ValueError: if ``requests`` is empty or two requests share a ``custom_id``.
    """

    if not requests:
        raise ValueError("A batch must contain at least one request")

    seen: set[str] = set()
    duplicates: list[str] = []
    for request in requests:
        if request.custom_id in seen and request.custom_id not in duplicates:
            duplicates.append(request.custom_id)
        seen.add(request.custom_id)

    if duplicates:
        raise ValueError("Duplicate batch custom_id: " + ", ".join(repr(d) for d in duplicates))
