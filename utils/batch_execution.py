"""Bounded wait, resume and cancel for batches, independent of any one tool.

A batch outlives the tool call that submitted it (see ``utils/batch_store.py``), so every
batch-capable tool does the same three things: submit and wait a bounded time, resume a
handle later, and cancel a handle. This module is that shared behaviour. It is deliberately
free of MCP types, request models and tool-specific rendering: tools call
:class:`BatchExecutor`, receive a :class:`BatchRunResult`, and render it themselves (the
helpers at the bottom cover the standard non-terminal, cancelled and error renderings).

All operations are synchronous (provider adapters use blocking HTTP). Async tools run them
with ``asyncio.to_thread``. The clock and the sleep function are injectable so tests are
instant and deterministic.

Bounded wait
------------
A call never blocks on a batch's 24-hour window. :class:`WaitPolicy` defaults to a 45 s wait
and clamps a caller's ``wait_seconds`` to 50 s. Those are the numbers ``tools/clink.py``
settled on after measuring MCP hosts (Claude Desktop's local-agent mode cancels a tool call
at about a minute), and the batch spec requires the maximum to "stay inside typical MCP host
tool-call deadlines"; a larger ceiling would invite the host to cancel the call mid-wait.

Polling schedule
----------------
Inside a wait the first provider check comes after 1 s (a resume checks immediately: the
batch may long be finished), then the delay backs off 1, 2, 4, 8, 15, 15, ... seconds, and
the number of poll rounds per wait is capped (:attr:`WaitPolicy.max_polls`) whatever
``wait_seconds`` was. Provider latency counts against the budget: the sleep before a round
is shortened by how long the previous round took, and no further round starts when the
remaining budget cannot cover one. Every provider call made during a wait also runs under a
hard ceiling (:attr:`WaitPolicy.call_grace_seconds` past the budget): a call still in flight
at the ceiling is abandoned (its thread finishes on its own; poll and fetch only read), the
poll is absorbed like any transient error and the fetch is reported pending, so the returned
wait is at most the clamped budget plus that small grace and never an adapter's whole HTTP
timeout. The results fetch retries only while budget remains. The clock starts when the call
starts, so submission and reconciliation time is charged to the budget too: a submit that took 6 s
of a 4 s wait leaves nothing to poll with and returns the handle at once. A batch-list call is
bounded by the same hard ceiling (never less than the grace). A *submit* is bounded separately and
independently of ``wait_seconds``: it gets the longest permitted wait plus the grace, never less than
the grace, because abandoning a slow but successful create would strand a paid batch (``wait_seconds=0``
means "do not wait for the result", not "give up on creating it"). A submit that does not answer is an
ambiguous outcome (intent kept, never retried); a list that does not answer is retried later. A rate-limited poll (``BatchRateLimited``) or a retryable provider
error is absorbed: ``retry_after`` lengthens the next delay (and ends the wait early when it
could not fit in the remaining budget), and the call still returns a result or the handle.

Cancellation semantics
----------------------
``cancel`` asks each running child's provider to cancel and then marks the record cancelled
locally. Local cancellation always stops polling and ignores results. Where the provider
cannot actually cancel (OpenRouter has no cancel endpoint) the report says so plainly: the
provider batch continues and is still billed. A child whose submission is unresolved (no
provider id) is left for reconciliation and reported as possibly still running.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from providers.shared.batch import (
    BatchFailed,
    BatchIncompleteResults,
    BatchNotFound,
    BatchNotReady,
    BatchOutcome,
    BatchProviderError,
    BatchRateLimited,
    BatchResultsExpired,
    CancelState,
    UnsupportedBatchOperation,
)
from providers.shared.batch_resolution import limits_from_record, note_from_record
from utils.batch_coordinator import (
    BatchCoordinator,
    BatchSubmissionError,
    CallTimedOut,
    ChildPlan,
    SubmissionOutcomeUnknown,
    call_within,
)
from utils.batch_store import (
    BatchChild,
    BatchRecord,
    BatchStore,
    ChildState,
    ChildStateChanged,
    LookupStatus,
    RecordState,
    batch_retention_days,
    get_batch_store,
)
from utils.model_resolution import resolution_metadata

if TYPE_CHECKING:
    from providers.base import ModelProvider

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_WAIT_POLICY",
    "BatchExecutor",
    "BatchHandleToolMismatch",
    "BatchRunResult",
    "BatchRunStatus",
    "ChildCancelReport",
    "ChildResult",
    "WaitPolicy",
    "build_default_executor",
    "build_status_payload",
    "describe_submission_error",
    "result_message",
]

#: Default and maximum seconds one call waits (see the module docstring for why 45 / 50).
DEFAULT_WAIT_SECONDS = 45.0
MAX_WAIT_SECONDS = 50.0

_RUNNING_CHILD_STATES = frozenset(
    {ChildState.VALIDATING, ChildState.IN_PROGRESS, ChildState.FINALIZING, ChildState.CANCELLING}
)
#: What a caller is told about a child whose submission has no provider batch recorded yet.
_UNCONFIRMED_SUBMISSION = (
    "its submission is not confirmed yet: the provider may or may not hold a batch for it, and the provider's "
    "batch list could not settle that so far"
)
_FETCHABLE_CHILD_STATES = frozenset({ChildState.COMPLETED, ChildState.EXPIRED, ChildState.FAILED})


# ----------------------------------------------------------------------
# Policy and result types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class WaitPolicy:
    """How long and how often one call waits on a batch."""

    default_seconds: float = DEFAULT_WAIT_SECONDS
    max_seconds: float = MAX_WAIT_SECONDS
    #: Delay before poll round 0, 1, 2, ...; the last value repeats.
    poll_delays: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 15.0)
    #: Hard cap on poll rounds in one wait, whatever the budget (each round polls every running child once).
    max_polls: int = 10
    #: Attempts to fetch a finished batch's results before reporting them pending.
    fetch_attempts: int = 3
    #: How far past the wait budget one provider call may still run before it is abandoned. Bounds
    #: the whole call to ``budget + grace`` however slow the provider is.
    call_grace_seconds: float = 5.0

    def budget(self, requested: Any) -> float:
        """The wait budget for a caller's ``wait_seconds``: default when absent or unusable, else clamped to ``[0, max]``."""

        if requested is None or isinstance(requested, bool):
            return min(self.default_seconds, self.max_seconds)
        try:
            value = float(requested)
        except (TypeError, ValueError):
            return min(self.default_seconds, self.max_seconds)
        if value != value:  # NaN
            return min(self.default_seconds, self.max_seconds)
        return max(0.0, min(value, self.max_seconds))

    def delay(self, round_index: int) -> float:
        """Seconds to wait before poll round ``round_index`` (0-based)."""

        delays = self.poll_delays or (1.0,)
        return delays[min(max(round_index, 0), len(delays) - 1)]


DEFAULT_WAIT_POLICY = WaitPolicy()


class BatchRunStatus(str, Enum):
    """What a submit, resume or cancel call found."""

    COMPLETED = "completed"
    RUNNING = "running"
    FAILED = "failed"
    CANCELLED = "cancelled"
    #: The provider's own window lapsed before the batch was processed.
    EXPIRED = "expired"
    #: Terminal, but the provider no longer retains the results (e.g. HTTP 410).
    RESULTS_EXPIRED = "results_expired"
    #: A submission could not be confirmed; an operator has to look at it.
    INDETERMINATE = "indeterminate"
    #: An accepted provider batch is not tracked; an operator has to look at it.
    ORPHANED = "orphaned"
    #: No record has ever existed under the handle.
    UNKNOWN_HANDLE = "unknown_handle"
    #: The record was purged at retention; only a tombstone remains.
    HANDLE_EXPIRED = "handle_expired"

    @property
    def is_terminal(self) -> bool:
        return self in (
            BatchRunStatus.COMPLETED,
            BatchRunStatus.FAILED,
            BatchRunStatus.CANCELLED,
            BatchRunStatus.EXPIRED,
            BatchRunStatus.RESULTS_EXPIRED,
        )


class BatchHandleToolMismatch(ValueError):
    """A handle was presented to a different tool than the one that issued it."""

    def __init__(self, handle: str, issued_by: str, called: str) -> None:
        super().__init__(
            f"Batch handle '{handle}' was issued by the '{issued_by}' tool; resume or cancel it by calling "
            f"'{issued_by}' (this is '{called}')."
        )
        self.handle = handle
        self.issued_by = issued_by
        self.called = called


@dataclass(frozen=True)
class ChildResult:
    """One provider batch as seen by a call.

    ``outcomes`` are mapped onto the model recorded at submission (see
    :meth:`BatchExecutor._recorded`). ``error`` is a terminal explanation when the child
    produced no usable results; ``pending_reason`` explains why a child that should have
    results could not deliver them *yet* (the call is then not terminal).
    """

    child: BatchChild
    outcomes: tuple[BatchOutcome, ...] = ()
    error: str | None = None
    results_expired: bool = False
    pending_reason: str | None = None
    poll_problem: str | None = None


@dataclass(frozen=True)
class ChildCancelReport:
    """What cancelling one child did, stated without optimism."""

    index: int
    provider: str
    model: str
    provider_batch_id: str | None
    action: str  # cancelled | cancelling | continues | already_terminal | unconfirmed | no_action
    state_after: ChildState
    detail: str
    batch_continues: bool = False


@dataclass(frozen=True)
class BatchRunResult:
    """Outcome of :meth:`BatchExecutor.submit_and_wait`, ``resume`` or ``cancel``."""

    status: BatchRunStatus
    handle: str
    record: BatchRecord | None = None
    children: tuple[ChildResult, ...] = ()
    cancel_reports: tuple[ChildCancelReport, ...] = ()
    notes: tuple[str, ...] = ()
    expired_at: float | None = None
    waited_seconds: float = 0.0
    polls: int = 0
    wait_budget_seconds: float = DEFAULT_WAIT_SECONDS

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    def outcomes_by_custom_id(self) -> dict[str, BatchOutcome]:
        """Every outcome returned, keyed by ``custom_id`` (callers never rely on order)."""

        return {outcome.custom_id: outcome for child in self.children for outcome in child.outcomes}

    def child_for_custom_id(self, custom_id: str) -> BatchChild | None:
        for child in self.children:
            if custom_id in child.child.custom_ids:
                return child.child
        return None


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------


class BatchExecutor:
    """Submit-and-wait, resume and cancel over a :class:`BatchCoordinator`."""

    def __init__(
        self,
        coordinator: BatchCoordinator,
        *,
        policy: WaitPolicy = DEFAULT_WAIT_POLICY,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._coordinator = coordinator
        self._policy = policy
        self._clock = clock
        self._sleep = sleep

    @property
    def coordinator(self) -> BatchCoordinator:
        return self._coordinator

    @property
    def store(self) -> BatchStore:
        return self._coordinator.store

    @property
    def policy(self) -> WaitPolicy:
        return self._policy

    # -- public operations -----------------------------------------------

    def submit_and_wait(
        self,
        tool_name: str,
        plans: Sequence[ChildPlan],
        *,
        continuation_id: str | None = None,
        payload: Any = None,
        wait_seconds: Any = None,
        fetch: bool = True,
    ) -> BatchRunResult:
        """Submit one provider batch per plan under one handle, then wait a bounded time.

        Returns the completed results when every child finished inside the wait, else the
        non-terminal result carrying the handle. The wait budget is measured from the start of
        this call, so the time the submission itself took is spent from it (a submission that
        used the whole budget returns the handle without polling). Each provider submit is
        bounded by the submit ceiling (the longest permitted wait plus the grace, whatever ``wait_seconds``
        was), so a hung provider cannot hold the call open but a slow successful submit is not abandoned.

        ``fetch=False`` stops after polling: a terminal record is reported by state only and no results
        are fetched, for a caller (consensus) that consumes each finished child's results itself with
        :meth:`fetch_child` and keeps them in its own durable state.

        Raises:
            ValueError: invalid plans (nothing persisted or sent).
            BatchSubmissionError / SubmissionOutcomeUnknown: the submission failed; see
                :func:`describe_submission_error`.
        """

        started = self._clock()
        budget = self._policy.budget(wait_seconds)
        handle = self._coordinator.submit(
            tool_name,
            plans,
            continuation_id=continuation_id,
            payload=payload,
            call_timeout=self._submit_timeout(started),
        )
        return self._advance(handle, budget, poll_first=False, started=started, fetch=fetch)

    def resume(
        self, handle: str, *, tool_name: str | None = None, wait_seconds: Any = None, fetch: bool = True
    ) -> BatchRunResult:
        """Resume ``handle``: results when terminal, otherwise wait again and report.

        Unresolved submission intent is reconciled **before** anything is polled. Never
        submits anything, however often it is repeated. ``fetch=False`` polls only and reports a terminal
        record by state, without fetching any child's results (see :meth:`submit_and_wait`).

        Raises:
            BatchHandleToolMismatch: ``tool_name`` is given and is not the issuing tool.
        """

        started = self._clock()
        budget = self._policy.budget(wait_seconds)
        lookup = self._coordinator.open_for_resume(handle, call_timeout=self._call_timeout(started, budget))
        early = self._unresolvable(lookup.status, handle, lookup.expired_at, budget)
        if early is not None:
            return early
        assert lookup.record is not None
        self._check_tool(lookup.record, handle, tool_name)
        return self._advance(handle, budget, poll_first=True, started=started, fetch=fetch)

    def fetch_child(self, child: BatchChild, *, budget_seconds: float = 10.0) -> ChildResult:
        """Fetch one provider batch's results now (the batch is already terminal), mapped onto its recorded model.

        For a caller that consumes results child by child. Like the wait loop, the provider call runs
        under a ceiling (``budget_seconds`` plus the call grace) and a fetch that cannot finish in time is
        reported as ``pending_reason`` rather than failing; an unusable batch is reported as ``error``.
        Never submits anything. The child must already be terminal (a resume with ``fetch=False`` polled it
        to that state); a child that is not terminal, or was cancelled, has no results and comes back empty.
        """

        deadline = self._clock() + max(0.0, budget_seconds)
        return self._collect_child(child, deadline)

    def cancel(self, handle: str, *, tool_name: str | None = None) -> BatchRunResult:
        """Cancel ``handle``: request provider cancellation per running child, then mark it locally.

        An already-terminal record is reported as it stands. Raises
        :class:`BatchHandleToolMismatch` like :meth:`resume`.
        """

        lookup = self._coordinator.open_for_resume(handle, call_timeout=self._call_timeout(self._clock(), 0.0))
        early = self._unresolvable(lookup.status, handle, lookup.expired_at, 0.0)
        if early is not None:
            return early
        record = lookup.record
        assert record is not None
        self._check_tool(record, handle, tool_name)

        if record.state.is_terminal:
            return self._report(handle, started=self._clock(), polls=0, budget=0.0, poll_problems={}, fetch=False)

        reports: list[ChildCancelReport] = []
        for child in record.children:
            reports.append(self._cancel_child(handle, child))

        fresh = self._require(handle)
        if not any(child.status is ChildState.COMPLETED for child in fresh.children):
            self.store.set_state(handle, RecordState.CANCELLED, detail="cancelled by the caller")
        else:
            self.store.recompute_state(handle)
        result = self._report(handle, started=self._clock(), polls=0, budget=0.0, poll_problems={}, fetch=False)
        if result.status is not BatchRunStatus.CANCELLED and any(r.action != "already_terminal" for r in reports):
            # Some children had finished before this call and the record holds their results, so it is not
            # marked cancelled; but this call did cancel the rest, and the response must say so rather than
            # report the whole operation as "completed". The finished children's results stay retrievable.
            result = dataclasses.replace(
                result,
                status=BatchRunStatus.CANCELLED,
                notes=(
                    *result.notes,
                    "Children that had already finished keep their results: call this tool again with ONLY the "
                    "batch_handle to retrieve them.",
                ),
            )
        return dataclasses.replace(result, cancel_reports=tuple(reports))

    def _call_timeout(self, started: float, budget: float) -> Callable[[], float]:
        """Seconds the next batch-list call (reconciliation) may take: the time left to the call's hard ceiling.

        The ceiling is the wait budget plus ``call_grace_seconds``, measured from ``started``. An
        unanswered list is only read again later, so it may follow the caller's wait: a resume that
        asked to wait 0 s is not held open by a provider that never answers a list.
        """

        hard_deadline = started + budget + self._policy.call_grace_seconds
        grace = self._policy.call_grace_seconds
        return lambda: max(grace, hard_deadline - self._clock())

    def _submit_timeout(self, started: float) -> Callable[[], float]:
        """Seconds the next provider *submit* may take, independent of the caller's ``wait_seconds``.

        A submit that is abandoned leaves a possibly billed batch with an unknown outcome (and, for a
        multi-batch operation, no handle at all), so it must not be cut off merely because the caller
        asked for a short wait: ``wait_seconds=0`` means "do not wait for the result", not "give up on
        creating the batch". The ceiling is therefore the longest wait a caller may ask for plus the
        grace, measured from ``started``, which is still inside a typical host tool-call deadline. A
        call that begins with little or none of that left still gets the grace, so a later batch of a
        multi-batch submission is not cut off after a slow first one.
        """

        grace = self._policy.call_grace_seconds
        hard_deadline = started + self._policy.max_seconds + grace
        return lambda: max(grace, hard_deadline - self._clock())

    # -- resolving a handle ------------------------------------------------

    @staticmethod
    def _unresolvable(
        status: LookupStatus, handle: str, expired_at: float | None, budget: float
    ) -> BatchRunResult | None:
        if status is LookupStatus.UNKNOWN:
            return BatchRunResult(BatchRunStatus.UNKNOWN_HANDLE, handle, wait_budget_seconds=budget)
        if status is LookupStatus.EXPIRED:
            return BatchRunResult(
                BatchRunStatus.HANDLE_EXPIRED, handle, expired_at=expired_at, wait_budget_seconds=budget
            )
        return None

    @staticmethod
    def _check_tool(record: BatchRecord, handle: str, tool_name: str | None) -> None:
        if tool_name and record.tool_name and record.tool_name != tool_name:
            raise BatchHandleToolMismatch(handle, record.tool_name, tool_name)

    def _require(self, handle: str) -> BatchRecord:
        record = self.store.get(handle)
        if record is None:
            raise LookupError(f"Batch record {handle} disappeared while it was being processed")
        return record

    # -- the wait loop --------------------------------------------------------

    @staticmethod
    def _pollable(child: BatchChild) -> bool:
        return child.provider_batch_id is not None and child.status in _RUNNING_CHILD_STATES

    def _advance(
        self, handle: str, budget: float, *, poll_first: bool, started: float | None = None, fetch: bool = True
    ) -> BatchRunResult:
        """Poll the handle's running children on the paced schedule until terminal or out of budget.

        ``started`` is when the whole call began (submission and reconciliation already spent part
        of the budget); it defaults to now.
        """

        started = self._clock() if started is None else started
        deadline = started + budget
        hard_deadline = deadline + self._policy.call_grace_seconds
        polls = 0
        rounds = 0
        min_delay = 0.0
        round_latency = 0.0
        poll_problems: dict[int, str] = {}

        while True:
            record = self._require(handle)
            if record.state is RecordState.CANCELLED or record.unwound:
                break
            pollable = [c for c in record.children if self._pollable(c) and c.index not in poll_problems]
            if not pollable:
                break
            if not (poll_first and rounds == 0):
                remaining = deadline - self._clock()
                # Provider latency spends the budget too: stop when the last round's duration would not fit.
                if remaining <= 0 or min_delay > remaining or (round_latency > 0 and remaining <= round_latency):
                    break
                self._sleep(min(max(self._policy.delay(rounds), min_delay), max(0.0, remaining - round_latency)))
            rounds += 1
            min_delay = 0.0
            round_started = self._clock()
            for child in pollable:
                polls += 1
                min_delay = max(min_delay, self._poll_child(handle, child, poll_problems, hard_deadline))
            round_latency = max(0.0, self._clock() - round_started)
            if rounds >= self._policy.max_polls or self._clock() >= deadline:
                break

        return self._report(
            handle, started=started, polls=polls, budget=budget, poll_problems=poll_problems, fetch=fetch
        )

    def _poll_child(self, handle: str, child: BatchChild, poll_problems: dict[int, str], hard_deadline: float) -> float:
        """Poll one child once and record its state. Returns a minimum delay before the next poll (0.0 for none).

        The provider call is abandoned at ``hard_deadline`` and treated as a transient error.
        """

        provider = self._coordinator.resolve_provider(child.provider)
        if provider is None or child.provider_batch_id is None:
            poll_problems[child.index] = f"provider '{child.provider}' is not configured on this server"
            return 0.0
        batch_id = child.provider_batch_id
        try:
            status = call_within(lambda: provider.poll_batch(batch_id), hard_deadline - self._clock())
        except CallTimedOut as exc:
            logger.info("Batch poll (%s) abandoned: %s; absorbing", child.provider, exc)
            return 0.0
        except BatchRateLimited as exc:
            logger.info("Batch poll rate limited (%s); absorbing", child.provider)
            return float(exc.retry_after) if exc.retry_after else 0.0
        except BatchNotFound:
            self._set_child_if_unchanged(
                handle,
                child,
                ChildState.FAILED,
                detail=f"provider '{child.provider}' no longer knows this batch (HTTP 404)",
            )
            return 0.0
        except BatchResultsExpired:
            # HTTP 410 on a poll: the provider has deleted the batch with its results (retention passed),
            # so it will never advance. Record it as terminal ("failed" claims no success); fetching it then
            # reports results_expired rather than a batch that is "running" forever.
            self._set_child_if_unchanged(
                handle,
                child,
                ChildState.FAILED,
                detail=f"provider '{child.provider}' no longer retains this batch or its results (HTTP 410)",
            )
            return 0.0
        except BatchProviderError as exc:
            if exc.retryable:
                logger.info("Batch poll failed transiently (%s): %s; absorbing", child.provider, exc)
                return 0.0
            poll_problems[child.index] = _scrub(child, str(exc))
            return 0.0
        except UnsupportedBatchOperation as exc:
            poll_problems[child.index] = _scrub(child, str(exc))
            return 0.0
        except Exception as exc:  # an adapter bug must not lose the handle
            logger.exception("Unexpected error polling %s batch for %s", child.provider, handle)
            poll_problems[child.index] = _scrub(child, f"unexpected error polling the provider: {exc}")
            return 0.0

        new_state = ChildState.from_batch_state(status.state)
        if new_state is not child.status:
            detail = _scrub(child, status.error_message) if status.error_message else None
            self._set_child_if_unchanged(handle, child, new_state, detail=detail)
        return 0.0

    def _set_child_if_unchanged(
        self, handle: str, child: BatchChild, status: ChildState, *, detail: str | None = None
    ) -> None:
        """Record a provider-observed state only if the child is still as this call read it.

        A concurrent cancel (or another resume) may have resolved the child while the provider
        call was in flight; their write wins, because a poll result is only an observation.
        """

        try:
            self.store.set_child_status(handle, child.index, status, detail=detail, expect_status=child.status)
        except ChildStateChanged:
            logger.info(
                "Child %s of %s changed while it was being polled; keeping the newer state", child.index, handle
            )

    # -- building the result -------------------------------------------------------

    def _report(
        self,
        handle: str,
        *,
        started: float,
        polls: int,
        budget: float,
        poll_problems: dict[int, str],
        fetch: bool,
    ) -> BatchRunResult:
        record = self._require(handle)
        waited = max(0.0, self._clock() - started)
        deadline = started + budget

        def base(
            status: BatchRunStatus, children: tuple[ChildResult, ...], notes: tuple[str, ...] = ()
        ) -> BatchRunResult:
            return BatchRunResult(
                status=status,
                handle=handle,
                record=record,
                children=children,
                notes=notes,
                waited_seconds=waited,
                polls=polls,
                wait_budget_seconds=budget,
            )

        plain = tuple(
            ChildResult(
                c,
                poll_problem=poll_problems.get(c.index),
                pending_reason=_UNCONFIRMED_SUBMISSION if c.status is ChildState.SUBMITTING else None,
            )
            for c in record.children
        )
        state = record.state
        if state is RecordState.CANCELLED:
            return base(BatchRunStatus.CANCELLED, plain)
        if state is RecordState.INDETERMINATE:
            return base(BatchRunStatus.INDETERMINATE, plain)
        if state is RecordState.ORPHANED:
            return base(BatchRunStatus.ORPHANED, plain)
        if state in (RecordState.SUBMITTING, RecordState.RUNNING):
            return base(BatchRunStatus.RUNNING, plain)
        if record.unwound:
            return base(BatchRunStatus.FAILED, plain, ("The submission was abandoned and unwound.",))

        # Terminal record: collect what each child produced.
        if not fetch:
            return base(_TERMINAL_STATUS[state], plain)
        results = tuple(self._collect_child(c, deadline) for c in record.children)
        if any(r.pending_reason for r in results):
            return base(BatchRunStatus.RUNNING, results)
        if not any(r.outcomes for r in results):
            if any(r.results_expired for r in results):
                return base(BatchRunStatus.RESULTS_EXPIRED, results)
        return base(_TERMINAL_STATUS[state], results)

    def _collect_child(self, child: BatchChild, deadline: float) -> ChildResult:
        if child.provider_batch_id is None:
            if child.status.is_terminal:
                return ChildResult(child, error=child.detail or "no provider batch was created for this submission")
            return ChildResult(child, pending_reason=child.detail or "the submission has not been confirmed")
        if child.status not in _FETCHABLE_CHILD_STATES:
            return ChildResult(child)  # locally/provider cancelled, or not terminal (not reached here)

        provider = self._coordinator.resolve_provider(child.provider)
        if provider is None:
            return ChildResult(
                child,
                pending_reason=f"provider '{child.provider}' is not configured on this server, so its results cannot be fetched",
            )
        return self._fetch_child(provider, child, deadline)

    def _fetch_child(self, provider: ModelProvider, child: BatchChild, deadline: float) -> ChildResult:
        batch_id = child.provider_batch_id
        assert batch_id is not None
        hard_deadline = deadline + self._policy.call_grace_seconds
        expected = child.custom_ids or None
        last_problem = ""
        for attempt in range(max(1, self._policy.fetch_attempts)):
            attempt_started = self._clock()
            try:
                outcomes = call_within(
                    lambda: provider.fetch_batch(batch_id, expected_custom_ids=expected),
                    hard_deadline - self._clock(),
                )
            except CallTimedOut:
                return ChildResult(
                    child,
                    pending_reason="the batch finished but fetching its results did not fit in this call's wait",
                )
            except BatchResultsExpired:
                return ChildResult(
                    child,
                    error="the provider no longer retains this batch's results (retention window passed)",
                    results_expired=True,
                )
            except BatchNotReady as exc:
                self._set_child_if_unchanged(child.handle, child, ChildState.from_batch_state(exc.state))
                return ChildResult(child, pending_reason=f"the provider reports the batch is still {exc.state.value}")
            except BatchFailed as exc:
                reason = _scrub(child, str(exc))
                if child.detail and child.detail not in reason:
                    reason = f"{child.detail} ({reason})"
                return ChildResult(child, error=reason)
            except BatchIncompleteResults as exc:
                return ChildResult(
                    child,
                    outcomes=tuple(self._recorded(child, o) for o in exc.outcomes),
                    error=_scrub(child, f"the provider returned fewer results than were submitted: {exc}"),
                )
            except BatchNotFound:
                return ChildResult(child, error=f"provider '{child.provider}' no longer holds this batch")
            except BatchProviderError as exc:
                if not exc.retryable:
                    return ChildResult(child, error=_scrub(child, str(exc)))
                last_problem = _scrub(child, str(exc))
                wait = (
                    float(exc.retry_after) if isinstance(exc, BatchRateLimited) and exc.retry_after else attempt + 1.0
                )
                now = self._clock()
                remaining = deadline - now
                # A retry that would likely take as long as the failed attempt did must also fit.
                if (
                    attempt + 1 >= self._policy.fetch_attempts
                    or remaining <= 0
                    or wait > remaining
                    or remaining <= now - attempt_started
                ):
                    break
                self._sleep(wait)
                continue
            except UnsupportedBatchOperation as exc:
                return ChildResult(child, error=_scrub(child, str(exc)))
            except Exception as exc:
                logger.exception("Unexpected error fetching %s batch results", child.provider)
                return ChildResult(child, error=_scrub(child, f"unexpected error fetching results: {exc}"))
            return ChildResult(child, outcomes=tuple(self._recorded(child, o) for o in outcomes))
        return ChildResult(
            child,
            pending_reason=f"the batch finished but its results could not be fetched right now ({last_problem})",
        )

    @staticmethod
    def _recorded(child: BatchChild, outcome: BatchOutcome) -> BatchOutcome:
        """Map an outcome onto the model recorded at submission, not whatever the adapter now believes.

        A fresh process forgot the caller's model spelling and the slug it submitted, so the
        response model name is overwritten with the recorded ``submitted_model`` and the
        recorded ``requested_model`` is kept in the metadata when it differs.
        """

        response = outcome.response
        if response is None:
            if outcome.error is not None:
                scrubbed = _scrub(child, outcome.error.message)
                if scrubbed != outcome.error.message:
                    return dataclasses.replace(outcome, error=dataclasses.replace(outcome.error, message=scrubbed))
            return outcome
        metadata = dict(response.metadata)
        if child.requested_model != child.submitted_model:
            metadata["requested_model"] = child.requested_model
        metadata.setdefault("batch_execution", True)
        return dataclasses.replace(
            outcome,
            response=dataclasses.replace(response, model_name=child.submitted_model, metadata=metadata),
        )

    # -- cancellation --------------------------------------------------------------

    def _cancel_child(self, handle: str, child: BatchChild) -> ChildCancelReport:
        def report(action: str, state: ChildState, detail: str, *, continues: bool = False) -> ChildCancelReport:
            if child.provider_batch_id:
                # Callers see only the handle (never the provider's id); the operator needs it to find the batch
                # at the provider. Logged for every outcome, loudly when the batch was NOT stopped and keeps
                # running (and billing).
                log = logger.warning if continues else logger.info
                log(
                    "Cancel of handle %s child %s: %s batch %s: %s%s",
                    handle,
                    child.index,
                    child.provider,
                    child.provider_batch_id,
                    action,
                    " (NOT stopped at the provider; it keeps running and billing)" if continues else "",
                )
            return ChildCancelReport(
                index=child.index,
                provider=child.provider,
                model=child.requested_model,
                provider_batch_id=child.provider_batch_id,
                action=action,
                state_after=state,
                detail=detail,
                batch_continues=continues,
            )

        if child.status.is_terminal:
            return report("already_terminal", child.status, f"already {child.status.value}; nothing to cancel")
        if child.provider_batch_id is None or child.status not in _RUNNING_CHILD_STATES:
            return report(
                "unconfirmed",
                child.status,
                "its submission is not confirmed (no provider batch id is recorded); if the provider did accept it, "
                "that batch continues and will still be billed, and it stays on the operator list",
                continues=True,
            )

        provider = self._coordinator.resolve_provider(child.provider)
        who = f"provider '{child.provider}'"
        if provider is None:
            detail = (
                f"{who} is not configured on this server, so cancellation could not be requested; the provider "
                "batch continues and will still be billed. Polling has stopped and its results will be ignored"
            )
            self.store.set_child_status(handle, child.index, ChildState.CANCELLED, detail=detail)
            return report("continues", ChildState.CANCELLED, detail, continues=True)

        try:
            outcome = provider.cancel_batch(child.provider_batch_id)
        except Exception as exc:
            detail = (
                f"cancellation could not be requested from {who} ({_scrub(child, str(exc))}); the provider batch may continue and "
                "will still be billed. Polling has stopped and its results will be ignored"
            )
            self.store.set_child_status(handle, child.index, ChildState.CANCELLED, detail=detail)
            return report("continues", ChildState.CANCELLED, detail, continues=True)

        if outcome.state is CancelState.CANCELLED:
            detail = f"cancelled at {who}"
            self.store.set_child_status(handle, child.index, ChildState.CANCELLED, detail=detail)
            return report("cancelled", ChildState.CANCELLED, detail)
        if outcome.state is CancelState.CANCELLING:
            detail = (
                f"{who} is still stopping the batch; work already processed may still be billed. "
                "Polling has stopped and its results will be ignored"
            )
            self.store.set_child_status(handle, child.index, ChildState.CANCELLED, detail=detail)
            return report("cancelling", ChildState.CANCELLED, detail, continues=True)
        if outcome.state is CancelState.ALREADY_TERMINAL:
            return self._report_already_terminal(handle, child, provider, report)

        # UNSUPPORTED: the provider has no cancel mechanism (OpenRouter).
        detail = (
            f"{who} has no cancel mechanism: the provider batch continues and will still be billed. "
            "It was marked cancelled locally; polling has stopped and its results will be ignored"
        )
        self.store.set_child_status(handle, child.index, ChildState.CANCELLED, detail=detail)
        return report("continues", ChildState.CANCELLED, detail, continues=True)

    def _report_already_terminal(
        self,
        handle: str,
        child: BatchChild,
        provider: ModelProvider,
        report: Callable[..., ChildCancelReport],
    ) -> ChildCancelReport:
        """The provider says the batch had already finished: learn which terminal state, never cancel it."""

        assert child.provider_batch_id is not None
        state = child.status
        try:
            status = provider.poll_batch(child.provider_batch_id)
            if status.state.is_terminal:
                state = ChildState.from_batch_state(status.state)
                self.store.set_child_status(handle, child.index, state)
        except Exception:
            logger.info("Could not confirm the terminal state of an already-finished batch", exc_info=True)
        return report("already_terminal", state, "the provider reports it had already finished; nothing was cancelled")


def _scrub(child: BatchChild, text: str) -> str:
    """Remove the provider's batch id from text that reaches a caller (the handle is the only id callers see)."""

    batch_id = child.provider_batch_id
    if batch_id and batch_id in text:
        return text.replace(batch_id, "<provider batch id withheld>")
    return text


_TERMINAL_STATUS: dict[RecordState, BatchRunStatus] = {
    RecordState.COMPLETED: BatchRunStatus.COMPLETED,
    RecordState.FAILED: BatchRunStatus.FAILED,
    RecordState.CANCELLED: BatchRunStatus.CANCELLED,
    RecordState.EXPIRED: BatchRunStatus.EXPIRED,
}


def build_default_executor(*, policy: WaitPolicy = DEFAULT_WAIT_POLICY) -> BatchExecutor:
    """An executor over the process-wide durable store and the default provider registry.

    Raises:
        BatchStoreUnavailable: the durable store cannot be opened (nothing falls back to memory).
    """

    return BatchExecutor(BatchCoordinator(get_batch_store()), policy=policy)


# ----------------------------------------------------------------------
# Standard renderings (tool-agnostic; tools wrap them in their own output type)
# ----------------------------------------------------------------------


def _iso(timestamp: float | None) -> str:
    if timestamp is None:
        return "an unknown time"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def describe_submission_error(error: BatchSubmissionError) -> str:
    """Caller-facing text for a failed submission; names the handle only when it is resumable."""

    text = str(error)
    if isinstance(error, SubmissionOutcomeUnknown) and error.handle:
        text += (
            f" The submission is retained under batch_handle '{error.handle}'; calling this tool again with ONLY "
            "that batch_handle confirms whether the provider accepted it. Do not resubmit the work: that could "
            "create duplicate paid batches."
        )
    return text


def _children_summary(result: BatchRunResult) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item in result.children:
        entry: dict[str, Any] = {
            "model": item.child.requested_model,
            "state": item.child.status.value,
            "requests": item.child.request_total,
        }
        # A name the provider resolved to a different model (a family alias under batch) says so, so the
        # caller can see what is being paid for before the batch finishes.
        entry.update(
            {
                key: value
                for key, value in resolution_metadata(item.child.requested_model, item.child.submitted_model).items()
                if key == "model_resolved"
            }
        )
        # What the submission resolved the model to, as recorded with the child: the limits that applied and the
        # newer batch models the alias did not pick. Read from the record only, so a resumed handle reports what
        # was true at submission whatever OpenRouter lists now; absent when nothing was recorded.
        limits = limits_from_record(item.child.resolution)
        if limits is not None:
            entry["limits"] = limits
        entry.update(note_from_record(item.child.resolution))
        note = item.pending_reason or item.poll_problem or item.error
        if note:
            entry["note"] = note
        elif item.child.detail and item.child.status in (
            ChildState.CANCELLED,
            ChildState.INDETERMINATE,
            ChildState.ORPHANED,
        ):
            entry["note"] = item.child.detail
        summary.append(entry)
    return summary


def result_message(
    result: BatchRunResult, *, tool_name: str, policy: WaitPolicy = DEFAULT_WAIT_POLICY, resume_hint: str = ""
) -> str:
    """The standard human-readable message for a result that is not a plain completion."""

    handle = result.handle
    status = result.status
    if status is BatchRunStatus.UNKNOWN_HANDLE:
        return (
            f"Unknown batch handle '{handle}': this server's batch store holds no operation under that handle. "
            "Check the handle was copied exactly; it is not an expired handle. Nothing was submitted."
        )
    if status is BatchRunStatus.HANDLE_EXPIRED:
        return (
            f"The results for batch handle '{handle}' have expired: the record was purged at its retention limit "
            f"({batch_retention_days():g} days) on {_iso(result.expired_at)}, so they can no longer be retrieved. "
            "The handle was valid; it is not unknown. Nothing was resubmitted."
        )
    if status is BatchRunStatus.RUNNING:
        waited = int(result.wait_budget_seconds)
        unconfirmed = [item for item in result.children if item.child.status is ChildState.SUBMITTING]
        if unconfirmed and len(unconfirmed) == len(result.children):
            lead = (
                "The batch submission is NOT confirmed: the provider may or may not hold a batch for it yet "
                "(nothing has been shown to be running), and the work is NOT finished."
            )
        elif unconfirmed:
            lead = (
                "The batch is still running at the provider; the work is NOT finished. Part of the submission is "
                "not confirmed either: the provider may or may not hold a batch for it."
            )
        else:
            lead = "The batch is still running at the provider; the work is NOT finished."
        text = (
            f"{lead} Call {tool_name} again with ONLY "
            f"batch_handle='{handle}' to keep waiting (each call waits up to {waited}s, adjustable with wait_seconds "
            f"up to {int(policy.max_seconds)}s) or with batch_handle plus cancel=true to stop it. Do not resubmit the "
            "task and do not report it done before the final result arrives."
        )
        pending = [item.pending_reason for item in result.children if item.pending_reason]
        problems = [item.poll_problem for item in result.children if item.poll_problem]
        if pending:
            text += " " + " ".join(sorted(set(pending))) + "."
        if problems:
            text += " Polling problems: " + "; ".join(sorted(set(problems))) + "."
        if resume_hint:
            text += " " + resume_hint
        return text
    if status is BatchRunStatus.CANCELLED:
        lines = [f"Batch '{handle}' is cancelled."]
        for report in result.cancel_reports:
            lines.append(f"- {report.model}: {report.detail}.")
        if not result.cancel_reports:
            for item in result.children:
                if item.child.detail:
                    lines.append(f"- {item.child.requested_model}: {item.child.detail}.")
        if any(r.batch_continues for r in result.cancel_reports):
            lines.append(
                "WARNING: at least one provider batch was NOT stopped and will still be billed by the provider."
            )
        lines.extend(result.notes)
        return "\n".join(lines)
    if status is BatchRunStatus.RESULTS_EXPIRED:
        return (
            f"The results of batch '{handle}' have expired at the provider (it keeps them about 30 days) and can no "
            "longer be fetched. Nothing was resubmitted."
        )
    if status is BatchRunStatus.EXPIRED:
        return (
            f"Batch '{handle}' expired at the provider before it was processed (its completion window lapsed). It "
            "was not resubmitted automatically."
        )
    if status is BatchRunStatus.FAILED:
        reasons = [
            f"{item.child.requested_model}: {item.error or item.child.detail or item.child.status.value}"
            for item in result.children
        ]
        return f"Batch '{handle}' failed. " + "; ".join(reasons)
    if status in (BatchRunStatus.INDETERMINATE, BatchRunStatus.ORPHANED):
        what = (
            "could not be confirmed (the provider may or may not hold a billable batch for it)"
            if status is BatchRunStatus.INDETERMINATE
            else "left a provider batch running that is not tracked"
        )
        details = [
            f"{item.child.provider} {item.child.requested_model}: {item.child.detail or item.child.status.value}"
            for item in result.children
            if item.child.status in (ChildState.INDETERMINATE, ChildState.ORPHANED)
        ]
        return (
            f"Batch '{handle}' needs operator attention: its submission {what}. It was not retried. "
            + ("; ".join(details) if details else "")
        ).strip()
    return f"Batch '{handle}' is {status.value}."


def build_status_payload(
    result: BatchRunResult, *, tool_name: str, policy: WaitPolicy = DEFAULT_WAIT_POLICY, resume_hint: str = ""
) -> dict[str, Any]:
    """The standard payload for a non-error, non-completed result (``running`` or ``cancelled``).

    Carries the caller-facing ``batch_handle`` only, never a provider batch identifier.
    """

    payload: dict[str, Any] = {
        "status": result.status.value,
        "batch_handle": result.handle,
        "tool": tool_name,
        "children": _children_summary(result),
        "message": result_message(result, tool_name=tool_name, policy=policy, resume_hint=resume_hint),
    }
    if result.status is BatchRunStatus.RUNNING:
        payload["resume_with"] = {"batch_handle": result.handle}
        payload["waited_seconds"] = round(result.waited_seconds, 1)
    if result.cancel_reports:
        payload["provider_batches_continue"] = any(r.batch_continues for r in result.cancel_reports)
        if result.status is not BatchRunStatus.CANCELLED:
            # A mixed outcome (some children had already finished): still say what cancelling did.
            payload["message"] += " Cancellation: " + "; ".join(
                f"{report.model}: {report.detail}" for report in result.cancel_reports
            )
    return payload
