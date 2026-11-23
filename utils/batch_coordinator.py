"""Submission saga and reconciliation for provider batches.

A local write and a remote submission cannot be made atomic: a crash or an ambiguous
network failure between the provider accepting a batch and the server recording it leaves
a billable batch nobody knows about. This module makes that recoverable rather than
silent.

Submission (:meth:`BatchCoordinator.submit`)
    1. Persist *intent* (record state ``submitting``; one child per model carrying the
       model, endpoint, request total, submitted custom_ids and the window start) **before**
       any provider call.
    2. Call ``submit_batch`` for each child, then record the provider id on the child.
    3. Failure handling:

       * a definite refusal (validation, or an HTTP status that proves non-acceptance: 400, 401, 402,
         403, 404, 413, 422, 429, ...) fails the child cleanly;
       * ``BatchProviderError(code="submission_outcome_unknown")`` (or any untyped failure
         of the call itself, a 408/425, any other 4xx outside the refusal list, a 5xx, or a 2xx that
         names no batch) is **ambiguous**: the intent
         stays, nothing is retried, and the caller gets :class:`SubmissionOutcomeUnknown`;
       * if the provider accepted a batch but recording its id fails, cancellation is
         requested; when the provider has no cancel mechanism (OpenRouter) or cancel fails,
         the batch is recorded ``orphaned`` (if the store is still writable) and the error
         names provider and batch id and says the batch continues and is still billed;
       * a failure partway through a multi-batch operation unwinds the earlier accepted
         children the same way and abandons the operation. No handle is returned on any
         failure except the ambiguous outcome of a *single*-batch submission, whose retained
         handle is named so it can be resumed. (An ambiguous multi-batch attempt keeps its
         intent for reconciliation, exposed as ``retained_handle`` for operators only.)

Reconciliation (:meth:`BatchCoordinator.reconcile_pending` / ``reconcile_handle``)
    Provider facts (OpenRouter, verified 2026-11-21): no idempotency key; a metadata-only
    batch list exists. So an unresolved intent is matched approximately and **never**
    resubmitted:

    * exactly one *unclaimed* listed batch whose model, endpoint, request total and
      creation time (inside the intent window) all match, with no competing unresolved
      intent and no entry the listing cannot verify -> adopted under the original handle;
    * zero matches, a listing that provably covers the window, **and** the window long past
      (the provider's list can lag an in-flight or just-accepted POST) -> intent closed out
      as ``failed`` (resuming the handle reports the failure); before that it stays
      unresolved so a later startup or resume tries again;
    * several matches, a match already claimed by another record, unverifiable entries, a
      possibly-truncated listing, or a provider with no list operation -> ``indeterminate``,
      retained and listed by :meth:`BatchStore.needs_attention`, never discarded;
    * a provider that is not configured, or a listing call that failed, leaves the intent
      unresolved so a later startup or resume tries again.

    Bounded provider calls
    ``submit_batch`` and ``list_batches`` run under a per-call ceiling supplied by
    :class:`utils.batch_execution.BatchExecutor` (a fixed :data:`DEFAULT_CALL_TIMEOUT_SECONDS`
    otherwise) so a hung provider cannot hold a tool call open. The submit ceiling does not depend on
    the caller's ``wait_seconds`` (a slow create is not abandoned because the caller declined to wait
    for the result). A listing that does not answer is simply not read (the child stays unresolved and is
    retried later). A submit that does not answer is **ambiguous** like any other unknowable outcome:
    the intent stays, nothing is retried, and if the abandoned call is later accepted its batch id is
    always recorded: on the child while it is unresolved or ``indeterminate`` (the id resolves it), as
    ``orphaned`` when the operation was abandoned or the child was already closed out as ``failed``.
    The reconcile lock is never held across a provider list call.

Two reconcilers can meet on one child (startup thread plus a resume, or two server
    processes on one store). Every decision is a compare-and-set on the child still being
    ``submitting``, serialised in-process by a lock, so the loser reports ``skipped`` instead
    of overwriting the winner. A handle whose submission is still running in this process is
    never reconciled.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from providers.openrouter_batch_wire import BATCH_ENDPOINT, SUBMISSION_OUTCOME_UNKNOWN, is_definite_submit_refusal
from providers.shared.batch import (
    BatchListEntry,
    BatchListing,
    BatchProviderError,
    BatchRequest,
    CancelState,
    UnsupportedBatchOperation,
    validate_batch_requests,
)
from utils.batch_store import (
    BatchChild,
    BatchRecord,
    BatchStore,
    BatchStoreError,
    BatchStoreUnavailable,
    ChildIntent,
    ChildState,
    ChildStateChanged,
    HandleLookup,
    LookupStatus,
    ProviderBatchAlreadyClaimed,
    default_batch_store_path,
    get_batch_store,
)

if TYPE_CHECKING:
    from providers.base import ModelProvider

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CALL_TIMEOUT_SECONDS",
    "BatchCoordinator",
    "BatchSubmissionError",
    "CallTimedOut",
    "ChildPlan",
    "OrphanedBatch",
    "ReconcileAction",
    "ReconcileResult",
    "SubmissionOutcomeUnknown",
    "call_within",
    "reconcile_at_startup",
    "registry_provider_resolver",
]

#: ``BatchProviderError.code`` an adapter raises when a submit POST's outcome is unknowable.
OUTCOME_UNKNOWN_CODE = SUBMISSION_OUTCOME_UNKNOWN

#: Intents younger than this are skipped by :meth:`BatchCoordinator.reconcile_pending`. Zero:
#: a restart right after a crash must still adopt. A submission running *in this process*
#: is protected by the in-flight set, and one in another process by the close-out grace and
#: the compare-and-set below (adoption is idempotent, close-out waits for the window).
DEFAULT_RECONCILE_MIN_AGE_SECONDS = 0.0

#: How long after an intent's window has fully closed a "no matching batch" verdict is
#: still withheld, so a provider list that lags a just-accepted batch cannot cause a
#: premature (irreversible) close-out.
DEFAULT_CLOSE_OUT_GRACE_SECONDS = 120.0

#: Tolerance when comparing the provider's creation time with the local intent window.
DEFAULT_CLOCK_SLACK_SECONDS = 10.0

#: Assumed upper bound of a submit call when its end was never recorded (a crash).
DEFAULT_SUBMIT_WINDOW_SECONDS = 300.0

#: Ceiling for one provider submit or list call when the caller supplies none: the longest wait a
#: caller may ask for (50 s) plus the 5 s grace, so no call is held longer than a tool call may be.
DEFAULT_CALL_TIMEOUT_SECONDS = 55.0

ProviderResolver = Callable[[str], "ModelProvider | None"]
#: Seconds the next provider call may take (re-evaluated before every call, so it shrinks as a wait is spent).
CallTimeout = Callable[[], float]

# Handles whose submission saga is running in this process right now (never reconciled), and a
# process-wide lock serialising reconcilers so they cannot interleave read-check-write.
_IN_FLIGHT: set[str] = set()
_IN_FLIGHT_LOCK = threading.Lock()
_RECONCILE_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# Bounded calls
# ----------------------------------------------------------------------


class CallTimedOut(Exception):
    """A provider call did not finish inside its ceiling (internal; never reaches a caller)."""


def call_within(
    function: Callable[[], Any], timeout: float, *, on_late_result: Callable[[Any], None] | None = None
) -> Any:
    """Run ``function`` and return its result, or raise :class:`CallTimedOut` after ``timeout`` seconds.

    The call runs on a daemon thread so a hung provider request cannot hold the tool call past its
    deadline; an abandoned request finishes (or times out at the HTTP layer) on its own. A read-only
    call (poll, fetch, list) needs nothing more. A call with a side effect (submit) passes
    ``on_late_result``: if the call returns after it was abandoned, that hook receives the value on
    the worker thread, so the effect is not lost.
    """

    if timeout <= 0:
        raise CallTimedOut("no time left")
    box: dict[str, Any] = {}
    lock = threading.Lock()
    state = {"abandoned": False, "finished": False}

    def run() -> None:
        try:
            value = function()
        except BaseException as exc:  # re-raised verbatim in the caller's thread
            with lock:
                box["error"] = exc
                state["finished"] = True
            return
        with lock:
            box["value"] = value
            state["finished"] = True
            late = state["abandoned"]
        if late and on_late_result is not None:
            try:
                on_late_result(value)
            except Exception:
                logger.exception("Handling the late result of an abandoned provider call failed")

    worker = threading.Thread(target=run, name="batch-provider-call", daemon=True)
    worker.start()
    worker.join(timeout)
    with lock:
        if not state["finished"]:
            state["abandoned"] = True
            raise CallTimedOut(f"still running after {timeout:.1f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ChildPlan:
    """One provider batch to submit: a provider, a requested model and its requests.

    ``model`` is the caller's spelling; it is passed to ``submit_batch`` untouched and
    persisted alongside the model the provider adapter reports it would submit.
    """

    provider: ModelProvider
    model: str
    requests: Sequence[BatchRequest]
    endpoint: str = BATCH_ENDPOINT
    #: What admission resolved ``model`` to (``provider.resolve_batch_model``), or ``None`` for a provider that
    #: resolves nothing. When set, submission uses exactly it (no second resolution) and it is persisted with the
    #: child record, so a resume reports what was submitted without consulting any catalog.
    resolution: Any = None


@dataclass(frozen=True)
class OrphanedBatch:
    """An accepted provider batch the server could not cancel or record."""

    provider: str
    batch_id: str
    detail: str

    def describe(self) -> str:
        return f"provider '{self.provider}' batch '{self.batch_id}' ({self.detail})"


class BatchSubmissionError(RuntimeError):
    """A batch submission failed; no handle is returned for it.

    ``orphans`` names every accepted provider batch left running (or finished and billed)
    so an operator can find it. ``handle`` is set only for the ambiguous outcome of a
    single-batch submission, where the record is deliberately retained and resumable.
    ``retained_handle`` names a retained record that is *not* resumable (an abandoned
    multi-batch attempt): for operator diagnostics and logs, never to hand to a caller.
    """

    def __init__(
        self,
        message: str,
        *,
        handle: str | None = None,
        orphans: Sequence[OrphanedBatch] = (),
        retained_handle: str | None = None,
    ) -> None:
        super().__init__(message)
        self.handle = handle
        self.retained_handle = retained_handle if retained_handle is not None else handle
        self.orphans = tuple(orphans)


class SubmissionOutcomeUnknown(BatchSubmissionError):
    """The provider may or may not hold a billable batch for the submission; it was not retried."""


class ReconcileAction(str, Enum):
    ADOPTED = "adopted"
    CLOSED_OUT = "closed_out"
    INDETERMINATE = "indeterminate"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ReconcileResult:
    """What reconciliation decided for one unresolved child."""

    handle: str
    child_index: int
    provider: str
    action: ReconcileAction
    provider_batch_id: str | None = None
    detail: str = ""


@dataclass
class _Unwind:
    orphan: OrphanedBatch | None = None
    note: str = ""


def provider_name_of(provider: ModelProvider) -> str:
    """The provider's stable name (``ProviderType`` value), falling back to its class name."""

    try:
        return str(provider.get_provider_type().value)
    except Exception:
        return type(provider).__name__


def registry_provider_resolver(name: str) -> ModelProvider | None:
    """Resolve a provider name (``ProviderType`` value) through the default registry."""

    try:
        from providers.registry import get_default_registry
        from providers.shared import ProviderType

        return get_default_registry().get_provider(ProviderType(name))
    except Exception:
        logger.debug("Could not resolve batch provider %r from the registry", name, exc_info=True)
        return None


def _is_ambiguous(exc: BaseException) -> bool:
    """True when a failed submit call may nonetheless have created a provider batch."""

    if isinstance(exc, BatchProviderError):
        if exc.code == OUTCOME_UNKNOWN_CODE:
            return True
        # Only a status that proves non-acceptance (400, 401, 402, 403, 404, 413, 422, 429 and
        # the like) is a definite refusal. No status at all, a 408/425 (a proxy may have
        # forwarded the body before timing out), any other 4xx, a 5xx (the server may have
        # created the batch before failing), or a non-error status (a 2xx that named no batch)
        # all leave a possibly billed batch behind.
        return not is_definite_submit_refusal(exc.status_code)
    # Local validation and "not supported" never reached the provider; anything else
    # (transport errors, timeouts, unexpected exceptions) cannot be assumed harmless.
    return not isinstance(exc, (ValueError, TypeError, UnsupportedBatchOperation))


# ----------------------------------------------------------------------
# Coordinator
# ----------------------------------------------------------------------


class BatchCoordinator:
    """Runs the submission saga and reconciles unresolved intent against the provider."""

    def __init__(
        self,
        store: BatchStore,
        *,
        provider_resolver: ProviderResolver | None = None,
        clock: Callable[[], float] = time.time,
        clock_slack_seconds: float = DEFAULT_CLOCK_SLACK_SECONDS,
        submit_window_seconds: float = DEFAULT_SUBMIT_WINDOW_SECONDS,
        close_out_grace_seconds: float = DEFAULT_CLOSE_OUT_GRACE_SECONDS,
        call_timeout_seconds: float | None = DEFAULT_CALL_TIMEOUT_SECONDS,
    ) -> None:
        self._store = store
        self._default_call_timeout = call_timeout_seconds
        self._resolve_provider: ProviderResolver = provider_resolver or registry_provider_resolver
        self._clock = clock
        self._slack = clock_slack_seconds
        self._submit_window = submit_window_seconds
        self._close_out_grace = close_out_grace_seconds

    @property
    def store(self) -> BatchStore:
        return self._store

    def resolve_provider(self, name: str) -> ModelProvider | None:
        """The configured provider named ``name`` (a ``ProviderType`` value), or ``None``.

        The same resolver the coordinator uses for reconciliation, exposed so the wait,
        resume and cancel paths reach a recorded child's provider the same way.
        """

        return self._resolve_provider(name)

    # ------------------------------------------------------------------
    # Submission saga
    # ------------------------------------------------------------------

    def submit(
        self,
        tool_name: str,
        plans: Sequence[ChildPlan],
        *,
        continuation_id: str | None = None,
        payload: Any = None,
        call_timeout: CallTimeout | None = None,
    ) -> str:
        """Submit one provider batch per plan under a single new handle and return it.

        ``call_timeout`` gives the seconds each ``submit_batch`` call may take (asked again before
        every call); without it every call gets :data:`DEFAULT_CALL_TIMEOUT_SECONDS`. A call that
        does not answer in time is an ambiguous outcome (see below), never a refusal.

        Raises:
            ValueError: empty plans, or a plan with no requests / duplicate custom_ids
                (before anything is persisted or sent).
            BatchSubmissionError: the durable store is unavailable or any submission
                failed. Nothing is left running except what the error names. No handle.
            SubmissionOutcomeUnknown: a submit call's outcome could not be determined;
                for a single-batch submission ``error.handle`` names the retained intent
                (none for an abandoned multi-batch attempt). Never retried.
        """

        plan_list = list(plans)
        if not plan_list:
            raise ValueError("A batch submission needs at least one child plan")
        for plan in plan_list:
            validate_batch_requests(plan.requests)

        if not self._store.is_durable:
            raise BatchSubmissionError(
                "Batch submission needs the durable batch store, which is not available; nothing was submitted."
            )

        # A model name a provider resolves dynamically (a family alias) must mean the same thing when the
        # intent is recorded and when each batch is submitted, for every plan of the operation: hold the
        # providers' resolution data fixed for the whole submission. What was submitted is persisted in the
        # record; resume, reconcile and cancel read it and never resolve again.
        with contextlib.ExitStack() as scopes:
            entered: list[int] = []
            for plan in plan_list:
                if id(plan.provider) in entered:
                    continue
                entered.append(id(plan.provider))
                factory = getattr(plan.provider, "batch_resolution_scope", None)
                if callable(factory):
                    try:
                        scopes.enter_context(factory())
                    except Exception:
                        logger.debug("batch_resolution_scope unavailable for %s", plan.provider, exc_info=True)

            intents = [self._intent_for(plan) for plan in plan_list]
            try:
                handle = self._store.create_record(tool_name, intents, continuation_id=continuation_id, payload=payload)
            except BatchStoreError as exc:
                raise BatchSubmissionError(
                    f"Could not persist the batch submission intent ({exc}); nothing was submitted."
                ) from exc

            with _IN_FLIGHT_LOCK:
                _IN_FLIGHT.add(handle)
            try:
                return self._run_saga(handle, plan_list, call_timeout)
            finally:
                with _IN_FLIGHT_LOCK:
                    _IN_FLIGHT.discard(handle)

    def _timeout_for_next_call(self, call_timeout: CallTimeout | None) -> float | None:
        if call_timeout is not None:
            return call_timeout()
        return self._default_call_timeout

    def _submit_bounded(self, handle: str, index: int, plan: ChildPlan, call_timeout: CallTimeout | None) -> str:
        """``plan.provider.submit_batch`` under its ceiling; a call that never answers is unknowable, not refused."""

        timeout = self._timeout_for_next_call(call_timeout)

        def submit() -> str:
            if plan.resolution is None:
                return plan.provider.submit_batch(plan.model, list(plan.requests))
            return plan.provider.submit_batch(plan.model, list(plan.requests), resolution=plan.resolution)

        if timeout is None:
            return submit()
        try:
            batch_id = call_within(
                submit,
                timeout,
                on_late_result=lambda late_id: self._record_late_acceptance(handle, index, plan, str(late_id)),
            )
        except CallTimedOut as exc:
            raise BatchProviderError(
                f"{provider_name_of(plan.provider)} did not answer the batch submission within {timeout:.0f}s "
                f"({exc}); a batch may or may not have been created",
                provider=provider_name_of(plan.provider),
                code=OUTCOME_UNKNOWN_CODE,
            ) from exc
        return str(batch_id)

    def _record_late_acceptance(self, handle: str, index: int, plan: ChildPlan, batch_id: str) -> None:
        """A submit that was abandoned at its ceiling was accepted after all: do not lose the batch.

        Runs on the abandoned call's worker thread. The child takes the id if it is still unresolved
        (so the handle polls it); when the operation was abandoned (or the record is gone) the batch
        is recorded ``orphaned`` where possible. If a reconciler resolved the child first (marked it
        ``indeterminate``, or closed it out as ``failed``) the id is still recorded: attached to an
        ``indeterminate`` child of a live operation, ``orphaned`` otherwise. Always logs provider and
        id, because this is billable work.
        """

        name = provider_name_of(plan.provider)
        try:
            outcome = self._attach_late_batch(handle, index, batch_id)
            logger.warning(
                "Abandoned %s submission under %s was accepted late as batch %s (%s)", name, handle, batch_id, outcome
            )
        except Exception:
            logger.error(
                "Abandoned %s submission under %s was accepted late as batch %s but could not be recorded; "
                "the provider batch continues and is billed",
                name,
                handle,
                batch_id,
                exc_info=True,
            )

    def _attach_late_batch(self, handle: str, index: int, batch_id: str) -> str:
        """Record a late-accepted batch id on its child whatever state a reconciler left the child in.

        Returns ``"adopted"`` or ``"orphaned"``. Raises only when the store refuses every write.
        """

        late = "accepted after the submission call had been abandoned at its time limit"
        for _ in range(3):  # the child can change state under us; each pass re-reads it
            record = self._store.get(handle)
            child = next((c for c in record.children if c.index == index), None) if record else None
            if record is None or child is None:
                # Nothing left to attach to (the record was purged): the provider batch is still billed.
                raise BatchStoreError(f"no record of child {index} under {handle} to hold batch {batch_id}")
            if child.provider_batch_id == batch_id:
                return "adopted"  # a reconciler already attributed this very batch to the child
            if child.provider_batch_id is not None:
                raise BatchStoreError(
                    f"child {index} under {handle} already holds batch {child.provider_batch_id}; "
                    f"the late batch {batch_id} belongs to nobody"
                )
            # An indeterminate child of a live operation was waiting for exactly this: the id resolves it.
            adopt = not record.unwound and child.status in (ChildState.SUBMITTING, ChildState.INDETERMINATE)
            status = ChildState.VALIDATING if adopt else ChildState.ORPHANED
            detail = late + (
                ""
                if adopt
                else (
                    "; its operation was already abandoned, so nobody is waiting on it"
                    if record.unwound
                    else f"; the child had already been resolved as {child.status.value}, so the batch is recorded "
                    "here for an operator"
                )
            )
            try:
                self._store.record_provider_batch(
                    handle, index, batch_id, status=status, detail=detail, expect_status=child.status
                )
            except ChildStateChanged:
                continue
            return "adopted" if adopt else "orphaned"
        raise BatchStoreError(f"child {index} under {handle} kept changing state; batch {batch_id} was not recorded")

    def _run_saga(self, handle: str, plan_list: list[ChildPlan], call_timeout: CallTimeout | None = None) -> str:
        accepted: list[tuple[int, ChildPlan, str]] = []
        for index, plan in enumerate(plan_list):
            try:
                batch_id = self._submit_bounded(handle, index, plan, call_timeout)
            except Exception as exc:
                ambiguous = _is_ambiguous(exc)
                self._best_effort(self._store.set_child_window_end, handle, index, self._clock())
                raise self._abort(
                    handle, plan_list, failed_index=index, accepted=accepted, cause=exc, ambiguous=ambiguous
                ) from exc

            try:
                self._store.record_provider_batch(handle, index, batch_id, intent_window_end=self._clock())
            except Exception as exc:
                unwound = self._unwind(handle, index, plan.provider, batch_id, cause=f"recording it failed: {exc}")
                raise self._abort(
                    handle,
                    plan_list,
                    failed_index=index,
                    accepted=accepted,
                    cause=exc,
                    ambiguous=False,
                    unrecorded=(unwound, plan, batch_id),
                ) from exc
            accepted.append((index, plan, batch_id))

        state = self._best_effort(self._store.recompute_state, handle)
        if state is None:
            logger.warning("Batch %s was accepted but its record state could not be refreshed", handle)
        return handle

    def _intent_for(self, plan: ChildPlan) -> ChildIntent:
        recorded: dict[str, Any] | None = None
        if plan.resolution is not None:
            # The resolution admission made is the one that is recorded and submitted; nothing resolves again.
            submitted = str(plan.resolution.slug)
            recorded = plan.resolution.to_record()
        else:
            try:
                submitted = plan.provider.batch_submission_model(plan.model)
            except Exception:
                submitted = plan.model
        return ChildIntent(
            provider=provider_name_of(plan.provider),
            requested_model=plan.model,
            submitted_model=submitted,
            endpoint=plan.endpoint,
            request_total=len(plan.requests),
            custom_ids=tuple(request.custom_id for request in plan.requests),
            resolution=recorded,
        )

    def _abort(
        self,
        handle: str,
        plans: Sequence[ChildPlan],
        *,
        failed_index: int,
        accepted: Sequence[tuple[int, ChildPlan, str]],
        cause: BaseException,
        ambiguous: bool,
        unrecorded: tuple[_Unwind, ChildPlan, str] | None = None,
    ) -> BatchSubmissionError:
        """Unwind a failed multi-step submission and build the error to raise (no handle)."""

        orphans: list[OrphanedBatch] = []
        notes: list[str] = []

        if unrecorded is not None:
            unwound, plan, batch_id = unrecorded
            if unwound.orphan is not None:
                orphans.append(unwound.orphan)
            if unwound.note:
                notes.append(unwound.note)

        for index, plan, batch_id in accepted:
            unwound = self._unwind(handle, index, plan.provider, batch_id, cause="a later submission failed")
            if unwound.orphan is not None:
                orphans.append(unwound.orphan)
            if unwound.note:
                notes.append(unwound.note)

        failed_plan = plans[failed_index]
        provider_name = provider_name_of(failed_plan.provider)
        if unrecorded is None and not ambiguous:
            self._best_effort(
                self._store.set_child_status,
                handle,
                failed_index,
                ChildState.FAILED,
                detail=f"submission refused: {cause}",
            )
        for later in range(failed_index + 1, len(plans)):
            self._best_effort(
                self._store.set_child_status,
                handle,
                later,
                ChildState.FAILED,
                detail="not submitted: an earlier submission in the same operation failed",
            )
        if len(plans) > 1:
            self._best_effort(self._store.mark_unwound, handle)

        something_to_keep = bool(orphans) or ambiguous or any(True for _ in accepted) or unrecorded is not None
        if something_to_keep:
            self._best_effort(self._store.recompute_state, handle)
        else:
            # Nothing reached the provider: do not retain the prompts of a submission that never happened.
            self._best_effort(self._store.delete_record, handle)

        tail = ""
        if orphans:
            tail += " Accepted provider batches left behind: " + "; ".join(o.describe() for o in orphans) + "."
        if notes:
            tail += " " + " ".join(notes)

        if ambiguous and len(plans) > 1:
            logger.error(
                "Abandoned multi-batch submission left an unresolved intent under %s (provider %s, model %s)",
                handle,
                provider_name,
                failed_plan.model,
            )
            message = (
                f"Batch submission outcome is UNKNOWN for provider '{provider_name}' model "
                f"'{failed_plan.model}': {cause}. The provider may or may not hold a billable batch for it. "
                "The submission was NOT retried and will not be. The multi-batch operation was abandoned and "
                "no handle is returned. The unresolved attempt is retained for operators and reconciled "
                "against the provider's batch list; if it turns out to exist it is cancelled where the "
                f"provider supports that, otherwise reported as orphaned. Do not resubmit blindly.{tail}"
            )
            return SubmissionOutcomeUnknown(message, handle=None, orphans=orphans, retained_handle=handle)
        if ambiguous:
            message = (
                f"Batch submission outcome is UNKNOWN for provider '{provider_name}' model "
                f"'{failed_plan.model}': {cause}. The provider may or may not hold a billable batch for it. "
                f"The submission was NOT retried and will not be. The intent is retained under handle "
                f"'{handle}'; resuming or inspecting that handle reconciles it against the provider's batch "
                f"list. Do not resubmit.{tail}"
            )
            return SubmissionOutcomeUnknown(message, handle=handle, orphans=orphans)

        message = (
            f"Batch submission failed for provider '{provider_name}' model '{failed_plan.model}': {cause}. "
            f"No handle was returned.{tail}"
        )
        return BatchSubmissionError(message, orphans=orphans)

    def _unwind(
        self,
        handle: str,
        index: int,
        provider: ModelProvider,
        batch_id: str,
        *,
        cause: str,
    ) -> _Unwind:
        """Cancel an accepted batch we are abandoning; record it ``orphaned`` if we cannot.

        Never issues anything but ``cancel_batch``; nothing is deleted at the provider.
        """

        name = provider_name_of(provider)
        failure: str | None = None
        try:
            outcome = provider.cancel_batch(batch_id)
        except Exception as exc:
            failure = f"cancellation failed ({exc}); the batch may still be running and billed"
        else:
            if outcome.state is CancelState.CANCELLED:
                self._best_effort(
                    self._store.record_provider_batch,
                    handle,
                    index,
                    batch_id,
                    status=ChildState.CANCELLED,
                    detail=f"cancelled during unwind: {cause}",
                )
                return _Unwind(note=f"Cancelled {name} batch '{batch_id}'.")
            if outcome.state is CancelState.CANCELLING:
                self._best_effort(
                    self._store.record_provider_batch,
                    handle,
                    index,
                    batch_id,
                    status=ChildState.CANCELLED,
                    detail=f"cancellation requested during unwind: {cause}",
                )
                return _Unwind(note=f"Cancellation was requested for {name} batch '{batch_id}' and is in progress.")
            if outcome.state is CancelState.ALREADY_TERMINAL:
                failure = "it had already finished at the provider; its work is billed and results are retained there"
            else:
                failure = f"{name} has no cancel mechanism; the batch continues to run and will still be billed" + (
                    f" ({outcome.detail})" if outcome.detail else ""
                )

        wrote = self._best_effort(
            self._store.record_provider_batch,
            handle,
            index,
            batch_id,
            status=ChildState.ORPHANED,
            detail=f"orphaned: {failure} ({cause})",
        )
        if wrote is None:
            logger.error("Could not record orphaned %s batch %s under %s", name, batch_id, handle)
        logger.error("Orphaned %s batch %s (handle %s): %s", name, batch_id, handle, failure)
        return _Unwind(orphan=OrphanedBatch(provider=name, batch_id=batch_id, detail=failure))

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile_pending(self, *, min_age_seconds: float = DEFAULT_RECONCILE_MIN_AGE_SECONDS) -> list[ReconcileResult]:
        """Reconcile every record with an unresolved (``submitting``) child older than the cutoff.

        Never creates or retries a submission. Never raises for a provider problem; a child
        that cannot be decided stays unresolved and is reported as ``skipped``.
        """

        listings: dict[str, BatchListing | None | UnsupportedBatchOperation] = {}
        results: list[ReconcileResult] = []
        for record in self._store.unresolved_intents(min_age_seconds=min_age_seconds):
            results.extend(self._reconcile_record(record, min_age_seconds, listings, None))
        return results

    def reconcile_handle(
        self, handle: str, *, min_age_seconds: float = 0.0, call_timeout: CallTimeout | None = None
    ) -> list[ReconcileResult]:
        """Reconcile one handle's unresolved intent (the resume-time entry point).

        ``call_timeout`` bounds the provider's batch-list call (see :meth:`submit`).
        """

        record = self._store.get(handle)
        if record is None:
            return []
        return self._reconcile_record(record, min_age_seconds, {}, call_timeout)

    def open_for_resume(self, handle: str, *, call_timeout: CallTimeout | None = None) -> HandleLookup:
        """Resolve a handle for resumption, reconciling unresolved intent **before** returning.

        The wait/resume path must call this first and poll only what it returns: a child
        still ``submitting`` has no provider id to poll. Reconciliation problems are logged,
        never raised; a child that could not be decided is simply still unresolved in the
        returned record.
        """

        lookup = self._store.lookup(handle)
        if lookup.status is not LookupStatus.FOUND or lookup.record is None:
            return lookup
        refreshed = False
        if any(child.status is ChildState.SUBMITTING for child in lookup.record.children):
            try:
                self.reconcile_handle(handle, call_timeout=call_timeout)
            except Exception:
                logger.exception("Reconciliation on resume of %s failed", handle)
            refreshed = True
        # The rolled-up record state is derived data. Heal it whenever it disagrees with the
        # children (a record written by an older build, or one left stale by a crash), whether
        # or not any child was still submitting, so the caller never reads a stale state.
        try:
            recomputed = self._store.recompute_state(handle)
        except BatchStoreError:
            logger.exception("Could not refresh the state of %s on resume", handle)
            recomputed = None
        if refreshed or (recomputed is not None and recomputed is not lookup.record.state):
            return self._store.lookup(handle)
        return lookup

    def _reconcile_record(
        self,
        record: BatchRecord,
        min_age_seconds: float,
        listings: dict[str, BatchListing | None | UnsupportedBatchOperation],
        call_timeout: CallTimeout | None,
    ) -> list[ReconcileResult]:
        with _IN_FLIGHT_LOCK:
            if record.handle in _IN_FLIGHT:
                return []  # this process is still submitting it; the saga resolves it itself
        cutoff = self._clock() - min_age_seconds
        results = []
        for child in record.children:
            if child.status is not ChildState.SUBMITTING or child.intent_window_start > cutoff:
                continue
            # The provider's batch list can take as long as its ceiling. Read it *before* taking the
            # lock so a reconciler stuck in a slow list cannot make every other resume wait on it
            # outside its own budget; the lock only has to cover the check-and-write that follows.
            self._ensure_listing(child.provider, listings, call_timeout)
            with _RECONCILE_LOCK:
                # Re-read under the lock: another reconciler may have resolved it since the caller looked.
                fresh = self._store.get(record.handle)
                current = next((c for c in fresh.children if c.index == child.index), None) if fresh else None
                if fresh is None or current is None or current.status is not ChildState.SUBMITTING:
                    continue
                try:
                    results.append(self._reconcile_child(fresh, current, listings, call_timeout))
                except ChildStateChanged:
                    results.append(
                        ReconcileResult(
                            record.handle,
                            child.index,
                            child.provider,
                            ReconcileAction.SKIPPED,
                            detail="resolved concurrently by another reconciler",
                        )
                    )
                except Exception as exc:
                    logger.exception("Reconciling %s child %d failed", record.handle, child.index)
                    results.append(
                        ReconcileResult(
                            record.handle, child.index, child.provider, ReconcileAction.SKIPPED, detail=f"error: {exc}"
                        )
                    )
        self._best_effort(self._store.recompute_state, record.handle)
        return results

    def _reconcile_child(
        self,
        record: BatchRecord,
        child: BatchChild,
        listings: dict[str, BatchListing | None | UnsupportedBatchOperation],
        call_timeout: CallTimeout | None = None,
    ) -> ReconcileResult:
        def result(action: ReconcileAction, detail: str, batch_id: str | None = None) -> ReconcileResult:
            return ReconcileResult(record.handle, child.index, child.provider, action, batch_id, detail)

        provider = self._resolve_provider(child.provider)
        if provider is None:
            detail = f"provider '{child.provider}' is not configured, so the intent cannot be checked yet"
            logger.warning("Batch %s intent unresolved: %s", record.handle, detail)
            return result(ReconcileAction.SKIPPED, detail)

        self._ensure_listing(child.provider, listings, call_timeout)  # a no-op once read before the lock
        listing = listings[child.provider]

        if isinstance(listing, UnsupportedBatchOperation):
            return self._mark_indeterminate(
                record,
                child,
                f"provider '{child.provider}' offers no way to enumerate batches, so the submission cannot be "
                "matched; it was not retried",
            )
        if listing is None:
            return result(ReconcileAction.SKIPPED, "the provider's batch list could not be read; will retry")

        lo = child.intent_window_start - self._slack
        hi = (
            child.intent_window_end
            if child.intent_window_end is not None
            else child.intent_window_start + self._submit_window
        ) + self._slack

        if listing.may_be_truncated:
            known = [entry.created_at for entry in listing.entries if entry.created_at is not None]
            if not known or min(known) > lo:
                return self._mark_indeterminate(
                    record,
                    child,
                    "the provider's batch list may be truncated and does not reach back to the submission time; "
                    "not enough to conclude whether a batch exists",
                )

        matches: list[BatchListEntry] = []
        unverifiable: list[BatchListEntry] = []
        lookalikes: list[BatchListEntry] = []
        for entry in listing.entries:
            verdict = self._classify(child, entry, lo, hi)
            if verdict == "match":
                matches.append(entry)
            elif verdict == "unverifiable" and self._store.find_claim(child.provider, entry.batch_id) is None:
                unverifiable.append(entry)
            elif verdict == "model_differs" and self._store.find_claim(child.provider, entry.batch_id) is None:
                lookalikes.append(entry)

        if unverifiable:
            return self._mark_indeterminate(
                record,
                child,
                f"{len(unverifiable)} listed batch(es) lack model, endpoint, size or time metadata and cannot be "
                "ruled out",
            )
        if not matches and lookalikes:
            # Same endpoint, size and creation time inside the window, only the model spelling
            # differs. The provider may report the model differently from the slug submitted, so
            # this could be our own batch: a close-out is irreversible and must not guess.
            return self._mark_indeterminate(
                record,
                child,
                f"{len(lookalikes)} listed batch(es) match the "
                "endpoint, size and time window but report a different model spelling than the one submitted; "
                "they may be this submission, so it cannot be closed out",
                log_detail=f"candidate provider batch ids: {', '.join(e.batch_id for e in lookalikes)}",
            )
        if not matches:
            if self._clock() < hi + self._close_out_grace:
                return result(
                    ReconcileAction.SKIPPED,
                    "no matching batch is listed yet, but the submission window closed too recently to rule one "
                    "out (the provider's list can lag); will retry",
                )
            detail = "reconciliation found no provider batch corresponding to this submission; it did not take effect"
            off_window = [
                entry
                for entry in listing.entries
                if self._matches_except_time(child, entry, lo, hi)
                and self._store.find_claim(child.provider, entry.batch_id) is None
            ]
            if off_window:
                # Not proof of identity, so not a reason to withhold the close-out (the window is the design's
                # rule), but a clock skew between this host and the provider would look exactly like this, so
                # the outcome must not read as certain, and the operator gets the candidate ids.
                detail += (
                    f". Note: {len(off_window)} unclaimed listed batch(es) matched this submission's model, "
                    "endpoint and size but were created outside its time window; if this host's clock is skewed "
                    "against the provider's, one of them may be this submission"
                )
                logger.warning(
                    "Batch %s child %s closed out as not taken effect, but unclaimed %s batch(es) match it apart "
                    "from creation time (possible clock skew): %s",
                    record.handle,
                    child.index,
                    child.provider,
                    ", ".join(e.batch_id for e in off_window),
                )
            self._store.set_child_status(
                record.handle,
                child.index,
                ChildState.FAILED,
                detail=detail,
                expect_status=ChildState.SUBMITTING,
            )
            return result(ReconcileAction.CLOSED_OUT, "no matching provider batch; intent closed out as failed")
        if len(matches) > 1:
            return self._mark_indeterminate(
                record,
                child,
                f"{len(matches)} listed batches match the intent; refusing to guess",
                log_detail=f"candidate provider batch ids: {', '.join(m.batch_id for m in matches)}",
            )

        entry = matches[0]
        claim = self._store.find_claim(child.provider, entry.batch_id)
        if claim == (child.handle, child.index):
            return result(ReconcileAction.SKIPPED, f"batch {entry.batch_id} was already adopted for this child")
        if claim is not None:
            return self._mark_indeterminate(
                record,
                child,
                "the only matching listed batch is already claimed by another batch operation",
                log_detail=f"provider batch {entry.batch_id} is claimed by record {claim[0]}",
            )
        competitor = self._competing_intent(child, lo, hi)
        if competitor is not None:
            return self._mark_indeterminate(
                record,
                child,
                "another unresolved submission matches the same model, endpoint and size, so the "
                "listed batch cannot be attributed to this one",
                log_detail=f"provider batch {entry.batch_id}; competing record {competitor[0]}",
            )

        status = ChildState.from_batch_state(entry.state) if entry.state is not None else ChildState.VALIDATING
        try:
            self._store.record_provider_batch(
                record.handle,
                child.index,
                entry.batch_id,
                status=status,
                detail="adopted by reconciliation from the provider's batch list",
                expect_status=ChildState.SUBMITTING,
            )
        except ProviderBatchAlreadyClaimed:
            return self._mark_indeterminate(
                record,
                child,
                "the matching listed batch was claimed by another batch operation meanwhile",
                log_detail=f"provider batch {entry.batch_id}",
            )
        logger.info("Adopted %s batch %s under handle %s", child.provider, entry.batch_id, record.handle)

        if record.unwound:
            # The operation was abandoned (a sibling failed); this batch is billed work nobody wants.
            unwound = self._unwind(
                record.handle,
                child.index,
                provider,
                entry.batch_id,
                cause="adopted after its operation was abandoned",
            )
            detail = unwound.orphan.describe() if unwound.orphan else unwound.note
            return result(ReconcileAction.ADOPTED, f"adopted, then unwound: {detail}", entry.batch_id)
        return result(ReconcileAction.ADOPTED, "exactly one matching unclaimed batch", entry.batch_id)

    def _ensure_listing(
        self,
        provider_name: str,
        listings: dict[str, BatchListing | None | UnsupportedBatchOperation],
        call_timeout: CallTimeout | None,
    ) -> None:
        """Read ``provider_name``'s batch list into ``listings`` once. Never raises, never holds a lock.

        A provider that is not configured is left out (the caller reports the child unresolved).
        """

        if provider_name in listings:
            return
        provider = self._resolve_provider(provider_name)
        if provider is None:
            return
        try:
            listings[provider_name] = self._list_bounded(provider, call_timeout)
        except UnsupportedBatchOperation as exc:
            listings[provider_name] = exc
        except Exception as exc:
            logger.warning("Could not list %s batches for reconciliation: %s", provider_name, exc)
            listings[provider_name] = None

    def _list_bounded(self, provider: ModelProvider, call_timeout: CallTimeout | None) -> BatchListing:
        """``provider.list_batches()`` under its ceiling; an unanswered call is an unreadable list (retried later)."""

        timeout = self._timeout_for_next_call(call_timeout)
        if timeout is None:
            return provider.list_batches()
        try:
            listing: BatchListing = call_within(provider.list_batches, timeout)
        except CallTimedOut as exc:
            raise TimeoutError(f"the provider's batch list did not answer within {timeout:.0f}s") from exc
        return listing

    def _classify(self, child: BatchChild, entry: BatchListEntry, lo: float, hi: float) -> str:
        """``match``, ``no_match`` (a field verifiably differs), ``model_differs`` or ``unverifiable``.

        ``model_differs``: every other field verifiably matches and only the model spelling
        does not (see the close-out guard in :meth:`_reconcile_child`).
        """

        model_differs = entry.model is not None and entry.model.strip().lower() != child.submitted_model.strip().lower()
        if entry.endpoint is not None and entry.endpoint != child.endpoint:
            return "no_match"
        if entry.total is not None and entry.total != child.request_total:
            return "no_match"
        if entry.created_at is not None and not (lo <= entry.created_at <= hi):
            return "no_match"
        if model_differs:
            fully_known = entry.endpoint is not None and entry.total is not None and entry.created_at is not None
            return "model_differs" if fully_known else "no_match"
        if entry.model is None or entry.endpoint is None or entry.total is None or entry.created_at is None:
            return "unverifiable"
        return "match"

    @staticmethod
    def _matches_except_time(child: BatchChild, entry: BatchListEntry, lo: float, hi: float) -> bool:
        """True when every known field matches ``child`` and only the creation time is outside the window."""

        if entry.created_at is None or lo <= entry.created_at <= hi:
            return False
        if entry.model is None or entry.endpoint is None or entry.total is None:
            return False
        return (
            entry.model.strip().lower() == child.submitted_model.strip().lower()
            and entry.endpoint == child.endpoint
            and entry.total == child.request_total
        )

    def _competing_intent(self, child: BatchChild, lo: float, hi: float) -> tuple[str, int] | None:
        """Another unresolved child that would match the same listed batches, if any."""

        for record in self._store.list_records():
            for other in record.children:
                if (record.handle, other.index) == (child.handle, child.index):
                    continue
                if not other.status.is_unresolved:
                    continue
                if (
                    other.provider == child.provider
                    and other.submitted_model.strip().lower() == child.submitted_model.strip().lower()
                    and other.endpoint == child.endpoint
                    and other.request_total == child.request_total
                ):
                    other_lo = other.intent_window_start - self._slack
                    other_hi = (
                        other.intent_window_end
                        if other.intent_window_end is not None
                        else other.intent_window_start + self._submit_window
                    ) + self._slack
                    if other_lo <= hi and lo <= other_hi:
                        return (record.handle, other.index)
        return None

    def _mark_indeterminate(
        self, record: BatchRecord, child: BatchChild, reason: str, *, log_detail: str | None = None
    ) -> ReconcileResult:
        """Mark ``child`` indeterminate. ``reason`` is stored and reaches callers, so it names no identifier;
        ``log_detail`` carries the provider batch ids and other handles an operator needs, server-side only."""

        self._store.set_child_status(
            record.handle, child.index, ChildState.INDETERMINATE, detail=reason, expect_status=ChildState.SUBMITTING
        )
        logger.warning(
            "Batch %s (provider %s) is INDETERMINATE and needs operator attention: %s%s",
            record.handle,
            child.provider,
            reason,
            f" [{log_detail}]" if log_detail else "",
        )
        return ReconcileResult(record.handle, child.index, child.provider, ReconcileAction.INDETERMINATE, detail=reason)

    # ------------------------------------------------------------------

    @staticmethod
    def _best_effort(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a store write that must not mask the failure being handled.

        Returns the call's result, or ``None`` if it raised (a ``None`` result of a
        void call is indistinguishable, which every caller here tolerates).
        """

        try:
            result = fn(*args, **kwargs)
        except Exception:
            logger.exception("Batch store write failed while handling a submission failure")
            return None
        return True if result is None else result


# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------


def reconcile_at_startup(
    *,
    store: BatchStore | None = None,
    provider_resolver: ProviderResolver | None = None,
    min_age_seconds: float = DEFAULT_RECONCILE_MIN_AGE_SECONDS,
    clock: Callable[[], float] = time.time,
) -> list[ReconcileResult]:
    """Reconcile unresolved batch intent when the server starts. Never raises.

    Adopts as soon as the server is up (no age guard): a restart seconds after a crash must
    still find the batch. Only an intent whose submission is running in this process is left
    alone, and a "nothing matches" verdict is withheld until the submission window has long
    closed (see :data:`DEFAULT_CLOSE_OUT_GRACE_SECONDS`).

    Does nothing (and creates nothing on disk) when no batch store exists yet. Logs a
    WARNING naming handle and provider for each record that needs operator attention.
    """

    try:
        if store is None:
            if not os.path.exists(default_batch_store_path()):
                return []
            store = get_batch_store()
        coordinator = BatchCoordinator(store, provider_resolver=provider_resolver, clock=clock)
        results = coordinator.reconcile_pending(min_age_seconds=min_age_seconds)
        for record in store.needs_attention():
            for child in record.children:
                if child.status in (ChildState.INDETERMINATE, ChildState.ORPHANED):
                    logger.warning(
                        "Batch %s needs operator attention: provider %s batch %s is %s (%s)",
                        record.handle,
                        child.provider,
                        child.provider_batch_id or "<unknown>",
                        child.status.value,
                        child.detail or "no detail",
                    )
        return results
    except BatchStoreUnavailable:
        logger.warning("Batch store could not be opened at startup; batch reconciliation skipped", exc_info=True)
    except Exception:
        logger.exception("Batch reconciliation at startup failed")
    return []
