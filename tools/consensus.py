"""
Consensus tool - concurrent multi-model consensus

A single step-1 call records the calling agent's analysis, consults every requested model
concurrently (each blinded to the others, with a per-model timeout and isolated errors, and
optionally as provider batches), and returns all verdicts for the calling agent to synthesize.
Later steps are accepted for compatibility but consult no model.

Key features:
- All models consulted concurrently inside step 1, responses returned in the order listed
- Context-aware file embedding
- Support for stance-based analysis (for/against/neutral)
- The calling agent, not the tool, synthesizes the final recommendation
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from mcp.types import TextContent

from config import TEMPERATURE_ANALYTICAL
from providers.shared.batch import BatchRequest
from systemprompts import CONSENSUS_PROMPT
from tools.shared.base_models import ConsolidatedFindings, WorkflowRequest
from tools.shared.batch_support import (
    EXECUTION_MODE_BATCH,
    BatchControlFields,
    BatchResumableToolMixin,
    admitted_resolution,
    batch_schema_properties,
    execution_mode_schema_property,
    normalize_execution_mode,
    require_inputs_unless_resuming,
)
from utils.batch_coordinator import ChildPlan
from utils.batch_execution import DEFAULT_WAIT_POLICY, BatchExecutor, BatchRunResult, BatchRunStatus, ChildResult
from utils.batch_store import BatchStore, BatchStoreError
from utils.circuit_breaker import ProviderUnavailable
from utils.consensus_batch import (
    ConsensusPayloadError,
    ConsensusState,
    build_payload,
    custom_id_for,
    entry_index,
    interrupted_sync_responses,
    merge_responses,
    ordered_responses,
    parse_payload,
    persisted_roster,
    response_key,
    responses_for_child,
    unanswered_indexes,
)
from utils.conversation_memory import MAX_CONVERSATION_TURNS, add_turn, create_thread, get_thread
from utils.model_resolution import resolution_metadata, trim_model_identifier

from .workflow.base import WorkflowTool

logger = logging.getLogger(__name__)

# Tool-specific field descriptions for consensus workflow
CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS = {
    "step": (
        "Consensus prompt: write the exact proposal/question every model will see (use 'Evaluate…', not meta commentary). "
        "It is sent verbatim, plus any relevant_files, to every consulted model and is echoed back as "
        "agent_analysis.initial_analysis."
    ),
    "step_number": (
        "Send 1. A single step-1 call consults every model in `models` concurrently and returns the finished consensus "
        "(consensus_complete true) with all verdicts; there are no per-model steps. (With execution_mode batch the "
        "call may instead return status running and a batch_handle; resume it with only that handle.) Later steps are not needed and "
        "consult no model. To run another round, send a new step 1 call (optionally with the earlier continuation_id "
        "to stay in the same thread); the consulted models never see earlier rounds or each other's answers, and "
        "every new round consults and bills all models again."
    ),
    "total_steps": (
        "Required; send 1. All models are consulted concurrently inside step 1, so the whole consultation is one step: "
        "there are no per-model steps and no server-side synthesis step (you synthesize the returned responses "
        "yourself). The server treats step 1 as total_steps 1 whatever value you send, and the response reports "
        "total_steps 1."
    ),
    "next_step_required": (
        "Required; send false. Step 1 consults every model and finishes the consensus: the server ignores this value "
        "on step 1, and the response (status consensus_workflow_complete, or all_providers_unavailable when every "
        "model failed; consensus_complete true) carries no next_step_required field. Do not call again to gather "
        "more responses; read accumulated_responses and synthesize them yourself. The one exception is "
        "execution_mode batch: a response with status running and a batch_handle means the verdicts are not ready, "
        "so call again with only that batch_handle (no next_step_required needed) until it finishes."
    ),
    "findings": (
        "Your own independent analysis, kept for your later synthesis. It is never sent to the consulted models and "
        "is echoed back as agent_analysis.findings."
    ),
    "relevant_files": "Optional supporting files that help the consensus analysis. Must be absolute full, non-abbreviated paths.",
    "models": (
        "Models to consult (provide at least two entries). All entries are consulted concurrently in step 1 and their "
        "verdicts are returned in the order listed (a model that fails yields an error entry without cancelling the "
        "others). Each entry may include model, stance (for/against/neutral), stance_prompt, and execution_mode. "
        "Each (model, stance) pair must be unique, e.g. [{'model':'gpt5','stance':'for'}, {'model':'pro','stance':'against'}]."
    ),
    "images": "Optional absolute image paths or base64 references that add helpful visual context.",
}


class ConsensusRequest(BatchControlFields, WorkflowRequest):
    """Request model for consensus workflow steps.

    The five workflow fields below are schema-optional so a call carrying only a
    ``batch_handle`` (resume or cancel) validates; :meth:`validate_step_one_requirements`
    enforces them, and the step-1 ``models`` roster, whenever the call starts or continues work.
    """

    # Required to start or continue work (runtime-enforced; a batch_handle call needs none of them)
    step: str | None = Field(None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step"])
    step_number: int | None = Field(None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step_number"])
    total_steps: int | None = Field(None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["total_steps"])
    next_step_required: bool | None = Field(
        None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["next_step_required"]
    )

    # Investigation tracking fields
    findings: str | None = Field(None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["findings"])
    confidence: str = Field(default="exploring", exclude=True, description="Not used")

    # Consensus-specific fields (only needed in step 1)
    models: list[dict] | None = Field(None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["models"])
    relevant_files: list[str] | None = Field(
        default_factory=list,
        description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["relevant_files"],
    )

    # Optional images for visual debugging
    images: list[str] | None = Field(default=None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["images"])

    # Override inherited fields to exclude them from schema
    temperature: float | None = Field(default=None, exclude=True)
    thinking_mode: str | None = Field(default=None, exclude=True)

    # Not used in consensus workflow
    files_checked: list[str] | None = Field(default_factory=list, exclude=True)
    relevant_context: list[str] | None = Field(default_factory=list, exclude=True)
    issues_found: list[dict] | None = Field(default_factory=list, exclude=True)
    hypothesis: str | None = Field(None, exclude=True)

    @model_validator(mode="after")
    def validate_step_one_requirements(self):
        """Enforce the runtime-required inputs, then step 1's models field and unique model+stance combinations.

        A call carrying a ``batch_handle`` resumes or cancels and needs none of them.
        """
        require_inputs_unless_resuming(
            self,
            "consensus",
            {
                "step": self.step,
                "step_number": self.step_number,
                "total_steps": self.total_steps,
                "next_step_required": self.next_step_required,
                "findings": self.findings,
            },
        )
        if self.is_resuming:
            return self

        if self.step_number == 1:
            if not self.models:
                raise ValueError("Step 1 requires 'models' field to specify which models to consult")

            # Check for unique model + stance combinations
            seen_combinations = set()
            for position, model_config in enumerate(self.models):
                # An entry carries the same execution_mode field, with the same accepted values, as the request.
                normalize_execution_mode(model_config.get("execution_mode"), field=f"models[{position}].execution_mode")
                model_name = model_config.get("model", "")
                # A missing, null or empty stance is the neutral stance (it gets the neutral prompt and
                # is recorded as "neutral"), so it collides with an explicit "neutral" for the same model.
                stance = model_config.get("stance") or "neutral"
                combination = f"{model_name}:{stance}"

                if combination in seen_combinations:
                    raise ValueError(
                        f"Duplicate model + stance combination found: {model_name} with stance '{stance}'. "
                        f"Each model + stance combination must be unique."
                    )
                seen_combinations.add(combination)

        return self


class ConsensusTool(BatchResumableToolMixin, WorkflowTool):
    """
    Consensus workflow tool for concurrent multi-model consensus gathering.

    Step 1 consults every specified model concurrently (each sees only the original proposal and
    files) and returns every verdict at once; the calling agent then synthesizes them into a
    unified recommendation. The tool itself does not synthesize.
    """

    # -- batch selection (execution_mode on the request and on each models[] entry) ----------

    def entry_execution_modes(self, arguments: dict[str, Any]) -> list[str | None]:
        """Each ``models[]`` entry's effective execution mode: its own ``execution_mode``, else the request's.

        Raises a tool error naming the entry whose value is not an accepted one.
        """

        try:
            default = normalize_execution_mode(arguments.get("execution_mode"))
            modes: list[str | None] = []
            roster = arguments.get("models")
            for position, entry in enumerate(roster if isinstance(roster, list) else []):
                own = normalize_execution_mode(
                    entry.get("execution_mode") if isinstance(entry, dict) else None,
                    field=f"models[{position}].execution_mode",
                )
                modes.append(own if own is not None else default)
            return modes
        except ValueError as exc:
            self._raise_batch_error(str(exc), code="invalid_execution_mode")
            return []  # unreachable

    def wants_batch_execution(self, arguments: dict[str, Any]) -> bool:
        """True when the request, or any single entry, selects batch: no entry may run synchronously by accident."""

        if super().wants_batch_execution(arguments):
            return True
        return EXECUTION_MODE_BATCH in self.entry_execution_modes(arguments)

    def requested_batch_models(self, arguments: dict[str, Any]) -> list[str]:
        """The models of the entries that run as a batch (every entry when the request itself selects batch)."""

        roster = arguments.get("models")
        models = roster if isinstance(roster, list) else []
        modes = self.entry_execution_modes(arguments)
        return [
            str(entry.get("model"))
            for entry, mode in zip(models, modes, strict=False)
            if mode == EXECUTION_MODE_BATCH and isinstance(entry, dict) and entry.get("model")
        ]

    def __init__(self):
        super().__init__()
        self.initial_prompt: str | None = None
        self.original_proposal: str | None = None  # Store the original proposal separately
        self.models_to_consult: list[dict] = []
        self.accumulated_responses: list[dict] = []
        self._current_arguments: dict[str, Any] = {}

    # -- batch execution (execution_mode="batch", on the request or on individual models[] entries) ----------
    #
    # Step 1 divides into one provider batch per distinct batch-selected model (entries that share a model
    # and differ in stance share its batch); entries that did not select batch run concurrently as they
    # always did. Everything a resumed run needs lives in the durable batch record (utils/consensus_batch.py
    # documents the payload), and the final response is rebuilt from that record by whichever instance
    # resumes the handle: nothing below the settle step reads this instance's attributes.

    # Serialises the read-merge-write of a record's payload within this process. Merging is idempotent
    # (a recorded response is final), so a second process racing on the same store can at worst repeat it.
    _payload_lock = threading.Lock()

    # Synchronous consultations of a mixed run that are still in flight, by batch handle. They run as a
    # task of their own (see execute_batch_start) so that a call cancelled by its host cannot lose a paid
    # verdict, and a resume in this process tells "still running" from "the server died while it ran".
    # Class-level: a resume may arrive on a different tool instance of the same process.
    _live_synchronous: dict[str, asyncio.Task] = {}

    # The least a call allows itself to fetch finished results once its own wait is spent, so a batch that
    # finished on the last poll is still collected; the total over all children is one shared budget.
    _FETCH_BUDGET_FLOOR_SECONDS = 5.0

    _BATCH_PASSTHROUGH_STATUSES = frozenset(
        {
            BatchRunStatus.UNKNOWN_HANDLE,
            BatchRunStatus.HANDLE_EXPIRED,
            BatchRunStatus.INDETERMINATE,
            BatchRunStatus.ORPHANED,
            BatchRunStatus.CANCELLED,
        }
    )

    async def execute_batch_start(self, arguments: dict[str, Any], providers: Any) -> list[Any]:
        """Run step 1 with its batch-selected consultations as provider batches (one per distinct model).

        Admission already checked that every batched model has a provider that can batch it and that
        nothing non-text is attached. Batches are submitted first (so a refused submission happens
        before any synchronous consultation is paid for), the synchronous entries then run
        concurrently, and every verdict they produce is written to the durable record before a
        non-terminal handle can be returned.
        """

        started = time.monotonic()
        request = ConsensusRequest(**arguments)
        if request.step_number != 1:
            self._raise_batch_error(
                "execution_mode='batch' applies to consensus step 1, the step that consults the models. Later "
                "steps make no model calls; omit execution_mode, or resume a batch with its batch_handle.",
                code="batch_applies_to_step_one",
            )
        roster: list[dict[str, Any]] = list(request.models or [])
        modes = self.entry_execution_modes(arguments)
        batched = [i for i, mode in enumerate(modes) if mode == EXECUTION_MODE_BATCH]
        synchronous = [i for i in range(len(roster)) if modes[i] != EXECUTION_MODE_BATCH]
        proposal = request.step or ""

        try:
            plans = self._plan_batches(request, roster, batched, providers, proposal)
        except Exception as exc:
            logger.exception("Could not prepare the consensus batch requests")
            self._raise_batch_error(
                f"Could not prepare the batched consultations ({exc}). Nothing was submitted and nothing was run.",
                code="batch_preparation_failed",
            )
            return []  # unreachable

        continuation_id = request.continuation_id
        if not continuation_id:
            clean_args = {k: v for k, v in arguments.items() if k != "_context"}
            continuation_id = create_thread(self.get_name(), clean_args)
        payload = build_payload(proposal, request.findings, persisted_roster(roster, modes), [])

        budget = DEFAULT_WAIT_POLICY.budget(request.wait_seconds)
        run = await self.run_batch_submission(
            plans,
            continuation_id=continuation_id,
            payload=payload,
            # A mixed run waits for its batches only after the synchronous consultations had their time.
            wait_seconds=0 if synchronous else request.wait_seconds,
            fetch=False,
        )
        handle = run.handle
        # The handle exists from here on and the batches are billed: say so where an operator can find it
        # even if this call is cancelled before it can return the handle to the caller.
        logger.info(
            "consensus: %d batch(es) submitted under batch_handle %s; resume with only that handle",
            len(plans),
            handle,
        )
        self._record_batch_turn(continuation_id, request)
        executor = self.get_batch_executor()

        if synchronous:
            self.store_initial_issue(proposal)
            task = self._start_synchronous_entries(executor, handle, [roster[i] for i in synchronous], request)
            # Wait for the synchronous models within the caller's wait budget, never longer, so the handle
            # reaches the caller before a host deadline. Neither the timeout nor a cancellation of this call
            # stops the task: it finishes, and writes its verdicts to the record, on its own.
            await asyncio.wait({task}, timeout=max(0.0, budget - (time.monotonic() - started)))
            if task.done():
                responses, failure = task.result()
                if failure is not None:
                    # Never hand back a handle while the finished verdicts exist only in memory.
                    self._raise_batch_error(
                        f"The synchronous consultations finished but their verdicts could not be saved ({failure}), "
                        "so no handle is returned. The batches are still running at the provider. The verdicts "
                        "follow so they are not lost:\n" + json.dumps(responses, indent=2, ensure_ascii=False),
                        handle=handle,
                        code="persist_failed",
                    )
            remaining = max(0.0, budget - (time.monotonic() - started))
            run = await self.resume_batch(executor, handle, remaining)
        else:
            fetch_budget = max(self._FETCH_BUDGET_FLOOR_SECONDS, budget - (time.monotonic() - started))
            run = await asyncio.to_thread(self._settle_run, executor, run, fetch_budget)
        return self.render_batch_result(run)

    def _start_synchronous_entries(
        self, executor: BatchExecutor, handle: str, entries: list[dict[str, Any]], request: ConsensusRequest
    ) -> asyncio.Task:
        """Consult the synchronous entries of a mixed run as a task that records its own verdicts.

        The task's result is ``(responses, failure)`` where ``failure`` is the error that kept the verdicts
        from being saved (``None`` when they were). It is registered as live under ``handle`` until it ends.
        """

        async def consult_and_record() -> tuple[list[dict], Exception | None]:
            responses = await self._consult_models_concurrently(entries, request)
            try:
                await asyncio.to_thread(self._persist_responses, executor.store, handle, responses)
            except (BatchStoreError, ConsensusPayloadError) as exc:
                logger.error(
                    "consensus: the synchronous verdicts for batch_handle %s could not be saved (%s); they are: %s",
                    handle,
                    exc,
                    json.dumps(responses, ensure_ascii=False),
                )
                return responses, exc
            return responses, None

        task = asyncio.get_running_loop().create_task(consult_and_record())
        self._live_synchronous[handle] = task

        def finished(done: asyncio.Task) -> None:
            if self._live_synchronous.get(handle) is done:
                del self._live_synchronous[handle]
            if not done.cancelled() and done.exception() is not None:
                logger.error(
                    "consensus: the synchronous consultations for batch_handle %s failed",
                    handle,
                    exc_info=done.exception(),
                )

        task.add_done_callback(finished)
        return task

    def _plan_batches(
        self,
        request: ConsensusRequest,
        roster: list[dict[str, Any]],
        batched: list[int],
        providers: Any,
        proposal: str,
    ) -> list[ChildPlan]:
        """One :class:`ChildPlan` per distinct batched model, with one request per (model, stance) entry.

        Mirrors what ``_consult_model`` sends a model synchronously (the original proposal plus any
        context files, the stance-enhanced system prompt, a temperature validated for the model), so a
        batched consultation asks exactly what a synchronous one would.
        """

        from utils.model_context import ModelContext

        groups: dict[str, list[int]] = {}
        for index in batched:
            groups.setdefault(trim_model_identifier(str(roster[index].get("model", ""))), []).append(index)

        plans: list[ChildPlan] = []
        for model, indexes in groups.items():
            provider = providers[model]
            # Each model of the run carries the limits of the slug it resolved to at admission.
            resolution = admitted_resolution(providers, model)
            model_context = ModelContext(
                model_name=model,
                provider=provider,
                capabilities=resolution.capabilities if resolution is not None else None,
            )
            prompt = proposal
            if request.relevant_files:
                file_content, _ = self._prepare_file_content_for_prompt(
                    request.relevant_files, None, "Context files", model_context=model_context
                )
                if file_content:
                    prompt = f"{prompt}\n\n=== CONTEXT FILES ===\n{file_content}\n=== END CONTEXT ==="
            temperature, warnings = self.validate_and_correct_temperature(self.get_default_temperature(), model_context)
            self.log_batch_temperature_warnings(warnings)
            requests = [
                BatchRequest(
                    custom_id=custom_id_for(index),
                    prompt=prompt,
                    system_prompt=self._get_stance_enhanced_prompt(
                        roster[index].get("stance", "neutral"), roster[index].get("stance_prompt")
                    ),
                    temperature=temperature,
                    extra={"model": model},
                )
                for index in indexes
            ]
            for batch_request in requests:
                self.check_batch_input_fits(
                    model_context,
                    resolution.slug if resolution is not None else model,
                    [batch_request.prompt, batch_request.system_prompt or ""],
                    resolution=resolution,
                )
            plans.append(ChildPlan(provider=provider, model=model, requests=requests, resolution=resolution))
        return plans

    def _record_batch_turn(self, continuation_id: str, request: ConsensusRequest) -> None:
        """Record the same assistant turn a synchronous step 1 leaves in the thread (best effort)."""

        try:
            add_turn(
                thread_id=continuation_id,
                role="assistant",
                content=self._extract_clean_workflow_content_for_history({"step_number": 1, "total_steps": 1}),
                tool_name=self.get_name(),
                files=request.relevant_files or None,
                model_metadata={"initial_request": request.step},
            )
        except Exception:
            logger.warning("Could not record the consensus turn for %s", continuation_id, exc_info=True)

    def _persist_responses(self, store: BatchStore, handle: str, responses: list[dict]) -> ConsensusState:
        """Merge ``responses`` into the record's payload and write it; return the stored state.

        Re-reads the record under the lock, so concurrent resumes never overwrite each other's
        responses, and writes only when something new was added.
        """

        with self._payload_lock:
            record = store.get(handle)
            if record is None:
                raise BatchStoreError(f"No live record under handle {handle}")
            state = parse_payload(record.payload)
            merged = merge_responses(state.accumulated_responses, responses)
            if merged != state.accumulated_responses:
                store.update_payload(
                    handle, build_payload(state.original_proposal, state.findings, state.models, merged)
                )
                state.accumulated_responses = merged
            return state

    async def resume_batch(self, executor: BatchExecutor, handle: str, wait_seconds: Any) -> BatchRunResult:
        """Poll only the outstanding provider batches, then settle: record what finished, report the rest."""

        tool_name = self.get_name()
        started = time.monotonic()
        budget = executor.policy.budget(wait_seconds)
        run = await asyncio.to_thread(
            executor.resume, handle, tool_name=tool_name, wait_seconds=wait_seconds, fetch=False
        )
        fetch_budget = max(self._FETCH_BUDGET_FLOOR_SECONDS, budget - (time.monotonic() - started))
        return await asyncio.to_thread(self._settle_run, executor, run, fetch_budget)

    def _settle_run(self, executor: BatchExecutor, run: BatchRunResult, fetch_budget: float) -> BatchRunResult:
        """Consume every finished child's results into the record, then say whether the run is complete.

        A child whose entries are all recorded is not touched again (no fetch, no poll), which is what
        makes repeated resumption free and lets a restart between completion and synthesis lose nothing.
        Fetching the finished children shares one ``fetch_budget`` (seconds) between them: the first is
        always attempted, a later one only while budget remains, and a child left unfetched keeps the run
        non-terminal so the next resume picks it up. Returns ``COMPLETED`` once every entry has a response
        (a failed one counts: its error is its response) and ``RUNNING`` while any is outstanding, including
        a synchronous consultation still running in this process; anything else (an unknown or expired
        handle, a cancelled or abandoned run, a submission needing an operator) passes through unchanged.
        """

        if run.status in self._BATCH_PASSTHROUGH_STATUSES or run.record is None or run.record.unwound:
            return run
        # Decide whether a synchronous task is still running BEFORE reading the record: a task that ended
        # by now has already written its verdicts, so the record read below then includes them.
        live = self._live_synchronous.get(run.handle)
        synchronous_running = live is not None and not live.done()
        record = executor.store.get(run.handle) or run.record
        try:
            state = parse_payload(record.payload)
        except ConsensusPayloadError as exc:
            self._raise_batch_error(
                f"This batch record cannot be resumed as a consensus run: {exc}.",
                handle=run.handle,
                code="payload_unreadable",
            )
            return run  # unreachable

        answered = {response_key(r.get("model", ""), r.get("stance")) for r in state.accumulated_responses}
        fresh: list[dict] = []
        pending: dict[int, ChildResult] = {}
        deadline = time.monotonic() + fetch_budget
        fetched = False
        for child in record.children:
            entries = [i for cid in child.custom_ids if (i := entry_index(cid)) is not None and i < len(state.models)]
            if all(
                response_key(state.models[i].get("model", ""), state.models[i].get("stance")) in answered
                for i in entries
            ):
                continue
            if not child.status.is_terminal:
                continue
            if fetched and time.monotonic() >= deadline:
                pending[child.index] = ChildResult(
                    child, pending_reason="its results were not fetched in this call; resume to fetch them"
                )
                continue
            fetched = True
            result = executor.fetch_child(child, budget_seconds=max(0.0, deadline - time.monotonic()))
            responses = responses_for_child(state, result)
            if responses is None:
                pending[child.index] = result
            else:
                fresh.extend(responses)

        if fresh:
            try:
                state = self._persist_responses(executor.store, run.handle, fresh)
            except (BatchStoreError, ConsensusPayloadError) as exc:
                self._raise_batch_error(
                    f"Finished consultations could not be saved to the batch record ({exc}). The handle is unchanged; "
                    "resume it again.",
                    handle=run.handle,
                    code="persist_failed",
                )
        current = executor.store.get(run.handle) or record
        outstanding = set(unanswered_indexes(state))
        batched = set(state.batched_indexes())
        waiting_on = (outstanding & batched) | (outstanding - batched if synchronous_running else set())
        if waiting_on:
            children = tuple(pending.get(item.child.index, item) for item in run.children)
            return dataclasses.replace(run, status=BatchRunStatus.RUNNING, record=current, children=children)
        return dataclasses.replace(run, status=BatchRunStatus.COMPLETED, record=current)

    def batch_status_extras(self, result: BatchRunResult) -> dict[str, Any]:
        """How far a consensus run has got, from its durable record (and the verdicts a cancelled run already holds)."""

        try:
            state = parse_payload(result.record.payload if result.record is not None else None)
        except ConsensusPayloadError:
            return {}
        outstanding = set(unanswered_indexes(state))
        label = lambda i: f"{state.models[i].get('model', '')}:{state.models[i].get('stance', 'neutral')}"  # noqa: E731
        extras: dict[str, Any] = {
            "consultations": {
                "total": len(state.models),
                "completed": len(state.models) - len(outstanding),
                "completed_models": [label(i) for i in range(len(state.models)) if i not in outstanding],
                "pending_models": [label(i) for i in sorted(outstanding)],
            }
        }
        if result.status is BatchRunStatus.CANCELLED and state.accumulated_responses:
            # A cancelled run still holds verdicts that were paid for (synchronous consultations, and batches
            # that finished before the cancel): hand them over rather than leaving them stranded in the record.
            extras["completed_responses"] = ordered_responses(state.models, state.accumulated_responses)
        if result.status is BatchRunStatus.CANCELLED:
            live = self._live_synchronous.get(result.handle)
            running_sync = sorted(outstanding - set(state.batched_indexes())) if live and not live.done() else []
            if running_sync:
                # Cancelling stops polling the batches; it does not stop a synchronous consultation already in
                # flight. It runs to completion, is billed in full, and its verdict is stored in the record.
                names = ", ".join(label(i) for i in running_sync)
                extras["synchronous_consultations_continue"] = [label(i) for i in running_sync]
                extras["message_note"] = (
                    f"The synchronous consultation(s) already running ({names}) are NOT stopped by this cancel: "
                    "they will run to completion and be billed in full; they are listed as pending above only "
                    "because they have not finished yet."
                )
        return extras

    def render_completed_batch(self, result: BatchRunResult) -> list[TextContent]:
        """The consensus response for a finished batched run, rebuilt from the durable record alone.

        Same fields as a synchronous run's step-1 response. It reads nothing from this instance
        (``original_proposal`` is ``None`` on one that never handled step 1): the proposal, findings,
        model-and-stance list and every verdict come from the persisted payload.
        """

        record = result.record
        try:
            state = parse_payload(record.payload if record is not None else None)
        except ConsensusPayloadError as exc:
            self._raise_batch_error(
                f"This batch record cannot be rendered as a consensus run: {exc}.",
                handle=result.handle,
                code="payload_unreadable",
            )
            return []  # unreachable
        assert record is not None

        responses = ordered_responses(
            state.models, merge_responses(state.accumulated_responses, interrupted_sync_responses(state))
        )
        response_data = self._step_one_response_data(
            step=state.original_proposal,
            findings=state.findings,
            proposal=state.original_proposal,
            responses=responses,
        )
        response_data = self.customize_workflow_response(
            response_data, SimpleNamespace(step_number=1), models=state.models, responses=responses
        )
        self._add_workflow_metadata(response_data, {})

        failed = [r for r in responses if r.get("status") != "success" and r.get("error") != "provider_unavailable"]
        if failed and len(failed) < len(responses):
            names = ", ".join(f"{r.get('model')}:{r.get('stance')}" for r in failed)
            response_data["next_steps"] += (
                f"\n\nNOTE: {len(failed)} consultation(s) failed ({names}) and have an error status in "
                "accumulated_responses. Synthesize from the available results only."
            )
        response_data["metadata"].update({"batch_handle": result.handle, "batch_execution": True})
        if record.continuation_id:
            offer = self._build_continuation_offer(record.continuation_id)
            if offer:
                response_data["continuation_offer"] = offer
        return [TextContent(type="text", text=json.dumps(response_data, indent=2, ensure_ascii=False))]

    def get_name(self) -> str:
        return "consensus"

    def get_annotations(self) -> dict[str, Any] | None:
        """Consensus is not read-only: it bills provider calls and, in batch mode, writes a durable record.

        A plain run only calls provider APIs and returns text, but ``execution_mode='batch'`` submits a billed
        provider batch and persists its state to the batch store (``.unison/batches.db``) so the run can be
        resumed with a ``batch_handle``. The workflow base class's read-only hint would hide that from clients.
        """

        return {"readOnlyHint": False}

    def get_description(self) -> str:
        return (
            "Builds multi-model consensus. A single call (step_number 1, total_steps 1, next_step_required false) "
            "consults every model in `models` concurrently, each with its own stance (for/against/neutral) and "
            "without seeing the others or their answers, then returns all verdicts together in accumulated_responses, "
            "in the order the models were listed. You then synthesize them into a recommendation; the tool does not "
            "synthesize for you. With execution_mode='batch' the batch-selected models run as provider batches and "
            "the call may return a batch_handle; resume it with batch_handle (and wait_seconds) until the verdicts "
            "are complete. Use for complex decisions, architectural choices, feature proposals, and technology "
            "evaluations."
        )

    def get_system_prompt(self) -> str:
        # For the CLI agent's initial analysis, use a neutral version of the consensus prompt
        return CONSENSUS_PROMPT.replace(
            "{stance_prompt}",
            """BALANCED PERSPECTIVE

Provide objective analysis considering both positive and negative aspects. However, if there is overwhelming evidence
that the proposal clearly leans toward being exceptionally good or particularly problematic, you MUST accurately
reflect this reality. Being "balanced" means being truthful about the weight of evidence, not artificially creating
50/50 splits when the reality is 90/10.

Your analysis should:
- Present all significant pros and cons discovered
- Weight them according to actual impact and likelihood
- If evidence strongly favors one conclusion, clearly state this
- Provide proportional coverage based on the strength of arguments
- Help the questioner see the true balance of considerations

Remember: Artificial balance that misrepresents reality is not helpful. True balance means accurate representation
of the evidence, even when it strongly points in one direction.""",
        )

    def get_default_temperature(self) -> float:
        return TEMPERATURE_ANALYTICAL

    def get_model_category(self) -> ToolModelCategory:
        """Consensus workflow requires extended reasoning"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_workflow_request_model(self):
        """Return the consensus workflow-specific request model."""
        return ConsensusRequest

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema for consensus workflow."""
        from .workflow.schema_builders import WorkflowSchemaBuilder

        # Consensus tool-specific field definitions
        consensus_field_overrides = {
            # Override standard workflow fields that need consensus-specific descriptions
            "step": {
                "type": "string",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step"],
            },
            "step_number": {
                "type": "integer",
                "minimum": 1,
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step_number"],
            },
            "total_steps": {
                "type": "integer",
                "minimum": 1,
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["total_steps"],
            },
            "next_step_required": {
                "type": "boolean",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["next_step_required"],
            },
            "findings": {
                "type": "string",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["findings"],
            },
            "relevant_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["relevant_files"],
            },
            # consensus-specific fields (not in base workflow)
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "stance": {"type": "string", "enum": ["for", "against", "neutral"], "default": "neutral"},
                        "stance_prompt": {"type": "string"},
                        "execution_mode": execution_mode_schema_property(),
                    },
                    "required": ["model"],
                },
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["models"],
                "minItems": 2,
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["images"],
            },
            # Resume / cancel a batch through this same tool (no separate job tool)
            **batch_schema_properties(),
        }

        # Provide guidance on available models similar to single-model tools
        model_description = (
            "When the user names a model, you MUST use that exact value or report the "
            "provider error—never swap in another option. Use the `listmodels` tool for the full roster."
        )

        summaries, total, restricted = self._get_ranked_model_summaries()
        remainder = max(0, total - len(summaries))
        if summaries:
            label = "Allowed models" if restricted else "Top models"
            top_line = "; ".join(summaries)
            if remainder > 0:
                top_line = f"{label}: {top_line}; +{remainder} more via `listmodels`."
            else:
                top_line = f"{label}: {top_line}."
            model_description = f"{model_description} {top_line}"
        else:
            model_description = (
                f"{model_description} No models detected—configure provider credentials or use the `listmodels` tool "
                "to inspect availability."
            )

        restriction_note = self._get_restriction_note()
        if restriction_note and (remainder > 0 or not summaries):
            model_description = f"{model_description} {restriction_note}."

        existing_models_desc = consensus_field_overrides["models"]["description"]
        consensus_field_overrides["models"]["description"] = f"{existing_models_desc} {model_description}"

        # Define excluded fields for consensus workflow
        excluded_workflow_fields = [
            # Runtime-enforced rather than schema-required, so a batch_handle-only call validates.
            # Dropping them here also drops them from the schema's ``required`` list; their
            # consensus-specific definitions are re-added through ``consensus_field_overrides``.
            "step",
            "step_number",
            "total_steps",
            "next_step_required",
            "findings",
            "files_checked",  # Not used in consensus workflow
            "relevant_context",  # Not used in consensus workflow
            "issues_found",  # Not used in consensus workflow
            "hypothesis",  # Not used in consensus workflow
            "confidence",  # Not used in consensus workflow
        ]

        excluded_common_fields = [
            "model",  # Consensus uses 'models' field instead
            "temperature",  # Not used in consensus workflow
            "thinking_mode",  # Not used in consensus workflow
        ]

        requires_model = self.requires_model()
        model_field_schema = self.get_model_field_schema() if requires_model else None
        auto_mode = self.is_effective_auto_mode() if requires_model else False

        return WorkflowSchemaBuilder.build_schema(
            tool_specific_fields=consensus_field_overrides,
            model_field_schema=model_field_schema,
            auto_mode=auto_mode,
            tool_name=self.get_name(),
            excluded_workflow_fields=excluded_workflow_fields,
            excluded_common_fields=excluded_common_fields,
            require_model=requires_model,
        )

    def get_required_actions(
        self, step_number: int, confidence: str, findings: str, total_steps: int, request=None
    ) -> list[str]:  # noqa: ARG002
        """Required actions for the calling agent, identical for every step.

        Retained only to satisfy the abstract contract: consensus never calls it, and returns its
        guidance through ``handle_work_completion``. Every model is consulted concurrently in step 1,
        so there is never a "wait for the next model" phase; the only action left is synthesis.
        The step and confidence arguments are kept for compatibility and not used.
        """
        return [
            "All models have been consulted concurrently; their verdicts are in accumulated_responses",
            "Synthesize all perspectives into a comprehensive recommendation",
            "Identify key points of agreement and disagreement",
            "Provide clear, actionable guidance based on the consensus",
        ]

    def should_call_expert_analysis(self, consolidated_findings, request=None) -> bool:
        """Consensus runs its own concurrent model consultations; there is no expert-analysis phase."""
        return False

    def prepare_expert_analysis_context(self, consolidated_findings) -> str:
        """Not used in consensus workflow."""
        return ""

    def requires_expert_analysis(self) -> bool:
        """Consensus workflow handles its own model consultations."""
        return False

    def requires_model(self) -> bool:
        """
        Consensus tool doesn't require model resolution at the MCP boundary.

        Uses it's own set of models

        Returns:
            bool: False
        """
        return False

    # Hook method overrides for consensus-specific behavior

    def prepare_step_data(self, request) -> dict:
        """Prepare consensus-specific step data."""
        step_data = {
            "step": request.step,
            "step_number": request.step_number,
            "findings": request.findings,
            "files_checked": [],  # Not used
            "relevant_files": request.relevant_files or [],
            "relevant_context": [],  # Not used
            "issues_found": [],  # Not used
            "confidence": "exploring",  # Not used, kept for compatibility
            "hypothesis": None,  # Not used
            "images": request.images or [],  # Now used for visual context
        }
        return step_data

    async def handle_work_completion(self, response_data: dict, request, arguments: dict) -> dict:  # noqa: ARG002
        """Handle consensus workflow completion - no expert analysis, just final synthesis."""
        response_data["consensus_complete"] = True
        response_data["status"] = "consensus_workflow_complete"

        # Prepare final synthesis data
        response_data["complete_consensus"] = {
            "initial_prompt": self.original_proposal if self.original_proposal else self.initial_prompt,
            "models_consulted": [m["model"] + ":" + m.get("stance", "neutral") for m in self.accumulated_responses],
            "total_responses": len(self.accumulated_responses),
            "consensus_confidence": "high",  # Consensus complete
        }

        response_data["next_steps"] = (
            "CONSENSUS GATHERING IS COMPLETE. You MUST now synthesize all perspectives and present:\n"
            "1. Key points of AGREEMENT across models\n"
            "2. Key points of DISAGREEMENT and why they differ\n"
            "3. Your final consolidated recommendation\n"
            "4. Specific, actionable next steps for implementation\n"
            "5. Critical risks or concerns that must be addressed"
        )

        return response_data

    def handle_work_continuation(self, response_data: dict, request) -> dict:  # noqa: ARG002
        """Handle a step-2+ call with ``next_step_required`` true.

        Step 1 returns before the base workflow, so it never reaches this method. Later steps run the
        generic workflow and consult no model, whatever ``total_steps`` the caller sent, so the response must
        not promise another consultation: it says the models were all consulted in step 1 and points at
        synthesis or a fresh step 1.
        """
        response_data["status"] = "ready_for_synthesis"
        response_data["next_steps"] = (
            "No further models will be consulted: every model is consulted concurrently in step 1. "
            "Synthesize the responses you already have, or start a new step 1 (optionally with the same "
            "continuation_id) to run another round."
        )

        return response_data

    async def execute_workflow(self, arguments: dict[str, Any]) -> list:
        """Consult all models concurrently on step 1; later steps run the generic workflow without model calls."""

        # Store arguments
        self._current_arguments = arguments

        # Validate request
        request = self.get_workflow_request_model()(**arguments)

        # Resolve existing continuation_id or create a new one on first step
        continuation_id = request.continuation_id

        if request.step_number == 1:
            if not continuation_id:
                clean_args = {k: v for k, v in arguments.items() if k != "_context"}
                continuation_id = create_thread(self.get_name(), clean_args)
                request.continuation_id = continuation_id
                arguments["continuation_id"] = continuation_id
                self.work_history = []
                self.consolidated_findings = ConsolidatedFindings()

            # Store the original proposal from step 1 - this is what all models should see
            self.store_initial_issue(request.step)
            self.initial_request = request.step
            self.models_to_consult = request.models or []

            # Dispatch all model consultations concurrently on step 1
            self.accumulated_responses = await self._consult_models_concurrently(self.models_to_consult, request)

            # All API calls complete on step 1; total_steps=1 for the dispatch step
            request.total_steps = 1

        # Step 1: concurrent dispatch complete — return all responses at once
        if request.step_number == 1:
            step_data = self.prepare_step_data(request)
            self.work_history.append(step_data)
            self._update_consolidated_findings(step_data)

            response_data = self._step_one_response_data(
                step=request.step,
                findings=request.findings,
                proposal=self.original_proposal if self.original_proposal else self.initial_prompt,
                responses=self.accumulated_responses,
            )

            # Add continuation information and workflow customization
            response_data = self.customize_workflow_response(response_data, request)
            self._add_workflow_metadata(response_data, arguments)

            if continuation_id:
                self.store_conversation_turn(continuation_id, response_data, request)
                continuation_offer = self._build_continuation_offer(continuation_id)
                if continuation_offer:
                    response_data["continuation_offer"] = continuation_offer

            return [TextContent(type="text", text=json.dumps(response_data, indent=2, ensure_ascii=False))]

        # Steps 2+: accepted for compatibility, no model calls
        return await super().execute_workflow(arguments)

    @staticmethod
    def _step_one_response_data(
        *, step: str | None, findings: str | None, proposal: str | None, responses: list[dict]
    ) -> dict:
        """The step-1 response body for a finished set of consultations.

        Pure: it reads only its arguments, never tool-instance state, so a batched run resumed on a
        fresh instance (where ``original_proposal`` is ``None``) builds the very same response a
        synchronous run does from the persisted record alone.
        """

        successful = [r for r in responses if r.get("status") == "success"]
        failed = [r for r in responses if r.get("status") != "success"]
        unavailable = [r for r in responses if r.get("error") == "provider_unavailable"]
        skipped_names = [r["model"] for r in unavailable]

        if not successful:
            response_data = {
                "status": "all_providers_unavailable",
                "step_number": 1,
                "total_steps": 1,
                "consensus_complete": True,
                "accumulated_responses": responses,
                "error": (
                    "No providers are currently available. "
                    f"All {len(responses)} provider(s) failed: "
                    f"{', '.join(r['model'] for r in responses)}. Try again later."
                ),
                "next_steps": (
                    "All providers are currently unavailable. "
                    "No consensus can be formed. Inform the user and suggest retrying later."
                ),
            }
        else:
            response_data = {
                "status": "consensus_workflow_complete",
                "step_number": 1,
                "total_steps": 1,
                "consensus_complete": True,
                "accumulated_responses": responses,
                "agent_analysis": {
                    "initial_analysis": step,
                    "findings": findings,
                },
                "complete_consensus": {
                    "initial_prompt": proposal,
                    "models_consulted": [f"{m['model']}:{m.get('stance', 'neutral')}" for m in successful],
                    "total_responses": len(successful),
                    "consensus_confidence": "high" if not failed else "reduced",
                },
            }
            if skipped_names:
                response_data["complete_consensus"]["skipped_providers"] = skipped_names
                response_data["complete_consensus"]["skipped_reason"] = "provider_unavailable"
            synthesis_steps = (
                "CONSENSUS GATHERING IS COMPLETE. All models were consulted concurrently. "
                "Synthesize all perspectives and present:\n"
                "1. Key points of AGREEMENT across models\n"
                "2. Key points of DISAGREEMENT and why they differ\n"
                "3. Your final consolidated recommendation\n"
                "4. Specific, actionable next steps for implementation\n"
                "5. Critical risks or concerns that must be addressed"
            )
            if skipped_names:
                synthesis_steps += (
                    f"\n\nNOTE: {len(skipped_names)} provider(s) were unavailable "
                    f"({', '.join(skipped_names)}) and skipped. "
                    "Synthesize from available results only."
                )
            response_data["next_steps"] = synthesis_steps
        return response_data

    def _build_continuation_offer(self, continuation_id: str) -> dict[str, Any] | None:
        """Create a continuation offer without exposing prior model responses."""
        try:
            from tools.models import ContinuationOffer

            thread = get_thread(continuation_id)
            if thread and thread.turns:
                remaining_turns = max(0, MAX_CONVERSATION_TURNS - len(thread.turns))
            else:
                remaining_turns = MAX_CONVERSATION_TURNS - 1

            # Provide a neutral note specific to consensus workflow
            note = (
                f"Consensus workflow can continue for {remaining_turns} more exchanges."
                if remaining_turns > 0
                else "Consensus workflow continuation limit reached."
            )

            continuation_offer = ContinuationOffer(
                continuation_id=continuation_id,
                note=note,
                remaining_turns=remaining_turns,
            )
            return continuation_offer.model_dump()
        except Exception:
            return None

    async def _consult_models_concurrently(self, model_configs: list[dict], request) -> list[dict]:
        """Dispatch all model consultations concurrently via asyncio.gather.

        Each model call is wrapped in a per-model timeout (120s, hard-coded below).
        Failures are isolated — one model's error does not cancel the others.
        """
        per_model_timeout = 120.0

        async def _consult_with_timeout(config: dict) -> dict:
            try:
                return await asyncio.wait_for(
                    self._consult_model(config, request),
                    timeout=per_model_timeout,
                )
            except asyncio.TimeoutError:
                model_name = config.get("model", "unknown")
                logger.warning("Model %s timed out after %ss", model_name, per_model_timeout)
                return {
                    "model": model_name,
                    "stance": config.get("stance", "neutral"),
                    "status": "error",
                    "error": f"timeout after {per_model_timeout}s",
                }

        tasks = [_consult_with_timeout(config) for config in model_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                model_name = model_configs[i].get("model", "unknown")
                logger.exception("Unexpected error consulting model %s", model_name, exc_info=result)
                responses.append(
                    {
                        "model": model_name,
                        "stance": model_configs[i].get("stance", "neutral"),
                        "status": "error",
                        "error": str(result),
                    }
                )
            else:
                responses.append(result)
        return responses

    async def _consult_model(self, model_config: dict, request) -> dict:
        """Consult a single model and return its response."""
        try:
            # Import and create ModelContext once at the beginning
            from utils.model_context import ModelContext

            # Get the provider for this model
            model_name = model_config["model"]
            provider = self.get_model_provider(model_name)

            # Create model context once and reuse for both file processing and temperature validation
            model_context = ModelContext(model_name=model_name)

            # Prepare the prompt with any relevant files
            # Use continuation_id=None for blinded consensus - each model should only see
            # original prompt + files, not conversation history or other model responses
            # CRITICAL: the prompt is the original step-1 proposal (never `findings`, the agent's own notes,
            # or another model's answer); consulted models are blinded to each other.
            prompt = self.original_proposal if self.original_proposal else self.initial_prompt
            if request.relevant_files:
                file_content, _ = self._prepare_file_content_for_prompt(
                    request.relevant_files,
                    None,  # Use None instead of request.continuation_id for blinded consensus
                    "Context files",
                    model_context=model_context,
                )
                if file_content:
                    prompt = f"{prompt}\n\n=== CONTEXT FILES ===\n{file_content}\n=== END CONTEXT ==="

            # Get stance-specific system prompt
            stance = model_config.get("stance", "neutral")
            stance_prompt = model_config.get("stance_prompt")
            system_prompt = self._get_stance_enhanced_prompt(stance, stance_prompt)

            # Validate temperature against model constraints (respects supports_temperature)
            validated_temperature, temp_warnings = self.validate_and_correct_temperature(
                self.get_default_temperature(), model_context
            )

            # Log any temperature corrections
            for warning in temp_warnings:
                logger.warning(warning)

            # Call the model with validated temperature (async path for concurrent dispatch)
            from utils.observability import instrumented_async_generate

            response = await instrumented_async_generate(
                provider,
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=validated_temperature,
                thinking_mode="medium",
                images=request.images if request.images else None,
            )

            return {
                "model": model_name,
                "stance": stance,
                "status": "success",
                "verdict": response.content,
                "metadata": {
                    "provider": provider.get_provider_type().value,
                    "model_name": model_name,
                    **resolution_metadata(model_name, getattr(response, "model_name", None)),
                },
            }

        except ProviderUnavailable as e:
            logger.warning(
                "Provider unavailable for model %s: %s",
                model_config.get("model", "unknown"),
                e,
            )
            return {
                "model": model_config.get("model", "unknown"),
                "stance": model_config.get("stance", "neutral"),
                "status": "error",
                "error": "provider_unavailable",
                "message": str(e),
            }
        except Exception as e:
            logger.exception("Error consulting model %s", model_config)
            return {
                "model": model_config.get("model", "unknown"),
                "stance": model_config.get("stance", "neutral"),
                "status": "error",
                "error": str(e),
            }

    def _get_stance_enhanced_prompt(self, stance: str, custom_stance_prompt: str | None = None) -> str:
        """Get the system prompt with stance injection."""
        base_prompt = CONSENSUS_PROMPT

        if custom_stance_prompt:
            return base_prompt.replace("{stance_prompt}", custom_stance_prompt)

        stance_prompts = {
            "for": """SUPPORTIVE PERSPECTIVE WITH INTEGRITY

You are tasked with advocating FOR this proposal, but with CRITICAL GUARDRAILS:

MANDATORY ETHICAL CONSTRAINTS:
- This is NOT a debate for entertainment. You MUST act in good faith and in the best interest of the questioner
- You MUST think deeply about whether supporting this idea is safe, sound, and passes essential requirements
- You MUST be direct and unequivocal in saying "this is a bad idea" when it truly is
- There must be at least ONE COMPELLING reason to be optimistic, otherwise DO NOT support it

WHEN TO REFUSE SUPPORT (MUST OVERRIDE STANCE):
- If the idea is fundamentally harmful to users, project, or stakeholders
- If implementation would violate security, privacy, or ethical standards
- If the proposal is technically infeasible within realistic constraints
- If costs/risks dramatically outweigh any potential benefits

YOUR SUPPORTIVE ANALYSIS SHOULD:
- Identify genuine strengths and opportunities
- Propose solutions to overcome legitimate challenges
- Highlight synergies with existing systems
- Suggest optimizations that enhance value
- Present realistic implementation pathways

Remember: Being "for" means finding the BEST possible version of the idea IF it has merit, not blindly supporting bad ideas.""",
            "against": """CRITICAL PERSPECTIVE WITH RESPONSIBILITY

You are tasked with critiquing this proposal, but with ESSENTIAL BOUNDARIES:

MANDATORY FAIRNESS CONSTRAINTS:
- You MUST NOT oppose genuinely excellent, common-sense ideas just to be contrarian
- You MUST acknowledge when a proposal is fundamentally sound and well-conceived
- You CANNOT give harmful advice or recommend against beneficial changes
- If the idea is outstanding, say so clearly while offering constructive refinements

WHEN TO MODERATE CRITICISM (MUST OVERRIDE STANCE):
- If the proposal addresses critical user needs effectively
- If it follows established best practices with good reason
- If benefits clearly and substantially outweigh risks
- If it's the obvious right solution to the problem

YOUR CRITICAL ANALYSIS SHOULD:
- Identify legitimate risks and failure modes
- Point out overlooked complexities
- Suggest more efficient alternatives
- Highlight potential negative consequences
- Question assumptions that may be flawed

Remember: Being "against" means rigorous scrutiny to ensure quality, not undermining good ideas that deserve support.""",
            "neutral": """BALANCED PERSPECTIVE

Provide objective analysis considering both positive and negative aspects. However, if there is overwhelming evidence
that the proposal clearly leans toward being exceptionally good or particularly problematic, you MUST accurately
reflect this reality. Being "balanced" means being truthful about the weight of evidence, not artificially creating
50/50 splits when the reality is 90/10.

Your analysis should:
- Present all significant pros and cons discovered
- Weight them according to actual impact and likelihood
- If evidence strongly favors one conclusion, clearly state this
- Provide proportional coverage based on the strength of arguments
- Help the questioner see the true balance of considerations

Remember: Artificial balance that misrepresents reality is not helpful. True balance means accurate representation
of the evidence, even when it strongly points in one direction.""",
        }

        stance_prompt = stance_prompts.get(stance, stance_prompts["neutral"])
        return base_prompt.replace("{stance_prompt}", stance_prompt)

    def customize_workflow_response(
        self,
        response_data: dict,
        request,
        *,
        models: list[dict] | None = None,
        responses: list[dict] | None = None,
    ) -> dict:
        """Customize response for consensus workflow.

        ``models`` and ``responses`` default to the instance's own; a batched run rebuilt from its
        persisted record passes them explicitly so nothing here reads state a fresh instance lacks.
        """
        accumulated = self.accumulated_responses if responses is None else responses
        # Store model responses in the response for tracking
        if accumulated:
            response_data["accumulated_responses"] = accumulated

        # consensus_workflow_status always repeats the response's own status, on every step, so the two can
        # never disagree: step 1 reports consensus_workflow_complete or all_providers_unavailable, a step 2+
        # call with next_step_required true reports ready_for_synthesis (handle_work_continuation), and one
        # with next_step_required false reports consensus_workflow_complete (handle_work_completion). The
        # fallback only covers a hand-built response that carries no status at all.
        fallback = "consensus_workflow_complete" if request.step_number == 1 else "ready_for_synthesis"
        response_data["consensus_workflow_status"] = response_data.get("status", fallback)

        # Customize metadata for consensus workflow
        self._customize_consensus_metadata(response_data, models=models)

        return response_data

    def _customize_consensus_metadata(self, response_data: dict, *, models: list[dict] | None = None) -> None:
        """
        Customize metadata for consensus workflow to accurately reflect multi-model nature.

        The default workflow metadata shows the model running Agent's analysis steps,
        but consensus is a multi-model tool that consults different models. We need
        to provide accurate metadata that reflects this.
        """
        if "metadata" not in response_data:
            response_data["metadata"] = {}

        metadata = response_data["metadata"]

        # Always preserve tool_name
        metadata["tool_name"] = self.get_name()

        roster = self.models_to_consult if models is None else models
        models_consulted = [f"{m['model']}:{m.get('stance', 'neutral')}" for m in roster] if roster else []

        # Every consensus response reports a finished consultation: all models are consulted in step 1, and a
        # later step consults none, so there is no in-progress shape (and nothing here depends on total_steps).
        metadata.update(
            {
                "workflow_type": "multi_model_consensus",
                "models_consulted": models_consulted,
                "consensus_complete": True,
                "total_models": len(roster) if roster else 0,
            }
        )

        # Remove the misleading single model metadata that shows Agent's execution model
        # instead of the models being consulted
        metadata.pop("model_used", None)
        metadata.pop("provider_used", None)

    def _add_workflow_metadata(self, response_data: dict, arguments: dict[str, Any]) -> None:
        """
        Override workflow metadata addition for consensus tool.

        The consensus tool doesn't use single model metadata because it's a multi-model
        workflow. Instead, we provide consensus-specific metadata that accurately
        reflects the models being consulted.
        """
        # Initialize metadata if not present
        if "metadata" not in response_data:
            response_data["metadata"] = {}

        # Add basic tool metadata
        response_data["metadata"]["tool_name"] = self.get_name()

        # The consensus-specific metadata is already added by _customize_consensus_metadata
        # which is called from customize_workflow_response. We don't add the standard
        # single-model metadata (model_used, provider_used) because it's misleading
        # for a multi-model consensus workflow.

        logger.debug(
            f"[CONSENSUS_METADATA] {self.get_name()}: Using consensus-specific metadata instead of single-model metadata"
        )

    def store_initial_issue(self, step_description: str):
        """Store initial prompt for model consultations."""
        self.original_proposal = step_description
        self.initial_prompt = step_description  # Keep for backward compatibility

    # Required abstract methods from BaseTool
    def get_request_model(self):
        """Return the consensus workflow-specific request model."""
        return ConsensusRequest

    async def prepare_prompt(self, request) -> str:  # noqa: ARG002
        """Not used - workflow tools use execute_workflow()."""
        return ""  # Workflow tools use execute_workflow() directly
