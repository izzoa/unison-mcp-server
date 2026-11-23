"""State and result mapping for a batched consensus run (add-batch-model-support, section 8).

A batched consensus run outlives the call that started it, so everything its synthesis needs lives
in the durable batch record (``utils/batch_store.py``) and is rebuilt from it by whichever tool
instance resumes the handle. This module is the pure part of that: what the record's payload holds,
how provider results become per-model responses, and how those responses are merged and ordered.
It reads no tool state and performs no I/O.

The payload is exactly four fields and nothing else (the spec bounds it; in particular it holds no
``work_history`` or other step history):

``original_proposal``
    The step-1 text every model was asked about.
``findings``
    The calling agent's own step-1 analysis.
``models``
    The model-and-stance list, in roster order. Each entry is ``{"model", "stance", "execution_mode"}``
    (plus ``stance_prompt`` when one was given); ``execution_mode`` is ``"batch"`` for an entry that
    runs as a provider batch and absent for one that ran synchronously.
``accumulated_responses``
    One response per model that has finished, in the shape a synchronous run produces
    (``model``, ``stance``, ``status``, ``verdict`` or ``error``, ``metadata``). A response is written
    here the moment it is known, so a restart never loses or re-bills a finished consultation.

A consultation is identified by its roster position: its request ``custom_id`` is ``consensus-<i>``.
Two entries that share a model (and so share one provider batch) differ by stance, and a
(model, stance) pair is unique within a roster, so ``(model, stance)`` also keys a response.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from providers.shared.batch_resolution import note_from_record
from utils.batch_execution import ChildResult
from utils.model_resolution import resolution_metadata

__all__ = [
    "EXECUTION_BATCH",
    "PAYLOAD_KEYS",
    "ConsensusPayloadError",
    "ConsensusState",
    "build_payload",
    "custom_id_for",
    "entry_index",
    "error_response",
    "interrupted_sync_responses",
    "merge_responses",
    "ordered_responses",
    "parse_payload",
    "persisted_roster",
    "response_key",
    "responses_for_child",
    "unanswered_indexes",
]

#: The only keys a batched consensus payload carries.
PAYLOAD_KEYS: tuple[str, ...] = ("original_proposal", "findings", "models", "accumulated_responses")

#: ``execution_mode`` value stored on a roster entry that runs as a provider batch.
EXECUTION_BATCH = "batch"

_CUSTOM_ID_PREFIX = "consensus-"


class ConsensusPayloadError(ValueError):
    """A record's payload is not a batched consensus payload (missing, mistyped or corrupt)."""


class ConsensusState:
    """A parsed payload: the four persisted fields, ready to rebuild the synthesis from."""

    __slots__ = ("accumulated_responses", "findings", "models", "original_proposal")

    def __init__(
        self,
        original_proposal: str,
        findings: str | None,
        models: list[dict[str, Any]],
        accumulated_responses: list[dict[str, Any]],
    ) -> None:
        self.original_proposal = original_proposal
        self.findings = findings
        self.models = models
        self.accumulated_responses = accumulated_responses

    def entry(self, index: int) -> dict[str, Any]:
        return self.models[index]

    def batched_indexes(self) -> list[int]:
        return [i for i, entry in enumerate(self.models) if entry.get("execution_mode") == EXECUTION_BATCH]


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------


def custom_id_for(index: int) -> str:
    """The request ``custom_id`` of the roster entry at ``index``."""

    return f"{_CUSTOM_ID_PREFIX}{index}"


def entry_index(custom_id: str) -> int | None:
    """The roster position a ``custom_id`` names, or ``None`` when it is not a consensus id."""

    if not custom_id.startswith(_CUSTOM_ID_PREFIX):
        return None
    tail = custom_id[len(_CUSTOM_ID_PREFIX) :]
    return int(tail) if tail.isdigit() else None


def response_key(model: Any, stance: Any) -> tuple[str, str]:
    """The key a response is stored under: its model and stance exactly as the roster named them."""

    return (str(model), str(stance if stance else "neutral"))


def _entry_key(entry: Mapping[str, Any]) -> tuple[str, str]:
    return response_key(entry.get("model", ""), entry.get("stance"))


# ----------------------------------------------------------------------
# The payload
# ----------------------------------------------------------------------


def persisted_roster(models: Sequence[Mapping[str, Any]], modes: Sequence[str | None]) -> list[dict[str, Any]]:
    """The model-and-stance list as persisted: each entry's model, stance and effective execution mode.

    ``modes`` is the effective mode per entry (its own ``execution_mode``, else the request's). A
    ``stance_prompt`` is kept because it changes what the model was asked; nothing else is.
    """

    roster: list[dict[str, Any]] = []
    for position, entry in enumerate(models):
        item: dict[str, Any] = {
            "model": str(entry.get("model", "")),
            "stance": str(entry.get("stance") or "neutral"),
        }
        if entry.get("stance_prompt"):
            item["stance_prompt"] = str(entry["stance_prompt"])
        mode = modes[position] if position < len(modes) else None
        if mode == EXECUTION_BATCH:
            item["execution_mode"] = EXECUTION_BATCH
        roster.append(item)
    return roster


def build_payload(
    original_proposal: str,
    findings: str | None,
    roster: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The record payload: exactly :data:`PAYLOAD_KEYS`, no step history."""

    return {
        "original_proposal": original_proposal,
        "findings": findings,
        "models": [dict(entry) for entry in roster],
        "accumulated_responses": [dict(response) for response in responses],
    }


def parse_payload(raw: Any) -> ConsensusState:
    """Read a record payload back, or raise :class:`ConsensusPayloadError` naming what is wrong."""

    if not isinstance(raw, dict):
        raise ConsensusPayloadError("the batch record carries no consensus payload")
    missing = [key for key in PAYLOAD_KEYS if key not in raw]
    if missing:
        raise ConsensusPayloadError(f"the consensus payload is missing {', '.join(missing)}")
    proposal, findings, models, responses = (raw[key] for key in PAYLOAD_KEYS)
    if not isinstance(proposal, str):
        raise ConsensusPayloadError("the consensus payload's original_proposal is not text")
    if not isinstance(models, list) or not all(isinstance(entry, dict) for entry in models):
        raise ConsensusPayloadError("the consensus payload's models list is malformed")
    if not isinstance(responses, list) or not all(isinstance(response, dict) for response in responses):
        raise ConsensusPayloadError("the consensus payload's accumulated_responses is malformed")
    return ConsensusState(proposal, findings if isinstance(findings, str) else None, list(models), list(responses))


# ----------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------


def error_response(entry: Mapping[str, Any], message: str) -> dict[str, Any]:
    """A failed consultation in the shape a synchronous run reports one (``status`` ``error``)."""

    return {
        "model": str(entry.get("model", "")),
        "stance": str(entry.get("stance") or "neutral"),
        "status": "error",
        "error": message,
    }


def _success_response(
    entry: Mapping[str, Any], verdict: str, provider: str, resolved_model: str, resolution: Any = None
) -> dict[str, Any]:
    model = str(entry.get("model", ""))
    return {
        "model": model,
        "stance": str(entry.get("stance") or "neutral"),
        "status": "success",
        "verdict": verdict,
        "metadata": {
            "provider": provider,
            "model_name": model,
            **resolution_metadata(model, resolved_model),
            # Newer batch models the alias did not pick, as recorded with the child at submission.
            **note_from_record(resolution),
            "execution_mode": EXECUTION_BATCH,
        },
    }


def responses_for_child(state: ConsensusState, result: ChildResult) -> list[dict[str, Any]] | None:
    """Per-model responses for the entries of one finished provider batch, or ``None`` if not yet usable.

    Error isolation: a failed request fails only its own entry; a batch that failed as a whole (or
    returned fewer results than were submitted) fails the entries it cannot answer and leaves the
    answered ones alone; and no failure here touches another provider batch's entries. ``None`` means
    the batch finished but its results could not be fetched right now: nothing is recorded and the
    run stays non-terminal.
    """

    if result.pending_reason:
        return None
    child = result.child
    outcomes = {outcome.custom_id: outcome for outcome in result.outcomes}
    if result.error:
        fallback = result.error
    else:
        detail = f": {child.detail}" if child.detail else ""
        fallback = (
            f"the batch ended {child.status.value} and returned no result for this consultation{detail}"
            if not result.outcomes
            else "the provider returned no result for this consultation"
        )

    responses: list[dict[str, Any]] = []
    for custom_id in child.custom_ids:
        index = entry_index(custom_id)
        if index is None or index >= len(state.models):
            continue
        entry = state.models[index]
        outcome = outcomes.get(custom_id)
        if outcome is None:
            responses.append(error_response(entry, fallback))
        elif outcome.success and outcome.response is not None and outcome.response.content:
            responses.append(
                _success_response(
                    entry, outcome.response.content, child.provider, outcome.response.model_name, child.resolution
                )
            )
        elif outcome.success:
            responses.append(error_response(entry, "the model returned an empty response"))
        else:
            message = outcome.error.message if outcome.error is not None else "the request failed"
            responses.append(error_response(entry, message))
    return responses


def merge_responses(existing: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``existing`` plus the ``new`` responses for models not yet answered; a recorded response is final.

    Repeating a merge (two resumes racing, a retry after a failed write) adds nothing twice.
    """

    merged = [dict(response) for response in existing]
    seen = {response_key(r.get("model", ""), r.get("stance")) for r in merged}
    for response in new:
        key = response_key(response.get("model", ""), response.get("stance"))
        if key not in seen:
            merged.append(dict(response))
            seen.add(key)
    return merged


def ordered_responses(
    models: Sequence[Mapping[str, Any]], responses: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """``responses`` in roster order (the order a synchronous run reports them), one per answered model."""

    by_key = {response_key(r.get("model", ""), r.get("stance")): dict(r) for r in responses}
    ordered = []
    for entry in models:
        response = by_key.get(_entry_key(entry))
        if response is not None:
            ordered.append(response)
    return ordered


def unanswered_indexes(state: ConsensusState) -> list[int]:
    """Roster positions with no recorded response yet."""

    answered = {response_key(r.get("model", ""), r.get("stance")) for r in state.accumulated_responses}
    return [i for i, entry in enumerate(state.models) if _entry_key(entry) not in answered]


def interrupted_sync_responses(state: ConsensusState) -> list[dict[str, Any]]:
    """Errors for synchronous entries that never recorded a result (the server stopped while they ran).

    Resuming never submits new work, so a synchronous consultation that did not finish before the
    process died is reported as failed rather than silently re-run at full price.
    """

    batched = set(state.batched_indexes())
    return [
        error_response(
            state.models[index],
            "this synchronous consultation has no recorded result (the server stopped while it ran, or its result "
            "could not be saved); it was not re-run, because resuming a batch never submits new work",
        )
        for index in unanswered_indexes(state)
        if index not in batched
    ]
