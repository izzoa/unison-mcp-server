"""Wire-format helpers for OpenRouter's asynchronous Batch API.

Everything here is pure (no I/O, no provider state) so the request envelope and the
response mapping can be tested and type-checked in isolation. :class:`OpenRouterProvider`
owns transport, model resolution and restrictions and calls into this module.

OpenRouter facts these helpers encode (verified against its documentation, 2026-11-21):

* ``POST {base}/batches`` takes ``{"endpoint", "model", "requests": [{"custom_id", "body"}]}``.
  The API stream-parses the envelope, so ``endpoint`` and ``model`` must be serialized
  *before* ``requests`` or it answers 400. :func:`serialize_envelope` builds the mapping in
  that order explicitly and never sorts keys.
* ``GET {base}/batches/{id}`` returns the batch object. Once ``completed`` it carries
  ``results`` inline, in arbitrary order, so outcomes are correlated on ``custom_id`` only.
* There is no cancel endpoint; that is handled by the provider, not here.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .shared import (
    BatchError,
    BatchFailed,
    BatchIncompleteResults,
    BatchListEntry,
    BatchListing,
    BatchNotFound,
    BatchNotReady,
    BatchOutcome,
    BatchProviderError,
    BatchRateLimited,
    BatchRequest,
    BatchResultsExpired,
    BatchState,
    BatchStatus,
    ModelResponse,
    ProviderType,
)
from .shared.batch_catalog import PASSTHROUGH_PARAMETERS

logger = logging.getLogger(__name__)

__all__ = [
    "BATCH_ENDPOINT",
    "BATCH_SUFFIX",
    "DEFINITE_SUBMIT_REFUSAL_STATUSES",
    "SUBMISSION_OUTCOME_UNKNOWN",
    "build_chat_body",
    "is_definite_submit_refusal",
    "parse_listing",
    "parse_results",
    "parse_status",
    "raise_for_http_error",
    "reject_invalid_temperature",
    "reject_non_text",
    "serialize_envelope",
    "strip_batch_suffix",
]

PROVIDER_NAME = "openrouter"

#: ``BatchProviderError.code`` for a submit whose outcome cannot be known: the provider may
#: hold a billable batch the caller never learned about. Never retried; reconciled instead.
SUBMISSION_OUTCOME_UNKNOWN = "submission_outcome_unknown"

#: HTTP statuses on a batch-create response that prove the batch was NOT created: the request was
#: refused for what it said (validation, auth, credits, size, rate limit), not lost on the way.
#: Every other outcome is ambiguous. That includes 408 (request timeout) and 425 (too early),
#: which proxies and gateways emit after the body may already have been forwarded, any other 4xx
#: not listed here, every 5xx and any non-error status. With no idempotency key a retry could
#: bill twice, so the intent is kept and reconciled instead.
DEFINITE_SUBMIT_REFUSAL_STATUSES: frozenset[int] = frozenset(
    {400, 401, 402, 403, 404, 405, 413, 414, 415, 422, 429, 431}
)

#: The request shape every batch entry uses; matches the synchronous chat-completions body.
BATCH_ENDPOINT = "/v1/chat/completions"

#: Catalog suffix marking a batch *pricing listing*. It is an eligibility signal, never sent.
BATCH_SUFFIX = ":batch"

# Sampling parameters the synchronous path forwards for models that accept sampling (one list, shared with the
# listing view, so a name the request builder sends is a name the view keeps).
_PASSTHROUGH_PARAMS = PASSTHROUGH_PARAMETERS

# ``BatchRequest.extra`` keys that would smuggle non-text content past the images check.
_NON_TEXT_EXTRA_KEYS = (
    "images",
    "image",
    "audio",
    "video",
    "file",
    "files",
    "attachment",
    "attachments",
    "document",
    "documents",
)

_RETRYABLE_ERROR_MARKERS = (
    "rate_limit",
    "overloaded",
    "server_error",
    "api_error",
    "timeout",
    "unavailable",
    "internal",
    "capacity",
)

_STATE_BY_NAME = {state.value: state for state in BatchState}


# ----------------------------------------------------------------------
# Slug mapping
# ----------------------------------------------------------------------


def strip_batch_suffix(identifier: str) -> str:
    """Map a caller spelling to the base slug OpenRouter's envelope expects.

    A trailing ``:batch`` (case-insensitive) is removed once; every other identifier is
    returned unchanged (whitespace trimmed). This is the single, deterministic half of the
    slug rule: ``vendor/model`` and ``vendor/model:batch`` both yield ``vendor/model``.
    Whether the model is *eligible* for batch is decided separately from its capabilities,
    never by remapping a model onto a different one.
    """

    cleaned = identifier.strip()
    if cleaned.lower().endswith(BATCH_SUFFIX):
        return cleaned[: -len(BATCH_SUFFIX)]
    return cleaned


# ----------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------


def reject_non_text(requests: Sequence[BatchRequest]) -> None:
    """Raise ``ValueError`` naming the first request that carries non-text content."""

    for request in requests:
        offending = request.has_non_text_content or any(request.extra.get(key) for key in _NON_TEXT_EXTRA_KEYS)
        if offending:
            raise ValueError(
                f"Batch request '{request.custom_id}' carries image or attachment content. "
                "OpenRouter batch execution is text-only; remove the attachments, or make the "
                "synchronous call by omitting execution_mode. Local files already read into the "
                "prompt as text are fine."
            )


def reject_invalid_temperature(requests: Sequence[BatchRequest]) -> None:
    """Raise ``ValueError`` naming the first request whose ``temperature`` is not a usable number.

    ``None`` means "the default". Anything else must be a real number: a ``bool`` (``True`` would be sent as the JSON
    literal ``true`` by a wide-range vendor), a string, or NaN (which a range clamp would turn into the upper bound)
    is refused locally rather than forwarded or silently reinterpreted. An infinite value is a number and is clamped to
    the model's range like any other out-of-range one.
    """

    for request in requests:
        value = request.temperature
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(
                f"Batch request '{request.custom_id}' has an invalid temperature {value!r}: "
                "expected a number (or None for the default)."
            )


def build_chat_body(
    request: BatchRequest,
    *,
    temperature: float | None,
    include_max_tokens: bool,
    listed_parameters: Collection[str] | None = None,
) -> dict[str, Any]:
    """Build one request body in the chat-completions shape the synchronous path sends.

    ``model`` is deliberately omitted: OpenRouter applies the batch-level model, and a
    body naming a different one is rejected.

    With no ``listed_parameters`` (no listing was consulted: a curated slug that has no listing anywhere, live or
    snapshot, or a derived entry whose listing gave no parameter list) the old rule stands: ``temperature=None``
    means the model takes no sampling parameters, in which case ``max_tokens`` and the passthrough parameters are
    omitted exactly as the synchronous path omits them. ``include_max_tokens`` is only read on this path.

    With ``listed_parameters`` (the names the model's own listing says it accepts) every optional parameter is
    decided on its own presence in that list, independently of the others: ``temperature`` is sent only when it is
    listed (and a value was given), ``max_tokens`` whenever ``max_tokens`` or ``max_completion_tokens`` is listed
    (a reasoning model that takes a token cap but no temperature keeps its cap), and each passthrough parameter
    only when listed. A parameter the listing does not name is never sent, and one it names is never dropped.
    """

    messages: list[dict[str, Any]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.prompt})

    body: dict[str, Any] = {"messages": messages}
    if listed_parameters is None:
        if temperature is not None:
            body["temperature"] = temperature
            if include_max_tokens and request.max_output_tokens:
                body["max_tokens"] = request.max_output_tokens
            for key in _PASSTHROUGH_PARAMS:
                if key in request.extra:
                    body[key] = request.extra[key]
        return body

    if temperature is not None and "temperature" in listed_parameters:
        body["temperature"] = temperature
    if request.max_output_tokens:
        if "max_tokens" in listed_parameters:
            body["max_tokens"] = request.max_output_tokens
        elif "max_completion_tokens" in listed_parameters:
            body["max_completion_tokens"] = request.max_output_tokens
    for key in _PASSTHROUGH_PARAMS:
        if key in request.extra and key in listed_parameters:
            body[key] = request.extra[key]
    return body


def serialize_envelope(model: str, entries: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    """Serialize a create-batch body with ``endpoint`` and ``model`` ahead of ``requests``.

    The mapping is built key by key in the required order (not from a sorted or merged
    dict) and ``json.dumps`` is called without ``sort_keys``, so the order does not depend
    on incidental mapping behaviour. ``entries`` are ``(custom_id, body)`` pairs.
    """

    envelope: dict[str, Any] = {}
    envelope["endpoint"] = BATCH_ENDPOINT
    envelope["model"] = model
    envelope["requests"] = [{"custom_id": custom_id, "body": dict(body)} for custom_id, body in entries]
    return json.dumps(envelope, separators=(",", ":"), ensure_ascii=False, sort_keys=False)


# ----------------------------------------------------------------------
# HTTP errors
# ----------------------------------------------------------------------


def _error_message(text: str) -> str:
    """Pull a human message out of an OpenRouter error body, falling back to the raw text."""

    try:
        payload = json.loads(text)
    except ValueError:
        return text.strip()[:500]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if payload.get("message"):
            return str(payload["message"])
    return text.strip()[:500]


def _retry_after(headers: Mapping[str, str]) -> float | None:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def is_definite_submit_refusal(status_code: int | None) -> bool:
    """True when ``status_code`` on a batch-create response proves no batch was created.

    Only the statuses in :data:`DEFINITE_SUBMIT_REFUSAL_STATUSES` qualify. ``None`` (no response
    at all) and everything else, 408 and 425 and 5xx included, may have left a billable batch.
    """

    return status_code in DEFINITE_SUBMIT_REFUSAL_STATUSES


def raise_for_http_error(status_code: int, headers: Mapping[str, str], text: str, *, action: str) -> None:
    """Raise a typed :class:`BatchProviderError` for a non-2xx batch response.

    429 and 5xx are retryable; 404, 410, 402 and the remaining 4xx are not. ``action``
    names the call ("submit", "poll", ...) for the message. For ``action="submit"`` only a
    status in :data:`DEFINITE_SUBMIT_REFUSAL_STATUSES` is a refusal; any other status
    (408, 425, other 4xx, 5xx, non-error) is typed ``submission_outcome_unknown``.
    """

    if 200 <= status_code < 300:
        return

    detail = _error_message(text)
    prefix = f"OpenRouter batch {action} failed (HTTP {status_code})"
    message = f"{prefix}: {detail}" if detail else prefix
    common: dict[str, Any] = {"provider": PROVIDER_NAME, "status_code": status_code}

    if action == "submit" and not is_definite_submit_refusal(status_code):
        # A 408/425 or a 5xx on a POST with no idempotency key (a 502/504 especially) does not say the
        # batch was not created, and a non-error non-2xx is not an answer at all.
        raise BatchProviderError(
            f"{message}. The batch may or may not have been created; reconcile before resubmitting.",
            code=SUBMISSION_OUTCOME_UNKNOWN,
            **common,
        )
    if status_code == 429:
        raise BatchRateLimited(message, retry_after=_retry_after(headers), provider=PROVIDER_NAME)
    if status_code == 404:
        raise BatchNotFound(message, **common)
    if status_code == 410:
        raise BatchResultsExpired(
            f"{prefix}: the batch's results are past OpenRouter's 30-day retention and are gone. {detail}".strip(),
            **common,
        )
    if status_code == 402:
        raise BatchProviderError(f"{message} (insufficient OpenRouter credits)", **common)
    if status_code == 413:
        raise BatchProviderError(f"{message} (too many requests or body too large; split the batch)", **common)
    if status_code >= 500:
        raise BatchProviderError(message, retryable=True, **common)
    raise BatchProviderError(message, **common)


# ----------------------------------------------------------------------
# Status and results
# ----------------------------------------------------------------------


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def parse_status(payload: Mapping[str, Any]) -> BatchStatus:
    """Map an OpenRouter batch object to a :class:`BatchStatus`.

    ``validating`` (the state of a freshly accepted ``202``) maps to a non-terminal state
    with zero completed requests: acceptance is not completion.
    """

    raw_status = payload.get("status")
    state = _STATE_BY_NAME.get(str(raw_status))
    if state is None:
        raise BatchProviderError(
            f"OpenRouter returned an unrecognised batch status {raw_status!r}",
            provider=PROVIDER_NAME,
        )

    counts = payload.get("request_counts")
    counts = counts if isinstance(counts, dict) else {}
    error = payload.get("error")
    error_message = error.get("message") if isinstance(error, dict) else None
    expires_at = payload.get("expires_at")

    return BatchStatus(
        state=state,
        provider=PROVIDER_NAME,
        total=_as_int(counts.get("total")),
        completed=_as_int(counts.get("completed")),
        failed=_as_int(counts.get("failed")),
        error_message=str(error_message) if error_message else None,
        expires_at=str(expires_at) if expires_at else None,
    )


def _epoch_seconds(value: Any) -> float | None:
    """Best-effort conversion of a provider timestamp (epoch s/ms or ISO-8601) to epoch seconds."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if number > 1e11 else number
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return float(text) / 1000.0 if float(text) > 1e11 else float(text)
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def parse_listing(payload: Any, *, limit: int) -> BatchListing:
    """Map ``GET /batches`` to a :class:`BatchListing` (metadata only; ``results`` is ignored).

    OpenRouter documents the page as ``{"object": "list", "data": [...], "first_id", "last_id",
    "has_more"}``, newest first, each item a metadata-only batch object (``id``, ``status``,
    ``model``, ``endpoint``, ``created_at`` as a Unix timestamp, ``request_counts.total``).
    ``has_more`` is authoritative for truncation when it is present; otherwise (a bare list, or
    a body without the flag) a page with ``limit`` or more items may be truncated. Rows without
    an ``id`` are skipped.
    """

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise BatchProviderError("OpenRouter batch list returned an unexpected body", provider=PROVIDER_NAME)

    entries: list[BatchListEntry] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        batch_id = row.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            continue
        counts = row.get("request_counts")
        total = counts.get("total") if isinstance(counts, dict) else None
        model = row.get("model")
        endpoint = row.get("endpoint")
        entries.append(
            BatchListEntry(
                batch_id=batch_id,
                state=_STATE_BY_NAME.get(str(row.get("status"))),
                model=model if isinstance(model, str) else None,
                endpoint=endpoint if isinstance(endpoint, str) else None,
                created_at=_epoch_seconds(row.get("created_at")),
                total=_as_int(total) if total is not None else None,
            )
        )
    has_more = payload.get("has_more") if isinstance(payload, dict) else None
    truncated = has_more if isinstance(has_more, bool) else len(rows) >= limit
    return BatchListing(entries=tuple(entries), may_be_truncated=truncated)


def _is_retryable(error_type: str | None, code: Any) -> bool:
    if isinstance(code, int) and not isinstance(code, bool):
        return code == 429 or code >= 500
    marker = (error_type or "").lower()
    if any(token in marker for token in _RETRYABLE_ERROR_MARKERS):
        return True
    return False


def _content_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "".join(str(p) for p in parts)
    return ""


def _usage(body_usage: Any) -> dict[str, int]:
    """Normalise an OpenAI-shaped usage object to the keys the synchronous path emits."""

    if not isinstance(body_usage, dict):
        return {}
    prompt = _as_int(body_usage.get("prompt_tokens"))
    completion = _as_int(body_usage.get("completion_tokens"))
    total = _as_int(body_usage.get("total_tokens")) or prompt + completion
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


def _map_item(
    item: Mapping[str, Any],
    *,
    batch_id: str,
    submitted_model: str,
    requested_model: str | None,
    friendly_name: str,
    fallback_usage: dict[str, int],
) -> BatchOutcome | None:
    custom_id = item.get("custom_id")
    if not isinstance(custom_id, str):
        logger.warning("OpenRouter batch %s returned a result without a custom_id; skipping it", batch_id)
        return None

    err = item.get("error")
    if isinstance(err, dict):
        error_type = err.get("error_type") or err.get("type")
        return BatchOutcome.failed(
            custom_id,
            BatchError(
                message=str(err.get("message") or "request failed"),
                code=str(error_type) if error_type else None,
                retryable=_is_retryable(str(error_type) if error_type else None, err.get("code")),
            ),
        )

    response = item.get("response")
    if not isinstance(response, dict):
        return BatchOutcome.failed(custom_id, BatchError(message="OpenRouter returned no response for this request"))

    status_code = response.get("status_code")
    body = response.get("body")
    body = body if isinstance(body, dict) else {}

    if isinstance(status_code, int) and not 200 <= status_code < 300:
        body_error = body.get("error")
        message = body_error.get("message") if isinstance(body_error, dict) else None
        return BatchOutcome.failed(
            custom_id,
            BatchError(
                message=str(message or f"request failed with HTTP {status_code}"),
                code=str(status_code),
                retryable=_is_retryable(None, status_code),
            ),
        )

    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message_obj = first.get("message")
    content = _content_text(message_obj) if isinstance(message_obj, dict) else ""
    if not content:
        return BatchOutcome.failed(
            custom_id,
            BatchError(
                message=f"OpenRouter returned empty content (finish_reason={first.get('finish_reason')})",
                code="empty_content",
            ),
        )

    metadata: dict[str, Any] = {
        "execution_mode": "batch",
        "batch_id": batch_id,
        "custom_id": custom_id,
        "submitted_model": submitted_model,
        "finish_reason": first.get("finish_reason"),
        "id": body.get("id"),
        "request_id": response.get("request_id"),
    }
    if requested_model is not None and requested_model != submitted_model:
        metadata["requested_model"] = requested_model

    return BatchOutcome.succeeded(
        custom_id,
        ModelResponse(
            content=content,
            usage=_usage(body.get("usage")) or dict(fallback_usage),
            model_name=submitted_model,
            friendly_name=friendly_name,
            provider=ProviderType.OPENROUTER,
            metadata=metadata,
        ),
    )


def parse_results(
    payload: Mapping[str, Any],
    *,
    batch_id: str,
    requested_model: str | None,
    friendly_name: str,
    submitted_model: str | None = None,
    expected_custom_ids: Sequence[str] | None = None,
) -> list[BatchOutcome]:
    """Turn a terminal batch object into per-request outcomes, correlated on ``custom_id``.

    ``submitted_model`` is the slug the adapter actually put in the envelope; it is what
    metadata reports when the provider's payload does not echo a model.

    Completeness: when ``expected_custom_ids`` is given, the result has exactly one outcome
    per expected id, in that order. An id the provider returned no result for becomes a
    failed ``missing_result`` outcome, an id returned more than once becomes a failed
    ``duplicate_result`` outcome (the copies cannot be trusted over one another), and
    results for ids that were never submitted are dropped with a warning. Without the ids
    the batch's ``request_counts.total`` is the only yardstick: fewer usable outcomes than
    that raises :class:`BatchIncompleteResults` carrying the outcomes that did arrive.

    Raises:
        BatchNotReady: the batch is not terminal; poll first.
        BatchFailed: the batch ended (failed / expired / cancelled by the provider) with no
            results to return.
        BatchIncompleteResults: fewer results than requests and no ids to attribute it to.
    """

    status = parse_status(payload)
    if not status.is_terminal:
        raise BatchNotReady(
            f"OpenRouter batch {batch_id} is still {status.state.value}; poll until it is terminal before fetching",
            state=status.state,
            provider=PROVIDER_NAME,
        )

    results = payload.get("results")
    if not isinstance(results, list):
        reason = status.error_message or f"batch ended as '{status.state.value}' without results"
        raise BatchFailed(
            f"OpenRouter batch {batch_id} {status.state.value}: {reason}",
            state=status.state,
            provider=PROVIDER_NAME,
        )

    model_label = str(payload.get("model") or submitted_model or "")
    batch_usage = _usage(payload.get("usage"))
    fallback_usage = batch_usage if len(results) == 1 else {}

    outcomes: dict[str, BatchOutcome] = {}
    duplicated: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        outcome = _map_item(
            item,
            batch_id=batch_id,
            submitted_model=model_label,
            requested_model=requested_model,
            friendly_name=friendly_name,
            fallback_usage=fallback_usage,
        )
        if outcome is None:
            continue
        if outcome.custom_id in outcomes:
            duplicated.add(outcome.custom_id)
            continue
        outcomes[outcome.custom_id] = outcome

    for custom_id in sorted(duplicated):
        logger.warning("OpenRouter batch %s returned several results for custom_id %r", batch_id, custom_id)
        outcomes[custom_id] = BatchOutcome.failed(
            custom_id,
            BatchError(
                message=f"OpenRouter returned more than one result for request '{custom_id}'",
                code="duplicate_result",
            ),
        )

    if expected_custom_ids is None:
        if len(outcomes) < status.total:
            raise BatchIncompleteResults(
                f"OpenRouter batch {batch_id} {status.state.value} with {status.total} requests but only "
                f"{len(outcomes)} usable results; pass the submitted custom_ids to attribute the gap",
                outcomes=list(outcomes.values()),
                provider=PROVIDER_NAME,
                code="incomplete_results",
            )
        return list(outcomes.values())

    ordered: list[BatchOutcome] = []
    for custom_id in expected_custom_ids:
        found = outcomes.pop(custom_id, None)
        if found is None:
            found = BatchOutcome.failed(
                custom_id,
                BatchError(
                    message=f"OpenRouter returned no result for request '{custom_id}' "
                    f"(batch {batch_id} ended as '{status.state.value}')",
                    code="missing_result",
                ),
            )
        ordered.append(found)
    for stray in outcomes:
        logger.warning(
            "OpenRouter batch %s returned a result for unsubmitted custom_id %r; dropping it", batch_id, stray
        )
    return ordered
