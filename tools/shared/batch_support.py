"""Tool-side support for batch execution: schema fields, request fields and the handle path.

A batch-capable tool carries ``batch_handle``, ``wait_seconds`` and ``cancel`` on its **own**
schema (the shape ``tools/clink.py`` uses for ``job_id``): there is no separate job tool. This
module owns the three shared pieces so every batch-capable tool does it identically.

* :func:`batch_control_schema_properties` - the three published control properties
  (``batch_handle``, ``wait_seconds``, ``cancel``); :func:`batch_schema_properties` adds
  ``execution_mode``, the one field that asks for batch execution.
* :class:`BatchControlFields` - the matching Pydantic fields (mixed into a request model) and
  :func:`require_inputs_unless_resuming`, the runtime enforcement that replaces the
  schema-level ``required`` list a handle-only call cannot satisfy.
* :class:`BatchResumableToolMixin` - ``execute`` interception: a call carrying a handle is a
  resumption or a cancellation and never reaches the tool's start-work path; a call asking for
  batch (``execution_mode="batch"``) is admitted (text-only, a provider that can batch the model)
  and handed to :meth:`BatchResumableToolMixin.execute_batch_start`, never run synchronously.
* :func:`execution_mode_rejection` - the clear refusal for a tool that has no batch path (clink,
  the multi-step workflow tools).

Execution mode and model identifiers
------------------------------------
Batch is requested ONLY through ``execution_mode``. A model identifier is never read for an
execution hint: the ``:batch`` suffix OpenRouter publishes is a pricing listing, not a routing
signal, so ``model="vendor/model:batch"`` without ``execution_mode`` runs synchronously and every
identifier reaches the provider exactly as supplied (whitespace trimmed).

What the MCP boundary must do (``handlers/tool_handlers.py``, task 7.8)
-----------------------------------------------------------------------
A handle-only call names no model. The boundary computes ``arguments.get("model") or
DEFAULT_MODEL`` and, for tools whose ``requires_model()`` is true (``chat``), resolves it,
validates the provider, allocates tokens and checks input size, which would resolve a
resumption against an unrelated default (and fail on a server with no provider for it). For a
call whose ``arguments`` carry a non-blank ``batch_handle`` (see :func:`is_batch_control_call`)
all of that must be skipped and ``tool.execute(arguments)`` called directly; the tool itself
never consults the boundary's resolution on this path. ``consensus`` already has
``requires_model() == False`` and so is not resolved at the boundary today.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from mcp.types import TextContent
from pydantic import BaseModel, Field, field_validator

from providers.shared.batch_resolution import BatchModelResolution
from tools.models import ToolOutput
from tools.shared.exceptions import ToolExecutionError
from utils.batch_coordinator import BatchSubmissionError, ChildPlan, SubmissionOutcomeUnknown
from utils.batch_execution import (
    DEFAULT_WAIT_POLICY,
    BatchExecutor,
    BatchHandleToolMismatch,
    BatchRunResult,
    BatchRunStatus,
    build_default_executor,
    build_status_payload,
    describe_submission_error,
    result_message,
)
from utils.batch_store import BatchStoreError

if TYPE_CHECKING:
    from providers.base import ModelProvider

logger = logging.getLogger(__name__)

__all__ = [
    "ACCEPTED_EXECUTION_MODES",
    "BATCH_CONTROL_KEYS",
    "BATCH_FIELD_DESCRIPTIONS",
    "EXECUTION_MODE_BATCH",
    "AdmittedBatchProviders",
    "BatchControlFields",
    "BatchRequestRejected",
    "BatchResumableToolMixin",
    "batch_control_schema_properties",
    "batch_schema_properties",
    "admitted_resolution",
    "execution_mode_rejection",
    "execution_mode_schema_property",
    "is_batch_control_call",
    "non_text_argument_names",
    "normalize_execution_mode",
    "reject_non_text_arguments",
    "require_inputs_unless_resuming",
    "resolve_at_admission",
    "select_batch_provider",
]

#: The one value ``execution_mode`` accepts. Absent or ``null`` means a normal synchronous call.
EXECUTION_MODE_BATCH = "batch"
ACCEPTED_EXECUTION_MODES: tuple[str, ...] = (EXECUTION_MODE_BATCH,)

#: Admission shorter than this is not charged to the wait budget (the common case: the catalog is cached).
_ADMISSION_CHARGE_FLOOR_SECONDS = 0.1

#: The request keys that make a call a resumption or cancellation rather than new work.
BATCH_CONTROL_KEYS = frozenset({"batch_handle", "wait_seconds", "cancel"})

BATCH_FIELD_DESCRIPTIONS = {
    "batch_handle": (
        "Resume a batch call that returned status='running'. Pass ONLY the batch_handle (optionally with "
        "wait_seconds or cancel); the call waits for the batch and returns the full result when finished. "
        "Omit when starting new work."
    ),
    "wait_seconds": (
        f"How long this call may wait for a batch before returning a still-running payload "
        f"(0-{int(DEFAULT_WAIT_POLICY.max_seconds)}, default {int(DEFAULT_WAIT_POLICY.default_seconds)}). "
        "Values above the maximum are clamped."
    ),
    "cancel": (
        "With batch_handle: cancel the batch. Where the provider has no cancel mechanism (OpenRouter) the batch is "
        "marked cancelled locally and polling stops, but the provider batch continues and is still billed."
    ),
}


EXECUTION_MODE_DESCRIPTION = (
    'Set to "batch" to run this through the provider\'s asynchronous Batch API. Batch execution costs less '
    "(about half the standard price) and takes longer: results may not be available within this call, in which "
    "case the response is status='running' with a batch_handle to resume (pass only batch_handle). Use it when "
    "the user asks for batch execution or batch models. Omit it (or pass null) for a normal immediate call. It is "
    "independent of the model: pass the model exactly as the user named it and never edit the model name to "
    "request batch. Under batch a short family alias (opus, sonnet, haiku, gemini, flash, gpt5, grok) resolves "
    "to the latest model of its family that OpenRouter lists as batch-capable (the response reports "
    "model_requested and model_resolved, and unclassified_newer_batch_models when OpenRouter also lists newer batch "
    "models the alias did not pick; to use one, pass its full model name); a full model name is used as given. Text-only (no images or "
    "attachments; local files passed as paths are fine) and only for models that support batch (see "
    "listmodels). On a consensus models[] entry, omitting it (or null) uses the "
    "request-level value; to batch only some entries, leave the request-level field unset and set it on those. "
    "On consensus it applies to step 1 only (the step that consults the models); later steps are refused. "
    "Under batch the temperature is sent only where the model's :batch listing names temperature. "
    "Not available on clink or on the other multi-step workflow tools."
)


def execution_mode_schema_property() -> dict[str, Any]:
    """The published JSON-schema property for ``execution_mode`` (same on a tool and on each model-list entry)."""

    return {
        "type": ["string", "null"],
        "enum": [*ACCEPTED_EXECUTION_MODES, None],
        "description": EXECUTION_MODE_DESCRIPTION,
    }


def batch_control_schema_properties() -> dict[str, dict[str, Any]]:
    """The published JSON-schema properties for batch handle, wait and cancel."""

    return {
        "batch_handle": {"type": "string", "description": BATCH_FIELD_DESCRIPTIONS["batch_handle"]},
        "wait_seconds": {"type": "number", "minimum": 0, "description": BATCH_FIELD_DESCRIPTIONS["wait_seconds"]},
        "cancel": {"type": "boolean", "default": False, "description": BATCH_FIELD_DESCRIPTIONS["cancel"]},
    }


def batch_schema_properties() -> dict[str, dict[str, Any]]:
    """Every property a batch-capable tool publishes: the control fields plus ``execution_mode``."""

    return {**batch_control_schema_properties(), "execution_mode": execution_mode_schema_property()}


def normalize_execution_mode(value: Any, *, field: str = "execution_mode") -> str | None:
    """``"batch"`` or ``None``; anything else raises ``ValueError`` naming the accepted values.

    ``None`` (field absent or ``null``) is the synchronous default. The value is matched exactly:
    a typo must not quietly cost double, so ``"Batch"`` or ``""`` is an error, not a guess.
    """

    if value is None:
        return None
    if isinstance(value, str) and value in ACCEPTED_EXECUTION_MODES:
        return value
    raise ValueError(
        f"Invalid {field} {value!r}: the accepted values are "
        + " or ".join(f'"{mode}"' for mode in ACCEPTED_EXECUTION_MODES)
        + " (omit the field or pass null for a normal synchronous call)."
    )


def is_batch_control_call(arguments: dict[str, Any]) -> bool:
    """True when ``arguments`` carry a non-blank ``batch_handle`` (a resumption or cancellation)."""

    return bool(str(arguments.get("batch_handle") or "").strip())


class BatchControlFields(BaseModel):
    """Pydantic fields every batch-capable request model carries.

    ``exclude=True`` keeps them out of ``model_dump()`` so a request that does not use them
    serialises exactly as it did before they existed.
    """

    batch_handle: str | None = Field(default=None, exclude=True, description=BATCH_FIELD_DESCRIPTIONS["batch_handle"])
    wait_seconds: float | None = Field(
        default=None, ge=0, exclude=True, description=BATCH_FIELD_DESCRIPTIONS["wait_seconds"]
    )
    cancel: bool = Field(default=False, exclude=True, description=BATCH_FIELD_DESCRIPTIONS["cancel"])
    execution_mode: str | None = Field(default=None, exclude=True, description=EXECUTION_MODE_DESCRIPTION)

    @field_validator("execution_mode", mode="before")
    @classmethod
    def _validate_execution_mode(cls, value: Any) -> str | None:
        return normalize_execution_mode(value)

    @property
    def requests_batch(self) -> bool:
        """True when ``execution_mode`` selects batch execution (a handle, if present, still takes precedence)."""

        return self.execution_mode == EXECUTION_MODE_BATCH

    @property
    def is_resuming(self) -> bool:
        """True when a handle was supplied: the call resumes or cancels instead of starting work."""

        return bool((self.batch_handle or "").strip())


def require_inputs_unless_resuming(request: BatchControlFields, tool_name: str, required: dict[str, Any]) -> None:
    """Runtime enforcement of fields the published schema no longer lists as required.

    ``required`` maps each field name to its parsed value; a ``None`` value is missing.
    Does nothing when the request carries a handle (it resumes; start-work inputs are
    ignored). Raises ``ValueError`` naming every missing field otherwise, which Pydantic
    surfaces as a validation error.
    """

    if request.is_resuming:
        return
    if request.cancel:
        raise ValueError(f"'cancel' requires 'batch_handle': {tool_name} has no batch to cancel without one.")
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"{tool_name} needs {', '.join(repr(m) for m in missing)} to start new work "
            "(or pass 'batch_handle' to resume or cancel a batch)."
        )


# ----------------------------------------------------------------------
# Admission: what a batch request may carry and which provider runs it
# ----------------------------------------------------------------------


class AdmittedBatchProviders(dict[str, "ModelProvider"]):
    """``{model identifier: provider}`` for an admitted batch request, plus what each provider resolved the model to.

    A plain ``dict`` to every existing reader. ``resolutions`` maps the same identifiers to the
    :class:`~providers.shared.batch_resolution.BatchModelResolution` the provider made once, at admission (only
    for providers that make one); planning reads the limits from it and submission sends exactly it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resolutions: dict[str, BatchModelResolution] = {}


def admitted_resolution(providers: Mapping[str, Any], model: str) -> BatchModelResolution | None:
    """The resolution admission made for ``model`` (a key of ``providers``), or ``None`` when it made none."""

    resolutions = getattr(providers, "resolutions", None)
    found = resolutions.get(model) if isinstance(resolutions, dict) else None
    return found if isinstance(found, BatchModelResolution) else None


def resolve_at_admission(provider: Any, model: str) -> BatchModelResolution | None:
    """Ask ``provider`` to resolve ``model`` under batch, once, inside its resolution scope.

    The slug, the limits derived for it and the note are all computed here from the one catalog the scope pins and
    are carried to planning and submission; nothing resolves a second time. ``None`` for a provider that resolves
    nothing. A refusal the provider only discovers now (it was admitted a moment ago) is a rejection with its text.
    """

    resolve = getattr(provider, "resolve_batch_model", None)
    if not callable(resolve):
        return None
    scope = getattr(provider, "batch_resolution_scope", None)
    try:
        with scope() if callable(scope) else nullcontext():
            resolution = resolve(model)
    except Exception as exc:
        raise BatchRequestRejected(str(exc), code="batch_unsupported_model") from exc
    return resolution if isinstance(resolution, BatchModelResolution) else None


class BatchRequestRejected(ValueError):
    """A batch request that must not be submitted; the message is caller-facing and says what to do instead."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


#: Request keys that carry non-text content. Batch execution is text-only.
_NON_TEXT_ARGUMENT_KEYS = ("images", "image", "audio", "video", "attachments", "documents")


def non_text_argument_names(arguments: Mapping[str, Any]) -> list[str]:
    """Names of the non-text content keys (images, audio, video, attachments, documents) that carry data.

    ``absolute_file_paths`` / ``relevant_files`` are deliberately not here: the tool reads those files
    and embeds them in the prompt as text, which batch execution permits.
    """

    return [key for key in _NON_TEXT_ARGUMENT_KEYS if arguments.get(key)]


def reject_non_text_arguments(tool_name: str, arguments: Mapping[str, Any]) -> None:
    """Raise :class:`BatchRequestRejected` when a batch request carries non-text content.

    Checked before any provider is contacted. The error names the content and the synchronous
    alternative; local files given as paths are not affected.
    """

    offending = non_text_argument_names(arguments)
    if offending:
        raise BatchRequestRejected(
            f"Batch execution is text-only, but this {tool_name} call carries non-text content "
            f"({', '.join(repr(name) for name in offending)}). Remove it, or make the same call without "
            "execution_mode to run it synchronously with the content intact. Local files passed as "
            "absolute_file_paths are read into the prompt as text and are fine in a batch.",
            code="non_text_content",
        )


def select_batch_provider(model_name: str, registry: Any) -> ModelProvider:
    """The provider that will batch ``model_name``, chosen by the registry's batch-aware selection.

    The identifier is passed through untouched (surrounding whitespace aside). Raises
    :class:`BatchRequestRejected` naming the model when no configured provider can batch it,
    and saying the same call without ``execution_mode`` runs it synchronously instead; it never
    falls back to the synchronous path on its own, because that would bill the caller twice what
    they asked for.
    """

    from utils.model_resolution import trim_model_identifier

    model = trim_model_identifier(model_name)
    provider: ModelProvider | None = registry.get_provider_for_model(model, require_batch=True)
    if provider is not None:
        return provider
    # A provider that deliberately refuses the model (a variant, a family with no batch member) says why; that
    # beats a generic message, and it holds even when the unresolved name itself looks unavailable (a family
    # alias whose pinned model the restriction policy forbids).
    explain = getattr(registry, "explain_batch_refusal", None)
    reason = explain(model) if callable(explain) else None
    if isinstance(reason, str) and reason:
        raise BatchRequestRejected(reason, code="batch_unsupported_model")
    if registry.get_provider_for_model(model) is None:
        raise BatchRequestRejected(
            f"Model '{model}' is not available with the current API keys, so it cannot be batched. "
            "Use the listmodels tool to see what is available.",
            code="model_unavailable",
        )
    raise BatchRequestRejected(
        f"Batch execution is unavailable for model '{model}': no configured provider can batch it. "
        "The same call without execution_mode will run it synchronously at standard pricing; "
        "or name a batch-capable model (listmodels shows which support batch).",
        code="batch_unsupported_model",
    )


_CLINK_REJECTION = (
    "execution_mode does not apply to clink: server-side batch execution does not apply to CLI bridge tools. "
    "clink hands the prompt to an external CLI that makes its own model request, so this server cannot submit a "
    "batch for it. Omit execution_mode; the model string you pass is forwarded to the CLI exactly as written."
)


def execution_mode_rejection(
    tool_name: str, arguments: Mapping[str, Any], *, accepts_execution_mode: bool
) -> str | None:
    """The refusal text for ``execution_mode`` sent to a tool with no batch path, else ``None``.

    ``execution_mode`` absent or ``null`` is a synchronous call and never refused. clink gets its own
    wording (server-side batch cannot apply to a CLI bridge); every other tool that does not publish the
    field (the multi-step workflow tools, listmodels, ...) gets the general one.
    """

    if accepts_execution_mode or arguments.get("execution_mode") is None:
        return None
    if tool_name == "clink":
        return _CLINK_REJECTION
    return (
        f"execution_mode is not supported by '{tool_name}': batch execution is available only on tools whose "
        "schema lists it (chat and consensus). Multi-step workflow tools run their analysis synchronously. "
        "Omit execution_mode to run this call normally."
    )


def _text_content(output: ToolOutput) -> list[TextContent]:
    return [TextContent(type="text", text=output.model_dump_json())]


class BatchResumableToolMixin:
    """Adds handle-based resume and cancel to a tool; place it before the tool's base class.

    ``class ChatTool(BatchResumableToolMixin, SimpleTool)`` makes ``execute`` route any call
    with a ``batch_handle`` to :meth:`execute_batch_control`; every other call falls through
    to the base class untouched. Tools override :meth:`render_completed_batch` to turn a
    completed :class:`BatchRunResult` into their own response, and
    :meth:`batch_resume_hint` to add tool-specific resume advice.
    """

    #: Tests inject an executor here; production builds one over the durable store lazily.
    _batch_executor: BatchExecutor | None = None

    def get_batch_executor(self) -> BatchExecutor:
        """The executor for this tool (injectable); raises ``BatchStoreUnavailable`` if the store cannot open."""

        if self._batch_executor is None:
            self._batch_executor = build_default_executor()
        return self._batch_executor

    def batch_resume_hint(self) -> str:
        """Extra tool-specific advice appended to a still-running response (default none)."""

        return ""

    def render_completed_batch(self, result: BatchRunResult) -> list[TextContent]:
        """Render a completed result (default: a JSON listing of every outcome by model)."""

        tool_name: str = self.get_name()  # type: ignore[attr-defined]
        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for item in result.children:
            for outcome in item.outcomes:
                if outcome.success and outcome.response is not None:
                    results.append(
                        {
                            "model": item.child.requested_model,
                            "custom_id": outcome.custom_id,
                            "content": outcome.response.content,
                            "usage": outcome.response.usage,
                        }
                    )
                else:
                    failures.append(
                        {
                            "model": item.child.requested_model,
                            "custom_id": outcome.custom_id,
                            "error": outcome.error.message if outcome.error else "request failed",
                            "retryable": bool(outcome.error and outcome.error.retryable),
                        }
                    )
            if item.error:
                # Also when some outcomes exist: a provider that returned fewer results than were
                # submitted must not look like a clean completion (the missing requests are named here).
                failures.append({"model": item.child.requested_model, "error": item.error})
        payload = {
            "status": "completed",
            "batch_handle": result.handle,
            "tool": tool_name,
            "results": results,
            "failures": failures,
        }
        output = ToolOutput(
            status="success",
            content=json.dumps(payload, indent=2),
            content_type="json",
            metadata={"tool_name": tool_name, "batch_handle": result.handle, "batch_execution": True},
            continuation_offer=None,
        )
        return _text_content(output)

    # ------------------------------------------------------------------

    # -- execution mode ----------------------------------------------------------

    def wants_batch_execution(self, arguments: dict[str, Any]) -> bool:
        """True when the call asks for batch execution (an invalid ``execution_mode`` is a tool error).

        Tools whose request names several models override this to look at each entry as well, so no
        entry that asked for batch can quietly run synchronously.
        """

        try:
            return normalize_execution_mode(arguments.get("execution_mode")) == EXECUTION_MODE_BATCH
        except ValueError as exc:
            self._raise_batch_error(str(exc), code="invalid_execution_mode")
            return False  # unreachable

    def requested_batch_models(self, arguments: dict[str, Any]) -> list[str]:
        """The models a batch call needs a provider for (default: the request's ``model``)."""

        model = arguments.get("model")
        return [str(model)] if model else []

    def get_batch_registry(self) -> Any:
        """The provider registry batch selection consults (the server's default; tests inject one)."""

        from providers.registry import get_default_registry

        return get_default_registry()

    @staticmethod
    def log_batch_temperature_warnings(warnings: Sequence[str]) -> None:
        """Log what temperature validation said for a batch call.

        "Does not accept a temperature" describes the model (its batch listing names none), not a mistake in the
        request: every default-temperature call to such a model would otherwise log a WARNING. It is INFO; a real
        correction (a clamped value) stays a WARNING.
        """

        for warning in warnings:
            if "does not accept a temperature" in warning:
                logger.info(warning)
            else:
                logger.warning(warning)

    def check_batch_input_fits(
        self, model_context: Any, slug: str, texts: Sequence[str], *, resolution: Any = None
    ) -> None:
        """Refuse, before any batch exists, input larger than the context window of the model that will run.

        ``model_context`` carries the resolved slug's own capabilities (not an alias's curated entry), so a prompt
        that outgrew an older curated window but fits the resolved model is admitted, and one beyond the resolved
        window is refused naming the slug and its limit. The size is an estimate (the provider's token counter, or
        a character heuristic); a zero or unknown window checks nothing, and neither does a window the resolution
        says is a stand-in (the older curated entry's or the generic default, because no listing gave one): a
        refusal against a guessed limit would shrink the model's usable context without telling the caller.
        """

        window_is_listed = getattr(resolution, "window_is_listed", None)
        if callable(window_is_listed) and not window_is_listed():
            return

        try:
            window = int(model_context.capabilities.context_window)
        except Exception:
            return
        if window <= 0:
            return
        tokens = sum(model_context.estimate_tokens(text) for text in texts if text)
        if tokens > window:
            self._raise_batch_error(
                f"The input is about {tokens:,} tokens, which is over the {window:,}-token context window of "
                f"{slug}. Nothing was submitted. Shorten it, pass fewer or smaller files, or name a model with a "
                "larger window.",
                code="input_too_large",
            )

    def admit_batch_request(self, arguments: dict[str, Any]) -> dict[str, ModelProvider]:
        """Refuse what cannot be batched and choose a provider for each model, before anything is submitted.

        Non-text content is rejected first (no provider is consulted), then every model the call
        batches must resolve to a provider that can batch it. Returns ``{model identifier: provider}`` (an
        :class:`AdmittedBatchProviders`, which also carries what each provider resolved the model to, once, here).
        Raises a tool error naming the problem and the synchronous alternative.
        """

        tool_name: str = self.get_name()  # type: ignore[attr-defined]
        try:
            reject_non_text_arguments(tool_name, arguments)
            models = self.requested_batch_models(arguments)
            if not models:
                raise BatchRequestRejected(
                    f"execution_mode='batch' needs a model to run: name a batch-capable model (listmodels shows "
                    f"which support batch), or omit execution_mode to run this {tool_name} call synchronously.",
                    code="model_required",
                )
            try:
                registry = self.get_batch_registry()
            except RuntimeError as exc:
                raise BatchRequestRejected(
                    f"Batch execution is unavailable: no provider registry is initialised ({exc}).",
                    code="registry_unavailable",
                ) from exc
            admitted = AdmittedBatchProviders()
            # Every model of the call resolves against ONE catalog: each chosen provider's resolution scope stays
            # entered for the whole loop, so the first lookup under it decides the data and the rest follow it (two
            # models of one consensus run cannot see different catalogs). The selection of the very first model
            # happens before its provider is known and so before the pin; its resolution below is what counts.
            with contextlib.ExitStack() as scopes:
                pinned: set[int] = set()
                for model in models:
                    from utils.model_resolution import trim_model_identifier

                    key = trim_model_identifier(model)
                    if key in admitted:
                        continue
                    provider = select_batch_provider(key, registry)
                    admitted[key] = provider
                    factory = getattr(provider, "batch_resolution_scope", None)
                    if callable(factory) and id(provider) not in pinned:
                        pinned.add(id(provider))
                        scopes.enter_context(factory())
                    resolution = resolve_at_admission(provider, key)
                    if resolution is not None:
                        admitted.resolutions[key] = resolution
            return admitted
        except BatchRequestRejected as exc:
            self._raise_batch_error(str(exc), code=exc.code)
            return {}  # unreachable

    async def run_batch_submission(
        self,
        plans: Sequence[ChildPlan],
        *,
        continuation_id: str | None,
        payload: Any,
        wait_seconds: Any,
        fetch: bool = True,
    ) -> BatchRunResult:
        """Submit ``plans`` as one handle and wait a bounded time, turning every failure into a tool error.

        The shared submission call for the tools' ``execute_batch_start``: it never returns a handle
        for a failed submission, names the handle only for the one ambiguous outcome that is resumable,
        and never retries. ``fetch=False`` leaves each finished child's results unfetched for a tool
        that consumes them itself (consensus).
        """

        tool_name: str = self.get_name()  # type: ignore[attr-defined]
        try:
            executor = self.get_batch_executor()
            return await asyncio.to_thread(
                executor.submit_and_wait,
                tool_name,
                list(plans),
                continuation_id=continuation_id,
                payload=payload,
                wait_seconds=wait_seconds,
                fetch=fetch,
            )
        except SubmissionOutcomeUnknown as exc:
            self._raise_batch_error(
                describe_submission_error(exc), handle=exc.handle, code="submission_outcome_unknown"
            )
        except BatchSubmissionError as exc:
            self._raise_batch_error(describe_submission_error(exc), code="submission_failed")
        except BatchStoreError as exc:
            self._raise_batch_error(
                f"The batch store is unavailable ({exc}); nothing was submitted and nothing was run.",
                code="store_unavailable",
            )
        except ValueError as exc:
            self._raise_batch_error(
                f"The batch request is invalid: {exc}. Nothing was submitted.", code="invalid_batch"
            )
        raise AssertionError("unreachable: _raise_batch_error always raises")  # pragma: no cover

    async def resume_batch(self, executor: BatchExecutor, handle: str, wait_seconds: Any) -> BatchRunResult:
        """Resume ``handle`` on ``executor`` (a bounded wait). Tools that consume results themselves override this."""

        tool_name: str = self.get_name()  # type: ignore[attr-defined]
        return await asyncio.to_thread(executor.resume, handle, tool_name=tool_name, wait_seconds=wait_seconds)

    def batch_status_extras(self, result: BatchRunResult) -> dict[str, Any]:
        """Extra fields a tool adds to its non-terminal status payload (default none)."""

        return {}

    async def execute_batch_start(self, arguments: dict[str, Any], providers: Mapping[str, ModelProvider]) -> list[Any]:
        """Start batch work for an admitted request. Each batch-capable tool implements this.

        ``providers`` maps each model identifier the call batches to the provider chosen for it.
        The default refuses plainly: a call that asked for batch must never be run synchronously
        at full price just because the tool has no batch path of its own yet.
        """

        tool_name: str = self.get_name()  # type: ignore[attr-defined]
        self._raise_batch_error(
            f"{tool_name} cannot start a batch yet: execution_mode='batch' was accepted and checked, but this "
            "tool has no batch submission path. Nothing was submitted and nothing was run; omit execution_mode "
            "for a normal synchronous call.",
            code="batch_start_not_implemented",
        )
        return []  # unreachable

    async def execute(self, arguments: dict[str, Any]) -> list[Any]:
        """Route a call: a handle resumes or cancels; ``execution_mode`` starts a batch; the rest is the tool's own path."""

        if is_batch_control_call(arguments):
            return await self.execute_batch_control(arguments)
        if arguments.get("cancel"):
            self._raise_batch_error(
                "'cancel' requires 'batch_handle': there is no batch to cancel without one.",
                code="cancel_without_handle",
            )
        if self.wants_batch_execution(arguments):
            # Admission may fetch OpenRouter's batch catalog (bounded, cached): keep that off the event loop.
            admission_started = time.monotonic()
            providers = await asyncio.to_thread(self.admit_batch_request, arguments)
            arguments = self._charge_admission_to_wait(arguments, time.monotonic() - admission_started)
            return await self.execute_batch_start(arguments, providers)
        result: list[Any] = await super().execute(arguments)  # type: ignore[misc]
        return result

    @staticmethod
    def _charge_admission_to_wait(arguments: dict[str, Any], elapsed: float) -> dict[str, Any]:
        """Spend a slow admission from the wait budget, so the whole call stays inside its documented ceiling.

        Admission can fetch OpenRouter's batch catalog (up to its timeout on a cold cache); the wait clock of
        the submission starts only after it, so without this the call could run that long past
        ``wait_seconds``. Admission that took a negligible time changes nothing, and a ``wait_seconds`` that is
        not a plain number is left for the request model to reject as before.
        """

        if elapsed < _ADMISSION_CHARGE_FLOOR_SECONDS:
            return arguments
        raw = arguments.get("wait_seconds")
        if isinstance(raw, bool) or not (raw is None or isinstance(raw, (int, float))):
            return arguments
        remaining = max(0.0, DEFAULT_WAIT_POLICY.budget(raw) - elapsed)
        return {**arguments, "wait_seconds": remaining}

    async def execute_batch_control(self, arguments: dict[str, Any]) -> list[Any]:
        """Resume (bounded wait) or cancel the handle in ``arguments``; never submits anything.

        A handle takes precedence over any start-work inputs sent alongside it: they are
        ignored (and logged), so a resumption can never create a second paid batch.
        """

        tool_name: str = self.get_name()  # type: ignore[attr-defined]
        handle = str(arguments.get("batch_handle") or "").strip()
        cancelling = bool(arguments.get("cancel"))
        ignored = sorted(
            key
            for key, value in arguments.items()
            if key not in BATCH_CONTROL_KEYS
            and not key.startswith("_")
            and key not in ("model", "continuation_id")
            and value not in (None, "", [], {})
        )
        if ignored:
            logger.warning(
                "%s: batch_handle supplied alongside %s; ignoring them so nothing is resubmitted",
                tool_name,
                ", ".join(ignored),
            )

        try:
            executor = self.get_batch_executor()
            if cancelling:
                result = await asyncio.to_thread(executor.cancel, handle, tool_name=tool_name)
            else:
                result = await self.resume_batch(executor, handle, arguments.get("wait_seconds"))
        except BatchHandleToolMismatch as exc:
            self._raise_batch_error(str(exc), handle=handle, code="wrong_tool")
        except BatchStoreError as exc:
            self._raise_batch_error(
                f"The batch store is unavailable ({exc}). The handle is unchanged; try again.",
                handle=handle,
                code="store_unavailable",
            )
        return self.render_batch_result(result, cancelling=cancelling, ignored_inputs=ignored)

    def render_batch_result(
        self, result: BatchRunResult, *, cancelling: bool = False, ignored_inputs: Sequence[str] = ()
    ) -> list[Any]:
        """Turn a :class:`BatchRunResult` into the tool's response (or raise a tool error).

        ``running`` and ``cancelled`` are normal responses (a standard JSON payload carrying
        the handle and how to resume). A completed non-cancel call goes through
        :meth:`render_completed_batch`. Unknown or expired handles, failed or expired
        batches and anything needing an operator are tool errors whose text says which.
        """

        tool_name: str = self.get_name()  # type: ignore[attr-defined]
        status = result.status
        if cancelling and status not in (BatchRunStatus.UNKNOWN_HANDLE, BatchRunStatus.HANDLE_EXPIRED):
            return self._status_response(result, tool_name, ignored_inputs, already=not result.cancel_reports)
        if status is BatchRunStatus.COMPLETED:
            return self.render_completed_batch(result)
        if status in (BatchRunStatus.RUNNING, BatchRunStatus.CANCELLED):
            return self._status_response(result, tool_name, ignored_inputs)
        self._raise_batch_error(
            result_message(result, tool_name=tool_name, resume_hint=self.batch_resume_hint()),
            handle=result.handle,
            code=status.value,
        )
        return []  # unreachable: _raise_batch_error always raises

    def _status_response(
        self, result: BatchRunResult, tool_name: str, ignored_inputs: Sequence[str], *, already: bool = False
    ) -> list[Any]:
        payload = build_status_payload(
            result, tool_name=tool_name, policy=DEFAULT_WAIT_POLICY, resume_hint=self.batch_resume_hint()
        )
        if already:
            payload["message"] = (
                f"Batch '{result.handle}' was already {result.status.value}; nothing was cancelled. "
                + str(payload["message"])
            )
        extras = dict(self.batch_status_extras(result))
        note = extras.pop("message_note", None)
        payload.update(extras)
        if note:
            payload["message"] = f"{payload['message']} {note}"
        if ignored_inputs:
            payload["ignored_inputs"] = list(ignored_inputs)
            payload[
                "message"
            ] += " The other inputs were ignored: a batch_handle always resumes, it never starts new work."
        output = ToolOutput(
            status="success",
            content=json.dumps(payload, indent=2),
            content_type="json",
            metadata={"tool_name": tool_name, "batch_handle": result.handle, "batch_status": result.status.value},
            continuation_offer=None,
        )
        return _text_content(output)

    def _raise_batch_error(self, message: str, *, handle: str | None = None, code: str = "batch_error") -> None:
        tool_name: str = self.get_name()  # type: ignore[attr-defined]
        metadata: dict[str, Any] = {"tool_name": tool_name, "batch_error": code}
        if handle:
            metadata["batch_handle"] = handle
        output = ToolOutput(
            status="error", content=message, content_type="text", metadata=metadata, continuation_offer=None
        )
        raise ToolExecutionError(output.model_dump_json())
