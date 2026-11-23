"""What a batch model name resolved to: the slug, the limits of that slug, and the note that goes with it.

Under ``execution_mode="batch"`` a short alias (``opus``) resolves to a slug (``anthropic/claude-opus-5.5``)
that ``conf/openrouter_models.json`` may not list. The limits applied to the call (context window, output cap,
temperature, which request parameters are sent) are then the slug's OWN, derived from its ``:batch`` listing in
OpenRouter's model list, never the older curated family entry's and never invented:

1. the live catalog's ``:batch`` listing the call pinned;
2. the committed snapshot (``conf/openrouter_batch_catalog.json``, the same view of each listing);
3. the live body's plain (non-``:batch``) listing, numbers and modalities only, never ``supported_parameters``;
4. the curated entry (the alias's family entry, or the generic defaults for a full slug the catalog does not
   list), with one WARNING.

The output cap and temperature rules are then bounded by :func:`apply_published_bounds` (the smallest valid cap
among those listings and LiteLLM, never above the context window).

A slug the curated catalog DOES list keeps its reviewed context window and flags, and its output cap and
temperature range are only ever narrowed by published figures (:func:`apply_published_bounds`), never raised.
Its request PARAMETERS are not kept: which optional parameters a request carries (``temperature``, ``top_p``, ``seed``, the token
cap spelling) is decided by the same listing chain, because the listing is what the batch endpoint accepts
(:func:`apply_listed_parameters`). Only when no listing exists at all does the curated entry decide, with one
WARNING.

Each field falls back on its own. :class:`BatchModelResolution` is the single value that carries the result: it is
built once at admission, handed to submission, and persisted (:meth:`BatchModelResolution.to_record`) with the child
record, so a resume reads what was recorded and never derives again.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .batch_catalog import MAX_UNCLASSIFIED_NOTE, merge_views, vendor_temperature_range, warn_once
from .model_capabilities import ModelCapabilities
from .provider_type import ProviderType
from .published_limits import bound_temperature, combine_output_caps, litellm_limits
from .temperature import FixedTemperatureConstraint, RangeTemperatureConstraint

__all__ = [
    "NOTE_KEY",
    "NOTE_OMITTED_KEY",
    "PARAMETERS_FIELD",
    "PUBLISHED_CAP_FIELD",
    "BatchModelResolution",
    "apply_listed_parameters",
    "apply_published_bounds",
    "derive_capabilities",
    "limits_from_record",
    "note_from_record",
    "resolve_capabilities",
]

#: Resolution-metadata key naming newer batch listings the alias did not pick.
NOTE_KEY = "unclassified_newer_batch_models"
#: Present beside :data:`NOTE_KEY` only when the list was cut.
NOTE_OMITTED_KEY = "unclassified_newer_batch_models_omitted"

#: The curated fields a view can leave unanswered, named as the WARNING names them.
_CAPABILITY_FIELDS = ("context_length", "max_completion_tokens", "supported_parameters", "input_modalities")

#: The key of ``BatchModelResolution.sources`` that says where a curated slug's parameter list came from.
PARAMETERS_FIELD = "supported_parameters"

#: The key of ``BatchModelResolution.sources`` that names the PUBLISHED source whose figure bounds the output cap:
#: ``live`` or ``snapshot`` (an OpenRouter listing) or ``litellm``. Absent when nothing published a cap.
PUBLISHED_CAP_FIELD = "max_output_published"

#: What a derived entry that lists ``temperature`` defaults to (the range itself is per vendor, see
#: :func:`providers.shared.batch_catalog.vendor_temperature_range`).
_DERIVED_TEMPERATURE_DEFAULT = 0.3
_DERIVED_NO_TEMPERATURE = FixedTemperatureConstraint(1.0)


@functools.cache
def _derived_temperature(low: float, high: float) -> RangeTemperatureConstraint:
    """The constraint for one vendor range, built once per distinct range so equal ranges share one object (the
    constraint classes compare by identity, and two derivations of one listing must give equal capabilities)."""

    return RangeTemperatureConstraint(low, high, _DERIVED_TEMPERATURE_DEFAULT)


_RECORD_VERSION = 1

logger = logging.getLogger(__name__)


def derive_capabilities(view: Mapping[str, Any], fallback: ModelCapabilities, *, slug: str) -> ModelCapabilities:
    """The capabilities of ``slug`` from its (merged) listing view; a field the view lacks comes from ``fallback``.

    Pure. ``fallback`` is the curated family entry for an alias, or the generic defaults for a full slug the
    curated catalog does not list. The result is a copy keyed to ``slug`` (no aliases, no family rule):

    * ``context_window`` from ``context_length``; ``max_output_tokens`` from ``max_completion_tokens`` and never
      above the window (the two can come from different sources);
    * ``supports_temperature`` is whether ``temperature`` is listed, with the range of the slug's vendor (the
      catalog publishes none: 0..1 for Anthropic and every vendor not in the table, 0..2 for OpenAI, Google and
      xAI; see :func:`providers.shared.batch_catalog.vendor_temperature_range`); a model without it gets the fixed
      constraint the other no-temperature entries use;
    * function calling, JSON mode and thinking flags from ``tools``, ``response_format``/``structured_outputs``
      and ``reasoning``/``include_reasoning``; ``supports_images`` from the ``image`` input modality;
    * ``supported_parameters`` records the listed names, which request construction obeys.
    """

    context = view.get("context_length")
    window = context if isinstance(context, int) else fallback.context_window
    cap = view.get("max_completion_tokens")
    output = cap if isinstance(cap, int) else fallback.max_output_tokens
    if window > 0:
        output = min(output, window)

    changes: dict[str, Any] = {"context_window": window, "max_output_tokens": output}
    listed = view.get("supported_parameters")
    if listed is not None:
        names = frozenset(listed)
        takes_temperature = "temperature" in names
        changes.update(
            supports_temperature=takes_temperature,
            temperature_constraint=(
                _derived_temperature(*vendor_temperature_range(slug)) if takes_temperature else _DERIVED_NO_TEMPERATURE
            ),
            supports_function_calling="tools" in names,
            supports_json_mode=bool(names & {"response_format", "structured_outputs"}),
            supports_extended_thinking=bool(names & {"reasoning", "include_reasoning"}),
            supported_parameters=tuple(sorted(names)),
        )
    inputs = view.get("input_modalities")
    if inputs is not None:
        changes["supports_images"] = "image" in inputs
    return dataclasses.replace(fallback, model_name=slug, aliases=[], batch_family=None, supports_batch=True, **changes)


def apply_listed_parameters(
    curated: ModelCapabilities,
    *,
    slug: str,
    sources: list[tuple[str, Mapping[str, Any] | None]],
) -> tuple[ModelCapabilities, str]:
    """A curated entry with the request parameters its slug's own ``:batch`` listing names (D9).

    Only request construction changes. The curated entry keeps its limits (context window, output cap), flags and
    temperature range; what is replaced is ``supported_parameters`` (so a body carries exactly the parameters the
    listing names, per parameter) and, when the listing names no ``temperature``, ``supports_temperature`` (so none
    is sent and the validator reports "unsupported" instead of clamping). A listing never WIDENS a reviewed entry:
    a curated ``supports_temperature: false`` stays false although a listing names it. The stored list then leaves ``temperature`` out, so the validator reports "unsupported" (INFO) instead of correcting a value that is never sent.

    ``sources`` is ``[("live", view), ("snapshot", view)]``, the same order a derived entry uses; the first view
    that has a parameter list answers. Returns the capabilities and where the list came from: ``"live"`` or
    ``"snapshot"``, or ``"curated"`` when no listing exists at all (the entry is returned untouched, with one
    WARNING per slug). Never raises for a malformed view.
    """

    merged, origin = merge_views(sources)
    listed = merged.get(PARAMETERS_FIELD)
    if not isinstance(listed, (list, tuple)) or not all(isinstance(name, str) for name in listed):
        listed = (
            None  # views are validated where they are built; a hand-made one that is not a list of names is missing
        )
    if listed is None:
        warn_once(
            ("curated-parameters", slug),
            "OpenRouter lists no batch parameters for %s (live or snapshot), so the curated entry decides which "
            "request parameters are sent",
            slug,
        )
        return curated, "curated"
    names = frozenset(listed)
    if not curated.supports_temperature:
        # A listing never widens a reviewed entry, so a ``temperature`` it names is not one this entry may send. Leaving
        # it in the list would make the validator believe the model accepts one and "correct" a value that is then
        # never sent (a misleading warning), so the stored list is the listed names the entry may act on.
        names -= {"temperature"}
    return (
        dataclasses.replace(
            curated,
            supports_temperature=curated.supports_temperature and "temperature" in names,
            supported_parameters=tuple(sorted(names)),
        ),
        origin[PARAMETERS_FIELD],
    )


def apply_published_bounds(
    capabilities: ModelCapabilities,
    *,
    slug: str,
    sources: list[tuple[str, Mapping[str, Any] | None]],
    stand_in_cap: bool,
) -> tuple[ModelCapabilities, str | None]:
    """``capabilities`` bounded by everything published about ``slug`` (D12, D13, D16): the output cap and the
    temperature rules. Returns the capabilities and the published source that bounds the output cap (``None`` when
    nothing published one).

    The output cap is the smallest valid figure among the OpenRouter listings in ``sources`` (live ``:batch``,
    snapshot, plain, in that order), LiteLLM, and the entry's own cap; on a disagreement the smallest wins. A cap that
    is a ``stand_in`` (the curated family entry's, or the generic default) is a guess about a different model and is
    replaced by the published figure rather than combined with it, so a real published number is never lowered to an
    invented one. Never above the context window. Temperature: see :func:`published_limits.bound_temperature`.
    """

    candidates: list[tuple[str, int | None]] = []
    for origin, view in sources:
        if view is not None:
            listed = view.get("max_completion_tokens")
            candidates.append((origin, listed if isinstance(listed, int) and not isinstance(listed, bool) else None))
    candidates.append(("litellm", litellm_limits(ProviderType.OPENROUTER, slug).output_cap))
    best = combine_output_caps(candidates)

    changes: dict[str, Any] = {}
    label: str | None = None
    if best is not None:
        figure, label = best
        own = capabilities.max_output_tokens
        effective = figure if stand_in_cap or own <= 0 else min(own, figure)
        if capabilities.context_window > 0:
            effective = min(effective, capabilities.context_window)
        if effective != own:
            changes["max_output_tokens"] = effective
    temperature_changes, _provenance = bound_temperature(ProviderType.OPENROUTER, slug, capabilities)
    changes.update(temperature_changes)
    return (dataclasses.replace(capabilities, **changes) if changes else capabilities), label


def resolve_capabilities(
    slug: str,
    *,
    sources: list[tuple[str, Mapping[str, Any] | None]],
    fallback: ModelCapabilities,
    fallback_label: str,
) -> tuple[ModelCapabilities, dict[str, str]]:
    """Derive ``slug``'s capabilities from ``sources`` (live, snapshot, then the plain listing), field by field, then ``fallback``.

    Returns the capabilities and ``{field: "live" | "snapshot" | fallback_label}``. ``fallback_label`` is
    ``"curated"`` (the alias's family entry) or ``"generic"`` (defaults for a full slug). When any capability
    field is answered by the fallback, one WARNING per ``(slug, set of fields)`` names them; a field the live
    listing lacked but the snapshot had is a DEBUG line only. Never raises for a malformed view.
    """

    merged, origin = merge_views(sources)
    capabilities = derive_capabilities(merged, fallback, slug=slug)
    where = {name: (fallback_label if origin[name] == "none" else origin[name]) for name in _CAPABILITY_FIELDS}
    capabilities, published = apply_published_bounds(
        capabilities, slug=slug, sources=sources, stand_in_cap=origin["max_completion_tokens"] == "none"
    )
    if published is not None:
        where[PUBLISHED_CAP_FIELD] = published
    gaps = sorted(name for name in _CAPABILITY_FIELDS if origin[name] == "snapshot")
    if gaps and sources and sources[0][0] != "snapshot":
        logger.debug(
            "OpenRouter %s listing for %s lacked %s; taken from the committed snapshot", sources[0][0], slug, gaps
        )
    borrowed = sorted(name for name in _CAPABILITY_FIELDS if origin[name] == "none")
    if borrowed:
        what = f"the older curated entry {fallback.model_name}" if fallback_label == "curated" else "generic defaults"
        warn_once(
            ("capabilities", slug, tuple(borrowed)),
            "OpenRouter batch limits for %s are partly taken from %s because no listing gave %s",
            slug,
            what,
            ", ".join(borrowed),
        )
    return capabilities, where


@dataclass(frozen=True)
class BatchModelResolution:
    """What one model name resolved to under batch: built once, carried to submission, persisted with the child.

    ``requested`` is the name the caller typed (trimmed, any ``:batch`` suffix kept off) and ``slug`` what is sent.
    ``capabilities`` are the limits applied to the call: derived from the slug's own listing (``derived``), or a
    curated entry whose context window and flags are kept (``derived`` false: a slug the curated catalog lists; its request
    PARAMETERS still come from the listing, see :func:`apply_listed_parameters`, so ``derived`` is NOT the "send
    only listed parameters" switch: ``supported_parameters`` is). ``sources`` records, per capability field,
    where it came from; for a curated slug it holds only ``supported_parameters`` (``live``, ``snapshot`` or
    ``curated``). ``unclassified`` / ``unclassified_omitted`` are the note: newer batch
    listings no family rule classifies, which the alias did not pick (empty unless the alias floated).
    """

    requested: str
    slug: str
    capabilities: ModelCapabilities
    floated: bool = False
    derived: bool = False
    catalog_origin: str = ""
    sources: Mapping[str, str] = field(default_factory=dict)
    unclassified: tuple[str, ...] = ()
    unclassified_omitted: int = 0

    def window_is_listed(self) -> bool:
        """True when the context window is a real figure for this model rather than a stand-in.

        A curated entry (its context window is kept) carries its reviewed window, and a derived one carries a listed window when
        its ``context_length`` came from the live or snapshot listing. A window answered by the curated family
        entry (an older, smaller model) or the generic defaults is a guess, and nothing may be refused against it.
        """

        return not self.derived or self.sources.get("context_length") in ("live", "snapshot")

    def output_cap_bound(self) -> tuple[int, str] | None:
        """``(cap, source)`` when a request's ``max_output_tokens`` is reduced to a figure somebody stated, else ``None``.

        The cap is stated when a published source bounds it (``live``, ``snapshot`` or ``litellm``, recorded under
        :data:`PUBLISHED_CAP_FIELD`), when it is the cap of a derived entry's own live or snapshot listing, or when it
        is a REVIEWED curated entry's cap (a reviewed figure is data in this repository, not an invented one). A
        stand-in (the curated family entry's cap or the generic default, for a model no listing and no published
        source speaks for) clamps nothing: it would reduce a request to a number nobody stated for this model.
        """

        cap = self.capabilities.max_output_tokens
        if cap <= 0:
            return None
        published = self.sources.get(PUBLISHED_CAP_FIELD)
        if published:
            return cap, published
        if not self.derived:
            return cap, "curated"
        source = self.sources.get("max_completion_tokens")
        return (cap, source) if source in ("live", "snapshot") else None

    def note(self) -> dict[str, Any]:
        """The bounded resolution-metadata note, empty when there is nothing to report."""

        if not self.unclassified:
            return {}
        note: dict[str, Any] = {NOTE_KEY: list(self.unclassified[:MAX_UNCLASSIFIED_NOTE])}
        omitted = self.unclassified_omitted + max(0, len(self.unclassified) - MAX_UNCLASSIFIED_NOTE)
        if omitted > 0:
            note[NOTE_OMITTED_KEY] = omitted
        return note

    def to_record(self) -> dict[str, Any]:
        """The compact JSON form stored on the child record (limits, parameters, field sources and the note)."""

        record: dict[str, Any] = {
            "v": _RECORD_VERSION,
            "slug": self.slug,
            "derived": self.derived,
            "floated": self.floated,
            "catalog": self.catalog_origin,
            "limits": {
                "context_window": self.capabilities.context_window,
                "max_output_tokens": self.capabilities.max_output_tokens,
            },
            "sources": dict(sorted(self.sources.items())),
        }
        if self.capabilities.supported_parameters is not None:
            record["parameters"] = list(self.capabilities.supported_parameters)
        record.update(self.note())
        return record


def _record_mapping(raw: Any) -> Mapping[str, Any]:
    return raw if isinstance(raw, Mapping) else {}


def limits_from_record(raw: Any) -> dict[str, int] | None:
    """``{context_window, max_output_tokens}`` read back from a stored resolution, or ``None`` (a NULL, an older
    row, or anything that is not two positive integers): "nothing recorded"."""

    limits = _record_mapping(_record_mapping(raw).get("limits"))
    window, output = limits.get("context_window"), limits.get("max_output_tokens")
    if (
        isinstance(window, int)
        and isinstance(output, int)
        and not isinstance(window, bool)
        and not isinstance(output, bool)
        and window > 0
        and output > 0
    ):
        return {"context_window": window, "max_output_tokens": output}
    return None


def note_from_record(raw: Any) -> dict[str, Any]:
    """The note stored with a resolution, re-bounded on the way out (at most 10 base slugs, a plain omitted count).

    ``{}`` for a NULL, an older row or a malformed value. Read by resume and by result interpretation: no catalog
    is consulted, so a handle reports what was true at submission.
    """

    record = _record_mapping(raw)
    slugs = record.get(NOTE_KEY)
    if not isinstance(slugs, list):
        return {}
    names = [name for name in slugs if isinstance(name, str) and name and ":" not in name]
    if not names:
        return {}
    note: dict[str, Any] = {NOTE_KEY: names[:MAX_UNCLASSIFIED_NOTE]}
    omitted = record.get(NOTE_OMITTED_KEY)
    cut = max(0, len(names) - MAX_UNCLASSIFIED_NOTE)
    if isinstance(omitted, int) and not isinstance(omitted, bool) and omitted > 0:
        cut += omitted
    if cut > 0:
        note[NOTE_OMITTED_KEY] = cut
    return note
