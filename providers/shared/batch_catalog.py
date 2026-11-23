"""Which OpenRouter models can be batched, and which one a family alias means right now.

OpenRouter lists a batch-priced copy of a model as a separate catalog entry whose id ends in
``:batch`` (``anthropic/claude-opus-5.5:batch``). The *batch catalog* is the set of base slugs that
have such an entry. Under ``execution_mode="batch"`` a short family alias (``opus``, ``flash``)
resolves to the latest member of its family that appears in that catalog, instead of a curated pin
that goes stale the day a newer model ships.

This module holds the pieces, none of which touch a provider or a tool:

* version ordering and family selection (pure functions, :class:`BatchFamilyRule`);
* the catalog value (:class:`BatchCatalog`) and the committed offline snapshot;
* :class:`BatchCatalogSource`, the lazy, cached, bounded-time live fetch of OpenRouter's public
  ``GET /api/v1/models`` (no credentials), which falls back to the snapshot and never raises into
  its callers;
* the *listing view* (:func:`reduce_listing`): the few fields of a ``:batch`` listing the server reads
  (context length, output cap, supported parameters, modalities), validated so a hostile or garbled body can
  never produce an unsafe value, and the same shape for the live body and the snapshot;
* :func:`unclassified_newer_listings`, the pure classification behind the "a newer model exists that the alias
  did not pick" note.

The fetch is never made at import, at provider construction, or by a capability query; only a batch
admission or resolution asks for the catalog.
"""

from __future__ import annotations

import functools
import importlib.resources
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from utils.env import get_env, suppress_env_vars

logger = logging.getLogger(__name__)

__all__ = [
    "BATCH_CATALOG_URL",
    "BATCH_SUFFIX",
    "BatchCatalog",
    "BatchCatalogFormatError",
    "BatchCatalogSource",
    "BatchFamilyRule",
    "DEFAULT_TEMPERATURE_RANGE",
    "LISTED_PARAMETERS",
    "MAX_UNCLASSIFIED_NOTE",
    "PASSTHROUGH_PARAMETERS",
    "VENDOR_TEMPERATURE_RANGES",
    "VIEW_FIELDS",
    "compare_key",
    "fetch_models_payload",
    "get_batch_catalog_source",
    "load_snapshot",
    "merge_views",
    "parse_catalog_payload",
    "parse_models_payload",
    "parse_version",
    "reduce_listing",
    "reset_batch_catalog_source",
    "reset_warn_once",
    "set_batch_catalog_source",
    "unclassified_newer_listings",
    "validate_vendor_temperature_ranges",
    "validate_view",
    "vendor_temperature_range",
    "warn_once",
]

#: OpenRouter's public model list. Needs no API key.
BATCH_CATALOG_URL = "https://openrouter.ai/api/v1/models"

#: Suffix of a batch-priced catalog entry.
BATCH_SUFFIX = ":batch"

#: ``OPENROUTER_BATCH_LIVE_CATALOG=false`` disables the live fetch (the snapshot and flags remain).
ENV_LIVE_CATALOG = "OPENROUTER_BATCH_LIVE_CATALOG"
#: Seconds a fetched catalog is trusted before the next admission refreshes it.
ENV_CATALOG_TTL = "OPENROUTER_BATCH_CATALOG_TTL_SECONDS"

DEFAULT_TTL_SECONDS = 3600.0
DEFAULT_NEGATIVE_TTL_SECONDS = 300.0
DEFAULT_TIMEOUT_SECONDS = 5.0
SNAPSHOT_FILENAME = "openrouter_batch_catalog.json"

#: Largest model-list body the live fetch accepts (decoded). OpenRouter's real list is under 1 MB; the cap
#: only bounds a misbehaving or intercepted endpoint, so neither memory nor parse time can run away.
MAX_BODY_BYTES = 8 * 1024 * 1024

_FALSE_WORDS = frozenset({"0", "false", "no", "off", "disabled", "disable"})
_TRUE_WORDS = frozenset({"1", "true", "yes", "on", "enabled", "enable"})

#: The longest slug believed. Real slugs are well under 60 characters; a hostile or garbled body could name one of
#: hundreds of kilobytes, which would then be copied into response metadata and log lines.
MAX_SLUG_CHARS = 128

#: What a ``:batch`` listing's base slug looks like: ``vendor/model`` in plain ASCII, no whitespace or control
#: characters (a leading ``~`` is OpenRouter's "latest" alias namespace). Anything else is not trusted as a slug.
_SLUG_RE = re.compile(
    rf"(?=.{{3,{MAX_SLUG_CHARS}}}\Z)~?[A-Za-z0-9][A-Za-z0-9._~+-]*/[A-Za-z0-9][A-Za-z0-9._~+:-]*", re.ASCII
)

#: A version is at most this many dot-separated components, each at most this many digits. Real versions are
#: ``5``, ``5.5``, ``4.10``; a longer one is garbage that must not outrank a real release.
_MAX_VERSION_PARTS = 4
_MAX_VERSION_DIGITS = 4
_VERSION_RE = re.compile(
    rf"[0-9]{{1,{_MAX_VERSION_DIGITS}}}(?:\.[0-9]{{1,{_MAX_VERSION_DIGITS}}}){{0,{_MAX_VERSION_PARTS - 1}}}"
)


class BatchCatalogFormatError(ValueError):
    """A catalog body or snapshot is not the shape OpenRouter's public list has."""


# ----------------------------------------------------------------------
# Version ordering and family selection (pure)
# ----------------------------------------------------------------------


def parse_version(text: str) -> tuple[int, ...] | None:
    """``"5.5"`` -> ``(5, 5)``; ``None`` when ``text`` is not plain ASCII dot-separated integers.

    Bounded: at most four components of at most four digits each, so a garbled or hostile slug such as
    ``claude-opus-99999999999999999999`` has no version and can never outrank a real release.
    """

    candidate = text.strip()
    if _VERSION_RE.fullmatch(candidate) is None:
        return None
    return tuple(int(part) for part in candidate.split("."))


def compare_key(version: tuple[int, ...], *, preview: bool) -> tuple[tuple[int, ...], int]:
    """Sort key where the larger value is the later model.

    Components compare numerically and position by position, with missing components counting as
    zero (trailing zeros are dropped so ``5`` and ``5.0`` are the same version): ``5.5 > 5 > 4.8`` and
    ``4.10 > 4.9``. At an equal version a general-availability slug beats a ``-preview`` one, while a
    newer preview still beats an older GA release (the version decides first).
    """

    trimmed = list(version)
    while trimmed and trimmed[-1] == 0:
        trimmed.pop()
    return tuple(trimmed), 0 if preview else 1


@functools.lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    # ASCII, so ``\d`` never matches a non-ASCII digit; callers use ``fullmatch``, so a trailing newline
    # (which ``$`` would let through) is part of the text that must match and fails it.
    return re.compile(pattern, re.ASCII)


@dataclass(frozen=True)
class BatchFamilyRule:
    """Which short aliases float, and which catalog slugs belong to their family.

    ``aliases`` names exactly the aliases of the owning catalog entry that resolve dynamically under
    batch; version-pinned aliases (``opus4.5``, ``o3``) and full slugs never float. ``pattern`` is
    anchored and must capture ``version`` (and may capture ``preview``); the tier is part of the
    pattern, so a flash rule cannot match ``-flash-lite`` nor a pro rule ``-flash``. A slug with no
    numeric version (a codename such as ``gemini-4-argon``) matches nothing, so it is ignored unless
    ``order`` names it: ``order`` ranks slugs explicitly, best first, and beats numeric order.
    ``label`` names the family in error text.
    """

    aliases: tuple[str, ...]
    pattern: str
    order: tuple[str, ...] = ()
    label: str = ""
    _aliases_lower: frozenset[str] = field(default=frozenset(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_aliases_lower", frozenset(alias.lower() for alias in self.aliases))

    @classmethod
    def from_config(
        cls,
        raw: Mapping[str, Any],
        *,
        model_name: str,
        entry_aliases: Sequence[str] = (),
    ) -> BatchFamilyRule:
        """Build and validate a rule from its JSON form; every problem names the catalog entry."""

        def fail(problem: str) -> BatchCatalogFormatError:
            return BatchCatalogFormatError(f"Invalid batch_family on model '{model_name}': {problem}")

        if not isinstance(raw, Mapping):
            raise fail("it must be an object")
        unknown = set(raw) - {"aliases", "pattern", "order", "label"}
        if unknown:
            raise fail("unsupported keys " + ", ".join(sorted(unknown)))

        aliases = raw.get("aliases")
        if not isinstance(aliases, list) or not aliases or not all(isinstance(a, str) and a.strip() for a in aliases):
            raise fail("'aliases' must be a non-empty list of names")
        belong = {alias.lower() for alias in entry_aliases}
        strangers = [alias for alias in aliases if alias.lower() not in belong]
        if strangers:
            raise fail(f"alias {', '.join(repr(a) for a in strangers)} is not an alias of this entry")

        pattern = raw.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise fail("'pattern' must be a regular expression")
        try:
            compiled = _compile(pattern)
        except re.error as exc:
            raise fail(f"'pattern' does not compile ({exc})") from exc
        if "version" not in compiled.groupindex:
            raise fail("'pattern' must capture a group named 'version'")

        order = raw.get("order", [])
        if not isinstance(order, list) or not all(isinstance(slug, str) and slug for slug in order):
            raise fail("'order' must be a list of slugs")
        label = raw.get("label", "")
        if not isinstance(label, str):
            raise fail("'label' must be text")
        return cls(
            aliases=tuple(str(a).strip() for a in aliases),
            pattern=pattern,
            order=tuple(order),
            label=label,
        )

    @property
    def line_stem(self) -> str | None:
        """The literal text a family's slugs begin with (``google/gemini-``), or ``None`` when it cannot be told.

        The text of ``pattern`` between its leading ``^`` and the ``version`` group, with escapes undone. It is
        the product line the rule describes (the vendor prefix narrowed to ``gemini-``, ``gpt-``,
        ``claude-opus-``), which is what the "newer, unclassified" note compares candidates against. A pattern
        that does not start with ``^``, has no version group, or has regular-expression syntax before it has
        no stem.
        """

        if not self.pattern.startswith("^"):
            return None
        end = self.pattern.find("(?P<version>")
        if end < 1:
            return None
        text = re.sub(r"\\(.)", r"\1", self.pattern[1:end])
        if not text or any(char in text for char in "()[]{}*+?|\\^$"):
            return None
        return text

    def floats(self, name: str) -> bool:
        """True when ``name`` (as typed) is one of the aliases that resolve dynamically."""

        return name.strip().lower() in self._aliases_lower

    def is_member(self, slug: str) -> bool:
        """True when ``slug`` belongs to the family (matches the pattern, or is named by ``order``)."""

        return self.rank(slug) is not None or slug.lower() in {named.lower() for named in self.order}

    def rank(self, slug: str) -> tuple[tuple[int, ...], int] | None:
        """The slug's :func:`compare_key`, or ``None`` when it is not a member of the family."""

        match = _compile(self.pattern).fullmatch(slug)
        if match is None:
            return None
        version = parse_version(match.group("version") or "")
        if version is None:
            return None
        preview = "preview" in match.re.groupindex and bool(match.group("preview"))
        return compare_key(version, preview=preview)

    def select(self, slugs: Iterable[str]) -> str | None:
        """The latest family member among ``slugs`` (``None`` when there is none).

        An ``order`` entry that is available wins over numeric order. Otherwise the highest
        :meth:`rank` wins, ties broken by slug so the answer never depends on iteration order.
        """

        available = list(dict.fromkeys(slugs))
        by_lower = {slug.lower(): slug for slug in available}
        for named in self.order:
            if named.lower() in by_lower:
                return by_lower[named.lower()]

        ranked: list[tuple[tuple[tuple[int, ...], int], str]] = []
        for slug in available:
            rank = self.rank(slug)
            if rank is not None:
                ranked.append((rank, slug))
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[0], item[1]))
        return ranked[-1][1]


# ----------------------------------------------------------------------
# Temperature range per vendor (pure): what a listing that names ``temperature`` leaves unsaid
# ----------------------------------------------------------------------

#: The range of a vendor not named in :data:`VENDOR_TEMPERATURE_RANGES`: the conservative one every vendor accepts.
DEFAULT_TEMPERATURE_RANGE: tuple[float, float] = (0.0, 1.0)

#: A listing says WHETHER a model takes ``temperature``, never the range. A derived entry whose listing names it
#: clamps to the range of the slug's vendor prefix (the part before ``/``): Anthropic takes 0..1, OpenAI, Google
#: and xAI take 0..2, and any vendor not named here (Qwen, Z.ai, DeepSeek, Moonshot, Mistral, Meta ...) gets
#: :data:`DEFAULT_TEMPERATURE_RANGE`. Keep it small: a vendor is added only when its documented range is wider.
VENDOR_TEMPERATURE_RANGES: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "anthropic": (0.0, 1.0),
        "openai": (0.0, 2.0),
        "google": (0.0, 2.0),
        "x-ai": (0.0, 2.0),
    }
)

#: The widest range any vendor entry may state (OpenRouter's documented ceiling for ``temperature``).
MAX_TEMPERATURE = 2.0


def validate_vendor_temperature_ranges(table: Mapping[str, tuple[float, float]]) -> None:
    """Refuse a table with an unusable entry, naming the vendor.

    A vendor key is a non-empty lower-case prefix with no ``/``; a range is two finite numbers with
    ``0 <= low < high <= MAX_TEMPERATURE``. A bad table is a load error, never a silent fallback.
    """

    for vendor, bounds in table.items():
        if not isinstance(vendor, str) or not vendor or vendor != vendor.strip().lower() or "/" in vendor:
            raise BatchCatalogFormatError(
                f"Invalid temperature range table: vendor {vendor!r} is not a lower-case prefix"
            )
        try:
            low, high = bounds
        except (TypeError, ValueError) as exc:
            raise BatchCatalogFormatError(
                f"Invalid temperature range for vendor '{vendor}': expected (low, high)"
            ) from exc
        numbers = (low, high)
        if not all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in numbers):
            raise BatchCatalogFormatError(
                f"Invalid temperature range for vendor '{vendor}': bounds must be finite numbers"
            )
        if not 0.0 <= low < high <= MAX_TEMPERATURE:
            raise BatchCatalogFormatError(
                f"Invalid temperature range for vendor '{vendor}': need 0 <= low < high <= {MAX_TEMPERATURE}"
            )


validate_vendor_temperature_ranges(VENDOR_TEMPERATURE_RANGES)


def vendor_temperature_range(
    slug: str, table: Mapping[str, tuple[float, float]] = VENDOR_TEMPERATURE_RANGES
) -> tuple[float, float]:
    """The ``(low, high)`` temperature range for ``slug`` from its vendor prefix (``DEFAULT_TEMPERATURE_RANGE`` when
    the vendor is not in the table, or the slug has no ``vendor/`` prefix)."""

    vendor, separator, _ = slug.strip().lower().partition("/")
    if not separator:
        return DEFAULT_TEMPERATURE_RANGE
    return table.get(vendor, DEFAULT_TEMPERATURE_RANGE)


# ----------------------------------------------------------------------
# The listing view (pure): what the server reads from a model listing
# ----------------------------------------------------------------------

#: Sampling parameters the batch request builder forwards when the model's listing names them.
PASSTHROUGH_PARAMETERS = ("top_p", "frequency_penalty", "presence_penalty", "seed", "stop")

#: The parameter names of a listing the server acts on; anything else a listing names is dropped from the view.
LISTED_PARAMETERS = frozenset(
    {
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        *PASSTHROUGH_PARAMETERS,
        "tools",
        "response_format",
        "structured_outputs",
        "reasoning",
        "include_reasoning",
    }
)

#: The five fields of a view, in the order they are reported.
VIEW_FIELDS = (
    "context_length",
    "max_completion_tokens",
    "supported_parameters",
    "input_modalities",
    "output_modalities",
)

#: Bounds a number or a list in a listing must meet to be believed. A value outside them is rejected whole (never
#: clamped: a clamped garbage value is still an invented one) and the field falls back to the next source.
MIN_CONTEXT_LENGTH = 1_024
#: A real output cap is at least a thousand tokens (the smallest any current OpenRouter listing or LiteLLM entry
#: publishes is 1,024 / 2,048); a value below this is a garbled listing (``100``, ``512``) that would cap a request to
#: a handful of tokens, so it is rejected like any other malformed number.
MIN_MAX_COMPLETION = 1_024
MAX_CONTEXT_LENGTH = 10_000_000
MAX_LISTED_PARAMETERS = 256
MAX_PARAMETER_CHARS = 64
MAX_MODALITIES = 16
MAX_MODALITY_CHARS = 32


def _plain_int(value: Any) -> int | None:
    """``value`` as an int when it is a JSON integer (an integral float counts); a bool, string, NaN or infinity does not."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _valid_context_length(value: Any) -> int | None:
    number = _plain_int(value)
    if number is None or not MIN_CONTEXT_LENGTH <= number <= MAX_CONTEXT_LENGTH:
        return None
    return number


def _valid_max_completion(value: Any, context: int | None) -> int | None:
    number = _plain_int(value)
    ceiling = context if context is not None else MAX_CONTEXT_LENGTH
    if number is None or not MIN_MAX_COMPLETION <= number <= ceiling:
        return None
    return number


def _string_list(value: Any, *, max_items: int, max_chars: int) -> list[str] | None:
    """``value`` when it is a list of at most ``max_items`` non-empty strings of at most ``max_chars`` characters."""

    if not isinstance(value, list) or len(value) > max_items:
        return None
    if not all(isinstance(item, str) and 0 < len(item) <= max_chars for item in value):
        return None
    return list(value)


def _valid_parameters(value: Any) -> list[str] | None:
    """The allowlisted, sorted parameter names of a listing; an empty list is valid (it means "send none")."""

    listed = _string_list(value, max_items=MAX_LISTED_PARAMETERS, max_chars=MAX_PARAMETER_CHARS)
    if listed is None:
        return None
    return sorted({name for name in listed if name in LISTED_PARAMETERS})


def _valid_modalities(value: Any) -> list[str] | None:
    listed = _string_list(value, max_items=MAX_MODALITIES, max_chars=MAX_MODALITY_CHARS)
    if listed is None:
        return None
    return sorted(set(listed))


def _dig(entry: Any, *path: str) -> Any:
    current = entry
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def reduce_listing(batch_entry: Any, base_entry: Any = None) -> dict[str, Any]:
    """The view of one model: ``{context_length, max_completion_tokens, supported_parameters, input_modalities,
    output_modalities}``, each valid or ``None``.

    ``batch_entry`` is the ``<slug>:batch`` listing (the one the batch endpoint accepts) and ``base_entry`` the
    plain ``<slug>`` listing of the same body (``None`` when there is none). ``supported_parameters`` come from
    the batch listing ONLY: the two disagree for most models (a model that takes ``temperature`` synchronously
    often does not under batch). The numbers and the modalities come from the batch listing and, when a field is
    missing or invalid there, from the base listing (the batch listing is a pricing copy of that model).

    A field that is missing, of the wrong type, out of bounds or malformed is ``None``; nothing is clamped or
    repaired, and no input makes this raise.
    """

    context = _valid_context_length(_dig(batch_entry, "context_length"))
    if context is None:
        context = _valid_context_length(_dig(base_entry, "context_length"))

    max_completion = _valid_max_completion(_dig(batch_entry, "top_provider", "max_completion_tokens"), context)
    if max_completion is None:
        max_completion = _valid_max_completion(_dig(base_entry, "top_provider", "max_completion_tokens"), context)

    view: dict[str, Any] = {
        "context_length": context,
        "max_completion_tokens": max_completion,
        "supported_parameters": _valid_parameters(_dig(batch_entry, "supported_parameters")),
    }
    for key in ("input_modalities", "output_modalities"):
        modalities = _valid_modalities(_dig(batch_entry, "architecture", key))
        if modalities is None:
            modalities = _valid_modalities(_dig(base_entry, "architecture", key))
        view[key] = modalities
    return view


def validate_view(raw: Any) -> dict[str, Any]:
    """A stored view (the snapshot's) re-validated with the same rules as a live body.

    A hand-edited file is as untrusted as a response. Anything that is not a mapping is an empty view (every
    field ``None``).
    """

    source = raw if isinstance(raw, Mapping) else {}
    context = _valid_context_length(source.get("context_length"))
    return {
        "context_length": context,
        "max_completion_tokens": _valid_max_completion(source.get("max_completion_tokens"), context),
        "supported_parameters": _valid_parameters(source.get("supported_parameters")),
        "input_modalities": _valid_modalities(source.get("input_modalities")),
        "output_modalities": _valid_modalities(source.get("output_modalities")),
    }


def merge_views(sources: Sequence[tuple[str, Mapping[str, Any] | None]]) -> tuple[dict[str, Any], dict[str, str]]:
    """Combine views in preference order, field by field.

    ``sources`` is ``[("live", view), ("snapshot", view)]``; each field takes the first source that has it (not
    ``None``), independently of the others, so one bad live field falls back alone. Returns the merged view and
    ``{field: name of the source it came from}`` (``"none"`` when no source has the field, which the caller
    answers from its curated entry).
    """

    merged: dict[str, Any] = {}
    origin: dict[str, str] = {}
    for key in VIEW_FIELDS:
        merged[key], origin[key] = None, "none"
        for name, view in sources:
            if view is not None and view.get(key) is not None:
                merged[key], origin[key] = view[key], name
                break
    return merged, origin


# ----------------------------------------------------------------------
# One WARNING per distinct situation per process
# ----------------------------------------------------------------------

_WARN_ONCE_LIMIT = 256
_warn_once_seen: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_warn_once_lock = threading.Lock()


def warn_once(key: tuple[Any, ...], message: str, *args: Any) -> bool:
    """Log ``message`` at WARNING the first time ``key`` is seen in this process; ``True`` when it logged.

    The seen set is bounded (the oldest key is forgotten past 256), so a long-running server cannot grow it
    without limit; the cost of forgetting is one repeated line.
    """

    with _warn_once_lock:
        if key in _warn_once_seen:
            _warn_once_seen.move_to_end(key)
            return False
        _warn_once_seen[key] = None
        while len(_warn_once_seen) > _WARN_ONCE_LIMIT:
            _warn_once_seen.popitem(last=False)
    logger.warning(message, *args)
    return True


def reset_warn_once() -> None:
    """Forget every seen key (test hook)."""

    with _warn_once_lock:
        _warn_once_seen.clear()


# ----------------------------------------------------------------------
# Catalog value and snapshot
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BatchCatalog:
    """An immutable set of base slugs OpenRouter lists as ``<slug>:batch``, with each one's listing view.

    ``origin`` is ``"live"`` (fetched from OpenRouter) or ``"snapshot"`` (the committed copy).
    ``views`` maps a slug to its :func:`reduce_listing` view (empty for a catalog that carries none, such as
    the flags-and-pins fallback or a format-1 snapshot). The value never changes after construction, so
    capturing one for a whole operation guarantees every model in it is resolved against the same data.
    """

    slugs: frozenset[str]
    origin: str = "snapshot"
    fetched_at: str | None = None
    views: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, compare=False)
    #: The two layers ``views`` was merged from, for a catalog parsed from a live body: what the ``:batch`` listing
    #: alone says, and what the plain listing of the same model says. Empty for a catalog that carries only merged
    #: views (the snapshot, a hand-built one). See :meth:`view_layers`.
    exact_views: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, compare=False)
    base_views: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, compare=False)
    _lower: frozenset[str] = field(default=frozenset(), init=False, repr=False, compare=False)
    _views_lower: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _exact_lower: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _base_lower: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_lower", frozenset(slug.lower() for slug in self.slugs))
        object.__setattr__(self, "_views_lower", {slug.lower(): view for slug, view in self.views.items()})
        object.__setattr__(self, "_exact_lower", {slug.lower(): view for slug, view in self.exact_views.items()})
        object.__setattr__(self, "_base_lower", {slug.lower(): view for slug, view in self.base_views.items()})

    def lists(self, slug: str) -> bool:
        """True when ``<slug>:batch`` is a listed batch model (case-insensitive)."""

        return slug.strip().lower() in self._lower

    def view_for(self, slug: str) -> Mapping[str, Any] | None:
        """The listing view of ``slug``, or ``None`` when this catalog carries none for it."""

        return self._views_lower.get(slug.strip().lower())

    def view_layers(self, slug: str) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        """``(batch listing alone, plain listing alone)`` for ``slug``, so a caller can rank them around other sources.

        The ``:batch`` listing is the one the batch endpoint serves and its numbers can differ from the plain
        listing's (a different context window is real for some models), so a value taken from the plain listing
        must not outrank the committed snapshot's batch-derived value. A catalog that carries only merged views
        returns ``(merged view, None)``.
        """

        key = slug.strip().lower()
        exact, base = self._exact_lower.get(key), self._base_lower.get(key)
        if exact is None and base is None:
            return self.view_for(slug), None
        return exact, base


def parse_models_payload(payload: Any) -> frozenset[str]:
    """Base slugs of the ``:batch`` entries in an OpenRouter ``/models`` body.

    Only well-formed slugs count (plain ASCII ``vendor/model``); an id with whitespace, a newline or other
    odd characters is dropped rather than trusted.

    Raises:
        BatchCatalogFormatError: the body is not a model list, or it lists models but not one ``:batch``
            entry. A truncated, filtered or error body must never replace a good catalog with an empty
            one (OpenRouter lists dozens of batch models; none at all means the body is degraded, and the
            caller keeps the last good catalog or the committed snapshot instead).
    """

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise BatchCatalogFormatError("the model list body has no 'data' array")
    ids = [entry["id"] for entry in data if isinstance(entry, dict) and isinstance(entry.get("id"), str)]
    if not ids:
        raise BatchCatalogFormatError("the model list holds no model ids")
    slugs = frozenset(
        base
        for base in (
            identifier[: -len(BATCH_SUFFIX)]
            for identifier in ids
            if identifier.endswith(BATCH_SUFFIX) and len(identifier) > len(BATCH_SUFFIX)
        )
        if _SLUG_RE.fullmatch(base)
    )
    if not slugs:
        raise BatchCatalogFormatError(
            f"the model list has {len(ids)} models but no usable '{BATCH_SUFFIX}' listing (a degraded body)"
        )
    return slugs


def parse_catalog_payload(payload: Any, *, origin: str = "live", fetched_at: str | None = None) -> BatchCatalog:
    """A :class:`BatchCatalog` from an OpenRouter ``/models`` body: the slugs of :func:`parse_models_payload` and
    the :func:`reduce_listing` view of each (its ``:batch`` listing, then the plain listing of the same body).

    Raises:
        BatchCatalogFormatError: as :func:`parse_models_payload`. A listing whose capability fields are garbled
            does not raise: those fields are ``None`` in its view.
    """

    slugs = parse_models_payload(payload)
    listings: dict[str, Any] = {}
    for entry in payload["data"]:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            listings[entry["id"]] = entry
    views = {slug: reduce_listing(listings.get(slug + BATCH_SUFFIX), listings.get(slug)) for slug in sorted(slugs)}
    exact = {slug: reduce_listing(listings.get(slug + BATCH_SUFFIX), None) for slug in views}
    base = {slug: reduce_listing(None, listings.get(slug)) for slug in views}
    return BatchCatalog(slugs, origin=origin, fetched_at=fetched_at, views=views, exact_views=exact, base_views=base)


def _snapshot_text() -> str | None:
    try:
        resource = importlib.resources.files("conf").joinpath(SNAPSHOT_FILENAME)
        return resource.read_text(encoding="utf-8")
    except Exception:
        pass
    path = Path(__file__).resolve().parents[2] / "conf" / SNAPSHOT_FILENAME
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def load_snapshot(path: Path | None = None) -> BatchCatalog | None:
    """The committed snapshot as a catalog, or ``None`` when it is missing or unreadable.

    ``path`` overrides the packaged ``conf/openrouter_batch_catalog.json`` (tests, the refresh script).
    A format-2 file carries ``model_data`` (a view per slug), re-validated with the rules a live body gets; a
    format-1 file (slugs only) loads with every view empty, so each field is missing and falls back.
    """

    text = _read(path) if path is not None else _snapshot_text()
    if text is None:
        return None
    try:
        document = json.loads(text)
        models = document["batch_models"]
        if not isinstance(models, list) or not all(isinstance(slug, str) and slug for slug in models):
            raise BatchCatalogFormatError("'batch_models' must be a list of slugs")
        fetched = document.get("fetched_at")
        listed = set(models)
        model_data = document.get("model_data")
        views: dict[str, Mapping[str, Any]] = {}
        if isinstance(model_data, dict):
            views = {slug: validate_view(raw) for slug, raw in model_data.items() if slug in listed}
        return BatchCatalog(
            frozenset(models),
            origin="snapshot",
            fetched_at=fetched if isinstance(fetched, str) else None,
            views=views,
        )
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("OpenRouter batch catalog snapshot is unreadable (%s); ignoring it", exc)
        return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# ----------------------------------------------------------------------
# Newer listings no family rule classifies (pure)
# ----------------------------------------------------------------------

#: The most slugs the "newer, unclassified" note names; the rest are counted, not listed.
MAX_UNCLASSIFIED_NOTE = 10

_LEAD_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+)*", re.ASCII)


def _lead_version(rest: str) -> tuple[tuple[int, ...], str] | None:
    """``("5.6-sol")`` -> ``((5, 6), "-sol")``: the plain version a slug's text begins with and what follows it.

    ``None`` when the text does not begin with a bounded plain version (``oss-120b``) or the version runs into
    letters (``4o``): such a slug cannot be told from a different product.
    """

    match = _LEAD_VERSION_RE.match(rest)
    if match is None:
        return None
    tail = rest[match.end() :]
    if tail and not tail.startswith("-"):
        return None
    version = parse_version(match.group(0))
    if version is None:
        return None
    return version, tail


def _tier_word(tail: str) -> str:
    """The first dash-separated word after a lead version (``flash`` in ``-flash-lite``), or ``""`` when none."""

    if not tail.startswith("-"):
        return ""
    return tail[1:].split("-", 1)[0].lower()


def _fold_pro_variants(names: tuple[str, ...]) -> tuple[str, ...]:
    """``names`` without the ``-pro`` sibling of a slug that is itself in ``names``, order kept.

    ``openai/gpt-6.2-sol`` and ``openai/gpt-6.2-sol-pro`` are one model to a reader choosing what to name; listing
    both spends half of a bounded note on duplicates and pushes older models out of it. A ``-pro`` slug whose plain
    sibling is not reported stays (it is then the only mention of that model). Folding is presentation only.
    """

    present = set(names)
    return tuple(name for name in names if not (name.endswith("-pro") and name[: -len("-pro")] in present))


def unclassified_newer_listings(
    *,
    pick: str,
    rule: BatchFamilyRule,
    catalog: BatchCatalog,
    rules: Iterable[BatchFamilyRule],
    allowed: Callable[[str], bool],
    limit: int = MAX_UNCLASSIFIED_NOTE,
) -> tuple[tuple[str, ...], int]:
    """Batch-listed models of the alias's product line that are newer than ``pick`` and that no family rule classifies.

    ``pick`` is the slug the alias resolved to by ``rule`` in ``catalog``; ``rules`` is every family rule of every
    curated entry; ``allowed(slug)`` is the restriction policy. A catalog slug is reported when ALL of these hold:

    1. it is batch-listed and starts with the rule's :attr:`~BatchFamilyRule.line_stem` (the vendor prefix narrowed
       to the product line, so ``gpt-oss-120b`` or ``o3`` is never reported for ``gpt5``);
    2. no rule of any curated entry matches it (a slug a rule matches is a member of some family, picked or not);
    3. it begins with a plain version strictly greater than ``pick``'s (``5.6-sol``, ``4-argon``), compared
       component-wise; no lead version, or an equal or lower one, is not reported;
    4. its first word after that version is not the tier word of a classified slug of the same line (``flash`` in
       ``3.9-flash-lite``, whereas ``sol`` or ``argon`` match nothing): a tier variant of a tier the rules already
       cover is not a new model;
    5. when the catalog carries output modalities for it they are exactly ``["text"]`` (an embeddings, image or
       audio listing is excluded); with none recorded it is kept;
    6. ``allowed`` admits it (the caller tests each candidate as the full slug a reader would have to type, because
       that is what the note advises).

    A ``-pro`` slug is folded into its plain sibling when both qualify (one model, named once), a slug with a
    colon (a variant spelling) or longer than :data:`MAX_SLUG_CHARS` is never reported, and a newer slug with no
    tier word at all (``gpt-6``) is reported.

    Returns ``(slugs, omitted)``: at most ``limit`` base slugs, newest version first then slug ascending, and the
    number cut. Pure: it never changes which model is chosen and never raises for a well-typed argument.
    """

    stem = rule.line_stem
    ranked = rule.rank(pick)
    if stem is None or ranked is None:
        return (), 0
    pick_version = ranked[0]
    stem_lower = stem.lower()
    rule_list = list(rules)

    in_line = [slug for slug in catalog.slugs if slug.lower().startswith(stem_lower)]
    classified = {slug for slug in in_line if any(candidate.is_member(slug) for candidate in rule_list)}
    classified_words: set[str] = set()
    for slug in classified:
        lead = _lead_version(slug[len(stem) :])
        if lead is not None:
            classified_words.add(_tier_word(lead[1]))
    # A classified slug with NO tier word (``gpt-5.5``) says nothing about a newer bare ``gpt-6``: the empty word is
    # not a tier, so it must not make every untiered newer slug look like a tier variant of something classified.
    classified_words.discard("")

    found: list[tuple[tuple[int, ...], str]] = []
    for slug in in_line:
        if slug in classified or ":" in slug or len(slug) > MAX_SLUG_CHARS:
            continue  # a variant spelling is not a base slug, and one too long to be real is not trusted
        lead = _lead_version(slug[len(stem) :])
        if lead is None:
            continue
        version, tail = lead
        key = compare_key(version, preview=False)[0]
        if key <= pick_version or _tier_word(tail) in classified_words:
            continue
        view = catalog.view_for(slug)
        outputs = view.get("output_modalities") if view is not None else None
        if outputs is not None and list(outputs) != ["text"]:
            continue
        if not allowed(slug):
            continue
        found.append((key, slug))

    found.sort(key=lambda item: item[1])
    found.sort(key=lambda item: item[0], reverse=True)
    names = _fold_pro_variants(tuple(slug for _, slug in found))
    return names[:limit], max(0, len(names) - limit)


# ----------------------------------------------------------------------
# Live source
# ----------------------------------------------------------------------


def _http_fetch(
    timeout: float,
    *,
    transport: Any = None,
    max_bytes: int = MAX_BODY_BYTES,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """``GET`` OpenRouter's public model list and return the decoded JSON. No credentials are sent.

    The single seam unit tests replace (and a harness guard makes fail loudly): nothing else in this
    module opens a socket. The body is streamed and bounded two ways: it may not exceed ``max_bytes``
    once decoded (the declared length is checked first, then the running total, so a body that lies or
    omits its length is still cut off), and the whole download, not each read, may not outlast
    ``timeout`` seconds. Parsing happens only after both limits held, so it is bounded too.
    ``transport`` is a test seam for an ``httpx`` mock transport.

    Raises:
        BatchCatalogFormatError: the body is larger than ``max_bytes``.
        TimeoutError: the download outlasted ``timeout``.
    """

    import httpx

    proxy_env_vars = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    deadline = clock() + timeout
    chunks: list[bytes] = []
    total = 0
    with suppress_env_vars(*proxy_env_vars):
        with httpx.Client(timeout=httpx.Timeout(timeout), follow_redirects=True, transport=transport) as client:
            with client.stream(
                "GET",
                BATCH_CATALOG_URL,
                headers={"Accept": "application/json", "User-Agent": "unison-mcp-server"},
            ) as response:
                response.raise_for_status()
                declared = response.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise BatchCatalogFormatError(f"the model list body declares {declared} bytes (limit {max_bytes})")
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise BatchCatalogFormatError(f"the model list body is larger than {max_bytes} bytes")
                    if clock() > deadline:
                        raise TimeoutError(f"the model list download outlasted {timeout:g}s")
                    chunks.append(chunk)
    return json.loads(b"".join(chunks))


def fetch_models_payload(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Any:
    """Fetch OpenRouter's public model list with the bounded, credential-free request (see :func:`_http_fetch`).

    The one entry point for both the live source and the refresh script; ``_http_fetch`` is looked up at call
    time so a test harness that replaces it covers both.
    """

    return _http_fetch(timeout)


_warned_values: set[str] = set()


def live_catalog_enabled() -> bool:
    """False when ``OPENROUTER_BATCH_LIVE_CATALOG`` says so (the fetch is then never attempted).

    Unset or empty means on. A value that is neither a recognised "on" nor "off" word fails closed: it is
    treated as off with a warning, because this switch is the privacy opt-out and a typo (``disable``,
    ``fasle``) must not leave the network request running. The snapshot and committed flags still serve.
    """

    value = get_env(ENV_LIVE_CATALOG)
    word = (value or "").strip().lower()
    if not word or word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    if word not in _warned_values:
        _warned_values.add(word)
        logger.warning(
            "%s=%r is not a recognised value (use true or false); treating it as false, so the live "
            "catalog is not fetched",
            ENV_LIVE_CATALOG,
            value,
        )
    return False


def _ttl_from_env() -> float:
    raw = get_env(ENV_CATALOG_TTL)
    if raw:
        try:
            value = float(raw)
            if value > 0 and math.isfinite(value):
                return value
        except ValueError:
            pass
        logger.warning(
            "%s=%r is not a positive, finite number of seconds; using %ss", ENV_CATALOG_TTL, raw, DEFAULT_TTL_SECONDS
        )
    return DEFAULT_TTL_SECONDS


class BatchCatalogSource:
    """Hands out the current :class:`BatchCatalog`: live when it can, the snapshot otherwise.

    Order of preference, per call to :meth:`current`:

    1. a live catalog fetched within the TTL;
    2. a fresh fetch (lazy, bounded by ``timeout_seconds`` on a daemon thread so a stalled connection
       can neither hang the caller nor outlive the ceiling), unless the last attempt failed within
       the negative TTL;
    3. the last good live catalog, kept when a refresh fails (a body with no ``:batch`` listing at all
       counts as a failure, so a degraded response never replaces a good catalog with an empty one);
    4. the committed snapshot;
    5. ``None``: the caller then uses the committed ``supports_batch`` flags and curated pins.

    It is thread-safe and never raises: every failure is logged once per failure window and degrades to
    the next step. Only one fetch runs at a time (concurrent callers of :meth:`current` wait for it and
    share its result), but that wait is on a separate fetch lock: :meth:`loaded` and :meth:`pin` take
    only a short state lock that no fetch ever holds, so a capability query or the start of another
    submission is never held up behind a network call. The fetch blocks its caller for at most the
    timeout, so async code calls :meth:`current` through ``to_thread``.
    """

    def __init__(
        self,
        *,
        fetcher: Callable[[float], Any] | None = None,
        snapshot: Callable[[], BatchCatalog | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        ttl_seconds: float | None = None,
        negative_ttl_seconds: float = DEFAULT_NEGATIVE_TTL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        live_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._fetcher = fetcher or fetch_models_payload
        self._snapshot_loader = snapshot or load_snapshot
        self._clock = clock
        self._ttl = ttl_seconds
        self._negative_ttl = negative_ttl_seconds
        self._timeout = timeout_seconds
        self._live_enabled = live_enabled or live_catalog_enabled
        # ``_state`` guards the fields below and is only ever held for a few instructions (never across a
        # fetch or a file read of any size); ``_fetch_lock`` serialises fetches and is taken by no reader.
        self._state = threading.RLock()
        self._fetch_lock = threading.Lock()
        self._live: BatchCatalog | None = None
        self._live_expires = 0.0
        self._retry_at = 0.0
        self._snapshot_cache: tuple[BatchCatalog | None] | None = None
        self._pins = 0
        self._pinned: tuple[BatchCatalog | None] | None = None

    # -- reads ---------------------------------------------------------------------

    def current(self) -> BatchCatalog | None:
        """The catalog to resolve against now (may fetch). ``None`` when nothing is known at all."""

        try:
            with self._state:
                if self._pins and self._pinned is not None:
                    return self._pinned[0]
            catalog = self._resolve()
            with self._state:
                if self._pins:
                    if self._pinned is None:  # the first lookup under a pin decides; later ones follow it
                        self._pinned = (catalog,)
                    return self._pinned[0]
            return catalog
        except Exception:  # pragma: no cover - the contract is that callers never see an error
            logger.exception("OpenRouter batch catalog lookup failed unexpectedly; using the snapshot")
            return self._snapshot()

    def snapshot_catalog(self) -> BatchCatalog | None:
        """The committed snapshot as a catalog (read once and kept; never fetches), or ``None`` when there is none.

        The second source of the per-field capability chain: a field the live listing lacks is answered from here.
        """

        return self._snapshot()

    def loaded(self) -> BatchCatalog | None:
        """What is already known, never fetching and never waiting for a fetch.

        A pinned or live catalog if held, else the snapshot. For capability queries (``listmodels``), which
        must make no network call and must not stall behind one in flight on another thread.
        """

        with self._state:
            if self._pins and self._pinned is not None:
                return self._pinned[0]
            if self._live is not None:
                return self._live
            return self._snapshot()

    @contextmanager
    def pin(self) -> Iterator[None]:
        """Hold one catalog for a whole operation.

        The first :meth:`current` inside the block decides the catalog and every later call (from any
        thread) gets the same object until the last pin is released, so the models of one submission
        cannot see different catalogs even if the TTL lapses mid-flight. Entering takes only the state
        lock, so it never waits for a fetch.
        """

        with self._state:
            self._pins += 1
        try:
            yield
        finally:
            with self._state:
                self._pins -= 1
                if self._pins <= 0:
                    self._pins = 0
                    self._pinned = None

    # -- internals -----------------------------------------------------------------

    def _ttl_seconds(self) -> float:
        return self._ttl if self._ttl is not None else _ttl_from_env()

    def _snapshot(self) -> BatchCatalog | None:
        with self._state:
            if self._snapshot_cache is None:
                try:
                    self._snapshot_cache = (self._snapshot_loader(),)
                except Exception as exc:
                    logger.warning("OpenRouter batch catalog snapshot could not be loaded (%s)", exc)
                    self._snapshot_cache = (None,)
            return self._snapshot_cache[0]

    def _fresh_live(self, now: float) -> BatchCatalog | None:
        with self._state:
            if self._live is not None and now < self._live_expires:
                return self._live
            return None

    def _resolve(self) -> BatchCatalog | None:
        if self._live_enabled():
            live = self._fresh_live(self._clock())
            if live is not None:
                return live
            if self._clock() >= self._retry_at:
                with self._fetch_lock:
                    # Another caller may have fetched (or failed) while this one waited for the lock.
                    now = self._clock()
                    live = self._fresh_live(now)
                    if live is not None:
                        return live
                    if now >= self._retry_at:
                        self._refresh(now)
            with self._state:
                if self._live is not None:
                    return self._live  # fresh, or the last good one kept after a failed refresh
        return self._snapshot()

    def _refresh(self, now: float) -> None:
        try:
            catalog = parse_catalog_payload(
                self._fetch_bounded(),
                origin="live",
                fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        except Exception as exc:
            with self._state:
                self._retry_at = now + self._negative_ttl
                fallback = "last fetched catalog" if self._live is not None else "committed snapshot"
            logger.warning(
                "OpenRouter batch catalog could not be fetched (%s: %s); using the %s and not retrying for %.0fs",
                type(exc).__name__,
                exc,
                fallback,
                self._negative_ttl,
            )
            return
        with self._state:
            self._live = catalog
            self._live_expires = now + self._ttl_seconds()
            self._retry_at = 0.0

    def _fetch_bounded(self) -> Any:
        """Run the fetcher on a daemon thread and wait at most the timeout for it."""

        box: dict[str, Any] = {}

        def run() -> None:
            try:
                box["payload"] = self._fetcher(self._timeout)
            except BaseException as exc:
                box["error"] = exc

        worker = threading.Thread(target=run, name="openrouter-batch-catalog", daemon=True)
        worker.start()
        worker.join(self._timeout)
        if worker.is_alive():
            raise TimeoutError(f"no answer within {self._timeout:g}s")
        error = box.get("error")
        if error is not None:
            if isinstance(error, Exception):
                raise error
            raise RuntimeError(f"the fetcher stopped with {error!r}")  # never let a BaseException escape a lookup
        return box["payload"]


# ----------------------------------------------------------------------
# Process-wide source
# ----------------------------------------------------------------------

_source: BatchCatalogSource | None = None
_source_lock = threading.Lock()


def get_batch_catalog_source() -> BatchCatalogSource:
    """The process-wide source (created on first use; creating it fetches nothing)."""

    global _source
    with _source_lock:
        if _source is None:
            _source = BatchCatalogSource()
        return _source


def set_batch_catalog_source(source: BatchCatalogSource | None) -> None:
    """Replace the process-wide source (tests inject a fixture-backed one)."""

    global _source
    with _source_lock:
        _source = source


def reset_batch_catalog_source() -> None:
    """Forget the process-wide source so the next use builds a fresh one."""

    set_batch_catalog_source(None)
