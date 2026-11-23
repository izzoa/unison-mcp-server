"""Published per-model limits: one offline lookup that every provider, tool and path shares (D12, D13, D16).

Two limits of a model are checked against what is PUBLISHED about it, never only against the curated entry:

* the **output cap** (``max_output_tokens``): the smallest valid value among the curated capability and what
  LiteLLM publishes for the model. On the BATCH path (``batch_resolution``) the OpenRouter listings (live ``:batch``,
  committed snapshot) are further sources; on the synchronous and streaming paths they are NOT (D17): the snapshot's
  figure is the ``:batch`` listing's and must not bound a request the plain listing would accept;
* the **temperature range**, in this order: LiteLLM's boolean ``supports_sampling_params`` (false on an
  Anthropic-family model means temperature 1 only); then an explicit curated constraint (fixed, discrete, none),
  kept as is; then a curated RANGE narrowed (never widened) by the range of a vendor named in the labelled
  fallback table. A vendor outside the table only fills a gap (a stand-in or derived capability) and never
  narrows a reviewed range.

What LiteLLM really publishes (researched 2026-11-22): per model it carries ``max_output_tokens`` (and the legacy
``max_tokens``), ``max_input_tokens`` and boolean ``supports_*`` flags. It carries **no numeric temperature range,
minimum, maximum or default** for any model. The only machine-readable temperature datum is the boolean
``supports_sampling_params`` (false means an Anthropic-family model takes temperature 1 only; LiteLLM's own
Anthropic code enforces exactly that). Everything else (Anthropic 0..1, OpenAI, Google and xAI 0..2) is
vendor-level knowledge that lives in vendor documentation and in code, not in data, so it stays here as the
documented, labelled FALLBACK (:data:`providers.shared.batch_catalog.VENDOR_TEMPERATURE_RANGES`, a table this
module reuses and does not copy). OpenRouter publishes no range either, only whether ``temperature`` is accepted.

Design guards (LiteLLM data is community-maintained and can be wrong or stale):

* **Offline and cheap.** The data is the extract ``conf/litellm_published_limits.json`` that
  ``scripts/refresh_litellm_models.py`` writes next to the model catalogs it already refreshes. It is read once per
  process under a lock; nothing is fetched at request time. A missing, oversized or corrupt file disables the check
  with one WARNING and behaviour is exactly what it was before the check existed.
* **Conservative key matching.** A model is matched to LiteLLM keys by exact key (provider-prefixed where that is
  how LiteLLM names it) and, for Anthropic names only, the single documented dot-to-dash normalisation of the
  version (``claude-opus-4.5`` -> ``claude-opus-4-5``, the rule LiteLLM applies itself). Never a prefix, a
  ``startswith`` or a fuzzy guess, so another model's limit cannot attach. Probes are tiered (provider-prefixed key,
  then the vendor's own key) but every tier is consulted and ALL hits combine by ``min``, so a stale entry under
  one key never hides a smaller figure under another. No match is no check (one DEBUG per model).
* **A garbled value is ignored**: zero, negative, NaN, a bool, a string, or an absurd number never shrinks a limit
  to nothing; the field is simply not published.
* **Min only.** A published limit larger than the curated one never raises it.
* **A value that disagrees** across sources: the smallest valid one wins, never silently exceeding any of them.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.resources
import json
import logging
import math
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .batch_catalog import (
    MAX_CONTEXT_LENGTH,
    MIN_MAX_COMPLETION,
    VENDOR_TEMPERATURE_RANGES,
    vendor_temperature_range,
    warn_once,
)
from .model_capabilities import ModelCapabilities
from .provider_type import ProviderType
from .temperature import FixedTemperatureConstraint, RangeTemperatureConstraint

__all__ = [
    "EXTRACT_FILENAME",
    "PublishedEntry",
    "PublishedIndex",
    "PublishedLimits",
    "apply_published_limits",
    "bound_temperature",
    "build_extract",
    "combine_output_caps",
    "litellm_limits",
    "published_output_caps",
    "reset_published_limits",
    "set_published_index",
    "valid_output_limit",
    "vendor_from_model_name",
    "vendor_of",
]

logger = logging.getLogger(__name__)

#: The committed extract of LiteLLM's model map (see ``scripts/refresh_litellm_models.py``).
EXTRACT_FILENAME = "litellm_published_limits.json"
#: Where the extract's data came from, written into it for a reader of the file.
EXTRACT_SOURCE_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
#: A file larger than this is not read (a hostile or runaway extract must not be parsed).
MAX_EXTRACT_BYTES = 8 * 1024 * 1024

#: A published output cap below this or above this is a garbled figure and is ignored (same bounds as a listing's).
#: The floor is 1,024 because the smallest cap any shipped LiteLLM entry or OpenRouter listing publishes is 1,024 /
#: 2,048: a figure under it is a typo (``100``, ``512``) that would shrink a real model to a handful of tokens.
MIN_PUBLISHED_OUTPUT = MIN_MAX_COMPLETION
MAX_PUBLISHED_OUTPUT = MAX_CONTEXT_LENGTH

#: ``litellm_provider`` values (and the ``vertex_ai`` prefix) whose entries the extract keeps: the providers this
#: repo can serve a model through, directly or by naming the vendor on OpenRouter.
_EXTRACT_PROVIDERS = frozenset(
    {"openai", "azure", "gemini", "xai", "openrouter", "anthropic", "mistral", "moonshot", "zai"}
)
_EXTRACT_MODES = frozenset({"chat", "responses"})


# ----------------------------------------------------------------------
# Validation of a published figure (pure)
# ----------------------------------------------------------------------


def valid_output_limit(value: Any) -> int | None:
    """``value`` as a published output cap when it is believable, else ``None``.

    A JSON integer (an integral float counts) from :data:`MIN_PUBLISHED_OUTPUT` to :data:`MAX_PUBLISHED_OUTPUT`.
    A bool, string, NaN, infinity, zero, a negative or an absurd figure is not a limit: it is ignored, never clamped,
    because a clamped garbage value is still an invented one and a tiny cap would shrink a limit to nothing.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        value = int(value)
    if not isinstance(value, int):
        return None
    if not MIN_PUBLISHED_OUTPUT <= value <= MAX_PUBLISHED_OUTPUT:
        return None
    return value


# ----------------------------------------------------------------------
# The index: what the extract says, keyed by LiteLLM's own keys
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PublishedEntry:
    """What the extract keeps about one LiteLLM key."""

    provider: str
    max_output_tokens: int | None = None
    #: LiteLLM's ``supports_sampling_params`` is explicitly ``false`` for this key.
    sampling_unsupported: bool = False


def _parse_entry(raw: Any) -> PublishedEntry | None:
    """One extract entry, leniently: a malformed entry is dropped, a malformed field is ignored."""

    if not isinstance(raw, Mapping):
        return None
    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider:
        return None
    cap = valid_output_limit(raw.get("max_output_tokens"))
    unsupported = raw.get("supports_sampling_params") is False
    if cap is None and not unsupported:
        return None
    return PublishedEntry(provider=provider, max_output_tokens=cap, sampling_unsupported=unsupported)


@dataclass(frozen=True)
class PublishedIndex:
    """An immutable index of LiteLLM keys to :class:`PublishedEntry`."""

    entries: Mapping[str, PublishedEntry] = field(default_factory=dict)
    fetched_at: str | None = None

    @classmethod
    def from_document(cls, document: Any) -> PublishedIndex:
        """Build an index from a parsed extract. Raises ``ValueError`` when the document is not an extract."""

        if not isinstance(document, Mapping) or not isinstance(document.get("entries"), Mapping):
            raise ValueError("a published-limits extract is an object with an 'entries' object")
        entries: dict[str, PublishedEntry] = {}
        for key, raw in document["entries"].items():
            if not isinstance(key, str) or not key:
                continue
            parsed = _parse_entry(raw)
            if parsed is not None:
                entries[key] = parsed
        fetched = document.get("fetched_at")
        return cls(entries=entries, fetched_at=fetched if isinstance(fetched, str) else None)

    @classmethod
    def from_text(cls, text: str) -> PublishedIndex:
        """Parse and validate an extract. Raises ``ValueError`` when it cannot be used."""

        if len(text) > MAX_EXTRACT_BYTES:
            raise ValueError(f"the extract is {len(text)} characters, over the {MAX_EXTRACT_BYTES} limit")
        return cls.from_document(json.loads(text))

    @classmethod
    def from_path(cls, path: Path) -> PublishedIndex:
        """Read and validate an extract file. Raises ``OSError`` or ``ValueError`` when it cannot be used."""

        size = path.stat().st_size
        if size > MAX_EXTRACT_BYTES:
            raise ValueError(f"the file is {size} bytes, over the {MAX_EXTRACT_BYTES} byte limit")
        return cls.from_text(path.read_text(encoding="utf-8"))


def _default_extract_path() -> Path:
    return Path(__file__).resolve().parents[2] / "conf" / EXTRACT_FILENAME


def _load_packaged() -> PublishedIndex:
    """The extract shipped with the package (``conf`` resource first, then the repository path)."""

    try:
        text = importlib.resources.files("conf").joinpath(EXTRACT_FILENAME).read_text(encoding="utf-8")
    except Exception:
        return PublishedIndex.from_path(_default_extract_path())
    return PublishedIndex.from_text(text)


_state_lock = threading.Lock()
_index: PublishedIndex | None = None
_index_loaded = False
_index_path: Path | None = None
_lookup_cache: dict[tuple[ProviderType, str], PublishedLimits] = {}
_apply_cache: dict[tuple[ProviderType, str, int], tuple[ModelCapabilities, ModelCapabilities]] = {}
_APPLY_CACHE_LIMIT = 2048
_debug_seen: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_DEBUG_SEEN_LIMIT = 512


def _get_index() -> PublishedIndex | None:
    """The process-wide index, loaded once under a lock. ``None`` (check disabled) when it cannot be read."""

    global _index, _index_loaded
    if _index_loaded:
        return _index
    with _state_lock:
        if _index_loaded:
            return _index
        path = _index_path if _index_path is not None else _default_extract_path()
        try:
            _index = PublishedIndex.from_path(_index_path) if _index_path is not None else _load_packaged()
        except (OSError, ValueError) as exc:
            _index = None
            warn_once(
                ("published-limits-unreadable", str(path)),
                "The LiteLLM published-limits extract %s cannot be used (%s); model limits are checked against the "
                "curated entries only",
                path.name,
                exc,
            )
        _index_loaded = True
        return _index


def set_published_index(index: PublishedIndex | Path | None) -> None:
    """Replace the process-wide index (test hook and the refresh script).

    A :class:`PublishedIndex` is used as given, a path is read lazily on first use, and ``None`` restores the
    packaged extract. Every cache is cleared.
    """

    global _index, _index_loaded, _index_path
    with _state_lock:
        if isinstance(index, PublishedIndex):
            _index, _index_loaded, _index_path = index, True, None
        elif isinstance(index, Path):
            _index, _index_loaded, _index_path = None, False, index
        else:
            _index, _index_loaded, _index_path = None, False, None
    reset_published_limits()


def reset_published_limits() -> None:
    """Forget every cached lookup and once-per-model log line (test hook); the loaded index is kept."""

    with _state_lock:
        _lookup_cache.clear()
        _apply_cache.clear()
        _debug_seen.clear()


# ----------------------------------------------------------------------
# Key matching (pure, conservative)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Probe:
    """One exact LiteLLM key to look for, and the ``litellm_provider`` values an entry under it must carry."""

    key: str
    providers: tuple[str, ...]
    provider_prefix: bool = False

    def accepts(self, provider: str) -> bool:
        if self.provider_prefix:
            return any(provider.startswith(prefix) for prefix in self.providers)
        return provider in self.providers


_CLAUDE_VERSION_DOT = re.compile(r"(?<=\d)\.(?=\d)")


def _anthropic_probes(name: str) -> list[_Probe]:
    probes = [_Probe(name, ("anthropic",))]
    if name.startswith("claude-"):
        dashed = _CLAUDE_VERSION_DOT.sub("-", name)
        if dashed != name:
            probes.append(_Probe(dashed, ("anthropic",)))
    return probes


def _native_probes(vendor: str, name: str) -> list[_Probe]:
    """The exact keys under which ``vendor`` models appear in LiteLLM's map (the vendor as OpenRouter names it)."""

    if vendor == "openai":
        return [_Probe(name, ("openai",)), _Probe(f"openai/{name}", ("openai",))]
    if vendor == "google":
        return [
            _Probe(f"gemini/{name}", ("gemini",)),
            _Probe(f"vertex_ai/{name}", ("vertex_ai",), provider_prefix=True),
        ]
    if vendor == "x-ai":
        return [_Probe(f"xai/{name}", ("xai",))]
    if vendor == "mistralai":
        return [_Probe(f"mistral/{name}", ("mistral",))]
    if vendor == "moonshotai":
        return [_Probe(f"moonshot/{name}", ("moonshot",))]
    if vendor == "z-ai":
        return [_Probe(f"zai/{name}", ("zai",))]
    if vendor == "anthropic":
        return _anthropic_probes(name)
    return []


_OPENAI_NAME = re.compile(r"^(gpt-|chatgpt|o\d+(-|$))")


def vendor_from_model_name(name: str) -> str | None:
    """The vendor a bare model NAME unambiguously carries (``claude*``/``anthropic.`` , ``gemini*``, ``gpt-*``/``o3``,
    ``grok*``), or ``None``. Used only where the provider itself serves many vendors (DIAL, a custom endpoint), so a
    local ``llama3.2`` or an arbitrary name is never given a vendor it does not carry."""

    lowered = name.strip().lower()
    if lowered.startswith(("anthropic.", "claude")):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "google"
    if _OPENAI_NAME.match(lowered):
        return "openai"
    if lowered.startswith("grok"):
        return "x-ai"
    return None


def vendor_of(provider: ProviderType, identifier: str) -> str | None:
    """The vendor whose documented temperature range applies to the model, or ``None`` when it cannot be told.

    A native provider names its vendor (Anthropic, Google, OpenAI and Azure OpenAI, xAI). OpenRouter names it in the
    slug (``anthropic/claude-opus-4.5``). DIAL and custom endpoints serve arbitrary models, so their vendor is taken
    only from an unambiguous model name (:func:`vendor_from_model_name`: DIAL's ``anthropic.claude-opus-4.1-...``);
    a name that says nothing keeps its curated constraint.
    """

    if provider is ProviderType.ANTHROPIC:
        return "anthropic"
    if provider is ProviderType.GOOGLE:
        return "google"
    if provider in (ProviderType.OPENAI, ProviderType.AZURE):
        return "openai"
    if provider is ProviderType.XAI:
        return "x-ai"
    if provider is ProviderType.OPENROUTER:
        vendor, separator, _ = identifier.strip().lower().partition("/")
        return vendor if separator and vendor else None
    if provider in (ProviderType.DIAL, ProviderType.CUSTOM):
        return vendor_from_model_name(identifier)
    return None


def _tiers(provider: ProviderType, identifier: str) -> list[list[_Probe]]:
    """The tiers of exact keys to try for a model, strongest first. Empty when no conservative key exists."""

    name = identifier.strip().lower()
    if not name or ":" in name or any(ch.isspace() for ch in name):
        # A variant (``:free``, an Ollama tag) is not the model LiteLLM lists under the bare name.
        return []
    if provider is ProviderType.OPENAI:
        return [_native_probes("openai", name)]
    if provider is ProviderType.AZURE:
        return [[_Probe(f"azure/{name}", ("azure",))]]
    if provider is ProviderType.GOOGLE:
        return [_native_probes("google", name)]
    if provider is ProviderType.XAI:
        return [_native_probes("x-ai", name)]
    if provider is ProviderType.ANTHROPIC:
        return [_anthropic_probes(name)]
    if provider is ProviderType.OPENROUTER:
        vendor, separator, bare = name.partition("/")
        if not separator or not bare:
            return []
        return [[_Probe(f"openrouter/{name}", ("openrouter",))], _native_probes(vendor, bare)]
    return []


@dataclass(frozen=True)
class PublishedLimits:
    """What the extract publishes for one model (``output_cap`` ``None`` and nothing else set = no check)."""

    output_cap: int | None = None
    sampling_unsupported: bool = False
    matched: tuple[str, ...] = ()


_NO_LIMITS = PublishedLimits()


def litellm_limits(provider: ProviderType, identifier: str, *, index: PublishedIndex | None = None) -> PublishedLimits:
    """What LiteLLM publishes for ``identifier`` as served by ``provider`` (cached; offline; never raises).

    Every tier of probes is consulted and all hits combine by ``min`` (a stale key never hides a smaller figure
    under another). ``index`` overrides the process-wide index (an explicit one is not cached).
    """

    cache_key = (provider, identifier)
    if index is None:
        cached = _lookup_cache.get(cache_key)
        if cached is not None:
            return cached
        index = _get_index()
        if index is None:
            return _NO_LIMITS
        explicit = False
    else:
        explicit = True

    hits: list[tuple[str, PublishedEntry]] = []
    for tier in _tiers(provider, identifier):
        hits.extend(
            (probe.key, entry)
            for probe in tier
            if (entry := index.entries.get(probe.key)) is not None and probe.accepts(entry.provider)
        )
    result = _NO_LIMITS
    if hits:
        caps = [entry.max_output_tokens for _, entry in hits if entry.max_output_tokens is not None]
        result = PublishedLimits(
            output_cap=min(caps) if caps else None,
            sampling_unsupported=any(entry.sampling_unsupported for _, entry in hits),
            matched=tuple(key for key, _ in hits),
        )

    if not result.matched:
        _note_unmatched(provider, identifier)
    if not explicit:
        with _state_lock:
            _lookup_cache[cache_key] = result
    return result


def _note_unmatched(provider: ProviderType, identifier: str) -> None:
    """One DEBUG line per model per process: no LiteLLM key matched, so nothing is checked against LiteLLM."""

    key = (provider, identifier)
    with _state_lock:
        if key in _debug_seen:
            return
        _debug_seen[key] = None
        while len(_debug_seen) > _DEBUG_SEEN_LIMIT:
            _debug_seen.popitem(last=False)
    logger.debug(
        "No LiteLLM published limit matches %s model '%s' by an exact key; nothing is checked",
        provider.value,
        identifier,
    )


def _info_once(key: tuple[Any, ...], message: str, *args: Any) -> None:
    """Log ``message`` at INFO the first time ``key`` is seen (bounded, like the DEBUG set)."""

    with _state_lock:
        if key in _debug_seen:
            return
        _debug_seen[key] = None
        while len(_debug_seen) > _DEBUG_SEEN_LIMIT:
            _debug_seen.popitem(last=False)
    logger.info(message, *args)


# ----------------------------------------------------------------------
# Output cap (pure)
# ----------------------------------------------------------------------


def combine_output_caps(
    candidates: Iterable[tuple[str, int | None]],
) -> tuple[int, str] | None:
    """The smallest valid figure among ``(source, value)`` pairs, with the source that gave it.

    ``candidates`` are given in source order (the batch path: OpenRouter live listing, committed snapshot, LiteLLM; the
    synchronous path: LiteLLM alone, D17); on a tie the earliest source is the one reported. An invalid value (see :func:`valid_output_limit`) is skipped. ``None`` when
    there is no valid figure at all: nothing is invented.
    """

    best: tuple[int, str] | None = None
    for source, value in candidates:
        valid = valid_output_limit(value)
        if valid is None:
            continue
        if best is None or valid < best[0]:
            best = (valid, source)
    return best


def published_output_caps(provider: ProviderType, identifier: str) -> list[tuple[str, int]]:
    """Every published output cap for the SYNCHRONOUS and streaming paths: LiteLLM's figure and nothing else (D17).

    The committed OpenRouter snapshot is deliberately not a source here. Its figure is the ``:batch`` listing's, which
    can be far below what the model accepts on a synchronous request (``moonshotai/kimi-k3``: 16,384 against 943,718
    on the plain listing), and the snapshot stores no plain-listing figure. The batch path reads the snapshot (and the
    live ``:batch`` listing) itself, in ``batch_resolution.apply_published_bounds``, and is unchanged. Offline.
    """

    litellm = litellm_limits(provider, identifier).output_cap
    return [("litellm", litellm)] if litellm is not None else []


# ----------------------------------------------------------------------
# Temperature (pure)
# ----------------------------------------------------------------------


@functools.cache
def _range(low: float, high: float, default: float) -> RangeTemperatureConstraint:
    """One constraint object per distinct range (the constraint classes compare by identity, so two derivations of one
    range must share an object or equal capabilities would compare unequal)."""

    return RangeTemperatureConstraint(low, high, default)


@functools.cache
def _fixed(value: float) -> FixedTemperatureConstraint:
    return FixedTemperatureConstraint(value)


def bound_temperature(
    provider: ProviderType, identifier: str, capabilities: ModelCapabilities
) -> tuple[dict[str, Any], str]:
    """The field changes (possibly none) that make a model's temperature rules respect what is published, and the
    provenance of the rule that applies: ``"litellm"``, ``"curated"`` or ``"vendor-fallback"``.

    Order: (1) LiteLLM's boolean ``supports_sampling_params: false`` on an Anthropic-family model means only
    temperature 1 is accepted: no temperature is sent and the constraint is fixed at 1.0 (the same shape the curated
    Anthropic entries use). It is not applied to other vendors: LiteLLM's own Gemini code does not consume it.
    (2) A curated explicit constraint (fixed, discrete, or a model that takes none) is kept as it is.
    (3) A curated RANGE is intersected with the range of a vendor NAMED in the fallback table (Anthropic 0..1,
    OpenAI, Google and xAI 0..2), the vendor taken from the provider, the OpenRouter slug, or (DIAL, custom) an
    unambiguous model name. A vendor outside the table (the "any other 0..1" row is a guess, not published data)
    narrows only a stand-in capability (an uncurated OpenRouter slug's generic 0..2), never a reviewed range; a
    model whose vendor cannot be told keeps its curated range. The intersection only narrows (it can never widen a
    curated range), and an empty intersection is ignored rather than shrinking the range to nothing.
    """

    vendor = vendor_of(provider, identifier)
    if vendor == "anthropic" and litellm_limits(provider, identifier).sampling_unsupported:
        if capabilities.supports_temperature or not isinstance(
            capabilities.temperature_constraint, FixedTemperatureConstraint
        ):
            return {"supports_temperature": False, "temperature_constraint": _fixed(1.0)}, "litellm"
        return {}, "litellm"

    constraint = capabilities.temperature_constraint
    if (
        not capabilities.supports_temperature
        or not isinstance(constraint, RangeTemperatureConstraint)
        or vendor is None
    ):
        return {}, "curated"
    if vendor not in VENDOR_TEMPERATURE_RANGES and not getattr(capabilities, "_is_generic", False):
        # D16 order: LiteLLM sampling flag, then an explicit curated constraint, then a curated range intersected
        # with the named vendor's fallback range. The "any other vendor 0..1" row is a guess, so it fills a gap (a stand-in capability or a derived entry) and never narrows a
        # range somebody reviewed.
        return {}, "curated"

    low_v, high_v = vendor_temperature_range(f"{vendor}/x")
    low = max(constraint.min_temp, low_v)
    high = min(constraint.max_temp, high_v)
    if not low <= high or (low, high) == (constraint.min_temp, constraint.max_temp):
        return {}, "curated"
    default = min(max(constraint.default_temp, low), high)
    return {"temperature_constraint": _range(low, high, default)}, "vendor-fallback"


# ----------------------------------------------------------------------
# The one hook every capability lookup goes through
# ----------------------------------------------------------------------


def apply_published_limits(
    provider: ProviderType, identifier: str, capabilities: ModelCapabilities
) -> ModelCapabilities:
    """``capabilities`` checked against the published limits of ``identifier`` (D16: every model, every path).

    Returns the SAME object when nothing changes, else a copy (registry objects are shared singletons and are never
    mutated) with the output cap reduced to LiteLLM's published figure (the only source on this path, D17) and the
    temperature rules bounded by :func:`bound_temperature`. A published figure larger than the curated cap changes nothing. Cached per
    ``(provider, identifier, object)`` so repeated lookups are O(1) and return the same copy.
    """

    if not isinstance(capabilities, ModelCapabilities) or not isinstance(identifier, str):
        return capabilities  # a stand-in object (a test double) is not something this check can reason about
    key = (provider, identifier, id(capabilities))
    with _state_lock:
        cached = _apply_cache.get(key)
    if cached is not None and cached[0] is capabilities:
        return cached[1]

    changes: dict[str, Any] = {}
    attributes: dict[str, Any] = {}
    stand_in = bool(getattr(capabilities, "_is_generic", False))
    cap = capabilities.max_output_tokens
    best = combine_output_caps(published_output_caps(provider, identifier)) if cap > 0 else None
    if best is not None:
        if stand_in:
            # The generic capability's cap is a made-up default, not a limit: it is never enforced, and a published
            # figure is recorded beside it (``_published_output_cap``) for ``_bound_output_tokens`` to enforce.
            attributes["_published_output_cap"] = best[0]
        if best[0] < cap:
            changes["max_output_tokens"] = best[0]
            warn_once(
                ("published-output-cap", provider.value, identifier, cap, best[0]),
                "Model %s %s: the %s output cap %d is above what %s publishes (%d); %d is used",
                provider.value,
                identifier,
                "default (stand-in)" if stand_in else "curated",
                cap,
                best[1],
                best[0],
                best[0],
            )
    temperature_changes, provenance = bound_temperature(provider, identifier, capabilities)
    changes.update(temperature_changes)
    if temperature_changes:
        attributes["_temperature_source"] = provenance
        constraint = temperature_changes.get("temperature_constraint")
        _info_once(
            ("published-temperature", provider.value, identifier, provenance),
            "Model %s %s: temperature rules follow the %s (%s)",
            provider.value,
            identifier,
            "LiteLLM supports_sampling_params flag" if provenance == "litellm" else "vendor fallback table",
            constraint.get_description() if constraint is not None else "no temperature",
        )

    if not changes and not attributes:
        result = capabilities
    else:
        result = dataclasses.replace(capabilities, **changes)
        # Attributes that are not dataclass fields (``_is_generic``) belong to the object, not to the fields.
        names = {f.name for f in dataclasses.fields(ModelCapabilities)}
        for attribute, value in vars(capabilities).items():
            if attribute not in names:
                setattr(result, attribute, value)
        for attribute, value in attributes.items():
            setattr(result, attribute, value)
    with _state_lock:
        if len(_apply_cache) >= _APPLY_CACHE_LIMIT:
            _apply_cache.clear()
        _apply_cache[key] = (capabilities, result)
    return result


# ----------------------------------------------------------------------
# The extract (written by the refresh script, validated by the loader)
# ----------------------------------------------------------------------


def _wanted_key(entry: Mapping[str, Any]) -> bool:
    provider = entry.get("litellm_provider")
    if not isinstance(provider, str):
        return False
    return (provider in _EXTRACT_PROVIDERS or provider.startswith("vertex_ai")) and entry.get("mode") in _EXTRACT_MODES


def build_extract(catalog: Mapping[str, Any], *, fetched_at: str) -> dict[str, Any]:
    """The extract document for a LiteLLM model map: only the keys of providers this repo can serve, only what the
    checks read (a valid output cap, and the sampling flag when it is false), sorted for a stable diff.

    The cap is ``max_output_tokens``; when a key has none, the legacy ``max_tokens`` stands in unless it equals the
    key's ``max_input_tokens`` (LiteLLM then means the input window, not an output cap). A figure that is not a
    believable cap is left out, exactly as the loader would ignore it.
    """

    entries: dict[str, dict[str, Any]] = {}
    for key in sorted(catalog):
        raw = catalog[key]
        if not isinstance(key, str) or not isinstance(raw, Mapping) or not _wanted_key(raw):
            continue
        cap = valid_output_limit(raw.get("max_output_tokens"))
        if cap is None:
            legacy = valid_output_limit(raw.get("max_tokens"))
            if legacy is not None and legacy != valid_output_limit(raw.get("max_input_tokens")):
                cap = legacy
        record: dict[str, Any] = {"provider": raw["litellm_provider"]}
        if cap is not None:
            record["max_output_tokens"] = cap
        if raw.get("supports_sampling_params") is False:
            record["supports_sampling_params"] = False
        if len(record) > 1:
            entries[key] = record
    return {
        "format": 1,
        "source": EXTRACT_SOURCE_URL,
        "fetched_at": fetched_at,
        "entries": entries,
    }
