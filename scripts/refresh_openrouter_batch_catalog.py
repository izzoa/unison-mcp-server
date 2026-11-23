#!/usr/bin/env python3
"""Refresh the committed snapshot of OpenRouter's batch listings and report catalog drift.

OpenRouter lists a batch-priced copy of a model as a separate entry whose id ends in ``:batch``.
``conf/openrouter_batch_catalog.json`` is the offline fallback for the live fetch the server makes
(see providers/shared/batch_catalog.py). This script:

1. fetches OpenRouter's public model list (``GET /api/v1/models``, no API key); a body that lists models
   but no ``:batch`` entry is treated as degraded and refused, and so is a list that shrinks the snapshot
   to less than half its size (pass ``--allow-shrink`` to accept a real delisting);
2. rewrites the snapshot (format 2): the sorted base slug of every ``<slug>:batch`` entry, and under
   ``model_data`` the capability view of each (context length, output cap, supported parameters, modalities;
   the same reduction the server applies to a live body, so the offline fallback has the same capabilities).
   The file is deterministic (sorted keys, one compact line per slug) and small; a body that drops ANY ONE
   capability field (context length, output cap, supported parameters, modalities) for more than half the slugs that
   had it is refused like a body that drops the slugs
   (``--allow-shrink`` overrides both);
3. reports drift between the list and the curated ``conf/openrouter_models.json``: a
   ``supports_batch`` flag that disagrees with the list, a ``batch_family`` rule that now matches
   nothing, and an alias whose curated pin is older than its family's latest batch-listed model; and per-slug
   capability drift against the previous snapshot (a slug whose temperature support, context length or other
   viewed field changed).

Exit codes (as scripts/refresh_litellm_models.py):
    0 - snapshot written (or, with --dry-run, would change)
    1 - error (nothing is written)
    2 - no change

Usage:
    python scripts/refresh_openrouter_batch_catalog.py
    python scripts/refresh_openrouter_batch_catalog.py --dry-run
    python scripts/refresh_openrouter_batch_catalog.py --allow-shrink
    python scripts/refresh_openrouter_batch_catalog.py --output-summary summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from providers.shared.batch_catalog import (  # noqa: E402
    BATCH_CATALOG_URL,
    VIEW_FIELDS,
    BatchCatalog,
    BatchCatalogFormatError,
    BatchFamilyRule,
    fetch_models_payload,
    parse_catalog_payload,
    validate_view,
)

SNAPSHOT_PATH = REPO_ROOT / "conf" / "openrouter_batch_catalog.json"
MODELS_PATH = REPO_ROOT / "conf" / "openrouter_models.json"

Fetcher = Callable[[], Any]

#: A refreshed list smaller than this fraction of the current snapshot is presumed degraded, not real.
SHRINK_LIMIT = 0.5


def fetch_models(timeout: float = 30.0) -> Any:
    """OpenRouter's public model list, decoded (no credentials; size and time bounded like the server's fetch)."""

    return fetch_models_payload(timeout)


#: The snapshot's schema: 1 held slugs only, 2 adds ``model_data`` (the capability view of every slug).
SNAPSHOT_FORMAT = 2

#: Most per-slug capability drift lines the report prints; the rest are counted.
MAX_DRIFT_LINES = 40


def build_snapshot(catalog: BatchCatalog, *, fetched_at: str) -> dict[str, Any]:
    """The snapshot document for ``catalog`` (slugs sorted, none ending in ``:batch``; a view for each)."""

    return {
        "format": SNAPSHOT_FORMAT,
        "fetched_at": fetched_at,
        "source": BATCH_CATALOG_URL,
        "note": (
            "Base slugs of every '<slug>:batch' listing in OpenRouter's public model list, and under model_data the "
            "capability view of each (context_length, max_completion_tokens, supported_parameters, modalities). "
            "Offline fallback for providers/shared/batch_catalog.py; refresh with "
            "scripts/refresh_openrouter_batch_catalog.py."
        ),
        "batch_models": sorted(catalog.slugs),
        "model_data": {slug: dict(catalog.views[slug]) for slug in sorted(catalog.slugs) if slug in catalog.views},
    }


def render_snapshot(document: dict[str, Any]) -> str:
    """The snapshot as text: deterministic, with each ``model_data`` entry on one compact line with sorted keys."""

    model_data = document["model_data"]
    head = {key: document[key] for key in ("format", "fetched_at", "source", "note", "batch_models")}
    text = json.dumps(head, indent=2)
    entries = ",\n".join(
        f"    {json.dumps(slug)}: {json.dumps(model_data[slug], sort_keys=True, separators=(',', ':'))}"
        for slug in sorted(model_data)
    )
    return text[:-2] + ',\n  "model_data": {\n' + entries + "\n  }\n}\n"


def _previous_snapshot(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The slugs and views of the snapshot on disk (empty when missing or unreadable; views re-validated)."""

    if not path.exists():
        return [], {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], {}
    slugs = [slug for slug in document.get("batch_models", []) if isinstance(slug, str)]
    raw = document.get("model_data")
    views = {slug: validate_view(raw[slug]) for slug in slugs if isinstance(raw, dict) and slug in raw}
    return slugs, views


def lost_fields(previous_views: Mapping[str, Mapping[str, Any]], catalog: BatchCatalog) -> dict[str, tuple[int, int]]:
    """Per view field, ``(had, lost)`` for each field a refresh would drop for more than half of the slugs that had it.

    A field is *lost* for a slug when the previous snapshot had a value and the new body has none. The check is per
    field, not per slug: a degraded body that drops ONE family of fields for every slug (``supported_parameters``,
    the output caps, the modalities) leaves each slug with some other field, yet it would silently revert the
    offline fallback for the dropped family (for ``supported_parameters``, to the legacy path that sends a
    temperature the model may not take). Only slugs still listed are counted; delistings are the slug-count
    guard's business.
    """

    report: dict[str, tuple[int, int]] = {}
    for name in VIEW_FIELDS:
        had = [slug for slug, view in previous_views.items() if view.get(name) is not None]
        lost = [slug for slug in had if (catalog.views.get(slug) or {}).get(name) is None]
        if had and len(lost) > len(had) * SHRINK_LIMIT:
            report[name] = (len(had), len(lost))
    return report


def capability_drift(previous: dict[str, dict[str, Any]], catalog: BatchCatalog) -> list[str]:
    """One line per slug present in both whose viewed fields changed, naming the slug and each changed field.

    A change in temperature support (the field that decides whether a parameter is sent) is named as such.
    """

    lines: list[str] = []
    for slug in sorted(set(previous) & catalog.slugs):
        old, new = previous[slug], catalog.views.get(slug, {})
        changes: list[str] = []
        old_params, new_params = old.get("supported_parameters"), new.get("supported_parameters")
        if old_params is not None and new_params is not None:
            before, after = "temperature" in old_params, "temperature" in new_params
            if before != after:
                changes.append(f"temperature support changed ({str(before).lower()} -> {str(after).lower()})")
        for name in VIEW_FIELDS:
            if old.get(name) != new.get(name) and not (name == "supported_parameters" and changes):
                changes.append(f"{name} changed ({old.get(name)} -> {new.get(name)})")
        if changes:
            lines.append(f"- `{slug}`: " + "; ".join(changes))
    return lines


def drift_report(slugs: frozenset[str], models_config: dict[str, Any]) -> list[str]:
    """Where the curated catalog disagrees with the batch listings, one line each (empty when it agrees)."""

    lowered = {slug.lower() for slug in slugs}
    lines: list[str] = []
    for entry in models_config.get("models", []):
        if not isinstance(entry, dict) or not entry.get("model_name"):
            continue
        name = str(entry["model_name"])
        listed = name.lower() in lowered
        flag = bool(entry.get("supports_batch", False))
        if flag and not listed:
            lines.append(f"- `{name}`: supports_batch is true but OpenRouter no longer lists `{name}:batch`")
        elif listed and not flag:
            lines.append(f"- `{name}`: OpenRouter lists `{name}:batch` but supports_batch is false")

        raw_rule = entry.get("batch_family")
        if raw_rule is None:
            continue
        try:
            rule = BatchFamilyRule.from_config(
                raw_rule, model_name=name, entry_aliases=[str(alias) for alias in entry.get("aliases", [])]
            )
        except BatchCatalogFormatError as exc:
            lines.append(f"- `{name}`: {exc}")
            continue
        latest = rule.select(slugs)
        if latest is None:
            lines.append(
                f"- `{name}`: the batch_family rule for {', '.join(rule.aliases)} matches no batch-listed slug"
            )
            continue
        pin_rank = rule.rank(name)
        latest_rank = rule.rank(latest)
        behind = pin_rank is None or (latest_rank is not None and pin_rank < latest_rank)
        if latest.lower() != name.lower() and behind:
            lines.append(
                f"- `{name}`: the pin behind {', '.join(rule.aliases)} is older than (or outside) the family's latest "
                f"batch model `{latest}`; under batch those aliases resolve to `{latest}`"
            )
    return lines


def _slug_diff(old: list[str], new: frozenset[str]) -> tuple[list[str], list[str]]:
    previous = set(old)
    return sorted(new - previous), sorted(previous - new)


def main(
    argv: list[str] | None = None,
    *,
    fetch: Fetcher | None = None,
    snapshot_path: Path = SNAPSHOT_PATH,
    models_path: Path = MODELS_PATH,
    today: str | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Refresh the committed OpenRouter batch catalog snapshot")
    parser.add_argument("--dry-run", action="store_true", help="Print the summary without writing the snapshot")
    parser.add_argument("--output-summary", type=str, help="Write the summary to this file path")
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help=f"Accept a list with fewer than {SHRINK_LIMIT:.0%} of the snapshot's batch models (default: refuse)",
    )
    args = parser.parse_args(argv)

    try:
        catalog = parse_catalog_payload((fetch or fetch_models)(), origin="live")
    except Exception as exc:  # transport, status, JSON or shape: nothing is written
        print(f"ERROR: could not read OpenRouter's model list: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    slugs = catalog.slugs

    previous, previous_views = _previous_snapshot(snapshot_path)
    if previous and len(slugs) < len(previous) * SHRINK_LIMIT and not args.allow_shrink:
        print(
            f"ERROR: the list has {len(slugs)} batch models against {len(previous)} in the snapshot (under "
            f"{SHRINK_LIMIT:.0%}); presuming a degraded response and writing nothing. Re-run with --allow-shrink "
            "if OpenRouter really delisted them.",
            file=sys.stderr,
        )
        return 1
    degraded = lost_fields(previous_views, catalog)
    if degraded and not args.allow_shrink:
        detail = "; ".join(f"{name} for {lost} of {had}" for name, (had, lost) in degraded.items())
        print(
            f"ERROR: the list drops capability fields that the snapshot has ({detail}; more than "
            f"{SHRINK_LIMIT:.0%} of the models that had each field); presuming a degraded response and writing "
            "nothing. Re-run with --allow-shrink if the list really changed.",
            file=sys.stderr,
        )
        return 1
    added, removed = _slug_diff(previous, slugs)
    drift_lines = capability_drift(previous_views, catalog)
    views_changed = {slug: dict(catalog.views.get(slug, {})) for slug in slugs} != {
        slug: previous_views.get(slug, {}) for slug in slugs
    }

    try:
        models_config = json.loads(models_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not read {models_path}: {exc}", file=sys.stderr)
        return 1
    drift = drift_report(slugs, models_config)

    lines = ["## OpenRouter batch catalog refresh\n", f"**Batch listings**: {len(slugs)}\n"]
    if added or removed:
        lines.append(f"**Added**: {len(added)} | **Removed**: {len(removed)}\n")
        lines.extend(f"- added `{slug}`" for slug in added)
        lines.extend(f"- removed `{slug}`" for slug in removed)
        lines.append("")
    lines.append("### Capability drift against the previous snapshot\n")
    if drift_lines:
        lines.extend(drift_lines[:MAX_DRIFT_LINES])
        if len(drift_lines) > MAX_DRIFT_LINES:
            lines.append(f"- ... and {len(drift_lines) - MAX_DRIFT_LINES} more")
    else:
        lines.append("No capability changes for the models in both lists.")
    lines.append("")
    lines.append("### Drift against conf/openrouter_models.json\n")
    lines.extend(drift or ["No drift: flags and family rules agree with the list."])
    summary = "\n".join(lines)

    if args.output_summary:
        Path(args.output_summary).write_text(summary + "\n", encoding="utf-8")
    print(summary)

    if not added and not removed and not views_changed and previous:
        print("\nNo change to the snapshot.")
        return 2
    if args.dry_run:
        print(f"\n[DRY RUN] Would write {snapshot_path}")
        return 0

    document = build_snapshot(catalog, fetched_at=today or date.today().isoformat())
    snapshot_path.write_text(render_snapshot(document), encoding="utf-8")
    print(f"\nWrote {snapshot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
