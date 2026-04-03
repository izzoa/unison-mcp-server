#!/usr/bin/env python3
"""Fetch the latest LiteLLM model catalog and merge into conf/*.json.

This script is designed to run in CI (weekly cron) or locally. It:
1. Fetches the live model_prices_and_context_window.json from LiteLLM's GitHub
2. Filters to chat-mode models for each supported provider
3. Merges with existing conf/*.json — updating capacity fields for known models,
   adding new models with inferred defaults, never deleting or overwriting curated fields
4. Writes updated JSON and prints a diff summary

Exit codes:
    0 — changes written
    1 — error
    2 — no changes detected

Usage:
    python scripts/refresh_litellm_models.py
    python scripts/refresh_litellm_models.py --dry-run
    python scripts/refresh_litellm_models.py --output-summary summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is importable so we can reuse litellm_adapter inference
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from providers.litellm_adapter import (  # noqa: E402
    infer_aliases,
    infer_intelligence_score,
    infer_temperature_constraint,
    infer_thinking_support,
)
from providers.shared.provider_type import ProviderType  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)

CONF_DIR = REPO_ROOT / "conf"

# Map litellm_provider values → (ProviderType, conf filename)
PROVIDER_MAP: dict[str, tuple[ProviderType, str]] = {
    "openai": (ProviderType.OPENAI, "openai_models.json"),
    "gemini": (ProviderType.GOOGLE, "gemini_models.json"),
    "azure": (ProviderType.AZURE, "azure_models.json"),
    "xai": (ProviderType.XAI, "xai_models.json"),
    "openrouter": (ProviderType.OPENROUTER, "openrouter_models.json"),
    "ollama": (ProviderType.CUSTOM, "custom_models.json"),
}

# Fields that LiteLLM is authoritative for (capacity/capability)
LITELLM_FIELDS = {
    "context_window",
    "max_output_tokens",
    "supports_images",
    "supports_function_calling",
    "supports_json_mode",
}

# Fields that are curated and must never be overwritten
CURATED_FIELDS = {
    "intelligence_score",
    "aliases",
    "friendly_name",
    "description",
    "max_thinking_tokens",
    "temperature_constraint",
    "supports_temperature",
    "allow_code_generation",
    "use_openai_response_api",
    "default_reasoning_effort",
    "max_image_size_mb",
}

# Provider-friendly labels for new model descriptions
PROVIDER_LABELS = {
    ProviderType.OPENAI: "OpenAI",
    ProviderType.GOOGLE: "Gemini",
    ProviderType.AZURE: "Azure",
    ProviderType.XAI: "X.AI",
    ProviderType.OPENROUTER: "OpenRouter",
    ProviderType.CUSTOM: "Custom",
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_litellm_catalog(max_retries: int = 2) -> dict:
    """Fetch the live LiteLLM model catalog JSON. Retries once on failure."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(LITELLM_URL, headers={"User-Agent": "unison-mcp-refresh/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Expected top-level dict from LiteLLM JSON")
            return data
        except (urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            if attempt < max_retries - 1:
                print(f"Fetch attempt {attempt + 1} failed: {exc}. Retrying in 30s...", file=sys.stderr)
                time.sleep(30)
            else:
                print(f"ERROR: Failed to fetch LiteLLM catalog after {max_retries} attempts: {exc}", file=sys.stderr)
                raise


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
def filter_models_by_provider(catalog: dict) -> dict[str, list[dict]]:
    """Filter catalog to chat-mode models grouped by conf filename.

    Returns: {conf_filename: [list of model dicts with flat names]}
    """
    result: dict[str, list[dict]] = {}

    for full_key, entry in catalog.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("mode") != "chat":
            continue

        litellm_provider = entry.get("litellm_provider")
        if litellm_provider not in PROVIDER_MAP:
            continue

        provider_type, conf_file = PROVIDER_MAP[litellm_provider]

        # Strip provider prefix from model name
        flat_name = full_key
        for prefix in (f"{litellm_provider}/", ):
            if flat_name.startswith(prefix):
                flat_name = flat_name[len(prefix):]
                break

        model_info = {
            "model_name": flat_name,
            "context_window": entry.get("max_input_tokens") or 0,
            "max_output_tokens": entry.get("max_output_tokens") or 0,
            "supports_images": bool(entry.get("supports_vision", False)),
            "supports_function_calling": bool(entry.get("supports_function_calling", False)),
            "supports_json_mode": bool(entry.get("supports_response_schema", False)),
            "_supports_reasoning": bool(entry.get("supports_reasoning", False)),
            "_provider_type": provider_type,
        }

        result.setdefault(conf_file, []).append(model_info)

    return result


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
class MergeResult:
    """Tracks changes made during a merge for summary generation."""

    def __init__(self, conf_file: str):
        self.conf_file = conf_file
        self.updated: list[dict] = []  # {"model": name, "field": f, "old": v, "new": v}
        self.added: list[str] = []
        self.not_in_litellm: list[str] = []

    @property
    def has_changes(self) -> bool:
        return bool(self.updated or self.added)


def merge_provider(
    conf_file: str,
    existing_data: dict,
    litellm_models: list[dict],
    provider_type: ProviderType,
) -> tuple[dict, MergeResult]:
    """Merge LiteLLM models into an existing conf JSON structure.

    Returns (updated_data, merge_result).
    """
    result = MergeResult(conf_file)

    models = list(existing_data.get("models", []))
    existing_by_name = {m["model_name"]: m for m in models}
    litellm_by_name = {m["model_name"]: m for m in litellm_models}

    # 1. Update existing models (capacity/capability fields only)
    for model in models:
        name = model["model_name"]
        litellm_entry = litellm_by_name.get(name)
        if not litellm_entry:
            result.not_in_litellm.append(name)
            continue

        for field in LITELLM_FIELDS:
            new_val = litellm_entry.get(field)
            if new_val is None or new_val == 0 or new_val is False:
                continue
            old_val = model.get(field)
            if old_val != new_val:
                result.updated.append({
                    "model": name,
                    "field": field,
                    "old": old_val,
                    "new": new_val,
                })
                model[field] = new_val

    # 2. Add new models
    for name, litellm_entry in litellm_by_name.items():
        if name in existing_by_name:
            continue

        info = {
            "context_window": litellm_entry.get("context_window", 0),
            "supports_images": litellm_entry.get("supports_images", False),
            "_litellm_supports_reasoning": litellm_entry.get("_supports_reasoning", False),
        }

        score = infer_intelligence_score(name, info)
        aliases = infer_aliases(name)
        thinks, think_tokens = infer_thinking_support(name, info)
        supports_temp, temp_constraint = infer_temperature_constraint(name, info)

        label = PROVIDER_LABELS.get(provider_type, "Unknown")

        new_entry: dict = {
            "model_name": name,
            "friendly_name": f"{label} ({name})",
            "aliases": aliases,
            "intelligence_score": score,
            "description": "Auto-discovered via LiteLLM refresh",
            "context_window": litellm_entry.get("context_window", 0),
            "max_output_tokens": litellm_entry.get("max_output_tokens", 0),
            "supports_images": litellm_entry.get("supports_images", False),
            "supports_function_calling": litellm_entry.get("supports_function_calling", False),
            "supports_json_mode": litellm_entry.get("supports_json_mode", False),
        }

        if thinks:
            new_entry["supports_extended_thinking"] = True
            new_entry["max_thinking_tokens"] = think_tokens

        if not supports_temp:
            new_entry["supports_temperature"] = False
            new_entry["temperature_constraint"] = "fixed"

        models.append(new_entry)
        result.added.append(name)

    updated_data = dict(existing_data)
    updated_data["models"] = models
    return updated_data, result


# ---------------------------------------------------------------------------
# Diff summary
# ---------------------------------------------------------------------------
def generate_summary(results: list[MergeResult]) -> str:
    """Generate a markdown diff summary for use as a PR body."""
    lines = ["## Model Catalog Refresh Summary\n"]

    total_updated = sum(len(r.updated) for r in results)
    total_added = sum(len(r.added) for r in results)
    total_missing = sum(len(r.not_in_litellm) for r in results)

    lines.append(f"**Updated fields**: {total_updated} | "
                 f"**New models**: {total_added} | "
                 f"**Not in LiteLLM**: {total_missing}\n")

    for r in results:
        if not r.updated and not r.added and not r.not_in_litellm:
            continue

        lines.append(f"### `{r.conf_file}`\n")

        if r.updated:
            lines.append("**Updated:**")
            for u in r.updated:
                lines.append(f"- `{u['model']}`.{u['field']}: `{u['old']}` → `{u['new']}`")
            lines.append("")

        if r.added:
            lines.append("**New models (needs curation):**")
            for name in r.added:
                lines.append(f"- `{name}` — inferred defaults, review intelligence_score and aliases")
            lines.append("")

        if r.not_in_litellm:
            lines.append("**Not found in LiteLLM** (may be pre-release, deprecated, or custom):")
            for name in r.not_in_litellm:
                lines.append(f"- `{name}`")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh model catalog from LiteLLM")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing files")
    parser.add_argument("--output-summary", type=str, help="Write summary to this file path")
    args = parser.parse_args()

    # Fetch
    try:
        catalog = fetch_litellm_catalog()
    except Exception:
        return 1

    print(f"Fetched {len(catalog)} entries from LiteLLM catalog")

    # Filter by provider
    provider_models = filter_models_by_provider(catalog)
    print(f"Filtered to {sum(len(v) for v in provider_models.values())} chat models "
          f"across {len(provider_models)} providers")

    # Merge each provider
    all_results: list[MergeResult] = []
    updated_files: dict[str, dict] = {}

    for conf_file, litellm_models in sorted(provider_models.items()):
        conf_path = CONF_DIR / conf_file
        if not conf_path.exists():
            print(f"  SKIP {conf_file}: file not found")
            continue

        with open(conf_path) as f:
            existing_data = json.load(f)

        provider_type = PROVIDER_MAP[next(
            k for k, (_, fname) in PROVIDER_MAP.items() if fname == conf_file
        )][0]

        updated_data, result = merge_provider(conf_file, existing_data, litellm_models, provider_type)
        all_results.append(result)

        if result.has_changes:
            updated_files[conf_file] = updated_data

        status = []
        if result.updated:
            status.append(f"{len(result.updated)} field updates")
        if result.added:
            status.append(f"{len(result.added)} new models")
        if result.not_in_litellm:
            status.append(f"{len(result.not_in_litellm)} not in LiteLLM")

        print(f"  {conf_file}: {', '.join(status) if status else 'no changes'}")

    # Generate summary
    summary = generate_summary(all_results)

    if args.output_summary:
        Path(args.output_summary).write_text(summary)
        print(f"Summary written to {args.output_summary}")

    if not updated_files:
        print("\nNo changes detected.")
        if not args.output_summary:
            print(summary)
        return 2

    # Write files (unless dry-run)
    if args.dry_run:
        print("\n[DRY RUN] Would update the following files:")
        for conf_file in updated_files:
            print(f"  conf/{conf_file}")
        print("\n" + summary)
        return 0

    for conf_file, data in updated_files.items():
        conf_path = CONF_DIR / conf_file
        with open(conf_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"  Wrote {conf_file}")

    print(f"\n{len(updated_files)} file(s) updated.")
    if not args.output_summary:
        print("\n" + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
