## Why

Model lists are maintained as static JSON files in `conf/` that require manual updates whenever providers release new models. This creates a persistent lag between provider announcements and model availability in Unison, and imposes ongoing maintenance burden for fields that are publicly known (context windows, capability flags, output limits). LiteLLM maintains a community-curated, frequently-updated model metadata database that can serve as the authoritative source for these fields, while Unison's JSON files shrink to override-only for custom fields like intelligence scores, aliases, and thinking token budgets.

## What Changes

- Add `litellm` as a project dependency for its bundled model metadata (`model_cost`, `get_model_info`, capability checks)
- Modify the registry `reload()` flow to merge LiteLLM-discovered models with JSON override files at startup
- For known models (in both LiteLLM and JSON): LiteLLM provides base capacity/capability fields, JSON provides custom overrides (intelligence_score, aliases, max_thinking_tokens, temperature_constraint, allow_code_generation, etc.)
- For auto-discovered models (in LiteLLM but not JSON): add them with inferred defaults for custom fields based on model name heuristics (family detection, size tier)
- For JSON-only models (not in LiteLLM): keep as-is for custom/pre-release/DIAL models
- Update `listmodels` tool to visually distinguish curated models from auto-discovered ones
- Slim down `conf/*.json` files to override-only format (remove fields LiteLLM provides)

## Capabilities

### New Capabilities
- `litellm-discovery`: Integration with LiteLLM's model metadata database for automatic model discovery, field mapping, and merge logic at startup
- `model-inference`: Heuristic engine for inferring Unison-specific fields (intelligence_score, aliases, thinking support) for auto-discovered models

### Modified Capabilities

## Impact

- **Dependencies**: New dependency on `litellm` Python package (adds ~10MB). Model freshness becomes coupled to litellm package version.
- **Configuration files**: All 7 `conf/*.json` files will be restructured to override-only format. Users with custom `*_MODELS_CONFIG_PATH` overrides will need to be aware that their JSON now supplements LiteLLM rather than being the sole source.
- **Code changes**: `providers/registries/base.py` (merge logic), `tools/listmodels.py` (auto-discovered display), new module for LiteLLM integration and inference heuristics.
- **Startup**: Near-zero latency impact — LiteLLM metadata is a bundled static dict, no network calls required.
- **Provider execution**: Unchanged — existing SDK-based providers (Gemini, OpenAI, Azure, xAI, etc.) continue to handle all API calls directly.
