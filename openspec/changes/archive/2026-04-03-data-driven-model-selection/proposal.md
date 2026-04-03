## Why

Provider `get_preferred_model()` methods contain hardcoded model name lists (e.g., `["gpt-5.1-codex", "gpt-5.2", "gpt-5-codex"]`) that go stale every time a provider releases new models. The same problem exists in the ultimate fallback (`"gemini-2.5-flash"` hardcoded in `registry.py:433`) and in XAI's `PRIMARY_MODEL`/`FALLBACK_MODEL` constants. These lists are a manual approximation of what `intelligence_score` and capability flags already express in `ModelCapabilities`. With LiteLLM auto-discovery now adding models dynamically, hardcoded preferences become a bottleneck — new models appear in the registry but are never preferred because no one updated the preference lists.

## What Changes

- Replace hardcoded model preference lists in `OpenAIModelProvider.get_preferred_model()` with capability-based selection: sort available models by `intelligence_score`, `supports_extended_thinking`, and name-pattern heuristics (e.g., "flash"/"mini" for fast tier)
- Replace hardcoded model preference lists in `GeminiModelProvider.get_preferred_model()` with the same capability-driven approach
- Replace `PRIMARY_MODEL`/`FALLBACK_MODEL` constants and hardcoded logic in `XAIModelProvider.get_preferred_model()` with capability-based selection
- Replace the ultimate fallback `"gemini-2.5-flash"` in `ModelProviderRegistry.get_preferred_fallback_model()` with a dynamic selection: highest-ranked model from the first available provider
- Move the shared selection logic into a reusable helper on `ModelProvider` base class so all providers benefit without duplicating sorting code

## Capabilities

### New Capabilities
- `capability-based-selection`: The shared selection logic that replaces hardcoded preference lists with `intelligence_score` and capability-flag-driven model ranking per tool category

### Modified Capabilities

## Impact

- **Provider code**: `providers/openai.py`, `providers/gemini.py`, `providers/xai.py` — `get_preferred_model()` methods rewritten. `providers/registry.py` — ultimate fallback made dynamic.
- **Base class**: `providers/base.py` — new shared helper for capability-based preference selection.
- **No configuration changes**: `conf/*.json` files and `ModelCapabilities` schema are unchanged. The `intelligence_score` and capability flags are already there — this change just uses them for selection instead of hardcoded lists.
- **Behavioral change**: Auto-mode model selection will prefer the highest-scored model for each category rather than following a hand-maintained order. In practice this should be an improvement, but the exact model selected may differ from the previous hardcoded preference.
