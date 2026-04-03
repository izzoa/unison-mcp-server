## Context

Three providers (OpenAI, Gemini, XAI) implement `get_preferred_model()` with hardcoded model name lists. When models change, these lists go stale and the "best" model for a category may not be selected. The `ModelCapabilities` dataclass already carries `intelligence_score` (1-20) and capability flags (`supports_extended_thinking`, `supports_images`, etc.) that encode the same information the hardcoded lists approximate.

## Goals / Non-Goals

**Goals:**
- Derive model preferences from capability metadata at runtime
- Eliminate all hardcoded model names from provider preference logic
- Provide a shared helper on `ModelProvider` so providers don't duplicate selection logic
- Make the ultimate fallback in `ModelProviderRegistry` dynamic

**Non-Goals:**
- Changing the `ModelCapabilities` schema or adding new fields
- Modifying `intelligence_score` values (those remain curated in JSON)
- Changing how tool categories are defined or assigned to tools

## Decisions

### 1. Shared `select_preferred_model()` on `ModelProvider` base class

**Decision**: Add a `select_preferred_model(category, allowed_models)` method to `ModelProvider` that selects the best model from `allowed_models` using capability metadata. Individual providers can still override `get_preferred_model()` if they need truly custom logic, but the default implementation covers all three categories.

**Rationale**: All three providers currently implement the same pattern (filter → sort → pick first). The only difference is which model names appear in the preference lists. Moving to capability-based selection makes this generic.

### 2. Selection strategy per category

**Decision**:
- `EXTENDED_REASONING`: Filter to models with `supports_extended_thinking == True`, then sort by `intelligence_score` descending. If none support thinking, fall back to highest `intelligence_score`.
- `FAST_RESPONSE`: Prefer models with "flash", "mini", "lite", or "fast" in the name (fast-tier indicators), then sort by `intelligence_score` descending within that tier. If no fast-tier models, fall back to lowest `intelligence_score` (cheapest).
- `BALANCED`: Sort all allowed models by `intelligence_score` descending, pick the highest.

**Rationale**: This mirrors what the hardcoded lists were doing — EXTENDED picks the smartest thinking model, FAST picks a fast-tier model, BALANCED picks the best overall. The `intelligence_score` is already curated to reflect these priorities.

### 3. Remove `PRIMARY_MODEL` / `FALLBACK_MODEL` constants from XAI

**Decision**: Delete the hardcoded constants and let the shared selection logic handle it.

### 4. Dynamic ultimate fallback

**Decision**: Replace `return "gemini-2.5-flash"` in `ModelProviderRegistry.get_preferred_fallback_model()` with: pick the highest-ranked model from all available providers using `get_capabilities_by_rank()`. If truly no models are available (edge case), raise an error rather than returning a potentially invalid hardcoded name.

## Risks / Trade-offs

**[Selection order changes]** → The exact model selected in auto-mode may differ from the previous hardcoded order. Mitigation: `intelligence_score` already reflects the intended ranking; any discrepancies indicate the scores need tuning, not that the selection logic is wrong.

**[Fast-tier heuristic]** → Name-pattern matching ("flash", "mini") is a heuristic that could miss models with unusual names. Mitigation: Auto-discovered models with unusual names will still be selected by `intelligence_score` fallback; the name heuristic only prioritizes, it doesn't exclude.
