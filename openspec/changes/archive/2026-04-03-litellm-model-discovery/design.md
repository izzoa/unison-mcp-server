## Context

Model metadata is currently stored in 7 static JSON files under `conf/` (one per provider), loaded once at startup by `CustomModelRegistryBase.reload()` in `providers/registries/base.py`. Each model entry carries ~20 fields of curated metadata. When providers release new models, the JSON files must be manually updated before Unison can use them.

LiteLLM ships a community-maintained model metadata database (`litellm.model_cost`) as a bundled static dict — no network calls required. It covers context windows, output limits, capability flags (vision, function calling, JSON mode), and pricing for 100+ providers. It updates with each litellm pip release, typically within hours to days of provider announcements.

The existing provider execution layer (Gemini SDK, OpenAI SDK, Azure SDK, xAI SDK, etc.) remains unchanged — this change only affects model discovery and metadata population.

## Goals / Non-Goals

**Goals:**
- Auto-discover new models at startup via LiteLLM's bundled metadata
- Use LiteLLM as the authoritative source for capacity/capability fields (context_window, max_output_tokens, supports_images, supports_function_calling, supports_json_mode)
- Preserve Unison-specific curated fields via slimmed-down JSON override files (intelligence_score, aliases, max_thinking_tokens, temperature_constraint, allow_code_generation, default_reasoning_effort, use_openai_response_api)
- Provide reasonable inferred defaults for auto-discovered models so they are usable immediately
- Distinguish curated vs auto-discovered models in `listmodels` output

**Non-Goals:**
- Replacing existing provider SDK execution with LiteLLM's `completion()` API
- Live/network-based model discovery (LiteLLM data is bundled, not fetched)
- Automatic intelligence score calibration (remains human-curated for override models)
- Changing how model restrictions or access policies work

## Decisions

### 1. LiteLLM as metadata source, not execution layer

**Decision**: Import `litellm.model_cost` and `litellm.get_model_info()` for metadata only. Do not use `litellm.completion()`.

**Rationale**: The existing provider implementations have significant custom logic — extended thinking budget management, response API routing, temperature constraint enforcement, per-provider model preferences by tool category. Replacing these with LiteLLM would require rebuilding that logic on top of LiteLLM's abstractions with uncertain benefit.

**Alternative considered**: Full LiteLLM provider replacement (Path B). Rejected because it would be a much larger change with higher risk and would lose fine-grained SDK control.

### 2. LiteLLM prefix → ProviderType mapping

**Decision**: Map LiteLLM's `provider/model` naming to Unison's flat names using a static prefix table:

| LiteLLM prefix | ProviderType | Strip prefix |
|---|---|---|
| `openai/` | OPENAI | `openai/gpt-5` → `gpt-5` |
| `gemini/` | GOOGLE | `gemini/gemini-3-pro` → `gemini-3-pro` |
| `vertex_ai/` | GOOGLE | (secondary prefix) |
| `azure/` | AZURE | `azure/gpt-5` → `gpt-5` |
| `xai/` | XAI | `xai/grok-4` → `grok-4` |
| `openrouter/` | OPENROUTER | `openrouter/...` → `...` |
| `ollama/` | CUSTOM | `ollama/llama3.2` → `llama3.2` |

**Rationale**: Keeps existing alias/model naming conventions intact. Users continue to type `gpt-5` not `openai/gpt-5`.

### 3. Merge strategy: LiteLLM base + JSON overrides

**Decision**: For each provider, during `reload()`:
1. Query `litellm.model_cost` filtered by the provider's LiteLLM prefix
2. Build base `ModelCapabilities` from LiteLLM data
3. Load `conf/*.json` as override entries
4. Merge: LiteLLM wins for capacity/capability fields it provides; JSON wins for Unison-specific fields it sets
5. Models only in LiteLLM → auto-discovered with inferred defaults
6. Models only in JSON → kept as-is (custom, pre-release, or DIAL models)

**Rationale**: LiteLLM's community-maintained data is more likely to be current than stale JSON. Unison-specific fields (intelligence_score, aliases, thinking tokens) have no upstream source and must remain curated.

**Alternative considered**: JSON always wins for all fields. Rejected per user decision — trust LiteLLM for the fields it covers.

### 4. Integration point: new LiteLLM adapter module

**Decision**: Create a new module `providers/litellm_adapter.py` that encapsulates all LiteLLM interaction. The registry base class calls into this adapter during `reload()`.

**Rationale**: Isolates the LiteLLM dependency to a single module. If LiteLLM is not installed or fails to load, the adapter returns empty results and the system falls back to JSON-only behavior (graceful degradation).

### 5. JSON files become override-only

**Decision**: Slim `conf/*.json` files to contain only fields that override or supplement LiteLLM data. Remove fields LiteLLM provides (context_window, max_output_tokens, supports_images, etc.) from JSON entries for known models.

**Rationale**: Reduces maintenance burden and eliminates a source of staleness. LiteLLM is trusted for these values per user decision.

## Risks / Trade-offs

**[LiteLLM version coupling]** → Model freshness is tied to `pip install --upgrade litellm`. Mitigation: Document the upgrade path; consider a periodic CI check that flags litellm updates.

**[LiteLLM data accuracy]** → If LiteLLM has incorrect metadata for a model, Unison inherits it. Mitigation: JSON override files can always override any LiteLLM field if needed. Log discrepancies at startup for visibility.

**[Dependency size]** → LiteLLM adds ~10MB to the install. Mitigation: The metadata functions used (`model_cost`, `get_model_info`) are lightweight; no heavy SDK imports needed at runtime.

**[Name collisions across providers]** → Two providers could have models with the same flat name after prefix stripping (e.g., `azure/gpt-5` and `openai/gpt-5` both become `gpt-5`). Mitigation: Each provider's registry only queries its own LiteLLM prefix, so collisions are provider-scoped (same as today).

**[Auto-discovered model quality]** → Inferred defaults may not be ideal (e.g., intelligence_score guess is wrong). Mitigation: Auto-discovered models are visually flagged in `listmodels` so users know they're using inferred settings. Curating a JSON override entry always takes precedence.
