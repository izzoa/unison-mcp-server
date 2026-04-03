## Context

The litellm-model-discovery change added runtime auto-discovery of models via `litellm.model_cost`, but the bundled pip data lags behind LiteLLM's live catalog. Meanwhile, `conf/*.json` files carry curated Unison-specific metadata (intelligence scores, aliases, thinking token budgets) that must be preserved.

LiteLLM publishes a continuously-updated JSON file at:
```
https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
```

This is the same data that gets bundled into pip releases, but fresher. The file is ~2MB, publicly accessible, and has a stable schema.

Existing CI/CD workflows: `test.yml` (PR tests), `semantic-release.yml` (push to main → version bump), `docker-release.yml` (release → Docker image). None currently modify `conf/` files.

## Goals / Non-Goals

**Goals:**
- Automatically surface new models and updated capacity/capability data weekly
- Preserve all curated Unison-specific fields (intelligence_score, aliases, max_thinking_tokens, temperature_constraint, allow_code_generation, description, friendly_name, use_openai_response_api, default_reasoning_effort)
- Generate a clear, reviewable PR with a summary of what changed
- Support manual trigger for on-demand refresh
- Keep the script deterministic — no AI, no non-determinism, no API keys required

**Non-Goals:**
- Auto-merging PRs (always requires human review)
- Inferring intelligence scores or aliases for new models (that's a separate curation tool concern)
- Replacing the runtime litellm_adapter.py discovery (this complements it by keeping JSON fresh)
- Modifying `conf/dial_models.json` (DIAL is not in LiteLLM)

## Decisions

### 1. Fetch from GitHub raw URL, not pip install

**Decision**: Fetch `model_prices_and_context_window.json` via HTTPS from GitHub, not by installing litellm and reading `model_cost`.

**Rationale**: The raw URL is always the latest commit. The pip package bundles a snapshot from the last release, which can be days to weeks behind. The script needs no dependencies beyond the standard library (`urllib` or `requests`).

**Alternative considered**: `pip install litellm && python -c "import litellm; ..."`. Rejected because it installs a large package (~10MB) just to read a JSON file, and the data is staler.

### 2. Merge strategy: update existing, add new, never delete

**Decision**: The script performs a three-way operation on each provider's JSON:
1. **Existing models (in both LiteLLM and JSON)**: Update only `_LITELLM_BASE_FIELDS` (context_window, max_output_tokens, supports_images, supports_function_calling, supports_json_mode). Leave all other fields untouched.
2. **New models (in LiteLLM but not JSON)**: Add with inferred defaults for Unison-specific fields (same heuristics as `litellm_adapter.py`). Mark clearly in the PR summary.
3. **Removed models (in JSON but not LiteLLM)**: Do NOT remove. They may be pre-release, custom, or intentionally kept. Flag in the PR summary for human review.

**Rationale**: Conservative by design. The script should never cause regressions — only additions and capacity updates. Humans decide what to remove.

### 3. Provider prefix filtering via litellm_provider field

**Decision**: Filter LiteLLM entries by the `litellm_provider` field (e.g., `"openai"`, `"gemini"`, `"xai"`) and `mode == "chat"`. Map to conf files using the same provider mapping from `litellm_adapter.py`.

**Rationale**: The `litellm_provider` field is the most reliable way to map models to our providers. Prefix-based matching (e.g., `openai/gpt-5`) can be ambiguous.

### 4. PR-based output, not direct commit

**Decision**: The workflow creates a branch, commits the changes, and opens a PR with a structured body. Never pushes directly to main.

**Rationale**: Model metadata affects auto-mode selection, fallback behavior, and tool capabilities. A bad merge could subtly break model selection. Human review catches issues like incorrect context windows or models that shouldn't be added.

### 5. Script reuses litellm_adapter.py inference logic

**Decision**: Import and reuse `infer_intelligence_score`, `infer_aliases`, `infer_thinking_support`, `infer_temperature_constraint`, and `infer_defaults` from `providers/litellm_adapter.py` for new model defaults.

**Rationale**: Avoids duplicating inference heuristics. The adapter already has tested logic for this. The script runs from the repo root, so imports work directly.

### 6. Workflow schedule: weekly on Mondays at 06:00 UTC

**Decision**: `cron: '0 6 * * 1'` with `workflow_dispatch` for manual runs.

**Rationale**: Weekly is frequent enough to catch new model releases quickly (providers rarely release more than a few models per week). Monday morning ensures the PR is ready for review at the start of the work week. Manual dispatch allows immediate refresh when a major model launch is known.

## Risks / Trade-offs

**[LiteLLM raw URL changes or goes down]** → The script retries once and exits with a clear error. The workflow is non-blocking — a failed run simply means no PR that week. No existing functionality is affected.

**[LiteLLM schema changes]** → The script validates the expected top-level structure (`models` array or key-value entries with `litellm_provider` field). If the schema doesn't match, it exits with a descriptive error rather than producing corrupt JSON.

**[PR noise]** → If nothing changed, the script detects an empty diff and skips PR creation. No unnecessary PRs.

**[New models with bad inferred defaults]** → The PR summary clearly labels new models as "needs curation" so the reviewer knows to check intelligence scores and aliases. Inferred defaults are conservative (e.g., `allow_code_generation=False`).

**[Stale PRs pile up]** → If a previous refresh PR is still open, the workflow checks for it and updates the existing PR branch instead of creating a new one.
