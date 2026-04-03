## Why

Model metadata in `conf/*.json` drifts out of date as providers release new models. The litellm-model-discovery change added runtime auto-discovery, but the curated JSON files — which carry the authoritative intelligence scores, aliases, and thinking token budgets — still require manual updates. A weekly CI cron job that fetches LiteLLM's live model catalog, merges it with existing JSON, and opens a PR would surface new models and updated specs automatically, reducing the window from "whenever someone remembers" to at most 7 days.

## What Changes

- Add a new GitHub Actions workflow (`litellm-refresh.yml`) running on a weekly cron schedule
- Create a deterministic Python script (`scripts/refresh_litellm_models.py`) that:
  - Fetches LiteLLM's `model_prices_and_context_window.json` from the GitHub raw URL (always fresher than the pip-bundled version)
  - Filters to chat-mode models for the 7 supported providers (OpenAI, Gemini, Azure, xAI, OpenRouter, Ollama, DIAL)
  - Merges with existing `conf/*.json`: updates capacity/capability fields for known models, adds new models with inferred defaults, preserves all curated Unison-specific fields
  - Generates a human-readable diff summary (new models, updated fields, removed models)
- The workflow creates a PR with the updated JSON files and summary, requiring human review before merge
- Add manual `workflow_dispatch` trigger so the refresh can be run on demand

## Capabilities

### New Capabilities
- `model-catalog-refresh`: The CI workflow, fetch script, merge logic, PR creation, and diff summary generation

### Modified Capabilities

## Impact

- **CI/CD**: New workflow file in `.github/workflows/`. Runs weekly, creates PRs against `main`. Requires `GITHUB_TOKEN` with PR write permissions (default for Actions).
- **Configuration files**: `conf/*.json` files will be updated by the script. Curated fields (intelligence_score, aliases, max_thinking_tokens, temperature_constraint, allow_code_generation, description, friendly_name) are never overwritten. Only capacity/capability fields and new model entries are touched.
- **New script**: `scripts/refresh_litellm_models.py` — standalone, no server dependencies, runs in CI with only `requests` (or `urllib`) needed.
- **No runtime changes**: The server code is unaffected. This only changes the source JSON files that feed into the existing registry system.
