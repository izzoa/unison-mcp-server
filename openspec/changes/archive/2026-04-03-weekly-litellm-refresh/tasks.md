## 1. Refresh Script

- [x] 1.1 Create `scripts/refresh_litellm_models.py` with argument parsing (--dry-run, --output-summary)
- [x] 1.2 Implement LiteLLM JSON fetch from `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` with retry logic and schema validation
- [x] 1.3 Implement provider filtering: filter fetched entries by `litellm_provider` field and `mode == "chat"`, mapping to Unison's 6 provider JSON files (skip DIAL)
- [x] 1.4 Implement JSON merge logic: load existing `conf/*.json`, update capacity/capability fields for known models, add new models with inferred defaults (reuse `providers/litellm_adapter.py` inference functions)
- [x] 1.5 Implement curated field protection: never overwrite intelligence_score, aliases, friendly_name, description, max_thinking_tokens, temperature_constraint, supports_temperature, allow_code_generation, use_openai_response_api, default_reasoning_effort, max_image_size_mb
- [x] 1.6 Implement diff summary generation: list updated models with old/new values, new models flagged as "needs curation", models not found in LiteLLM
- [x] 1.7 Write updated JSON files back to `conf/*.json` (preserving `_README` block and formatting)
- [x] 1.8 Exit with appropriate status codes: 0 = changes written, 1 = error, 2 = no changes

## 2. GitHub Actions Workflow

- [x] 2.1 Create `.github/workflows/litellm-refresh.yml` with weekly cron trigger (`0 6 * * 1`) and `workflow_dispatch`
- [x] 2.2 Add job steps: checkout, set up Python, run refresh script
- [x] 2.3 Add PR creation step: create branch `chore/litellm-model-refresh`, commit updated files, open PR with diff summary as body using `gh pr create`
- [x] 2.4 Add logic to detect and update an existing open refresh PR instead of creating duplicates
- [x] 2.5 Add no-op handling: skip PR creation when script exits with status 2 (no changes)

## 3. Testing

- [x] 3.1 Write unit tests for the fetch + parse logic (mock the HTTP response with sample LiteLLM JSON)
- [x] 3.2 Write unit tests for the merge logic: existing model update, new model addition, curated field preservation, missing-from-LiteLLM flagging
- [x] 3.3 Write unit tests for diff summary generation
- [x] 3.4 Write integration test: run the full script against actual `conf/*.json` in dry-run mode, verify output is valid JSON
- [x] 3.5 Run `ruff check` and fix any linting issues in the new script and tests
