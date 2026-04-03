## ADDED Requirements

### Requirement: Weekly CI workflow fetches LiteLLM model catalog
The system SHALL include a GitHub Actions workflow that runs on a weekly cron schedule (`0 6 * * 1`) and fetches the latest LiteLLM model catalog from the public GitHub raw URL.

#### Scenario: Successful weekly fetch
- **WHEN** the weekly cron triggers on Monday at 06:00 UTC
- **THEN** the workflow fetches `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` and passes the data to the refresh script

#### Scenario: Manual trigger
- **WHEN** a maintainer triggers the workflow via `workflow_dispatch`
- **THEN** the workflow runs immediately with the same behavior as the cron trigger

#### Scenario: Fetch failure
- **WHEN** the LiteLLM URL is unreachable or returns a non-200 status
- **THEN** the workflow retries once after 30 seconds, and if still failing, exits with a clear error message without creating a PR or modifying any files

### Requirement: Refresh script merges LiteLLM data with existing JSON
The system SHALL include a script at `scripts/refresh_litellm_models.py` that merges fetched LiteLLM data into the existing `conf/*.json` files. The script SHALL be deterministic and require no AI models or API keys.

#### Scenario: Existing model with updated capacity fields
- **WHEN** a model (e.g., `gpt-5`) exists in both LiteLLM data and `conf/openai_models.json`, and LiteLLM reports a different `context_window` value
- **THEN** the script updates `context_window` in the JSON file to match LiteLLM, and preserves all Unison-specific fields (intelligence_score, aliases, max_thinking_tokens, temperature_constraint, allow_code_generation, description, friendly_name, use_openai_response_api, default_reasoning_effort) unchanged

#### Scenario: New model discovered
- **WHEN** a model exists in LiteLLM data but has no entry in the corresponding `conf/*.json` file
- **THEN** the script adds a new entry with LiteLLM-provided capacity/capability fields and inferred defaults for Unison-specific fields (using the same heuristics as `providers/litellm_adapter.py`)

#### Scenario: Model in JSON but not in LiteLLM
- **WHEN** a model exists in `conf/*.json` but not in the fetched LiteLLM data
- **THEN** the script does NOT remove it, and flags it in the diff summary as "not found in LiteLLM — may be pre-release, deprecated, or custom"

#### Scenario: DIAL provider skipped
- **WHEN** the script processes provider JSON files
- **THEN** `conf/dial_models.json` is never modified because DIAL models are not present in LiteLLM

#### Scenario: No changes detected
- **WHEN** the fetched LiteLLM data produces no differences from the current `conf/*.json` files
- **THEN** the script exits successfully with a "no changes" message and the workflow does not create a PR

### Requirement: Curated fields are never overwritten
The refresh script SHALL never modify the following fields in existing model entries: `intelligence_score`, `aliases`, `friendly_name`, `description`, `max_thinking_tokens`, `temperature_constraint`, `supports_temperature`, `allow_code_generation`, `use_openai_response_api`, `default_reasoning_effort`, `max_image_size_mb`.

#### Scenario: LiteLLM data would change a curated field
- **WHEN** LiteLLM data contains a field that maps to a curated Unison-specific field for an existing model
- **THEN** the existing value in the JSON file is preserved and the LiteLLM value is ignored

### Requirement: Diff summary is generated for PR body
The script SHALL generate a structured, human-readable summary of all changes suitable for use as a PR body.

#### Scenario: Mix of updates and new models
- **WHEN** the refresh finds 2 updated models and 3 new models
- **THEN** the summary lists updated models with their changed fields and old/new values, lists new models with their inferred defaults flagged as "needs curation", and notes any models not found in LiteLLM

### Requirement: Workflow creates a PR with changes
The workflow SHALL commit updated JSON files to a dedicated branch and open a PR against `main` with the diff summary as the PR body. If an existing refresh PR is already open, the workflow SHALL update it instead of creating a duplicate.

#### Scenario: First refresh with changes
- **WHEN** the script produces changes and no existing refresh PR is open
- **THEN** the workflow creates a new branch (`chore/litellm-model-refresh`), commits the updated files, and opens a PR titled "chore: weekly model catalog refresh" with the diff summary

#### Scenario: Existing refresh PR is open
- **WHEN** the script produces changes and an existing refresh PR from a previous run is still open
- **THEN** the workflow force-pushes to the existing branch and updates the PR body with the new diff summary

#### Scenario: Script produces no changes
- **WHEN** the script exits with a "no changes" status
- **THEN** the workflow does not create or update any PR
