## Context

After the `data-driven-model-selection` change, no tests are currently failing due to hardcoded model names. However, 581 hardcoded model name references remain across 51 test files. The next time `conf/*.json` is updated (manually or via the weekly LiteLLM refresh), many of these tests will break.

## Goals / Non-Goals

**Goals:**
- Create `tests/model_test_helpers.py` with utilities that query the live registry for model names
- Replace hardcoded model name strings in test assertions with calls to these helpers or property-based checks
- Ensure all tests still pass after the refactoring

**Non-Goals:**
- Modifying production code
- Changing what the tests verify — only how they express assertions
- Achieving 100% elimination of model names (some are legitimately testing specific model behavior)

## Decisions

### 1. Test helper module with dynamic model queries

**Decision**: Create `tests/model_test_helpers.py` that provides:
- `get_any_model(provider_type)` — returns any valid model name from the provider
- `get_flagship_model(provider_type)` — returns the highest-intelligence-scored model
- `get_flash_model(provider_type)` — returns a flash/fast-tier model
- `get_model_with_thinking(provider_type)` — returns a model with `supports_extended_thinking`

These query the actual `MODEL_CAPABILITIES` at test time, so they always reflect the current JSON.

### 2. Three categories of hardcoded references

**Keep as-is:** References in mock data, fixture setup, and test data that construct fake model entries (e.g., creating a mock response with `model_name="gpt-5"`). These aren't assertions about real model names — they're arbitrary test data.

**Replace with helpers:** Assertions that check specific model names from provider queries (e.g., `assert model == "gemini-2.5-flash"`). These should use property checks or dynamic helpers.

**Replace with property checks:** Assertions about capabilities (e.g., `assert caps.context_window == 400000`). These should use ranges or minimum thresholds.

### 3. Batch by test file, not by pattern

Work through files one at a time in order of most references first. This minimizes context switches and makes it easier to verify each file passes independently.

## Risks / Trade-offs

**[Weaker assertions]** → Property-based assertions are less precise than exact name matches. Mitigation: The assertions still verify the same behavior — they just allow for model names to change. A test that says "auto-mode picks a model with thinking support" is actually a better test than "auto-mode picks gpt-5.1-codex".

**[Large diff]** → 51 files changed. Mitigation: Each file is an independent, reviewable change. No production code is touched.
