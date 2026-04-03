## Why

There are 303 hardcoded model name references across 40 test files (e.g., `assert model == "gemini-2.5-flash"`, `assert "gpt-5.1-codex" in capabilities`). Every time a model is renamed, replaced, or re-ranked — whether by a manual JSON edit, the weekly LiteLLM refresh, or the data-driven selection change — dozens of tests break despite no actual regression. Tests should verify behavior and properties ("auto-mode picks a valid model with extended thinking support") rather than specific model names ("auto-mode picks gpt-5.1-codex").

## What Changes

- Refactor test assertions across ~40 files to assert **properties** instead of **exact model names**:
  - "Model exists" assertions → check that a model with the expected trait exists (e.g., `any("pro" in m for m in caps)`)
  - "Alias resolves" assertions → check that the alias resolves to a valid model (`resolve("pro") is not None`)
  - "Auto-mode picks" assertions → check that the selected model is valid and has the expected capabilities (`provider.validate_model_name(model)` and capability checks)
  - "Capabilities match" assertions → check ranges or minimums instead of exact values (`caps.context_window >= 100_000`)
- Extract common model name references into test fixtures or helper constants derived from the live registry, so the few tests that legitimately need a specific model name get it dynamically
- Add a `tests/model_test_helpers.py` module with utilities like `get_any_pro_model(provider)`, `get_any_flash_model(provider)`, `get_flagship_model(provider)` that query the actual registry

## Capabilities

### New Capabilities
- `test-model-helpers`: Shared test utilities for dynamically querying model names and capabilities from the live registry, replacing hardcoded model name constants

### Modified Capabilities

## Impact

- **Test files**: ~40 files in `tests/` modified. Purely test-side changes — no production code affected.
- **Test behavior**: Tests become resilient to model catalog updates. A weekly LiteLLM refresh PR that adds/updates models will no longer break unrelated tests.
- **Test coverage**: No reduction in coverage. Tests still verify the same behaviors (alias resolution works, auto-mode selects appropriately, capabilities are structured correctly) — just without asserting specific model names.
- **Dependency**: Best done after the `data-driven-model-selection` change lands, since that change will also shift which models get selected in auto-mode tests.
