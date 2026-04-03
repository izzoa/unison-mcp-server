## 1. Test Helper Module

- [x] 1.1 Create `tests/model_test_helpers.py` with `get_any_model()`, `get_flagship_model()`, `get_flash_model()`, `get_model_with_thinking()` functions

## 2. Refactor Test Files (batch by highest reference count)

- [x] 2.1 Refactor `tests/test_xai_provider.py` (~70 refs)
- [x] 2.2 Refactor `tests/test_supported_models_aliases.py` (~34 refs)
- [x] 2.3 Refactor `tests/test_openai_provider.py` (~34 refs)
- [x] 2.4 Refactor `tests/test_providers.py` (~12 refs)
- [x] 2.5 Refactor remaining test files with hardcoded model names (batch the rest)

## 3. Validation

- [x] 3.1 Run full test suite and verify 0 regressions
- [x] 3.2 Run ruff check on all modified files
