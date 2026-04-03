## 1. Shared Selection Helper

- [x] 1.1 Add `select_preferred_model(category, allowed_models)` method to `ModelProvider` base class in `providers/base.py`
- [x] 1.2 Implement EXTENDED_REASONING selection: filter by `supports_extended_thinking`, sort by `intelligence_score` descending
- [x] 1.3 Implement FAST_RESPONSE selection: prefer fast-tier name patterns ("flash", "mini", "lite", "fast"), sort by `intelligence_score` within tier
- [x] 1.4 Implement BALANCED selection: sort by `intelligence_score` descending, pick highest

## 2. Provider Refactoring

- [x] 2.1 Replace `OpenAIModelProvider.get_preferred_model()` hardcoded lists with delegation to `select_preferred_model()`
- [x] 2.2 Replace `GeminiModelProvider.get_preferred_model()` hardcoded lists with delegation to `select_preferred_model()`
- [x] 2.3 Replace `XAIModelProvider.get_preferred_model()` hardcoded logic and remove `PRIMARY_MODEL` / `FALLBACK_MODEL` constants
- [x] 2.4 Replace ultimate fallback `"gemini-2.5-flash"` in `ModelProviderRegistry.get_preferred_fallback_model()` with dynamic selection

## 3. Testing

- [x] 3.1 Update existing tests that assert specific model names from `get_preferred_model()` to assert valid model properties instead
- [x] 3.2 Run full test suite and fix any regressions
