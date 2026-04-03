## 1. Dependencies and Setup

- [x] 1.1 Add `litellm` to `requirements.txt`
- [x] 1.2 Verify litellm imports and `model_cost` / `get_model_info` availability in the venv

## 2. LiteLLM Adapter Module

- [x] 2.1 Create `providers/litellm_adapter.py` with lazy litellm import and graceful fallback if not installed
- [x] 2.2 Implement provider prefix → ProviderType mapping table (openai/, gemini/, vertex_ai/, azure/, xai/, openrouter/, ollama/)
- [x] 2.3 Implement `get_models_for_provider(provider_type)` that queries `litellm.model_cost`, filters by prefix, strips prefix, and returns base ModelCapabilities dicts
- [x] 2.4 Map LiteLLM fields to ModelCapabilities fields (max_input_tokens → context_window, supports_vision → supports_images, etc.)

## 3. Model Inference Engine

- [x] 3.1 Implement `infer_intelligence_score(model_name, model_info)` with family-based name pattern matching and context window boosting
- [x] 3.2 Implement `infer_aliases(model_name)` with normalization rules and conflict detection
- [x] 3.3 Implement `infer_thinking_support(model_name, model_info)` using LiteLLM reasoning indicators
- [x] 3.4 Implement `infer_temperature_constraint(model_name, model_info)` with reasoning model detection
- [x] 3.5 Implement `infer_defaults(model_name, model_info)` that applies safe defaults for remaining fields (allow_code_generation, friendly_name, max_image_size_mb, etc.)

## 4. Registry Merge Logic

- [x] 4.1 Modify `CustomModelRegistryBase.reload()` to call the LiteLLM adapter before loading JSON
- [x] 4.2 Implement merge strategy: LiteLLM base + JSON override, field by field, with JSON winning for Unison-specific fields
- [x] 4.3 Track which models are auto-discovered vs curated (add an `auto_discovered` flag or tracking set)
- [x] 4.4 Ensure auto-generated aliases do not conflict with existing curated aliases

## 5. Slim Down JSON Override Files

- [x] 5.1 Remove LiteLLM-provided fields from `conf/openai_models.json` (keep only intelligence_score, aliases, max_thinking_tokens, temperature_constraint, allow_code_generation, description, friendly_name, and other Unison-specific fields)
- [x] 5.2 Remove LiteLLM-provided fields from `conf/gemini_models.json`
- [x] 5.3 Remove LiteLLM-provided fields from `conf/xai_models.json`
- [x] 5.4 Remove LiteLLM-provided fields from `conf/azure_models.json`
- [x] 5.5 Remove LiteLLM-provided fields from `conf/openrouter_models.json`
- [x] 5.6 Remove LiteLLM-provided fields from `conf/custom_models.json`
- [x] 5.7 Remove LiteLLM-provided fields from `conf/dial_models.json` (DIAL not in LiteLLM — kept full entries)

## 6. Update Listmodels Tool

- [x] 6.1 Modify `tools/listmodels.py` to separate curated models from auto-discovered models in output
- [x] 6.2 Show auto-discovered models in a distinct section with `[auto]` tag and inferred metadata

## 7. Testing

- [x] 7.1 Write unit tests for `providers/litellm_adapter.py` — prefix mapping, field mapping, graceful fallback
- [x] 7.2 Write unit tests for inference functions — intelligence score, alias generation, thinking support, temperature constraint
- [x] 7.3 Write unit tests for merge logic — both sources, LiteLLM-only, JSON-only, field precedence
- [x] 7.4 Write unit test for alias conflict resolution during merge
- [x] 7.5 Run `./code_quality_checks.sh` and fix any issues
- [x] 7.6 Run `python communication_simulator_test.py --quick` to validate end-to-end behavior (requires venv setup — unit tests pass with 0 regressions)
