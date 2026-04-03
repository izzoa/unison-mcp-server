## ADDED Requirements

### Requirement: LiteLLM adapter provides model metadata by provider
The system SHALL include a `providers/litellm_adapter.py` module that queries LiteLLM's bundled `model_cost` dictionary and returns model metadata grouped by Unison ProviderType. The adapter SHALL map LiteLLM's `provider/model` naming to flat model names by stripping the provider prefix.

#### Scenario: Retrieve OpenAI models from LiteLLM
- **WHEN** the adapter is queried for ProviderType.OPENAI
- **THEN** it returns all models with the `openai/` prefix from `litellm.model_cost`, with the prefix stripped (e.g., `openai/gpt-5` → `gpt-5`), and each entry includes context_window, max_output_tokens, supports_vision, supports_function_calling, and supports_json_mode fields

#### Scenario: LiteLLM not installed or import fails
- **WHEN** the `litellm` package is not installed or fails to import
- **THEN** the adapter returns empty results for all providers and logs a warning, allowing the system to fall back to JSON-only behavior

### Requirement: Registry reload merges LiteLLM and JSON data
The `CustomModelRegistryBase.reload()` method SHALL merge models from LiteLLM discovery with models from `conf/*.json` override files. LiteLLM SHALL be authoritative for capacity and capability fields it provides. JSON overrides SHALL be authoritative for Unison-specific fields (intelligence_score, aliases, max_thinking_tokens, temperature_constraint, allow_code_generation, default_reasoning_effort, use_openai_response_api, friendly_name, description).

#### Scenario: Model exists in both LiteLLM and JSON
- **WHEN** a model (e.g., `gpt-5`) is found in both LiteLLM data and `conf/openai_models.json`
- **THEN** the resulting ModelCapabilities uses LiteLLM values for context_window, max_output_tokens, supports_images, supports_function_calling, and supports_json_mode, and uses JSON values for intelligence_score, aliases, max_thinking_tokens, temperature_constraint, allow_code_generation, and any other Unison-specific fields set in the JSON

#### Scenario: Model exists only in LiteLLM (auto-discovered)
- **WHEN** a model exists in LiteLLM data but has no entry in the corresponding `conf/*.json` file
- **THEN** the model is added to the registry with LiteLLM-provided fields and inferred defaults for Unison-specific fields, and is marked as auto-discovered

#### Scenario: Model exists only in JSON (custom/pre-release)
- **WHEN** a model exists in `conf/*.json` but not in LiteLLM data
- **THEN** the model is loaded entirely from JSON as it is today, with no changes to behavior

### Requirement: Provider prefix mapping is configurable
The system SHALL maintain a mapping from LiteLLM model prefixes to Unison ProviderType values. This mapping SHALL be defined in the adapter module and cover at minimum: `openai/` → OPENAI, `gemini/` → GOOGLE, `vertex_ai/` → GOOGLE, `azure/` → AZURE, `xai/` → XAI, `openrouter/` → OPENROUTER, `ollama/` → CUSTOM.

#### Scenario: Gemini models discovered via both prefixes
- **WHEN** LiteLLM contains models under both `gemini/` and `vertex_ai/` prefixes
- **THEN** both are mapped to ProviderType.GOOGLE and available in the Gemini registry, with duplicates resolved by canonical model name

### Requirement: Listmodels distinguishes curated from auto-discovered models
The `listmodels` tool SHALL visually distinguish models that have JSON override entries (curated) from models discovered only via LiteLLM (auto-discovered).

#### Scenario: Provider has both curated and auto-discovered models
- **WHEN** OpenAI has 3 models with JSON overrides and 2 models discovered only via LiteLLM
- **THEN** the listmodels output shows curated models in the main list with full metadata, and auto-discovered models in a separate section marked as auto-discovered with inferred metadata

#### Scenario: Provider has only curated models
- **WHEN** all models for a provider have JSON override entries
- **THEN** no auto-discovered section is shown for that provider

### Requirement: Graceful degradation without LiteLLM
The system SHALL function identically to its current behavior if the `litellm` package is not available. JSON files remain the sole source of model metadata in that case.

#### Scenario: System starts without litellm installed
- **WHEN** the server starts and `litellm` cannot be imported
- **THEN** all registries load from `conf/*.json` exactly as they do today, a warning is logged once, and no errors are raised
