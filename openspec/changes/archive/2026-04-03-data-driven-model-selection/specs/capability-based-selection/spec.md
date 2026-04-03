## ADDED Requirements

### Requirement: Shared capability-based model selection on base provider
The `ModelProvider` base class SHALL provide a `select_preferred_model(category, allowed_models)` method that selects the best model from `allowed_models` based on `ModelCapabilities` metadata.

#### Scenario: Extended reasoning category
- **WHEN** `select_preferred_model(EXTENDED_REASONING, allowed_models)` is called
- **THEN** the method returns the model with the highest `intelligence_score` among those with `supports_extended_thinking == True`, or the highest `intelligence_score` overall if none support extended thinking

#### Scenario: Fast response category
- **WHEN** `select_preferred_model(FAST_RESPONSE, allowed_models)` is called
- **THEN** the method returns the highest-scored model whose name contains a fast-tier indicator ("flash", "mini", "lite", "fast"), or the lowest-scored model if no fast-tier models exist

#### Scenario: Balanced category
- **WHEN** `select_preferred_model(BALANCED, allowed_models)` is called
- **THEN** the method returns the model with the highest `intelligence_score` from all allowed models

#### Scenario: Single model available
- **WHEN** `allowed_models` contains exactly one model
- **THEN** the method returns that model regardless of category

#### Scenario: Empty allowed list
- **WHEN** `allowed_models` is empty
- **THEN** the method returns None

### Requirement: Providers use capability-based selection by default
All provider `get_preferred_model()` implementations SHALL delegate to `select_preferred_model()` instead of using hardcoded model name lists.

#### Scenario: OpenAI provider selects by capability
- **WHEN** `OpenAIModelProvider.get_preferred_model(EXTENDED_REASONING, allowed_models)` is called
- **THEN** the result is determined by `intelligence_score` and `supports_extended_thinking`, not by a hardcoded preference list

#### Scenario: XAI provider no longer uses PRIMARY_MODEL constant
- **WHEN** `XAIModelProvider.get_preferred_model(category, allowed_models)` is called
- **THEN** the result is determined by capability metadata, and the `PRIMARY_MODEL` / `FALLBACK_MODEL` constants are removed

### Requirement: Dynamic ultimate fallback in registry
`ModelProviderRegistry.get_preferred_fallback_model()` SHALL NOT contain a hardcoded model name as the ultimate fallback.

#### Scenario: No provider returns a preference
- **WHEN** no provider's `get_preferred_model()` returns a result but models are available
- **THEN** the registry returns the first model (alphabetically sorted) from the first provider with available models

#### Scenario: No models available at all
- **WHEN** no providers have any available models
- **THEN** the registry raises a `ValueError` instead of returning a hardcoded model name
