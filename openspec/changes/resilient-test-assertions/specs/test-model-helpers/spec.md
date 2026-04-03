## ADDED Requirements

### Requirement: Test helper module provides dynamic model queries
The system SHALL include a `tests/model_test_helpers.py` module with functions that query the live registry for model names, so test files can reference models dynamically instead of hardcoding names.

#### Scenario: Get any valid model for a provider
- **WHEN** a test calls `get_any_model(ProviderType.OPENAI)`
- **THEN** it returns a valid model name string from the OpenAI provider's current MODEL_CAPABILITIES

#### Scenario: Get flagship model for a provider
- **WHEN** a test calls `get_flagship_model(ProviderType.GOOGLE)`
- **THEN** it returns the model with the highest `intelligence_score` from the Gemini provider

#### Scenario: Get fast-tier model for a provider
- **WHEN** a test calls `get_flash_model(ProviderType.GOOGLE)`
- **THEN** it returns a model whose name contains "flash", "mini", "lite", or "fast"

#### Scenario: Get model with thinking support
- **WHEN** a test calls `get_model_with_thinking(ProviderType.OPENAI)`
- **THEN** it returns a model with `supports_extended_thinking == True`, or None if no such model exists

### Requirement: Test assertions use properties not hardcoded names
Tests that verify model selection, alias resolution, or capability queries SHALL assert properties and validity rather than exact model name strings.

#### Scenario: Model selection test resilient to catalog changes
- **WHEN** `conf/gemini_models.json` is updated with a new flagship model
- **THEN** all existing tests pass without modification
