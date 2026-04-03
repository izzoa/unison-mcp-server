## ADDED Requirements

### Requirement: Intelligence score inference from model name
The system SHALL infer an `intelligence_score` (1-20) for auto-discovered models based on model name pattern matching and available metadata.

#### Scenario: Model name contains "pro" or "ultra"
- **WHEN** an auto-discovered model has "pro" or "ultra" in its name
- **THEN** the inferred intelligence_score is 16

#### Scenario: Model name contains "flash" or "fast"
- **WHEN** an auto-discovered model has "flash" or "fast" in its name
- **THEN** the inferred intelligence_score is 12

#### Scenario: Model name contains "mini" or "lite"
- **WHEN** an auto-discovered model has "mini" or "lite" in its name
- **THEN** the inferred intelligence_score is 10

#### Scenario: Model name contains "nano"
- **WHEN** an auto-discovered model has "nano" in its name
- **THEN** the inferred intelligence_score is 7

#### Scenario: Model name matches no known pattern
- **WHEN** an auto-discovered model name does not match any known family pattern
- **THEN** the inferred intelligence_score defaults to 12

#### Scenario: Large context window boosts score
- **WHEN** an auto-discovered model has context_window >= 1,000,000
- **THEN** the inferred intelligence_score is at least 14, regardless of name-based inference

### Requirement: Alias generation for auto-discovered models
The system SHALL generate basic aliases for auto-discovered models by applying normalization rules to the model name.

#### Scenario: Model name with hyphens and version numbers
- **WHEN** an auto-discovered model is named `gpt-5.2-mini`
- **THEN** aliases include at minimum a version without hyphens (e.g., `gpt5.2-mini`) and a version without dots (e.g., `gpt-52-mini`)

#### Scenario: Generated alias conflicts with existing alias
- **WHEN** a generated alias for an auto-discovered model conflicts with an alias already registered to a curated model
- **THEN** the conflicting alias is discarded and the auto-discovered model is registered without that alias

### Requirement: Extended thinking inference
The system SHALL infer `supports_extended_thinking` and `max_thinking_tokens` for auto-discovered models based on LiteLLM capability data.

#### Scenario: LiteLLM indicates reasoning support
- **WHEN** LiteLLM's model info indicates reasoning/thinking support for an auto-discovered model
- **THEN** the model is created with `supports_extended_thinking=True` and `max_thinking_tokens=32768` as a safe default

#### Scenario: LiteLLM does not indicate reasoning support
- **WHEN** LiteLLM's model info does not indicate reasoning support
- **THEN** the model is created with `supports_extended_thinking=False` and `max_thinking_tokens=0`

### Requirement: Temperature constraint inference
The system SHALL infer a `temperature_constraint` for auto-discovered models based on model name patterns and LiteLLM metadata.

#### Scenario: Model name suggests reasoning-only model
- **WHEN** an auto-discovered model name contains patterns like "o3", "o4", or "-reasoning" and LiteLLM does not indicate temperature support
- **THEN** the inferred temperature_constraint is "fixed"

#### Scenario: Standard model with no special indicators
- **WHEN** an auto-discovered model has no reasoning-specific name patterns
- **THEN** the inferred temperature_constraint defaults to "range" (0.0-2.0)

### Requirement: Safe defaults for remaining fields
The system SHALL apply safe, conservative defaults for Unison-specific fields not covered by other inference rules.

#### Scenario: Auto-discovered model default field values
- **WHEN** an auto-discovered model has no JSON override
- **THEN** the following defaults are applied: `allow_code_generation=False`, `use_openai_response_api=False`, `default_reasoning_effort=None`, `max_image_size_mb=20.0` (if supports_images is True, else 0.0), `friendly_name` generated from provider name and model name
