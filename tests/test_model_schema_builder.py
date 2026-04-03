"""
Unit tests for the ModelSchemaBuilder class.

Tests cover model schema generation, auto-mode detection, model normalization,
context window formatting, and error message composition.
"""

from unittest.mock import MagicMock, patch


class TestModelSchemaBuilderInstantiation:
    """Test basic instantiation and attribute access."""

    def test_instantiation_with_tool_name_only(self):
        """ModelSchemaBuilder can be created with just a tool name."""
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        assert builder.tool_name == "chat"
        assert builder.tool is None

    def test_instantiation_with_mock_tool(self):
        """ModelSchemaBuilder stores the tool reference for delegation."""
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        mock_tool = MagicMock()
        mock_tool.get_name.return_value = "analyze"
        builder = ModelSchemaBuilder(tool_name="analyze", tool=mock_tool)
        assert builder.tool_name == "analyze"
        assert builder.tool is mock_tool


class TestNormalizeModelIdentifier:
    """Test the static _normalize_model_identifier method."""

    def test_simple_name(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._normalize_model_identifier("gpt-4o") == "gpt-4o"

    def test_uppercase_to_lowercase(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._normalize_model_identifier("GPT-4o") == "gpt-4o"

    def test_strips_provider_prefix(self):
        """Slash-separated provider prefixes should be stripped."""
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._normalize_model_identifier("openai/gpt-4o") == "gpt-4o"

    def test_strips_colon_suffix(self):
        """Colon-separated version suffixes should be stripped."""
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._normalize_model_identifier("llama3:latest") == "llama3"

    def test_both_slash_and_colon(self):
        """Combined provider prefix and version suffix."""
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        result = ModelSchemaBuilder._normalize_model_identifier("meta/llama3:70b")
        # Colon is stripped first (from the full lowered string), then slash
        assert result == "llama3"

    def test_empty_string(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._normalize_model_identifier("") == ""

    def test_name_with_dots_and_dashes(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._normalize_model_identifier("gemini-2.5-flash") == "gemini-2.5-flash"


class TestFormatContextWindow:
    """Test the static _format_context_window method."""

    def test_zero_tokens_returns_none(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(0) is None

    def test_negative_tokens_returns_none(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(-100) is None

    def test_none_tokens_returns_none(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(None) is None

    def test_small_token_count(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(512) == "512 ctx"

    def test_exact_thousands(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(8000) == "8K ctx"

    def test_non_exact_thousands(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(8192) == "8.2K ctx"

    def test_exact_million(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(1_000_000) == "1M ctx"

    def test_non_exact_million(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(1_500_000) == "1.5M ctx"

    def test_two_million(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(2_000_000) == "2M ctx"

    def test_128k(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        assert ModelSchemaBuilder._format_context_window(128_000) == "128K ctx"


class TestIsEffectiveAutoMode:
    """Test auto mode detection logic."""

    def test_explicit_auto_mode(self):
        """When DEFAULT_MODEL is 'auto', should return True."""
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        with (
            patch("config.DEFAULT_MODEL", "auto"),
            patch(
                "providers.registry.ModelProviderRegistry.get_provider_for_model",
                return_value=None,
            ),
        ):
            builder = ModelSchemaBuilder(tool_name="chat")
            assert builder.is_effective_auto_mode() is True

    def test_available_model_not_auto(self):
        """When DEFAULT_MODEL is a real available model, should return False."""
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        mock_provider = MagicMock()
        with (
            patch("config.DEFAULT_MODEL", "gemini-2.5-flash"),
            patch(
                "providers.registry.ModelProviderRegistry.get_provider_for_model",
                return_value=mock_provider,
            ),
        ):
            builder = ModelSchemaBuilder(tool_name="chat")
            assert builder.is_effective_auto_mode() is False

    def test_unavailable_model_falls_back_to_auto(self):
        """When DEFAULT_MODEL is set but provider is unavailable, should return True."""
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        with (
            patch("config.DEFAULT_MODEL", "nonexistent-model-xyz"),
            patch(
                "providers.registry.ModelProviderRegistry.get_provider_for_model",
                return_value=None,
            ),
        ):
            builder = ModelSchemaBuilder(tool_name="chat")
            assert builder.is_effective_auto_mode() is True


class TestShouldRequireModelSelection:
    """Test runtime model selection requirement checks."""

    def test_auto_model_name_requires_selection(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        assert builder._should_require_model_selection("auto") is True

    def test_auto_case_insensitive(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        assert builder._should_require_model_selection("AUTO") is True
        assert builder._should_require_model_selection("Auto") is True

    def test_available_model_does_not_require_selection(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        mock_provider = MagicMock()
        with patch(
            "providers.registry.ModelProviderRegistry.get_provider_for_model",
            return_value=mock_provider,
        ):
            builder = ModelSchemaBuilder(tool_name="chat")
            assert builder._should_require_model_selection("gemini-2.5-flash") is False

    def test_unavailable_model_requires_selection(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        with patch(
            "providers.registry.ModelProviderRegistry.get_provider_for_model",
            return_value=None,
        ):
            builder = ModelSchemaBuilder(tool_name="chat")
            assert builder._should_require_model_selection("nonexistent-model") is True


class TestBuildModelUnavailableMessage:
    """Test error message composition for unavailable models."""

    def _make_builder_with_mock_tool(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        # Create a mock ToolModelCategory enum value
        mock_category = MagicMock()
        mock_category.value = "balanced"

        mock_tool = MagicMock()
        mock_tool.get_name.return_value = "analyze"
        mock_tool.get_model_category.return_value = mock_category

        builder = ModelSchemaBuilder(tool_name="analyze", tool=mock_tool)
        return builder

    def test_message_contains_model_name(self):
        builder = self._make_builder_with_mock_tool()
        with (
            patch(
                "providers.ModelProviderRegistry.get_preferred_fallback_model",
                return_value="gemini-2.5-flash",
            ),
            patch.object(
                builder,
                "_format_available_models_list",
                return_value="gemini-2.5-flash; gpt-4o",
            ),
        ):
            msg = builder._build_model_unavailable_message("bad-model")
            assert "bad-model" in msg
            assert "not available" in msg

    def test_message_contains_suggested_model(self):
        builder = self._make_builder_with_mock_tool()
        with (
            patch(
                "providers.ModelProviderRegistry.get_preferred_fallback_model",
                return_value="gemini-2.5-flash",
            ),
            patch.object(
                builder,
                "_format_available_models_list",
                return_value="gemini-2.5-flash; gpt-4o",
            ),
        ):
            msg = builder._build_model_unavailable_message("bad-model")
            assert "gemini-2.5-flash" in msg
            assert "analyze" in msg

    def test_message_contains_category(self):
        builder = self._make_builder_with_mock_tool()
        with (
            patch(
                "providers.ModelProviderRegistry.get_preferred_fallback_model",
                return_value="gemini-2.5-flash",
            ),
            patch.object(
                builder,
                "_format_available_models_list",
                return_value="gemini-2.5-flash",
            ),
        ):
            msg = builder._build_model_unavailable_message("missing-model")
            assert "balanced" in msg

    def test_message_contains_available_models(self):
        builder = self._make_builder_with_mock_tool()
        with (
            patch(
                "providers.ModelProviderRegistry.get_preferred_fallback_model",
                return_value="gemini-2.5-flash",
            ),
            patch.object(
                builder,
                "_format_available_models_list",
                return_value="gemini-2.5-flash; gpt-4o; grok-3",
            ),
        ):
            msg = builder._build_model_unavailable_message("missing-model")
            assert "gemini-2.5-flash; gpt-4o; grok-3" in msg


class TestBuildAutoModeRequiredMessage:
    """Test auto-mode required message composition."""

    def _make_builder_with_mock_tool(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        mock_category = MagicMock()
        mock_category.value = "extended_reasoning"

        mock_tool = MagicMock()
        mock_tool.get_name.return_value = "thinkdeep"
        mock_tool.get_model_category.return_value = mock_category

        builder = ModelSchemaBuilder(tool_name="thinkdeep", tool=mock_tool)
        return builder

    def test_message_mentions_auto_mode(self):
        builder = self._make_builder_with_mock_tool()
        with (
            patch(
                "providers.ModelProviderRegistry.get_preferred_fallback_model",
                return_value="o3",
            ),
            patch.object(
                builder,
                "_format_available_models_list",
                return_value="o3; gemini-2.5-pro",
            ),
        ):
            msg = builder._build_auto_mode_required_message()
            assert "auto mode" in msg.lower()
            assert "required" in msg.lower()

    def test_message_contains_suggested_model(self):
        builder = self._make_builder_with_mock_tool()
        with (
            patch(
                "providers.ModelProviderRegistry.get_preferred_fallback_model",
                return_value="o3",
            ),
            patch.object(
                builder,
                "_format_available_models_list",
                return_value="o3; gemini-2.5-pro",
            ),
        ):
            msg = builder._build_auto_mode_required_message()
            assert "o3" in msg
            assert "thinkdeep" in msg

    def test_message_contains_category(self):
        builder = self._make_builder_with_mock_tool()
        with (
            patch(
                "providers.ModelProviderRegistry.get_preferred_fallback_model",
                return_value="o3",
            ),
            patch.object(
                builder,
                "_format_available_models_list",
                return_value="o3",
            ),
        ):
            msg = builder._build_auto_mode_required_message()
            assert "extended_reasoning" in msg

    def test_message_contains_available_models(self):
        builder = self._make_builder_with_mock_tool()
        with (
            patch(
                "providers.ModelProviderRegistry.get_preferred_fallback_model",
                return_value="o3",
            ),
            patch.object(
                builder,
                "_format_available_models_list",
                return_value="o3; gemini-2.5-pro; gpt-4o",
            ),
        ):
            msg = builder._build_auto_mode_required_message()
            assert "o3; gemini-2.5-pro; gpt-4o" in msg


class TestGetRestrictionNote:
    """Test restriction note generation from environment variables."""

    def test_no_restrictions_returns_none(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        with patch("utils.env.get_env", return_value=None):
            assert builder._get_restriction_note() is None

    def test_single_provider_restriction(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")

        def mock_get_env(key):
            if key == "OPENAI_ALLOWED_MODELS":
                return "gpt-4o, gpt-4o-mini"
            return None

        with patch("utils.env.get_env", side_effect=mock_get_env):
            note = builder._get_restriction_note()
            assert note is not None
            assert "OpenAI" in note
            assert "gpt-4o" in note
            assert "gpt-4o-mini" in note

    def test_multiple_provider_restrictions(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")

        def mock_get_env(key):
            if key == "OPENAI_ALLOWED_MODELS":
                return "gpt-4o"
            if key == "GOOGLE_ALLOWED_MODELS":
                return "gemini-2.5-flash"
            return None

        with patch("utils.env.get_env", side_effect=mock_get_env):
            note = builder._get_restriction_note()
            assert note is not None
            assert "OpenAI" in note
            assert "Google" in note


class TestFormatAvailableModelsList:
    """Test the human-friendly model list formatter."""

    def test_no_models_returns_guidance(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        with patch.object(builder, "_get_ranked_model_summaries", return_value=([], 0, False)):
            result = builder._format_available_models_list()
            assert "No models detected" in result

    def test_models_within_limit(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        summaries = ["model-a (score 90)", "model-b (score 80)"]
        with patch.object(builder, "_get_ranked_model_summaries", return_value=(summaries, 2, False)):
            result = builder._format_available_models_list()
            assert "model-a (score 90)" in result
            assert "model-b (score 80)" in result
            assert "more" not in result

    def test_models_exceeding_limit_shows_remainder(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        summaries = ["model-a (score 90)", "model-b (score 80)"]
        # total=5, displayed=2, remainder=3
        with patch.object(builder, "_get_ranked_model_summaries", return_value=(summaries, 5, False)):
            result = builder._format_available_models_list()
            assert "+3 more" in result
            assert "listmodels" in result


class TestGetModelFieldSchema:
    """Test the main schema generation method."""

    def test_auto_mode_schema_type_is_string(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        with (
            patch.object(builder, "is_effective_auto_mode", return_value=True),
            patch.object(builder, "_get_ranked_model_summaries", return_value=([], 0, False)),
            patch.object(builder, "_get_restriction_note", return_value=None),
        ):
            schema = builder.get_model_field_schema()
            assert schema["type"] == "string"
            assert "auto model selection" in schema["description"].lower()

    def test_non_auto_mode_includes_default_model(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        with (
            patch("config.DEFAULT_MODEL", "gemini-2.5-flash"),
            patch.object(builder, "is_effective_auto_mode", return_value=False),
            patch.object(builder, "_get_ranked_model_summaries", return_value=([], 0, False)),
            patch.object(builder, "_get_restriction_note", return_value=None),
        ):
            schema = builder.get_model_field_schema()
            assert schema["type"] == "string"
            assert "gemini-2.5-flash" in schema["description"]

    def test_auto_mode_with_summaries(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        summaries = ["gpt-4o (score 95, 128K ctx, thinking)"]
        with (
            patch.object(builder, "is_effective_auto_mode", return_value=True),
            patch.object(builder, "_get_ranked_model_summaries", return_value=(summaries, 1, False)),
            patch.object(builder, "_get_restriction_note", return_value=None),
        ):
            schema = builder.get_model_field_schema()
            assert "gpt-4o (score 95" in schema["description"]
            assert "Top models" in schema["description"]

    def test_auto_mode_with_restrictions(self):
        from tools.shared.model_schema_builder import ModelSchemaBuilder

        builder = ModelSchemaBuilder(tool_name="chat")
        summaries = ["gpt-4o (score 95)"]
        with (
            patch.object(builder, "is_effective_auto_mode", return_value=True),
            patch.object(builder, "_get_ranked_model_summaries", return_value=(summaries, 1, True)),
            patch.object(builder, "_get_restriction_note", return_value=None),
        ):
            schema = builder.get_model_field_schema()
            assert "Allowed models" in schema["description"]
