"""Tests for the weekly LiteLLM model refresh script."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the script modules
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import refresh_litellm_models as refresh  # noqa: E402

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------
SAMPLE_CATALOG = {
    "sample_spec": {"mode": "documentation"},
    "openai/gpt-5": {
        "litellm_provider": "openai",
        "mode": "chat",
        "max_input_tokens": 400000,
        "max_output_tokens": 128000,
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_response_schema": True,
        "supports_reasoning": False,
    },
    "openai/gpt-99-turbo": {
        "litellm_provider": "openai",
        "mode": "chat",
        "max_input_tokens": 500000,
        "max_output_tokens": 200000,
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_response_schema": True,
        "supports_reasoning": False,
    },
    "gemini/gemini-9-pro": {
        "litellm_provider": "gemini",
        "mode": "chat",
        "max_input_tokens": 2000000,
        "max_output_tokens": 65536,
        "supports_vision": True,
        "supports_function_calling": True,
        "supports_response_schema": True,
        "supports_reasoning": True,
    },
    "openai/dall-e-3": {
        "litellm_provider": "openai",
        "mode": "image_generation",
        "max_input_tokens": 0,
        "max_output_tokens": 0,
    },
}


class TestFetchAndParse:
    """Test fetch + parse logic with mocked HTTP."""

    def test_filter_chat_models_only(self):
        result = refresh.filter_models_by_provider(SAMPLE_CATALOG)
        all_names = [m["model_name"] for models in result.values() for m in models]
        assert "dall-e-3" not in all_names

    def test_filter_by_provider(self):
        result = refresh.filter_models_by_provider(SAMPLE_CATALOG)
        assert "openai_models.json" in result
        assert "gemini_models.json" in result
        openai_names = [m["model_name"] for m in result["openai_models.json"]]
        assert "gpt-5" in openai_names
        assert "gpt-99-turbo" in openai_names

    def test_strips_provider_prefix(self):
        result = refresh.filter_models_by_provider(SAMPLE_CATALOG)
        openai_names = [m["model_name"] for m in result["openai_models.json"]]
        assert "gpt-5" in openai_names
        assert "openai/gpt-5" not in openai_names

    def test_maps_litellm_fields(self):
        result = refresh.filter_models_by_provider(SAMPLE_CATALOG)
        gpt5 = next(m for m in result["openai_models.json"] if m["model_name"] == "gpt-5")
        assert gpt5["context_window"] == 400000
        assert gpt5["max_output_tokens"] == 128000
        assert gpt5["supports_images"] is True
        assert gpt5["supports_function_calling"] is True
        assert gpt5["supports_json_mode"] is True

    def test_ignores_unknown_providers(self):
        catalog = {
            "deepseek/deepseek-v3": {
                "litellm_provider": "deepseek",
                "mode": "chat",
                "max_input_tokens": 64000,
            }
        }
        result = refresh.filter_models_by_provider(catalog)
        assert not any("deepseek" in f for f in result)

    @patch("refresh_litellm_models.urllib.request.urlopen")
    def test_fetch_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"test": "data"}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = refresh.fetch_litellm_catalog()
        assert result == {"test": "data"}

    @patch("refresh_litellm_models.urllib.request.urlopen")
    def test_fetch_retries_on_failure(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = [
            urllib.error.URLError("Network error"),
            urllib.error.URLError("Still failing"),
        ]
        with pytest.raises(urllib.error.URLError):
            refresh.fetch_litellm_catalog(max_retries=2)
        assert mock_urlopen.call_count == 2


class TestMergeLogic:
    """Test the merge logic: updates, additions, curated field protection."""

    def _make_existing(self, models):
        return {"_README": {"description": "Test"}, "models": models}

    def test_update_existing_model_capacity(self):
        existing = self._make_existing(
            [
                {
                    "model_name": "gpt-5",
                    "aliases": ["gpt5"],
                    "intelligence_score": 16,
                    "description": "Curated description",
                    "context_window": 100000,
                    "max_output_tokens": 50000,
                    "supports_images": False,
                    "supports_function_calling": False,
                    "supports_json_mode": False,
                }
            ]
        )
        litellm_models = [
            {
                "model_name": "gpt-5",
                "context_window": 400000,
                "max_output_tokens": 128000,
                "supports_images": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
            }
        ]
        from providers.shared.provider_type import ProviderType

        updated, result = refresh.merge_provider("openai_models.json", existing, litellm_models, ProviderType.OPENAI)

        model = updated["models"][0]
        assert model["context_window"] == 400000
        assert model["max_output_tokens"] == 128000
        assert model["supports_images"] is True
        assert len(result.updated) == 5

    def test_curated_fields_preserved(self):
        existing = self._make_existing(
            [
                {
                    "model_name": "gpt-5",
                    "aliases": ["gpt5", "five"],
                    "intelligence_score": 16,
                    "friendly_name": "My Custom Name",
                    "description": "My curated description",
                    "context_window": 100000,
                    "max_output_tokens": 50000,
                    "supports_images": False,
                    "supports_function_calling": False,
                    "supports_json_mode": False,
                }
            ]
        )
        litellm_models = [
            {
                "model_name": "gpt-5",
                "context_window": 400000,
                "max_output_tokens": 128000,
                "supports_images": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
            }
        ]
        from providers.shared.provider_type import ProviderType

        updated, _ = refresh.merge_provider("openai_models.json", existing, litellm_models, ProviderType.OPENAI)

        model = updated["models"][0]
        assert model["aliases"] == ["gpt5", "five"]
        assert model["intelligence_score"] == 16
        assert model["friendly_name"] == "My Custom Name"
        assert model["description"] == "My curated description"

    def test_new_model_added(self):
        existing = self._make_existing([])
        litellm_models = [
            {
                "model_name": "gpt-99-turbo",
                "context_window": 500000,
                "max_output_tokens": 200000,
                "supports_images": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "_supports_reasoning": False,
                "_provider_type": None,
            }
        ]
        from providers.shared.provider_type import ProviderType

        updated, result = refresh.merge_provider("openai_models.json", existing, litellm_models, ProviderType.OPENAI)

        assert len(updated["models"]) == 1
        assert result.added == ["gpt-99-turbo"]
        new_model = updated["models"][0]
        assert new_model["model_name"] == "gpt-99-turbo"
        assert new_model["context_window"] == 500000
        assert "intelligence_score" in new_model
        assert new_model["friendly_name"] == "OpenAI (gpt-99-turbo)"

    def test_missing_from_litellm_flagged(self):
        existing = self._make_existing(
            [
                {
                    "model_name": "custom-internal",
                    "context_window": 8192,
                    "max_output_tokens": 4096,
                    "supports_images": False,
                    "supports_function_calling": False,
                    "supports_json_mode": False,
                }
            ]
        )
        from providers.shared.provider_type import ProviderType

        _, result = refresh.merge_provider("openai_models.json", existing, [], ProviderType.OPENAI)

        assert result.not_in_litellm == ["custom-internal"]

    def test_readme_preserved(self):
        readme = {"description": "Test metadata", "docs": "https://example.com"}
        existing = {"_README": readme, "models": []}
        from providers.shared.provider_type import ProviderType

        updated, _ = refresh.merge_provider("openai_models.json", existing, [], ProviderType.OPENAI)

        assert updated["_README"] == readme


class TestDiffSummary:
    """Test diff summary generation."""

    def test_summary_with_updates_and_additions(self):
        r = refresh.MergeResult("openai_models.json")
        r.updated = [
            {"model": "gpt-5", "field": "context_window", "old": 100000, "new": 400000},
        ]
        r.added = ["gpt-99-turbo"]
        r.not_in_litellm = ["custom-internal"]

        summary = refresh.generate_summary([r])

        assert "Updated fields**: 1" in summary
        assert "New models**: 1" in summary
        assert "Not in LiteLLM**: 1" in summary
        assert "gpt-5" in summary
        assert "100000" in summary
        assert "400000" in summary
        assert "gpt-99-turbo" in summary
        assert "needs curation" in summary
        assert "custom-internal" in summary

    def test_empty_summary(self):
        r = refresh.MergeResult("openai_models.json")
        summary = refresh.generate_summary([r])
        assert "Updated fields**: 0" in summary


class TestIntegration:
    """Integration test: run against actual conf/*.json in dry-run."""

    def test_dry_run_against_real_conf(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "refresh_litellm_models.py"), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "Fetched" in result.stdout
        assert "Filtered" in result.stdout
        assert "[DRY RUN]" in result.stdout

    def test_output_summary_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            summary_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "refresh_litellm_models.py"),
                    "--dry-run",
                    "--output-summary",
                    summary_path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0
            summary = Path(summary_path).read_text()
            assert "Model Catalog Refresh Summary" in summary
        finally:
            Path(summary_path).unlink(missing_ok=True)
