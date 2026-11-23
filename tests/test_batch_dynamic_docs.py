"""Section 14.15: the dynamic batch resolution is documented where users and agents look.

Pure file and schema reads; no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.chat import ChatTool
from tools.consensus import ConsensusTool
from tools.shared.batch_support import EXECUTION_MODE_DESCRIPTION

REPO = Path(__file__).resolve().parent.parent


def _read(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


class TestDynamicResolutionIsDocumented:
    @pytest.mark.parametrize("path", ["docs/tools/batch.md", ".env.example", "docs/configuration.md", "CHANGELOG.md"])
    def test_the_opt_out_variable_is_mentioned(self, path):
        assert "OPENROUTER_BATCH_LIVE_CATALOG" in _read(path)

    @pytest.mark.parametrize("path", ["docs/tools/batch.md", ".env.example", "docs/configuration.md"])
    def test_the_ttl_variable_is_mentioned(self, path):
        assert "OPENROUTER_BATCH_CATALOG_TTL_SECONDS" in _read(path)

    def test_the_batch_guide_replaces_the_alias_caveat(self):
        text = _read("docs/tools/batch.md")
        assert "resolves to OpenRouter's `anthropic/claude-opus-4.5`" not in text  # the old caveat
        assert "model_requested" in text and "model_resolved" in text
        assert "refresh_openrouter_batch_catalog.py" in text and "conf/openrouter_batch_catalog.json" in text
        assert "Variants are refused" in text and "Version-pinned aliases and full slugs never float" in text

    def test_the_batch_guide_documents_the_catalog_limits_and_the_caveats_the_review_found(self):
        text = _read("docs/tools/batch.md")
        assert "8 MB cap" in text and "spent from the call's `wait_seconds`" in text
        assert "lists models but not one `:batch` entry" in text
        assert "treated as `false` with a warning" in text and "positive, finite number" in text
        assert "--allow-shrink" in text
        assert "cost cap" in text  # an alias allow-list admits the newest, possibly dearer, model under batch
        # Section 15 (D8) reverses the section-14 sentence: a resolved slug no longer borrows its family entry's limits.
        assert "borrows its family entry's limits" not in text
        assert "A resolved model uses the limits of its own listing, not its family entry's" in text
        assert "live catalog's `:batch` listing, then the committed snapshot" in text
        assert "then the plain listing of the same live body" in text and "with one warning" in text

    def test_the_batch_guide_documents_the_newer_model_note_and_the_full_slug_route(self):
        text = _read("docs/tools/batch.md")
        assert "unclassified_newer_batch_models" in text and "unclassified_newer_batch_models_omitted" in text
        assert "type its full slug" in text and "google/gemini-4-argon" in text
        assert "the product-line narrowing" in text  # the stated refinement of "same vendor"
        assert "never appears on a synchronous call" in text
        assert "an alarm with known limits" in text  # the false-positive and false-negative profile is stated
        assert "carries each model's limits and parameter list" in text  # the snapshot's new content

    def test_the_batch_guide_states_the_curated_parameter_rule_the_vendor_ranges_and_the_note_limits(self):
        text = _read("docs/tools/batch.md")
        # D9: a curated slug keeps its limits but sends only the parameters its listing names
        assert "A curated slug keeps its limits but sends only the parameters its `:batch` listing names" in text
        assert "`openai/o3`, `o3-mini`, `o4-mini` and `gpt-5.2-pro`" in text and "silently dropped before" in text
        assert "narrower than a live listing" in text
        assert "keeps its curated entry" not in text  # the stale blanket wording
        # D10: per-vendor range, with the conservative fallback
        assert "`anthropic` 0 to 1" in text and "`openai`, `google` and `x-ai` 0 to 2" in text
        assert "any other vendor 0 to 1" in text
        # D11: the stated limits of a product-line-only note
        assert "claude-fable-5.x" in text and "no numeric lead" in text and "deliberate limits" in text

    def test_the_changelog_states_the_curated_parameter_change_and_the_vendor_ranges_under_unreleased(self):
        text = _read("CHANGELOG.md")
        unreleased = text[text.index("## [Unreleased]") : text.index("\n## [", text.index("## [Unreleased]") + 5)]
        assert "a curated slug sends only the parameters its `:batch` listing names" in unreleased
        assert "`openai/o3`, `o3-mini`, `o4-mini` and `gpt-5.2-pro`" in unreleased
        assert "clamped to its vendor's range instead of 0 to 1 for all" in unreleased
        assert "A catalog slug the curated file lists keeps its curated entry." not in unreleased

    def test_no_document_still_says_a_resolved_model_borrows_limits_or_gets_generic_ones_under_batch(self):
        for path in ("docs/tools/batch.md", "README.md", "docs/configuration.md", "CHANGELOG.md"):
            text = _read(path)
            assert "borrows its family entry's limits" not in text, path

    def test_the_environment_documentation_names_the_fail_closed_value_handling(self):
        for path in (".env.example", "docs/configuration.md"):
            assert "any value other than a true/false word" in _read(path).lower()

    def test_the_schema_description_states_the_alias_behaviour_on_every_batch_capable_tool(self):
        assert "resolves to the latest model of its family" in EXECUTION_MODE_DESCRIPTION
        assert "unclassified_newer_batch_models" in EXECUTION_MODE_DESCRIPTION
        assert "model_requested" in EXECUTION_MODE_DESCRIPTION and "model_resolved" in EXECUTION_MODE_DESCRIPTION
        for tool in (ChatTool(), ConsensusTool()):
            published = tool.get_input_schema()["properties"]["execution_mode"]["description"]
            assert published == EXECUTION_MODE_DESCRIPTION

    def test_the_changelog_entries_sit_under_unreleased(self):
        changelog = _read("CHANGELOG.md")
        unreleased = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
        sections = dict(_sections(unreleased))
        assert any("dynamic model resolution" in entry for entry in sections["Added"])
        assert any("now resolve dynamically" in entry for entry in sections["Changed"])
        assert any("`gemini` and `flash` are no longer refused" in entry for entry in sections["Fixed"])
        # section 15: live-derived capabilities and the unclassified-newer note
        assert any(
            "carries the limits of its own listing" in entry and "unclassified_newer_batch_models" in entry
            for entry in sections["Added"]
        )
        assert any(
            "no longer borrows the older curated family entry's limits" in entry for entry in sections["Changed"]
        )
        assert any("no longer gets the generic 32K limits" in entry for entry in sections["Fixed"])

    def test_the_catalog_readme_documents_the_field_and_the_fallback_role_of_the_flag(self):
        import json

        readme = json.loads(_read("conf/openrouter_models.json"))["_README"]["field_descriptions"]
        assert "batch_family" in readme
        assert "offline fallback" in readme["supports_batch"]


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """``### Name`` sections of a changelog block, each as its list of bullet entries (Added may recur)."""

    found: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            found.setdefault(current, [])
        elif current is not None and line.startswith("- "):
            found[current].append(line)
    return list(found.items())
