"""Unit tests for the README mockup generator."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

# Add scripts/ to sys.path so we can import the generator module.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_mockups import (  # noqa: E402
    ALLOWED_KINDS,
    PALETTES,
    REQUIRED_SCENE_FIELDS,
    SceneValidationError,
    collect_glyphs,
    compute_height,
    generate_all,
    load_scene,
    validate_scene,
    visual_lines,
)

FONT_PATH = REPO_ROOT / "scripts" / "fonts" / "JetBrainsMono-Regular.ttf"
TEMPLATE_PATH = REPO_ROOT / "scripts" / "build_mockups_template.svg.j2"
SCENES_DIR = REPO_ROOT / "docs" / "mockup-scenes"


@pytest.fixture
def minimal_scene() -> dict[str, Any]:
    return {
        "id": "test",
        "title": "~/test — claude code",
        "description": "A minimal valid scene.",
        "width": 820,
        "lines": [{"kind": "prompt", "text": "hello"}],
    }


@pytest.fixture
def scene_yaml(tmp_path: Path) -> Path:
    """Returns a writable path with the basename ``test.yaml`` for validation."""
    return tmp_path / "test.yaml"


def write_scene(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSceneValidation:
    def test_minimal_valid_scene_passes(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        validate_scene(minimal_scene, scene_yaml)  # no exception

    @pytest.mark.parametrize("field", sorted(REQUIRED_SCENE_FIELDS))
    def test_missing_required_field_rejected(self, minimal_scene: dict[str, Any], scene_yaml: Path, field: str) -> None:
        del minimal_scene[field]
        with pytest.raises(SceneValidationError, match=f"missing required fields.*{field}"):
            validate_scene(minimal_scene, scene_yaml)

    def test_extra_top_level_field_rejected(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        minimal_scene["bogus"] = 1
        with pytest.raises(SceneValidationError, match="unknown top-level fields.*bogus"):
            validate_scene(minimal_scene, scene_yaml)

    def test_id_must_match_filename(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        minimal_scene["id"] = "different"
        with pytest.raises(SceneValidationError, match="'id' .* must match filename stem"):
            validate_scene(minimal_scene, scene_yaml)

    def test_unknown_kind_rejected(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        minimal_scene["lines"] = [{"kind": "rainbow", "text": "x"}]
        with pytest.raises(SceneValidationError, match="must be one of"):
            validate_scene(minimal_scene, scene_yaml)

    def test_prompt_requires_text(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        minimal_scene["lines"] = [{"kind": "prompt"}]
        with pytest.raises(SceneValidationError, match="'text' required"):
            validate_scene(minimal_scene, scene_yaml)

    def test_unknown_color_rejected(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        minimal_scene["lines"] = [{"kind": "output", "text": "hi", "color": "rainbow"}]
        with pytest.raises(SceneValidationError, match="unknown color"):
            validate_scene(minimal_scene, scene_yaml)

    def test_tree_requires_items(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        minimal_scene["lines"] = [{"kind": "tree", "items": []}]
        with pytest.raises(SceneValidationError, match="'items' required"):
            validate_scene(minimal_scene, scene_yaml)

    def test_tool_call_meta_must_be_string_list(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        minimal_scene["lines"] = [{"kind": "tool_call", "spinner": "⠋", "name": "x", "meta": [1, 2]}]
        with pytest.raises(SceneValidationError, match="'meta' must be a list of strings"):
            validate_scene(minimal_scene, scene_yaml)

    def test_box_requires_string_lines(self, minimal_scene: dict[str, Any], scene_yaml: Path) -> None:
        minimal_scene["lines"] = [{"kind": "box", "title": "x", "lines": [1, 2]}]
        with pytest.raises(SceneValidationError, match="'lines' must be a list of strings"):
            validate_scene(minimal_scene, scene_yaml)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestLayout:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ({"kind": "blank"}, 1),
            ({"kind": "prompt", "text": "x"}, 1),
            ({"kind": "output", "text": "x"}, 1),
            ({"kind": "status", "icon": "◆", "text": "x"}, 1),
            ({"kind": "tool_call", "spinner": "⠋", "name": "x"}, 1),
            ({"kind": "tree", "items": ["a", "b", "c"]}, 3),
            ({"kind": "box", "title": "t", "lines": ["a", "b"]}, 4),
        ],
    )
    def test_visual_lines(self, line: dict[str, Any], expected: int) -> None:
        assert visual_lines(line) == expected

    def test_compute_height_includes_chrome_and_padding(self, minimal_scene: dict[str, Any]) -> None:
        h = compute_height(minimal_scene)
        # 32 (titlebar) + 22 (top pad) + 1*22 (one line) + 16 (bottom pad)
        assert h == 32 + 22 + 22 + 16


# ---------------------------------------------------------------------------
# Glyph collection
# ---------------------------------------------------------------------------


class TestGlyphCollection:
    def test_collects_chars_from_all_line_kinds(self) -> None:
        scene = {
            "id": "x",
            "title": "TitleText",
            "description": "Desc with chars.",
            "width": 820,
            "lines": [
                {"kind": "prompt", "text": "abc"},
                {"kind": "output", "text": "def"},
                {"kind": "tool_call", "spinner": "S", "name": "Nm", "meta": ["m1", "m2"]},
                {"kind": "tree", "items": ["it1", {"text": "it2"}]},
                {"kind": "status", "icon": "I", "text": "Stat"},
                {"kind": "box", "title": "Bx", "lines": ["bl1"]},
                {"kind": "blank"},
            ],
        }
        glyphs = collect_glyphs(scene)
        for expected in "TitleTextDescabcdefSNmm1m2it1it2IStatBxbl1":
            assert expected in glyphs, f"{expected!r} missing"


# ---------------------------------------------------------------------------
# Rendering + accessibility
# ---------------------------------------------------------------------------


class TestRendering:
    @pytest.fixture(scope="class")
    def generated(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
        """Generates the hero scene into a tmp dir; returns (light_path, dark_path)."""
        out_dir = tmp_path_factory.mktemp("mockups")
        scenes_dir = tmp_path_factory.mktemp("scenes")
        # Copy real hero.yaml so we test the actual production scene.
        hero_src = SCENES_DIR / "hero.yaml"
        (scenes_dir / "hero.yaml").write_text(hero_src.read_text(), encoding="utf-8")
        generate_all(scenes_dir, out_dir, TEMPLATE_PATH, FONT_PATH)
        return out_dir / "hero-light.svg", out_dir / "hero-dark.svg"

    def test_produces_both_variants(self, generated: tuple[Path, Path]) -> None:
        light, dark = generated
        assert light.exists() and light.stat().st_size > 1000
        assert dark.exists() and dark.stat().st_size > 1000

    def test_output_is_valid_xml(self, generated: tuple[Path, Path]) -> None:
        for path in generated:
            ET.parse(path)  # raises ParseError if invalid

    def test_accessibility_metadata_first(self, generated: tuple[Path, Path]) -> None:
        """<title> and <desc> MUST be the first two children of <svg> for screen readers."""
        for path in generated:
            tree = ET.parse(path)
            root = tree.getroot()
            children = list(root)
            assert len(children) >= 2
            assert children[0].tag.endswith("title")
            assert children[1].tag.endswith("desc")
            assert children[0].text and "claude code" in children[0].text
            assert children[1].text and len(children[1].text.strip()) > 10

    def test_light_and_dark_differ_only_in_colors(self, generated: tuple[Path, Path]) -> None:
        light, dark = generated
        # Strip every fill="#xxx" and stroke="#xxx" attribute; remaining
        # content should be byte-identical between light and dark.
        import re

        attr_re = re.compile(r'(fill|stroke)="#[0-9a-f]+"')
        light_stripped = attr_re.sub("", light.read_text())
        dark_stripped = attr_re.sub("", dark.read_text())
        # Also strip the font base64 blob (will be identical, but we don't care
        # about diffing it).
        font_re = re.compile(r"base64,[A-Za-z0-9+/=]+\)")
        light_stripped = font_re.sub("base64,X)", light_stripped)
        dark_stripped = font_re.sub("base64,X)", dark_stripped)
        assert light_stripped == dark_stripped

    def test_palette_keys_match_across_themes(self) -> None:
        assert set(PALETTES["light"].keys()) == set(PALETTES["dark"].keys())


# ---------------------------------------------------------------------------
# Determinism (via subprocess so PYTHONHASHSEED re-exec actually applies)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeat_run_produces_identical_bytes(self, tmp_path: Path) -> None:
        out_dir_a = tmp_path / "a"
        out_dir_b = tmp_path / "b"
        # Run the generator twice via subprocess so the PYTHONHASHSEED re-exec
        # branch in main() actually executes.
        for out_dir in (out_dir_a, out_dir_b):
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "build_mockups.py"),
                    "--scenes-dir",
                    str(SCENES_DIR),
                    "--output-dir",
                    str(out_dir),
                    "--template",
                    str(TEMPLATE_PATH),
                    "--font",
                    str(FONT_PATH),
                ],
                check=True,
                capture_output=True,
            )
            assert result.returncode == 0
        for variant in ("hero-dark.svg", "hero-light.svg"):
            assert (out_dir_a / variant).read_bytes() == (out_dir_b / variant).read_bytes()


# ---------------------------------------------------------------------------
# Smoke checks for fixture data
# ---------------------------------------------------------------------------


class TestRealScenes:
    def test_all_scenes_load_and_validate(self) -> None:
        for path in sorted(SCENES_DIR.glob("*.yaml")):
            scene = load_scene(path)
            validate_scene(scene, path)

    def test_all_kinds_documented(self) -> None:
        # Sanity: the set of kinds the generator allows should match what
        # docs/mockup-scenes/README.md documents. We can't easily parse the
        # markdown, so just assert the set matches the constant.
        expected = {"prompt", "output", "tool_call", "tree", "status", "blank", "box"}
        assert ALLOWED_KINDS == expected
