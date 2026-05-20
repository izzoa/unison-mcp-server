"""SVG terminal mockup generator for the README.

Reads scene YAML files from ``docs/mockup-scenes/`` and emits two SVG files
per scene (light + dark Catppuccin palette) into ``docs/assets/mockups/``.
JetBrains Mono Regular is subsetted to the glyphs actually used across all
scenes and base64-embedded once per SVG via ``@font-face``.

Run: ``python scripts/build_mockups.py``

See ``docs/mockup-scenes/README.md`` for the scene YAML schema.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

# fontTools' subsetter iterates over sets internally, so the byte output of
# subset_font() depends on Python's hash seed. Re-exec with PYTHONHASHSEED=0
# the first time so the generator produces deterministic SVGs across runs
# (the CI drift check byte-compares output).
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import jinja2  # noqa: E402
import yaml  # noqa: E402
from fontTools.subset import Subsetter  # type: ignore[import-untyped]  # noqa: E402
from fontTools.ttLib import TTFont  # type: ignore[import-untyped]  # noqa: E402

# ---------------------------------------------------------------------------
# Layout constants (px, applied at font-size 14)
# ---------------------------------------------------------------------------

TITLE_BAR_HEIGHT = 32
CONTENT_PAD_TOP = 22
CONTENT_PAD_BOTTOM = 16
SIDE_PAD = 24
LINE_HEIGHT = 22
PROMPT_PREFIX_W = 14
TREE_PREFIX_W = 14
TREE_TEXT_OFFSET = 46
TOOL_NAME_OFFSET = 22
STATUS_TEXT_OFFSET = 22

# ---------------------------------------------------------------------------
# Catppuccin palettes (Latte = light, Mocha = dark)
# https://github.com/catppuccin/catppuccin (OFL 1.1)
# ---------------------------------------------------------------------------

PALETTES: dict[str, dict[str, str]] = {
    "light": {  # Catppuccin Latte
        "bg": "#eff1f5",
        "titlebar_bg": "#dce0e8",
        "border": "#ccd0da",
        "default": "#4c4f69",
        "muted": "#9ca0b0",
        "prompt": "#04a5e5",
        "info": "#7287fd",
        "success": "#40a02b",
        "warning": "#df8e1d",
        "error": "#d20f39",
        "accent": "#ea76cb",
    },
    "dark": {  # Catppuccin Mocha
        "bg": "#1e1e2e",
        "titlebar_bg": "#181825",
        "border": "#313244",
        "default": "#cdd6f4",
        "muted": "#7f849c",
        "prompt": "#89dceb",
        "info": "#b4befe",
        "success": "#a6e3a1",
        "warning": "#f9e2af",
        "error": "#f38ba8",
        "accent": "#f5c2e7",
    },
}

ALLOWED_COLORS = set(PALETTES["light"].keys())
ALLOWED_KINDS = {"prompt", "output", "tool_call", "tree", "status", "blank", "box"}
REQUIRED_SCENE_FIELDS = {"id", "title", "description", "width", "lines"}

# Chrome / status glyphs that may not appear in any scene text but must be in the font subset.
CHROME_GLYPHS = set("›─├└┌┐┘╭╮╰╯→↳●⠋⠙⠹⠸✓✗⚠◆⚡⚖")


class SceneValidationError(ValueError):
    """Raised when a scene YAML fails validation."""


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def load_scene(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SceneValidationError(f"{path}: top-level must be a mapping")
    return raw


def validate_scene(scene: dict[str, Any], path: Path) -> None:
    keys = set(scene.keys())
    missing = REQUIRED_SCENE_FIELDS - keys
    if missing:
        raise SceneValidationError(f"{path}: missing required fields: {sorted(missing)}")
    extra = keys - REQUIRED_SCENE_FIELDS
    if extra:
        raise SceneValidationError(f"{path}: unknown top-level fields: {sorted(extra)}")

    if not isinstance(scene["id"], str) or not scene["id"]:
        raise SceneValidationError(f"{path}: 'id' must be a non-empty string")
    if scene["id"] != path.stem:
        raise SceneValidationError(f"{path}: 'id' ({scene['id']!r}) must match filename stem ({path.stem!r})")
    if not isinstance(scene["title"], str):
        raise SceneValidationError(f"{path}: 'title' must be a string")
    if not isinstance(scene["description"], str) or not scene["description"].strip():
        raise SceneValidationError(f"{path}: 'description' must be a non-empty string")
    if not isinstance(scene["width"], int) or scene["width"] <= 0:
        raise SceneValidationError(f"{path}: 'width' must be a positive integer")
    if not isinstance(scene["lines"], list):
        raise SceneValidationError(f"{path}: 'lines' must be a list")

    for i, line in enumerate(scene["lines"]):
        if not isinstance(line, dict):
            raise SceneValidationError(f"{path}: lines[{i}] must be a mapping")
        kind = line.get("kind")
        if kind not in ALLOWED_KINDS:
            raise SceneValidationError(f"{path}: lines[{i}].kind ({kind!r}) must be one of {sorted(ALLOWED_KINDS)}")
        _validate_line(line, i, path)


def _validate_line(line: dict[str, Any], index: int, path: Path) -> None:
    kind = line["kind"]
    ctx = f"{path}: lines[{index}] ({kind})"
    if kind == "blank":
        return
    if kind in ("prompt", "output"):
        if not isinstance(line.get("text"), str):
            raise SceneValidationError(f"{ctx}: 'text' required (string)")
        color = line.get("color")
        if color is not None and color not in ALLOWED_COLORS:
            raise SceneValidationError(f"{ctx}: unknown color {color!r}")
        return
    if kind == "tool_call":
        for field in ("spinner", "name"):
            if not isinstance(line.get(field), str):
                raise SceneValidationError(f"{ctx}: {field!r} required (string)")
        meta = line.get("meta", [])
        if not isinstance(meta, list) or not all(isinstance(m, str) for m in meta):
            raise SceneValidationError(f"{ctx}: 'meta' must be a list of strings")
        return
    if kind == "tree":
        items = line.get("items")
        if not isinstance(items, list) or not items:
            raise SceneValidationError(f"{ctx}: 'items' required (non-empty list)")
        for j, item in enumerate(items):
            if isinstance(item, str):
                continue
            if isinstance(item, dict):
                if not isinstance(item.get("text"), str):
                    raise SceneValidationError(f"{ctx}: items[{j}].text required (string)")
                color = item.get("color")
                if color is not None and color not in ALLOWED_COLORS:
                    raise SceneValidationError(f"{ctx}: items[{j}].color {color!r} unknown")
            else:
                raise SceneValidationError(f"{ctx}: items[{j}] must be a string or {{text, color}} mapping")
        return
    if kind == "status":
        for field in ("icon", "text"):
            if not isinstance(line.get(field), str):
                raise SceneValidationError(f"{ctx}: {field!r} required (string)")
        color = line.get("color")
        if color is not None and color not in ALLOWED_COLORS:
            raise SceneValidationError(f"{ctx}: unknown color {color!r}")
        return
    if kind == "box":
        if not isinstance(line.get("title"), str):
            raise SceneValidationError(f"{ctx}: 'title' required (string)")
        box_lines = line.get("lines")
        if not isinstance(box_lines, list) or not all(isinstance(b, str) for b in box_lines):
            raise SceneValidationError(f"{ctx}: 'lines' must be a list of strings")
        return


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def visual_lines(line: dict[str, Any]) -> int:
    kind = line["kind"]
    if kind in ("prompt", "output", "tool_call", "status", "blank"):
        return 1
    if kind == "tree":
        items = line["items"]
        assert isinstance(items, list)
        return len(items)
    if kind == "box":
        box_lines = line["lines"]
        assert isinstance(box_lines, list)
        return 2 + len(box_lines)
    raise ValueError(f"unknown kind: {kind}")


def compute_height(scene: dict[str, Any]) -> int:
    total = sum(visual_lines(line) for line in scene["lines"])
    return TITLE_BAR_HEIGHT + CONTENT_PAD_TOP + total * LINE_HEIGHT + CONTENT_PAD_BOTTOM


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _text(x: int, y: int, content: str, fill: str, anchor: str = "start") -> str:
    anchor_attr = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'  <text x="{x}" y="{y}" fill="{fill}"{anchor_attr}>{xml_escape(content)}</text>'


def render_lines(scene: dict[str, Any], palette: dict[str, str]) -> str:
    fragments: list[str] = []
    y = TITLE_BAR_HEIGHT + CONTENT_PAD_TOP + LINE_HEIGHT  # baseline of first content line
    width = scene["width"]

    for line in scene["lines"]:
        y = _render_line(line, fragments, y, width, palette)

    return "\n".join(fragments)


def _render_line(
    line: dict[str, Any],
    fragments: list[str],
    y: int,
    width: int,
    palette: dict[str, str],
) -> int:
    kind = line["kind"]

    if kind == "blank":
        return y + LINE_HEIGHT

    if kind == "prompt":
        fragments.append(_text(SIDE_PAD, y, "›", palette["prompt"]))
        fragments.append(_text(SIDE_PAD + PROMPT_PREFIX_W, y, line["text"], palette["default"]))
        return y + LINE_HEIGHT

    if kind == "output":
        color = palette[line.get("color", "default")]
        fragments.append(_text(SIDE_PAD, y, line["text"], color))
        return y + LINE_HEIGHT

    if kind == "tool_call":
        fragments.append(_text(SIDE_PAD, y, line["spinner"], palette["info"]))
        fragments.append(_text(SIDE_PAD + TOOL_NAME_OFFSET, y, line["name"], palette["info"]))
        meta = line.get("meta", [])
        if meta:
            meta_text = "  ·  ".join(meta)
            fragments.append(_text(width - SIDE_PAD, y, meta_text, palette["muted"], anchor="end"))
        return y + LINE_HEIGHT

    if kind == "tree":
        items = line["items"]
        for i, item in enumerate(items):
            prefix = "└─" if i == len(items) - 1 else "├─"
            if isinstance(item, dict):
                item_text = item["text"]
                color_key = item.get("color", "default")
            else:
                item_text = item
                color_key = "default"
            fragments.append(_text(SIDE_PAD + PROMPT_PREFIX_W, y, prefix, palette["muted"]))
            fragments.append(_text(SIDE_PAD + TREE_TEXT_OFFSET, y, item_text, palette[color_key]))
            y += LINE_HEIGHT
        return y

    if kind == "status":
        color = palette[line.get("color", "default")]
        fragments.append(_text(SIDE_PAD, y, line["icon"], color))
        fragments.append(_text(SIDE_PAD + STATUS_TEXT_OFFSET, y, line["text"], color))
        return y + LINE_HEIGHT

    if kind == "box":
        # Box drawn as full-width rectangle with title in the top border.
        inset = SIDE_PAD * 2
        box_width = width - 2 * inset
        title_text = line["title"]
        # Each character is ~8.4px wide at font-size 14; round to ints for SVG.
        char_w = 8
        inner_chars = max(2, box_width // char_w - 2)
        title_chars = len(title_text)
        dashes_right = max(1, inner_chars - title_chars - 4)
        top = "┌─ " + title_text + " " + "─" * dashes_right + "┐"
        fragments.append(_text(inset, y, top, palette["muted"]))
        y += LINE_HEIGHT
        for box_line in line["lines"]:
            content = "│ " + box_line + " " * max(0, inner_chars - len(box_line) - 3) + " │"
            fragments.append(_text(inset, y, content, palette["muted"]))
            y += LINE_HEIGHT
        bottom = "└" + "─" * (inner_chars + 2) + "┘"
        fragments.append(_text(inset, y, bottom, palette["muted"]))
        y += LINE_HEIGHT
        return y

    raise ValueError(f"unknown kind: {kind}")


def collect_glyphs(scene: dict[str, Any]) -> set[str]:
    chars: set[str] = set()
    chars.update(scene["title"])
    chars.update(scene["description"])
    for line in scene["lines"]:
        kind = line["kind"]
        if kind in ("prompt", "output"):
            chars.update(line.get("text", ""))
        elif kind == "tool_call":
            chars.update(line.get("spinner", ""))
            chars.update(line.get("name", ""))
            for m in line.get("meta", []):
                chars.update(m)
        elif kind == "tree":
            for item in line["items"]:
                if isinstance(item, dict):
                    chars.update(item["text"])
                else:
                    chars.update(item)
        elif kind == "status":
            chars.update(line.get("icon", ""))
            chars.update(line.get("text", ""))
        elif kind == "box":
            chars.update(line.get("title", ""))
            for bl in line.get("lines", []):
                chars.update(bl)
    return chars


# ---------------------------------------------------------------------------
# Font subsetting
# ---------------------------------------------------------------------------


def subset_font(font_path: Path, glyphs: set[str]) -> bytes:
    """Subset TTF to just the unicode codepoints in ``glyphs`` and return bytes.

    fontTools' subsetter has internal state that varies across processes (some
    of which is ASLR-affected), so the output bytes can differ between separate
    Python invocations even with PYTHONHASHSEED pinned. The drift check would
    fail constantly if we re-subsetted on every generation. ``generate_all``
    works around this by reading a pre-built subset from disk via
    ``load_or_build_font_cache``; this function is only invoked when the cache
    is missing or being rebuilt with ``--rebuild-font-cache``.
    """
    font = TTFont(str(font_path))
    subsetter = Subsetter()
    subsetter.populate(unicodes=sorted({ord(c) for c in glyphs}))
    subsetter.subset(font)
    head = font["head"]
    head.created = 0
    head.modified = 0
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def load_or_build_font_cache(
    font_path: Path,
    needed_glyphs: set[str],
    cache_path: Path | None = None,
    rebuild: bool = False,
) -> str:
    """Read the cached subset (base64) from disk, or rebuild it if requested.

    The cache lives at ``scripts/fonts/JetBrainsMono-subset.b64`` by default
    and is checked in alongside the source font. Validates that the cached
    subset covers every glyph that scenes need; raises ``SceneValidationError``
    with a rebuild instruction if a glyph is missing.
    """
    if cache_path is None:
        cache_path = font_path.parent / "JetBrainsMono-subset.b64"

    if rebuild or not cache_path.exists():
        font_bytes = subset_font(font_path, needed_glyphs)
        b64 = base64.b64encode(font_bytes).decode("ascii")
        cache_path.write_text(b64 + "\n", encoding="utf-8")
        return b64

    cached_b64 = cache_path.read_text(encoding="utf-8").strip()
    cached_font = TTFont(io.BytesIO(base64.b64decode(cached_b64)))
    cached_codepoints = set(cached_font.getBestCmap().keys())
    # Compare against the FULL source font, not just the cached subset — some
    # glyphs (Braille spinners, for example) aren't in JetBrains Mono at all,
    # and the browser will font-stack-fall-back to ui-monospace for those.
    # Only warn when a glyph is in the source font but missing from the cache,
    # which means the cache needs rebuilding.
    source_font = TTFont(str(font_path))
    source_codepoints = set(source_font.getBestCmap().keys())
    missing_from_cache = sorted(
        {c for c in needed_glyphs if ord(c) in source_codepoints and ord(c) not in cached_codepoints}
    )
    if missing_from_cache:
        raise SceneValidationError(
            f"Font cache is missing glyphs that JetBrains Mono supports: {missing_from_cache!r}. "
            f"Rebuild the cache with: python scripts/build_mockups.py --rebuild-font-cache"
        )
    # Glyphs not in the source font fall through to the browser's font stack
    # (ui-monospace, SFMono-Regular, Menlo, Consolas, monospace) — no warning.
    return cached_b64


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------


def render_scene(
    scene: dict[str, Any],
    template: jinja2.Template,
    font_b64: str,
    theme: str,
) -> str:
    palette = PALETTES[theme]
    content_svg = render_lines(scene, palette)
    height = compute_height(scene)
    rendered = template.render(
        width=scene["width"],
        width_half=scene["width"] // 2,
        height=height,
        title_escaped=xml_escape(scene["title"]),
        description_escaped=xml_escape(scene["description"].strip()),
        content_svg=content_svg,
        font_base64=font_b64,
        palette=palette,
    )
    # Ensure a single trailing newline for POSIX-friendly files.
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def generate_all(
    scenes_dir: Path,
    output_dir: Path,
    template_path: Path,
    font_path: Path,
    rebuild_font_cache: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_paths = sorted(scenes_dir.glob("*.yaml"))
    scenes: list[tuple[Path, dict[str, Any]]] = []
    for p in scene_paths:
        scene = load_scene(p)
        validate_scene(scene, p)
        scenes.append((p, scene))

    all_glyphs = set(CHROME_GLYPHS)
    for _, scene in scenes:
        all_glyphs.update(collect_glyphs(scene))

    font_b64 = load_or_build_font_cache(font_path, all_glyphs, rebuild=rebuild_font_cache)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        keep_trailing_newline=False,
    )
    template = env.get_template(template_path.name)

    written: list[Path] = []
    for _, scene in scenes:
        for theme in ("dark", "light"):
            svg = render_scene(scene, template, font_b64, theme)
            out = output_dir / f"{scene['id']}-{theme}.svg"
            out.write_text(svg, encoding="utf-8")
            written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SVG terminal mockups for the README")
    parser.add_argument("--scenes-dir", type=Path, default=Path("docs/mockup-scenes"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/mockups"))
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("scripts/build_mockups_template.svg.j2"),
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("scripts/fonts/JetBrainsMono-Regular.ttf"),
    )
    parser.add_argument(
        "--rebuild-font-cache",
        action="store_true",
        help="Re-subset the font from --font and overwrite the cache file. "
        "Required after adding scene glyphs not covered by the existing cache.",
    )
    args = parser.parse_args(argv)

    try:
        written = generate_all(
            args.scenes_dir,
            args.output_dir,
            args.template,
            args.font,
            rebuild_font_cache=args.rebuild_font_cache,
        )
    except SceneValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {len(written)} SVG files in {args.output_dir}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
