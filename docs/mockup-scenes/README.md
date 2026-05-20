# Mockup scenes — authoring guide

This directory holds the **source of truth** for the terminal mockup SVGs that appear in the project's [README.md](../../README.md). Each `.yaml` file here describes one mockup scene; the generator at [`scripts/build_mockups.py`](../../scripts/build_mockups.py) renders every scene into two SVG files (light + dark palette) under [`docs/assets/mockups/`](../assets/mockups/).

**You never edit the SVGs by hand.** Edit the YAML, re-run the generator, commit both.

```bash
python scripts/build_mockups.py
git add docs/mockup-scenes/<scene>.yaml docs/assets/mockups/<scene>-{light,dark}.svg
```

CI fails if checked-in SVGs are out of sync with the YAML — the `code_quality_checks.sh` script enforces this locally too.

## File location & naming

```
docs/mockup-scenes/<scene-id>.yaml          # source
docs/assets/mockups/<scene-id>-light.svg    # generated
docs/assets/mockups/<scene-id>-dark.svg     # generated
```

`<scene-id>` must be kebab-case (e.g., `hero`, `clink-subagent`, `gallery-api-lookup-before`). It becomes part of the output filename.

## Schema

```yaml
id: hero                          # required, must match the filename stem
title: ~/myapp — claude code      # required, rendered in the window title bar
description: |                    # required, becomes <desc> for screen readers
  Multi-model handoff: O3 debugs a race condition while Gemini Pro
  validates the fix. Both share continuation thread 8f3e2a.
width: 820                        # required, SVG width in pixels (height auto-derived)
lines:                            # required, list of typed line entries (see "Line kinds")
  - { kind: prompt, text: "debug this race condition with o3..." }
  - { kind: blank }
  - kind: tool_call
    spinner: ⠋
    name: "unison · debug"
    meta: ["o3", "thread 8f3e2a"]
```

### Required fields

| Field | Type | Notes |
|---|---|---|
| `id` | string | kebab-case, must match filename stem |
| `title` | string | shown in window chrome title bar |
| `description` | string | one or more sentences; embedded as SVG `<desc>` for accessibility |
| `width` | integer | pixels; height is computed from line count + padding |
| `lines` | list | sequence of line entries (see kinds below) |

Missing or extra top-level fields cause the generator to fail with the file path and the offending field name. There are no optional top-level fields.

## Line kinds

Each entry in `lines` must have a `kind` field from this enumeration:

### `prompt`

User input. Rendered with a `›` prefix in the prompt color.

```yaml
- { kind: prompt, text: "debug this race condition with o3..." }
```

Fields: `text` (required, string).

### `output`

Plain terminal output text. Rendered in default foreground color unless `color` is set.

```yaml
- { kind: output, text: "Two models, one thread, one commit-ready fix." }
- { kind: output, text: "✓ tests pass", color: success }
```

Fields: `text` (required, string), `color` (optional, see "Color tokens").

### `tool_call`

A Unison tool invocation header. Rendered with a spinner glyph, the tool name in the info color, and right-aligned metadata pills.

```yaml
- kind: tool_call
  spinner: ⠋
  name: "unison · debug"
  meta: ["o3", "thread 8f3e2a"]
```

Fields:
- `spinner` (required, string): typically `⠋`, `⠙`, `⠹`, or `⠸` — one spinner-frame glyph.
- `name` (required, string): the tool identifier (e.g., `unison · chat`, `unison · clink`).
- `meta` (optional, list of strings): right-aligned chips with model name, thread ID, etc.

### `tree`

A bulleted indented list rendered with box-drawing prefixes (`├` and `└`). The last item gets the `└` prefix; earlier items get `├`.

```yaml
- kind: tree
  items:
    - "walking concurrent_writer.py  847 LOC"
    - "hypothesis: TOCTOU on fd between line 231→234"
    - { text: "confidence: high", color: success }
```

Each item is either a string or a `{text, color}` object. The optional `color` overrides the default tree-item color.

### `status`

A summary line with an icon prefix in a colored chip.

```yaml
- { kind: status, icon: "◆", text: "1 blocker · 1 nit", color: warning }
```

Fields: `icon` (required, string — one glyph like `◆`, `✓`, `⚠`, `⚡`, `⚖`), `text` (required, string), `color` (optional, see "Color tokens").

### `blank`

A blank vertical-space line. No fields besides `kind`.

```yaml
- { kind: blank }
```

### `box`

A nested box-drawn panel (used for "subagent finished" frames and similar). Rendered with `┌─` / `│` / `└─` borders and a centered title.

```yaml
- kind: box
  title: "subagent finished"
  lines:
    - "1 critical · 3 medium · 0 low"
    - "codex context: unchanged"
```

Fields: `title` (required, string), `lines` (required, list of strings).

## Color tokens

Line `color` fields use semantic tokens, not raw hex values. The template maps each token to a palette-specific color so the same scene renders consistently across light and dark variants.

| Token | Semantic |
|---|---|
| `default` | Default foreground (light: dark gray, dark: off-white) |
| `muted` | De-emphasized text (chrome details, meta pills) |
| `prompt` | The `›` prefix on prompt lines |
| `info` | Tool call names, thread IDs |
| `success` | `✓`, "passed", "current" |
| `warning` | `⚠`, "nit", "stale" |
| `error` | `✗`, "blocker", "failed" |
| `accent` | Highlighted callouts, status chips |

To add a new token, edit `scripts/build_mockups_template.svg.j2` (both palette dicts) and document it here.

## Window chrome

Every scene gets the same window chrome — rounded corners, three traffic-light dots (red/yellow/green), a centered title bar — defined once in the template. You can't override chrome from a scene YAML; per-scene customization is limited to `title`, `width`, `description`, and `lines`. To change chrome appearance for all scenes, edit the template.

## Before/after pairs (API-Lookup, Challenge)

For the `gallery-api-lookup-{before,after}` and `gallery-challenge-{before,after}` scene pairs, the **before** variant's title bar omits the `+ unison` suffix so a reader can tell at a glance which side represents the with-Unison state:

```yaml
# gallery-api-lookup-before.yaml
title: ~/myapp — claude code

# gallery-api-lookup-after.yaml
title: ~/myapp — claude code + unison
```

The window chrome (size, palette, layout) stays identical across the pair — the title bar is the only signal.

## Accessibility

The scene's `title` and `description` fields are embedded as `<title>` and `<desc>` SVG elements (as the first children of the root `<svg>`), so screen readers announce them before describing the visual content. Write the `description` like an alt-text caption: one or two sentences describing what the mockup demonstrates, not what it looks like.

## Adding a new scene

1. Create `docs/mockup-scenes/<scene-id>.yaml` matching the schema above.
2. Run `python scripts/build_mockups.py` to generate `docs/assets/mockups/<scene-id>-light.svg` and `docs/assets/mockups/<scene-id>-dark.svg`.
3. If the generator reports `Font cache is missing glyphs`, rebuild the font subset cache (see "Font cache" below).
4. Reference the SVGs in `README.md` via the `<picture>` pattern (see existing mockups for examples).
5. Commit the scene YAML, both SVGs, the cache file (if rebuilt), and the README change in one commit.

If you change an existing scene, just regenerate — the deterministic generator produces byte-identical output for unchanged input.

## Font cache

JetBrains Mono Regular is checked in at `scripts/fonts/JetBrainsMono-Regular.ttf`. The generator does **not** re-subset the font on every run — instead, a pre-built subset lives at `scripts/fonts/JetBrainsMono-subset.b64` (base64-encoded, ~35KB, diff-able as text) and is embedded directly into every SVG via `@font-face`.

Why: fontTools' subsetter has internal state that varies across processes (ASLR-affected memory layout sneaks into the binary output), so re-subsetting on every run would produce different bytes each time and break the CI drift check.

When to rebuild the cache:

- **A new scene introduces a glyph not yet in the cache.** The generator will refuse to run and tell you to rebuild.
- **You change the source font** (unlikely).

How to rebuild:

```bash
python scripts/build_mockups.py --rebuild-font-cache
git add scripts/fonts/JetBrainsMono-subset.b64
```

Commit the regenerated cache alongside the scene change.

Glyphs that JetBrains Mono doesn't support (e.g., the `⠋` / `⠙` Braille-pattern spinner glyphs) fall through to the browser's font stack (`ui-monospace`, `SFMono-Regular`, `Menlo`, `Consolas`, `monospace`) — no rebuild needed.
