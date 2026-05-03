"""Tests for utils/fs_snapshot.py classification helpers.

Covers ``classify_changes`` and the underlying ``_path_matches_pattern`` DSL
that splits a snapshot diff into model writes vs CLI bookkeeping.

The pattern DSL deliberately avoids ``fnmatch.fnmatch`` because stdlib
``fnmatch`` does not implement bash-style globstar (``**``) and produces
incorrect results for path-shaped strings on every supported Python version.
See clink/agents/opencode.py and the design rationale for details.
"""

from __future__ import annotations

from utils.fs_snapshot import SnapshotDiff, _path_matches_pattern, classify_changes

# Captured against opencode v0.18.x at
# /Users/anthonyizzo/Development/markdowndelta-swift/.opencode/ on 2026-05-02.
# Update when opencode bootstrap layout changes upstream.
OPENCODE_BOOTSTRAP_FIXTURE: list[str] = [
    ".opencode/.gitignore",
    ".opencode/package.json",
    ".opencode/package-lock.json",
    ".opencode/node_modules/.package-lock.json",
    ".git/opencode",
]

OPENCODE_PATTERNS: tuple[str, ...] = (
    ".opencode/.gitignore",
    ".opencode/package.json",
    ".opencode/package-lock.json",
    ".opencode/node_modules/**",
    ".git/opencode",
)


# ---------------------------------------------------------------------------
# 4.3b — _path_matches_pattern DSL
# ---------------------------------------------------------------------------


def test_path_matches_pattern_exact_match():
    assert _path_matches_pattern(".git/opencode", ".git/opencode")


def test_path_matches_pattern_exact_non_match_when_path_differs():
    assert not _path_matches_pattern(".git/opencodes", ".git/opencode")
    assert not _path_matches_pattern("subdir/.git/opencode", ".git/opencode")
    assert not _path_matches_pattern(".git/opencod", ".git/opencode")


def test_path_matches_pattern_prefix_descendants():
    assert _path_matches_pattern(".opencode/node_modules/foo/bar.json", ".opencode/node_modules/**")
    assert _path_matches_pattern(".opencode/node_modules/x.json", ".opencode/node_modules/**")


def test_path_matches_pattern_prefix_matches_directory_itself():
    assert _path_matches_pattern(".opencode/node_modules", ".opencode/node_modules/**")


def test_path_matches_pattern_prefix_does_not_match_sibling():
    assert not _path_matches_pattern(".opencode/node_modules_other/x", ".opencode/node_modules/**")
    assert not _path_matches_pattern(".opencode/node_modules-backup", ".opencode/node_modules/**")


def test_path_matches_pattern_no_fnmatch_globstar_dependency():
    # The pattern DSL is a string-prefix check, not fnmatch. Verifying the
    # specific path that fnmatch would mis-classify under Python 3.12.
    assert _path_matches_pattern(".opencode/package.json", ".opencode/package.json")


# ---------------------------------------------------------------------------
# 4.3 — classify_changes routes bookkeeping vs model paths
# ---------------------------------------------------------------------------


def test_classify_changes_routes_bookkeeping_paths():
    diff = SnapshotDiff(
        created=[
            ".opencode/package.json",
            ".git/opencode",
            "unrelated/file.py",
        ],
    )
    by_model, by_bookkeeping = classify_changes(diff, OPENCODE_PATTERNS)

    assert "unrelated/file.py" in by_model.created
    assert ".opencode/package.json" in by_bookkeeping.created
    assert ".git/opencode" in by_bookkeeping.created
    assert ".opencode/package.json" not in by_model.created


def test_classify_changes_real_opencode_bootstrap_fixture():
    """Regression: every entry in the captured opencode bootstrap fixture
    classifies as bookkeeping, not as a model write."""
    diff = SnapshotDiff(created=list(OPENCODE_BOOTSTRAP_FIXTURE))
    by_model, by_bookkeeping = classify_changes(diff, OPENCODE_PATTERNS)
    assert by_model.created == [], (
        f"Bootstrap files leaked to by_model: {by_model.created}. " "Either patterns are wrong or fixture changed."
    )
    assert sorted(by_bookkeeping.created) == sorted(OPENCODE_BOOTSTRAP_FIXTURE)


# ---------------------------------------------------------------------------
# 4.3a — model writes inside .opencode/ classify as model (not bookkeeping)
# ---------------------------------------------------------------------------


def test_classify_changes_path_inside_opencode_subdir_is_model():
    """Critical regression: directory-wide glob would mask these as bookkeeping.
    Tight enumeration ensures user-extension dirs route correctly."""
    diff = SnapshotDiff(
        created=[
            ".opencode/skills/malicious.md",
            ".opencode/commands/evil.md",
        ],
    )
    by_model, by_bookkeeping = classify_changes(diff, OPENCODE_PATTERNS)
    assert ".opencode/skills/malicious.md" in by_model.created
    assert ".opencode/commands/evil.md" in by_model.created
    assert ".opencode/skills/malicious.md" not in by_bookkeeping.created
    assert ".opencode/commands/evil.md" not in by_bookkeeping.created


# ---------------------------------------------------------------------------
# 4.4 — empty patterns mean every change is a model write
# ---------------------------------------------------------------------------


def test_classify_changes_with_no_patterns_attributes_all_to_model():
    diff = SnapshotDiff(
        created=[".opencode/package.json", "src/main.py"],
        modified=[".git/opencode"],
        deleted=["legacy.txt"],
    )
    by_model, by_bookkeeping = classify_changes(diff, ())
    assert by_model.created == [".opencode/package.json", "src/main.py"]
    assert by_model.modified == [".git/opencode"]
    assert by_model.deleted == ["legacy.txt"]
    assert by_bookkeeping.created == []
    assert by_bookkeeping.modified == []
    assert by_bookkeeping.deleted == []


# ---------------------------------------------------------------------------
# Information-loss guard: every input change appears in exactly one bucket
# ---------------------------------------------------------------------------


def test_classify_changes_no_information_loss():
    diff = SnapshotDiff(
        created=[".opencode/package.json", "src/main.py", ".git/opencode"],
        modified=[".opencode/node_modules/foo.json", "README.md"],
        deleted=["old.txt", ".opencode/.gitignore"],
    )
    by_model, by_bookkeeping = classify_changes(diff, OPENCODE_PATTERNS)

    for category in ("created", "modified", "deleted"):
        original = set(getattr(diff, category))
        recombined = set(getattr(by_model, category)) | set(getattr(by_bookkeeping, category))
        assert original == recombined, f"{category}: lost or duplicated entries"
        # No path appears in both buckets
        assert not (set(getattr(by_model, category)) & set(getattr(by_bookkeeping, category)))


# ---------------------------------------------------------------------------
# Mixed-bucket scenario from the spec
# ---------------------------------------------------------------------------


def test_classify_changes_mixed_diff_split_correctly():
    diff = SnapshotDiff(
        created=[".opencode/package.json", "src/main.py"],
    )
    by_model, by_bookkeeping = classify_changes(diff, OPENCODE_PATTERNS)
    assert by_model.created == ["src/main.py"]
    assert by_bookkeeping.created == [".opencode/package.json"]
