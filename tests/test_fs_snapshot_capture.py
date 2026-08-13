"""Tests for capture_snapshot traversal behavior: pruning, budgets, stats.

The read-only verifier walks the CLI's working directory twice per call.
Observed live: an unpruned, unbounded walk of a large OneDrive-synced repo
took 60-90s per snapshot — longer than some MCP hosts' tool timeouts — so
the walk prunes bulk directories and honors a wall-clock budget.
"""

from __future__ import annotations

import os

from utils.fs_snapshot import PRUNED_DIR_NAMES, SnapshotStats, capture_snapshot


def test_pruned_dirs_are_skipped(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.txt").write_text("x")
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "big.js").write_text("y")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref")

    snap = capture_snapshot(tmp_path, include_ignored=True)

    keys = set(snap)
    assert os.path.join("src", "a.txt") in keys
    assert not any(k.startswith("node_modules") for k in keys)
    assert not any(k.startswith(".git") for k in keys)


def test_prune_disabled_records_everything(tmp_path):
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "dep.js").write_text("y")

    snap = capture_snapshot(tmp_path, include_ignored=True, prune_dir_names=frozenset())

    assert os.path.join("node_modules", "dep.js") in snap


def test_time_budget_marks_partial(tmp_path):
    for i in range(50):
        (tmp_path / f"f{i}.txt").write_text("x")
    stats = SnapshotStats()

    snap = capture_snapshot(tmp_path, include_ignored=True, time_budget_seconds=0.0, stats=stats)

    assert stats.truncated_by_time is True
    assert stats.truncated is True
    assert len(snap) <= 1


def test_stats_populated_on_full_walk(tmp_path):
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    stats = SnapshotStats()

    snap = capture_snapshot(tmp_path, include_ignored=True, stats=stats)

    assert len(snap) == 2
    assert stats.entry_count == 2
    assert stats.truncated is False
    assert stats.elapsed_seconds >= 0.0


def test_entry_cap_flags_truncation(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("x")
    stats = SnapshotStats()

    snap = capture_snapshot(tmp_path, include_ignored=True, max_entries=3, stats=stats)

    assert stats.truncated_by_entries is True
    assert len(snap) == 3


def test_symlinks_recorded_not_followed(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("data")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    snap = capture_snapshot(tmp_path, include_ignored=True)

    assert "real.txt" in snap
    assert "link.txt" in snap


def test_pruned_set_covers_observed_bulk_dirs():
    for name in (".git", "node_modules", ".venv", "__pycache__"):
        assert name in PRUNED_DIR_NAMES
