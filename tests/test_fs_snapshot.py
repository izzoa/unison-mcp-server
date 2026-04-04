"""Tests for utils/fs_snapshot.py — snapshot capture, diffing, gitignore, and transient filtering."""

from __future__ import annotations

import os

from utils.fs_snapshot import SnapshotDiff, _is_transient, capture_snapshot, diff_snapshots


class TestCaptureSnapshot:
    def test_captures_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.py").write_text("world")
        snap = capture_snapshot(tmp_path)
        assert "a.txt" in snap
        assert "b.py" in snap
        assert len(snap) == 2

    def test_captures_mtime_and_size(self, tmp_path):
        (tmp_path / "file.txt").write_text("12345")
        snap = capture_snapshot(tmp_path)
        mtime_ns, size = snap["file.txt"]
        assert size == 5
        assert mtime_ns > 0

    def test_respects_max_depth(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (tmp_path / "a" / "shallow.txt").write_text("s")
        (deep / "deep.txt").write_text("d")
        snap = capture_snapshot(tmp_path, max_depth=2)
        assert os.path.join("a", "shallow.txt") in snap
        assert os.path.join("a", "b", "c", "d", "deep.txt") not in snap

    def test_respects_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
        (tmp_path / "app.py").write_text("code")
        (tmp_path / "server.log").write_text("log")
        build = tmp_path / "build"
        build.mkdir()
        (build / "output.js").write_text("js")
        snap = capture_snapshot(tmp_path)
        assert "app.py" in snap
        assert "server.log" not in snap
        assert os.path.join("build", "output.js") not in snap

    def test_excludes_transient_files(self, tmp_path):
        (tmp_path / "real.py").write_text("code")
        (tmp_path / "cache.pyc").write_text("bytecode")
        (tmp_path / ".DS_Store").write_bytes(b"\x00")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.cpython-312.pyc").write_text("cached")
        snap = capture_snapshot(tmp_path)
        assert "real.py" in snap
        assert "cache.pyc" not in snap
        assert ".DS_Store" not in snap
        assert os.path.join("__pycache__", "mod.cpython-312.pyc") not in snap

    def test_empty_directory(self, tmp_path):
        snap = capture_snapshot(tmp_path)
        assert snap == {}

    def test_nonexistent_directory(self):
        snap = capture_snapshot("/nonexistent/path")
        assert snap == {}


class TestDiffSnapshots:
    def test_detects_created_files(self):
        before = {"a.txt": (100, 10)}
        after = {"a.txt": (100, 10), "b.txt": (200, 20)}
        diff = diff_snapshots(before, after)
        assert diff.created == ["b.txt"]
        assert diff.modified == []
        assert diff.deleted == []

    def test_detects_modified_files(self):
        before = {"a.txt": (100, 10)}
        after = {"a.txt": (200, 15)}
        diff = diff_snapshots(before, after)
        assert diff.modified == ["a.txt"]
        assert diff.created == []
        assert diff.deleted == []

    def test_detects_deleted_files(self):
        before = {"a.txt": (100, 10), "b.txt": (100, 10)}
        after = {"a.txt": (100, 10)}
        diff = diff_snapshots(before, after)
        assert diff.deleted == ["b.txt"]

    def test_no_changes(self):
        snap = {"a.txt": (100, 10)}
        diff = diff_snapshots(snap, snap)
        assert not diff.has_changes

    def test_has_changes_property(self):
        diff = SnapshotDiff(created=["new.txt"])
        assert diff.has_changes

    def test_to_dict(self):
        diff = SnapshotDiff(created=["a"], modified=["b"], deleted=["c"])
        assert diff.to_dict() == {"created": ["a"], "modified": ["b"], "deleted": ["c"]}

    def test_combined_changes(self):
        before = {"keep.txt": (100, 10), "modify.txt": (100, 10), "delete.txt": (100, 5)}
        after = {"keep.txt": (100, 10), "modify.txt": (200, 15), "create.txt": (300, 8)}
        diff = diff_snapshots(before, after)
        assert diff.created == ["create.txt"]
        assert diff.modified == ["modify.txt"]
        assert diff.deleted == ["delete.txt"]


class TestIsTransient:
    def test_pyc_files(self):
        assert _is_transient("module.pyc")
        assert _is_transient(os.path.join("pkg", "module.pyc"))

    def test_pycache_directory(self):
        assert _is_transient(os.path.join("__pycache__", "mod.cpython-312.pyc"))

    def test_log_files(self):
        assert _is_transient("server.log")

    def test_ds_store(self):
        assert _is_transient(".DS_Store")

    def test_normal_files_not_transient(self):
        assert not _is_transient("main.py")
        assert not _is_transient(os.path.join("src", "app.py"))
