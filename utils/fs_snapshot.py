"""Lightweight filesystem snapshot utility for read-only verification.

Captures directory state as ``{relative_path: (mtime_ns, size)}`` dicts
and diffs two snapshots to detect created, modified, or deleted files.
Used by the clink tool's read-only sandbox to verify that an external CLI
did not modify files despite sandbox flags.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Files/patterns that change frequently for reasons unrelated to the CLI
TRANSIENT_PATTERNS: list[str] = [
    "*.pyc",
    "__pycache__",
    "*.log",
    ".DS_Store",
    "*.swp",
    "*.swo",
    "*~",
    ".pytest_cache",
]


@dataclass
class SnapshotDiff:
    """Result of comparing two filesystem snapshots."""

    created: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.created or self.modified or self.deleted)

    def to_dict(self) -> dict:
        return {
            "created": self.created,
            "modified": self.modified,
            "deleted": self.deleted,
        }


def _path_matches_pattern(rel_path: str, pattern: str) -> bool:
    """Match ``rel_path`` against a single pattern using the explicit DSL.

    A pattern is one of:

    - An exact relative path (matched by string equality).
    - A directory prefix ending in ``"/**"`` (matches the prefix path itself
      OR any descendant of it).

    No ``fnmatch`` involvement — stdlib ``fnmatch`` does not implement
    bash-style globstar and produces incorrect results for path-shaped
    strings on every supported Python version.
    """
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return rel_path == prefix or rel_path.startswith(prefix + "/")
    return rel_path == pattern


def classify_changes(
    diff: SnapshotDiff,
    ignore_patterns: tuple[str, ...],
) -> tuple[SnapshotDiff, SnapshotDiff]:
    """Split a snapshot diff into ``(by_model, by_cli_bookkeeping)``.

    Each path in ``diff`` is checked against ``ignore_patterns`` using the
    per-CLI matching rules implemented by :func:`_path_matches_pattern`. Paths
    that match any pattern classify as CLI bookkeeping; everything else
    classifies as a model write.

    The two returned diffs together contain every change present in the input
    (no information loss). Empty buckets are returned as ``SnapshotDiff()``
    defaults; the function never returns ``None`` for either side.
    """
    by_model = SnapshotDiff()
    by_bookkeeping = SnapshotDiff()

    def _route(paths: list[str], target_attr: str) -> None:
        for path in paths:
            matched = any(_path_matches_pattern(path, p) for p in ignore_patterns)
            target = by_bookkeeping if matched else by_model
            getattr(target, target_attr).append(path)

    _route(diff.created, "created")
    _route(diff.modified, "modified")
    _route(diff.deleted, "deleted")

    return by_model, by_bookkeeping


def _is_transient(rel_path: str) -> bool:
    """Check if a path matches a transient file pattern."""
    name = os.path.basename(rel_path)
    parts = rel_path.replace("\\", "/").split("/")
    for pattern in TRANSIENT_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
    return False


def _load_gitignore_patterns(directory: Path) -> list[str]:
    """Load .gitignore patterns from a directory, if present."""
    gitignore = directory / ".gitignore"
    if not gitignore.is_file():
        return []
    patterns = []
    try:
        for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    except OSError:
        pass
    return patterns


def _is_gitignored(rel_path: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any gitignore pattern."""
    name = os.path.basename(rel_path)
    parts = rel_path.replace("\\", "/").split("/")
    for pattern in patterns:
        clean = pattern.rstrip("/")
        if fnmatch.fnmatch(name, clean):
            return True
        if fnmatch.fnmatch(rel_path, clean):
            return True
        if any(fnmatch.fnmatch(part, clean) for part in parts):
            return True
    return False


def capture_snapshot(
    directory: str | Path,
    max_depth: int = 3,
) -> dict[str, tuple[int, int]]:
    """Capture a filesystem snapshot of a directory.

    Returns a dict mapping relative file paths to ``(mtime_ns, size)`` tuples.
    Respects ``.gitignore`` patterns and excludes transient files.

    Args:
        directory: Root directory to snapshot.
        max_depth: Maximum directory depth to traverse (default 3).

    Returns:
        Dict of ``{relative_path: (mtime_ns, size)}``.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        return {}

    gitignore_patterns = _load_gitignore_patterns(root)
    snapshot: dict[str, tuple[int, int]] = {}

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(current.iterdir())
        except PermissionError:
            return

        for entry in entries:
            try:
                rel = str(entry.relative_to(root))
            except ValueError:
                continue

            if _is_gitignored(rel, gitignore_patterns):
                continue
            if _is_transient(rel):
                continue

            if entry.is_symlink():
                continue
            if entry.is_file():
                try:
                    stat = entry.stat()
                    snapshot[rel] = (stat.st_mtime_ns, stat.st_size)
                except OSError:
                    pass
            elif entry.is_dir():
                _walk(entry, depth + 1)

    _walk(root, 1)
    return snapshot


def diff_snapshots(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> SnapshotDiff:
    """Compare two snapshots and return the differences.

    Args:
        before: Snapshot taken before execution.
        after: Snapshot taken after execution.

    Returns:
        SnapshotDiff with created, modified, and deleted path lists.
    """
    before_keys = set(before)
    after_keys = set(after)

    created = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)
    modified = sorted(path for path in before_keys & after_keys if before[path] != after[path])

    return SnapshotDiff(created=created, modified=modified, deleted=deleted)
