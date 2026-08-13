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
import time
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

# Bulk directories pruned from snapshot traversal by default. These are
# dependency stores, VCS internals, and build outputs: they routinely hold
# tens of thousands of entries that exhaust the entry cap (destroying coverage
# of the files that matter) and dominate wall-clock cost — observed at 60-90s
# per snapshot on a large OneDrive-synced repo, which starved the actual CLI
# call of the host's tool-timeout budget. Security tradeoff, stated honestly:
# a CLI write hidden inside one of these directories goes undetected — but
# under the entry cap on any large repo it already did, while also making the
# rest of the verification blind. Pass ``prune_dir_names=frozenset()`` to
# disable pruning.
PRUNED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".cache",
        ".next",
        ".turbo",
        ".gradle",
        "dist",
        "build",
        "target",
    }
)


@dataclass
class SnapshotStats:
    """Out-parameter describing how a snapshot traversal actually went.

    Callers that report verification coverage pass an instance to
    :func:`capture_snapshot`, which fills it in. Kept separate from the
    returned mapping so existing callers (and test doubles returning plain
    dicts) stay valid.
    """

    entry_count: int = 0
    truncated_by_entries: bool = False
    truncated_by_time: bool = False
    elapsed_seconds: float = 0.0

    @property
    def truncated(self) -> bool:
        return self.truncated_by_entries or self.truncated_by_time


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
    max_depth: int | None = None,
    *,
    include_ignored: bool = False,
    max_entries: int = 50_000,
    time_budget_seconds: float | None = None,
    prune_dir_names: frozenset[str] = PRUNED_DIR_NAMES,
    stats: SnapshotStats | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Capture a filesystem snapshot of a directory.

    Returns a dict mapping relative file paths to ``(mtime_ns, ctime_ns, size)``
    tuples. ``ctime_ns`` is included so a content edit that restores the
    original mtime and byte length (an evasion of the old ``(mtime_ns, size)``
    key) still registers as a change — ctime updates on any write and cannot be
    portably restored. Symlinks are recorded (never followed) so
    creating/retargeting/deleting one is detected.

    The walk uses ``os.scandir`` (type/stat data comes from the directory read
    on most platforms instead of per-entry syscalls) and prunes bulk
    directories (:data:`PRUNED_DIR_NAMES`) by default — both matter enormously
    on network/cloud-synced filesystems where each stat is slow.

    Args:
        directory: Root directory to snapshot.
        max_depth: Maximum directory depth to traverse. ``None`` (default) means
            no depth limit. Read-only verification must not silently miss writes
            nested below an arbitrary depth, so it relies on this default;
            traversal is still bounded by ``max_entries``.
        include_ignored: When True, do NOT skip ``.gitignore``'d or transient
            files. Read-only verification passes this so a write to a gitignored
            path (e.g. ``.env``, ``secrets/``) or a ``*.log`` cannot evade
            detection. When False (default) those files are excluded to reduce
            noise.
        max_entries: Safety cap on the number of files recorded; if exceeded a
            warning is logged and coverage becomes partial (never silent).
        time_budget_seconds: Optional wall-clock ceiling for the traversal.
            When exceeded the walk stops where it is, a warning is logged, and
            coverage becomes partial — a bounded verification delay instead of
            an unbounded stall that can outlive the caller's own timeout.
        prune_dir_names: Directory basenames skipped entirely (not recorded,
            not descended into). Defaults to :data:`PRUNED_DIR_NAMES`; pass
            ``frozenset()`` for the previous unpruned behavior.
        stats: Optional :class:`SnapshotStats` the traversal fills in, for
            callers that report verification coverage.

    Returns:
        Dict of ``{relative_path: (mtime_ns, ctime_ns, size)}``.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        return {}

    gitignore_patterns: list[str] = [] if include_ignored else _load_gitignore_patterns(root)
    snapshot: dict[str, tuple[int, int, int]] = {}
    truncated_entries = False
    truncated_time = False
    started = time.monotonic()
    root_str = str(root)

    def _out_of_time() -> bool:
        return time_budget_seconds is not None and (time.monotonic() - started) >= time_budget_seconds

    def _walk(current: str, depth: int) -> None:
        nonlocal truncated_entries, truncated_time
        if max_depth is not None and depth > max_depth:
            return
        try:
            with os.scandir(current) as scanner:
                entries = sorted(scanner, key=lambda e: e.name)
        except OSError:
            return

        for entry in entries:
            if len(snapshot) >= max_entries:
                truncated_entries = True
                return
            if _out_of_time():
                truncated_time = True
                return

            rel = os.path.relpath(entry.path, root_str)

            if not include_ignored:
                if _is_gitignored(rel, gitignore_patterns):
                    continue
                if _is_transient(rel):
                    continue

            try:
                if entry.is_symlink():
                    # Record the link itself without following it, so a symlink
                    # created/retargeted/deleted by the CLI is detected instead
                    # of being an invisible write channel.
                    st = entry.stat(follow_symlinks=False)
                    snapshot[rel] = (st.st_mtime_ns, st.st_ctime_ns, st.st_size)
                elif entry.is_file():
                    st = entry.stat()
                    snapshot[rel] = (st.st_mtime_ns, st.st_ctime_ns, st.st_size)
                elif entry.is_dir():
                    if entry.name in prune_dir_names:
                        continue
                    _walk(entry.path, depth + 1)
            except OSError:
                continue

            if truncated_entries or truncated_time:
                return

    _walk(root_str, 1)

    if stats is not None:
        stats.entry_count = len(snapshot)
        stats.truncated_by_entries = truncated_entries
        stats.truncated_by_time = truncated_time
        stats.elapsed_seconds = time.monotonic() - started

    if truncated_entries:
        logger.warning(
            "Filesystem snapshot of %s hit the %d-entry cap; read-only verification coverage is partial",
            root,
            max_entries,
        )
    if truncated_time:
        logger.warning(
            "Filesystem snapshot of %s exceeded the %.1fs time budget after %d entries; "
            "read-only verification coverage is partial",
            root,
            time_budget_seconds,
            len(snapshot),
        )
    return snapshot


def diff_snapshots(
    before: dict[str, tuple[int, ...]],
    after: dict[str, tuple[int, ...]],
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
