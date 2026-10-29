"""Derive the working directory for a clink call, without any host cooperation.

``derive_working_dir`` is a pure function over explicit inputs: it takes every candidate directory the caller
knows about and returns the one the CLI should run in, or raises ``WorkingDirError``. It never touches MCP,
globals, asyncio or conversation storage, which keeps it table-testable with real temporary repositories.

Precedence (first tier that yields a valid directory wins)::

    explicit > continuation > files > roots > manifest > env > server_cwd

Files select the project the caller means; host-declared roots are a *fence* around every derived directory and
a *selector* only when no usable files were passed. Launch-time candidates (continuation, manifest, env, server
cwd) are declared directories: they must be narrow, inside the fence, contain every file used for selection and be
the same checkout as those files.

Four outcomes are terminal (an error, never a fall-through to a lower tier), because falling through would run an
approvals-bypassed CLI somewhere unrelated to the evidence: ``ambiguous``, ``roots_conflict``,
``fallback_refused`` and budget exhaustion (``underivable`` with a timeout message).

Every filesystem probe goes through ``os.path.*`` / ``os.stat`` helpers wrapped in ``try/except`` (never
``Path.is_dir()`` / raw ``Path.resolve()``, which leak ``PermissionError`` and NUL-byte ``ValueError``), is counted,
and is preceded by a cooperative deadline check. The marker search never scans a directory: it stats a handful of
marker names per ancestor, at most ``max_depth`` ancestors up, and stops at a broad directory without evaluating it.

The roots helpers at the end of the module (``parse_roots_mode``, ``file_uri_to_dir``, ``roots_from_uris``) are pure
string functions: they turn the ``CLINK_MCP_ROOTS`` value and the ``file://`` URIs a host declared into the plain
directory paths ``derive_working_dir`` takes as ``roots``, and never touch the filesystem or raise.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import unquote_to_bytes

__all__ = [
    "DEFAULT_ROOTS_MODE",
    "FALLBACK_ROOTS_MODE",
    "ROOTS_MODES",
    "SOURCES",
    "Derivation",
    "WorkingDirError",
    "broad_directory_reason",
    "derive_working_dir",
    "file_uri_to_dir",
    "find_project_root",
    "parse_roots_mode",
    "roots_from_uris",
]

SOURCES = ("explicit", "continuation", "files", "roots", "manifest", "env", "server_cwd")

# A VCS root outranks nested package markers (monorepos): agents expect `git rev-parse --show-toplevel`.
VCS_MARKERS = (".git", ".hg", ".svn", ".jj")
PROJECT_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "mix.exs",
    "deno.json",
    "CMakeLists.txt",
)

# System directories: the directory and everything beneath it is refused (unless beneath the user's own home).
_POSIX_SYSTEM_DIRS = (
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/boot",
    "/var",
    "/root",
    "/proc",
    "/sys",
    "/dev",
    "/run",
)
_WINDOWS_SYSTEM_DIRS = (
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
)
# Shared containers: only the exact directory is refused; a project inside (/opt/x, /tmp/x) is fine.
# /System/Volumes/Data is the data volume's root (the firmlink target behind /Users, /Library, ...).
_SHARED_CONTAINERS = (
    "/Users",
    "/home",
    "/Volumes",
    "/mnt",
    "/media",
    "/opt",
    "/srv",
    "/Library",
    "/Applications",
    "/System",
    "/System/Volumes/Data",
    "/private",
    "/tmp",
    "/var/tmp",
)

HOME_REASON = "the user's home directory"
_HOME_WARNING = "working_dir is the user's home directory: the CLI can read and write everything in it"
_POLICIES = ("allow", "warn", "refuse")
_MAX_LISTED = 5
_MAX_SHOWN = 200  # characters of one caller-supplied path echoed back in a message
_MAX_CANDIDATES = 20  # entries of ``WorkingDirError.candidates``
_MAX_PATH_CHARS = 4096  # PATH_MAX on Linux; nothing longer can name a directory
_MAX_PATH_CHARS_NT = 32767


class Derivation(NamedTuple):
    """A successful derivation: the resolved directory, which tier chose it, and what to tell the caller."""

    path: Path
    source: str
    warnings: list[str]


class WorkingDirError(Exception):
    """A refusal. ``code`` is machine-readable (ambiguous, roots_conflict, fallback_refused, underivable,
    invalid_explicit); ``candidates`` helps the caller pick a ``working_dir``; ``tried`` lists the tiers visited."""

    def __init__(
        self,
        code: str,
        message: str,
        candidates: Sequence[Path] = (),
        tried: Sequence[str] = (),
        warnings: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.candidates: list[Path] = list(candidates)
        self.tried: list[str] = list(tried)
        self.warnings: list[str] = list(warnings)


class _BudgetExhausted(Exception):
    """The cooperative deadline passed before the next filesystem probe."""


def _short(raw: object) -> str:
    """``repr`` of a caller-supplied path, truncated so a huge argument cannot balloon a message."""
    text = repr(str(raw))
    return text if len(text) <= _MAX_SHOWN else text[:_MAX_SHOWN] + "...'"


def _clean(text: str) -> str:
    """Escape control characters (a directory name may hold a newline) so a message cannot forge extra lines."""
    return "".join(c if c.isprintable() else c.encode("unicode_escape").decode() for c in text)


def _under(p: Path, base: Path) -> bool:
    return p == base or base in p.parents


def _system_dir_names(nt: bool) -> tuple[str, ...]:
    """Guard entries for the platform; a seam so the Windows list is testable without Windows."""
    return _POSIX_SYSTEM_DIRS + (_WINDOWS_SYSTEM_DIRS if nt else ())


@dataclass
class _GuardEntry:
    kind: str  # "system" (the dir and everything beneath) | "container" (exact) | "temp" (exact)
    path: Path
    ident: tuple[int, int] | None  # (st_dev, st_ino); None when the entry does not exist on this machine


@dataclass
class _Ctx:
    """Per-call state: configuration, the probe helpers, and their caches and counters."""

    home_raw: Path | str | None
    temp_raw: list[str]
    max_depth: int
    deadline: float | None
    nt: bool
    home: Path | None = None  # resolved lazily with the guard (so the deadline covers it)
    temp_roots: list[Path] = field(default_factory=list)
    marker_cache: dict[Path, tuple[str, str] | None] = field(default_factory=dict)
    # Probe counters: `probes` is the marker/ancestor search (independent of how many files were passed),
    # `file_probes` is the per-file normalisation, `setup_probes` the fixed cost of building the guard.
    probes: int = 0
    file_probes: int = 0
    setup_probes: int = 0
    _stat_cache: dict[str, os.stat_result | None] = field(default_factory=dict)
    _root_cache: dict[Path, tuple[Path, str] | None] = field(default_factory=dict)
    _broad_cache: dict[Path, str | None] = field(default_factory=dict)
    _guard: list[_GuardEntry] | None = None
    _home_ident: tuple[int, int] | None = None
    _home_ancestors: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total_probes(self) -> int:
        return self.probes + self.file_probes + self.setup_probes

    # ----- probes (each: deadline check, count, try/except) ---------------------------------------

    def _tick(self, kind: str) -> None:
        if self.deadline is not None and not time.monotonic() < self.deadline:  # a NaN deadline is expired
            raise _BudgetExhausted
        if kind == "file":
            self.file_probes += 1
        elif kind == "setup":
            self.setup_probes += 1
        else:
            self.probes += 1

    def realpath(self, p: str | Path, kind: str = "search") -> Path | None:
        self._tick(kind)
        try:
            if len(str(p)) > (_MAX_PATH_CHARS_NT if self.nt else _MAX_PATH_CHARS):
                return None  # realpath cannot be interrupted and is quadratic in the length; no directory is this long
            return Path(os.path.realpath(p))
        except (OSError, RuntimeError, ValueError, TypeError):  # symlink loops, permissions, NUL bytes, bad types
            return None

    def stat(self, p: str | Path, kind: str = "search") -> os.stat_result | None:
        key = str(p)
        if key in self._stat_cache:
            return self._stat_cache[key]
        self._tick(kind)
        try:
            st: os.stat_result | None = os.stat(key)
        except (OSError, RuntimeError, ValueError):
            st = None
        self._stat_cache[key] = st
        return st

    def isdir(self, p: str | Path, kind: str = "search") -> bool:
        st = self.stat(p, kind)
        return st is not None and stat.S_ISDIR(st.st_mode)

    def isfile(self, p: str | Path, kind: str = "search") -> bool:
        st = self.stat(p, kind)
        return st is not None and stat.S_ISREG(st.st_mode)

    def lexists(self, p: str | Path) -> bool:
        self._tick("search")
        try:
            os.lstat(p)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def access(self, p: str | Path) -> bool:
        self._tick("file")
        try:
            return os.access(p, os.R_OK | os.X_OK)
        except (OSError, ValueError):
            return False

    def ident(self, p: str | Path, kind: str = "search") -> tuple[int, int] | None:
        st = self.stat(p, kind)
        return None if st is None else (st.st_dev, st.st_ino)

    # ----- the broadness guard (D3) ---------------------------------------------------------------

    def _ensure_guard(self) -> list[_GuardEntry]:
        if self._guard is not None:
            return self._guard
        if self.home_raw is not None:
            self.home = self.realpath(self.home_raw, "setup")
        self.temp_roots = [t for t in (self.realpath(r, "setup") for r in self.temp_raw) if t is not None]
        entries: list[_GuardEntry] = []
        for kind, names in (
            ("system", _system_dir_names(self.nt)),
            ("container", _SHARED_CONTAINERS),
            ("temp", tuple(str(t) for t in self.temp_roots)),
        ):
            for name in names:
                entries.append(_GuardEntry(kind, Path(name), self.ident(name, "setup")))
        if self.home is not None:
            self._home_ident = self.ident(self.home, "setup")
            # A home that is a filesystem root, a shared container or a temp directory (HOME= or HOME=/ in a
            # container or daemon) is not "the user's own tree": it would switch off the system-directory guard.
            if self.home.parent == self.home or any(
                e.kind in ("container", "temp") and self._same(e, self.home, self._home_ident) for e in entries
            ):
                self.home = None
                self._home_ident = None
        if self.home is not None:
            for a in self.home.parents:
                ai = self.ident(a, "setup")
                if ai is not None:
                    self._home_ancestors.append(ai)
        self._guard = entries
        return entries

    @staticmethod
    def _same(entry: _GuardEntry, p: Path, pid: tuple[int, int] | None) -> bool:
        """Identity comparison; entries that do not exist here are skipped, an unstat-able candidate falls back
        to text equality."""
        if entry.ident is None:
            return False
        if pid is None:
            return str(p) == str(entry.path)
        return entry.ident == pid

    def _is_home(self, p: Path) -> bool:
        if self.home is None:
            return False
        if self._home_ident is not None:
            pid = self.ident(p)
            return pid is not None and pid == self._home_ident
        return p == self.home

    def broad_reason(self, p: Path) -> str | None:
        """Why ``p`` (an already-resolved directory) must not be a working directory; None when acceptable."""
        if p in self._broad_cache:
            return self._broad_cache[p]
        reason = self._broad_reason(p)
        self._broad_cache[p] = reason
        return reason

    def _broad_reason(self, p: Path) -> str | None:
        if p.parent == p:
            return "a filesystem or drive root"
        guard = self._ensure_guard()
        pid = self.ident(p)
        if self._is_home(p):
            return HOME_REASON
        if self.home is not None:
            if pid is not None and pid in self._home_ancestors:
                return "an ancestor of the user's home directory"
            if pid is None and p in self.home.parents:
                return "an ancestor of the user's home directory"
        for e in guard:
            if e.kind == "container" and self._same(e, p, pid):
                return f"a shared container ({e.path})"
            if e.kind == "temp" and self._same(e, p, pid):
                return "the shared temp directory"
        # Everything strictly beneath the user's own home is the user's tree, never a "system directory"
        # (a root user's home is /root, and /root/project must work): the home and its ancestors are not matched,
        # but a system directory strictly beneath the home still is.
        chain = [p, *p.parents]
        home_at = next((i for i, a in enumerate(chain) if i > 0 and self._is_home(a)), len(chain))
        for a in chain[:home_at]:
            aid = self.ident(a)
            for e in guard:
                if e.kind == "system" and self._same(e, a, aid):
                    return f"a system directory ({e.path})" if a == p else f"beneath the system directory {e.path}"
        return None

    def under_temp(self, p: Path) -> bool:
        self._ensure_guard()
        return any(_under(p, t) for t in self.temp_roots)

    # ----- project markers (D4) -------------------------------------------------------------------

    def marker_at(self, d: Path) -> tuple[str, str] | None:
        """('vcs', name) | ('project', name) | None. A ``.git`` FILE counts only when it is a ``gitdir:`` pointer
        (worktrees and submodules); the pointer is never followed, the worktree top is the root. A pointer that
        cannot be read counts as one (fail closed to the narrower directory)."""
        if d in self.marker_cache:
            return self.marker_cache[d]
        found: tuple[str, str] | None = None
        for name in VCS_MARKERS:
            m = d / name
            if name == ".git":
                if self.isdir(m):
                    found = ("vcs", name)
                elif self.isfile(m) and self._is_gitdir_pointer(m):
                    found = ("vcs", name)
            elif self.lexists(m):
                found = ("vcs", name)
            if found:
                break
        if not found:
            for name in PROJECT_MARKERS:
                if self.lexists(d / name):
                    found = ("project", name)
                    break
        self.marker_cache[d] = found
        return found

    def _is_gitdir_pointer(self, m: Path) -> bool:
        self._tick("search")
        try:
            with open(m, "rb") as fh:
                return fh.read(256).startswith(b"gitdir:")
        except (OSError, ValueError):
            return True

    def project_root(self, start: Path) -> tuple[Path, str] | None:
        """Nearest VCS root at or above ``start``; else the nearest package marker; else None. Looks at ``start``
        and at most ``max_depth`` ancestors, and stops at a broad directory without evaluating it."""
        if start in self._root_cache:
            return self._root_cache[start]
        cur = start
        nearest_pkg: tuple[Path, str] | None = None
        result: tuple[Path, str] | None = None
        for _ in range(self.max_depth + 1):
            if self.broad_reason(cur):
                break
            m = self.marker_at(cur)
            if m and m[0] == "vcs":
                result = (cur, m[1])
                break
            if m and nearest_pkg is None:
                nearest_pkg = (cur, m[1])
            if cur.parent == cur:
                break
            cur = cur.parent
        if result is None:
            result = nearest_pkg
        self._root_cache[start] = result
        return result

    def vcs_root_between(self, start: Path, candidate: Path) -> Path | None:
        """A VCS root strictly below ``candidate`` on the way up from ``start`` (not bounded by ``max_depth``:
        a worktree nested far below the checkout it was created from is still a different checkout)."""
        cur = start
        while cur != candidate:
            m = self.marker_at(cur)
            if m and m[0] == "vcs":
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        return None


def _make_ctx(
    home: Path | str | None,
    temp_roots: Sequence[Path | str] | None,
    max_depth: int,
    deadline: float | None,
    nt: bool | None = None,
) -> _Ctx:
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        max_depth = 12
    if home is None:
        try:
            home = Path.home()
        except (RuntimeError, KeyError, OSError):
            home = None
    raw_temp: Sequence[Path | str] = (
        temp_roots if temp_roots is not None else ("/tmp", "/var/tmp", tempfile.gettempdir())  # noqa: S108
    )
    return _Ctx(home, [str(t) for t in raw_temp], max_depth, deadline, os.name == "nt" if nt is None else nt)


def broad_directory_reason(
    path: str | Path, *, home: Path | str | None = None, temp_roots: Sequence[Path | str] | None = None
) -> str | None:
    """Why ``path`` is a broad directory (None when acceptable). The user's home directory is reported as broad;
    a path that is relative or cannot be resolved is not acceptable either (fail closed)."""
    ctx = _make_ctx(home, temp_roots, 12, None)
    if not os.path.isabs(str(path)):
        return "not an absolute path"
    rp = ctx.realpath(path, "file")
    return "a path that cannot be resolved" if rp is None else ctx.broad_reason(rp)


def find_project_root(
    path: str | Path,
    *,
    home: Path | str | None = None,
    temp_roots: Sequence[Path | str] | None = None,
    max_depth: int = 12,
) -> Path | None:
    """The project root the files tier would derive for one existing path, or None."""
    ctx = _make_ctx(home, temp_roots, max_depth, None)
    rp = ctx.realpath(path, "file")
    st = None if rp is None else ctx.stat(rp, "file")
    if rp is None or st is None:
        return None
    hit = ctx.project_root(rp if stat.S_ISDIR(st.st_mode) else rp.parent)
    return hit[0] if hit else None


@dataclass
class _Ev:
    raw: str
    path: Path  # realpath
    start: Path  # directory the marker walk starts from (the path itself when it is a directory)


def _listed(items: Sequence[object]) -> str:
    shown = ", ".join(str(i) for i in items[:_MAX_LISTED])
    more = len(items) - _MAX_LISTED
    return shown + (f" and {more} more" if more > 0 else "")


def _listed_paths(items: Sequence[str]) -> str:
    """Like ``_listed`` for caller-supplied paths: each shown one is quoted and truncated by ``_short``."""
    shown = ", ".join(_short(i) for i in items[:_MAX_LISTED])
    more = len(items) - _MAX_LISTED
    return shown + (f" and {more} more" if more > 0 else "")


class _Deriver:
    """One derivation run: the inputs, the warnings/tried trail, and one method per tier."""

    def __init__(
        self,
        ctx: _Ctx,
        *,
        explicit: str | Path | None,
        roots: Sequence[str | Path] | None,
        file_paths: Sequence[str | Path],
        continuation_dir: str | Path | None,
        continuation_source: str | None,
        manifest_dir: str | Path | None,
        env_default: str | Path | None,
        server_cwd: str | Path | None,
        server_cwd_substituted: bool,
        fallback_policy: str | None,
        extra_warnings: Sequence[str],
    ) -> None:
        self.ctx = ctx
        self.explicit = explicit
        self.roots = roots
        self.file_paths = file_paths
        self.continuation_dir = continuation_dir
        self.continuation_source = continuation_source
        self.manifest_dir = manifest_dir
        self.env_default = env_default
        self.server_cwd = server_cwd
        self.server_cwd_substituted = server_cwd_substituted
        self.fallback_policy = fallback_policy
        self.extra = list(extra_warnings)
        self.warnings: list[str] = []
        self.tried: list[str] = []
        self.evid: list[_Ev] = []  # every existing absolute file passed
        self.sel: list[_Ev] = []  # the files that take part in selection and containment (non-scratch)
        self.fence: list[Path] = []

    # ----- result helpers -------------------------------------------------------------------------

    def _ok(self, path: Path, source: str) -> Derivation:
        return Derivation(path, source, [_clean(w) for w in self.extra + self.warnings])

    def _fail(self, code: str, message: str, candidates: Sequence[Path] = ()) -> WorkingDirError:
        return WorkingDirError(
            code, _clean(message), candidates, self.tried, [_clean(w) for w in self.extra + self.warnings]
        )

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    # ----- driver ---------------------------------------------------------------------------------

    def run(self) -> Derivation:
        self._normalize_files()
        if self.explicit is not None and str(self.explicit).strip():
            return self._explicit(str(self.explicit).strip())
        self.tried.append("explicit: not given")
        policy = self._policy()
        self._select_evidence()
        self._build_fence()

        result = (
            self._continuation()
            or self._files()
            or self._roots_selector()
            or self._declared_tier("manifest", self.manifest_dir, "manifest")
            or self._declared_tier("env", self.env_default, "environment default")
            or self._server_cwd(policy)
        )
        if result is None:
            raise self._underivable()
        return result

    def _underivable(self) -> WorkingDirError:
        detail = f" Details: {'; '.join(self.extra + self.warnings)}." if (self.extra or self.warnings) else ""
        return self._fail(
            "underivable",
            "Could not derive a safe working_dir. Pass working_dir (an absolute path to the project root) or set "
            f"CLINK_DEFAULT_WORKING_DIR. Tried: {'; '.join(self.tried)}.{detail}",
        )

    def timeout_error(self) -> WorkingDirError:
        return self._fail(
            "underivable",
            "The project-root search timed out before a working directory could be derived. "
            f"Pass working_dir (an absolute path to the project root). Tried: {'; '.join(self.tried)}.",
        )

    def _policy(self) -> str:
        raw = self.fallback_policy
        text = "" if raw is None else str(raw).strip().lower()
        if text in _POLICIES:
            return text
        if text:
            self.warn(f"CLINK_WORKING_DIR_FALLBACK={_short(raw)} is not one of allow, warn, refuse; using warn")
        return "warn"

    # ----- inputs ---------------------------------------------------------------------------------

    def _normalize_files(self) -> None:
        seen: set[Path] = set()
        relative: list[str] = []
        missing: list[str] = []
        for entry in self.file_paths or ():
            raw = str(entry)
            if not os.path.isabs(raw):
                relative.append(raw)
                continue
            rp = self.ctx.realpath(raw, "file")
            st = None if rp is None else self.ctx.stat(rp, "file")
            if rp is None or st is None:
                missing.append(raw)
                continue
            if rp in seen:
                continue
            seen.add(rp)
            self.evid.append(_Ev(raw, rp, rp if stat.S_ISDIR(st.st_mode) else rp.parent))
        # One warning per kind, naming the first few: a call passing thousands of bad paths must not produce
        # thousands of warnings (each is also logged and echoed in the error message).
        if relative:
            self.warn(f"ignored {len(relative)} non-absolute file path(s): {_listed_paths(relative)}")
        if missing:
            self.warn(
                f"ignored {len(missing)} nonexistent or inaccessible file path(s) (not used to derive "
                f"working_dir): {_listed_paths(missing)}"
            )

    def _select_evidence(self) -> None:
        """Scratch files (under a temp directory, no project root within the bound) take no part in selection or
        containment; they are reported, not used."""
        scratch = [e for e in self.evid if self.ctx.project_root(e.start) is None and self.ctx.under_temp(e.path)]
        scratch_ids = {id(e) for e in scratch}  # not `e not in scratch`: that is a quadratic list scan
        self.sel = [e for e in self.evid if id(e) not in scratch_ids]
        if scratch:
            self.warn(
                f"{len(scratch)} file(s) under the temp directory have no project root and were ignored for "
                f"selection: {_listed([str(e.path) for e in scratch])}"
            )

    def _build_fence(self) -> None:
        for entry in self.roots or ():
            raw = str(entry)
            if not os.path.isabs(raw):
                self.warn(f"ignored root {_short(raw)}: not an absolute path")
                continue
            rp = self.ctx.realpath(raw)
            if rp is None or not self.ctx.isdir(rp):
                self.warn(f"ignored root {_short(raw)}: does not exist or is not a directory")
                continue
            if rp not in self.fence:
                self.fence.append(rp)

    def _in_fence(self, p: Path) -> bool:
        return any(_under(p, r) for r in self.fence)

    def _fence_text(self) -> str:
        return "[" + ", ".join(str(r) for r in self.fence) + "]"

    # ----- 1. explicit ----------------------------------------------------------------------------

    def _explicit(self, raw: str) -> Derivation:
        """Validated, never fenced by roots, always returned as the resolved path."""
        self.tried.append("explicit: given")
        if not os.path.isabs(raw):
            raise self._fail("invalid_explicit", f"working_dir must be an absolute path, got: {_short(raw)}")
        rp = self.ctx.realpath(raw, "file")
        st = None if rp is None else self.ctx.stat(rp, "file")
        if rp is None or st is None:
            raise self._fail("invalid_explicit", f"working_dir does not exist or cannot be accessed: {_short(raw)}")
        if not stat.S_ISDIR(st.st_mode):
            raise self._fail("invalid_explicit", f"working_dir is not a directory (it is a file): {_short(raw)}")
        if not self.ctx.access(rp):
            raise self._fail(
                "invalid_explicit", f"working_dir is not accessible (read/search permission): {_short(raw)}"
            )
        why = self.ctx.broad_reason(rp)
        if why == HOME_REASON:
            self.warn(_HOME_WARNING)
        elif why:
            raise self._fail("invalid_explicit", f"working_dir {_short(raw)} refused: it is {why}")
        outside = [str(e.path) for e in self.evid if not _under(e.path, rp)]
        if outside:
            self.warn(f"{len(outside)} file(s) lie outside working_dir {rp}: {_listed(outside)}")
        return self._ok(rp, "explicit")

    # ----- declared candidates --------------------------------------------------------------------

    def _declared(self, raw_in: str | Path | None, label: str, *, allow_home: bool = False) -> Path | None:
        """Validate one declared directory; a rejected one is skipped WITH a warning (silent only when unset)."""
        if raw_in is None or not str(raw_in).strip():
            return None
        raw = str(raw_in)
        if not os.path.isabs(raw):
            self.warn(f"{label} directory {_short(raw)} not used: it is not an absolute path")
            return None
        rp = self.ctx.realpath(raw)
        if rp is None or not self.ctx.isdir(rp):
            self.warn(f"{label} directory {_short(raw)} not used: it does not exist or is not a directory")
            return None
        if not self.ctx.access(rp):
            self.warn(f"{label} directory {rp} not used: it is not accessible (read/search permission)")
            return None
        why = self.ctx.broad_reason(rp)
        home_ok = why == HOME_REASON and allow_home
        if why and not home_ok:
            self.warn(f"{label} directory {_short(raw)} not used: it is {why}")
            return None
        if self.fence and not self._in_fence(rp):
            self.warn(
                f"{label} directory {rp} not used: it is outside the client's declared roots {self._fence_text()}"
            )
            return None
        missing = [str(e.path) for e in self.sel if not _under(e.path, rp)]
        if missing:
            self.warn(f"{label} directory {rp} not used: it does not contain the call's files ({_listed(missing)})")
            return None
        for start in dict.fromkeys(e.start for e in self.sel):
            other = self.ctx.vcs_root_between(start, rp)
            if other is not None:
                self.warn(
                    f"{label} directory {rp} not used: the call's files are in a different checkout ({other}) "
                    "nested inside it"
                )
                return None
        if home_ok:
            self.warn(_HOME_WARNING)
        return rp

    def _declared_tier(self, tier: str, raw: str | Path | None, label: str) -> Derivation | None:
        if raw is None or not str(raw).strip():
            self.tried.append(f"{tier}: not set")
            return None
        rp = self._declared(raw, label)
        if rp is None:
            self.tried.append(f"{tier}: skipped")
            return None
        return self._ok(rp, tier)

    # ----- 2. continuation ------------------------------------------------------------------------

    def _continuation(self) -> Derivation | None:
        if self.continuation_dir is None or not str(self.continuation_dir).strip():
            self.tried.append("continuation: no stored directory")
            return None
        rp = self._declared(
            self.continuation_dir, "stored continuation", allow_home=self.continuation_source == "explicit"
        )
        if rp is None:
            self.tried.append("continuation: stored directory not reused")
            return None
        return self._ok(rp, "continuation")

    # ----- 3. files (selector) fenced by roots ----------------------------------------------------

    def _files(self) -> Derivation | None:
        if not self.sel:
            self.tried.append("files: no usable files passed")
            return None
        if self.fence:
            outside = [e for e in self.sel if not self._in_fence(e.path)]
            if outside:
                raise self._fail(
                    "roots_conflict",
                    f"The files passed ({_listed([str(e.path) for e in outside])}) lie outside every root the "
                    f"client declared {self._fence_text()}. Pass working_dir explicitly to override.",
                    self.fence,
                )
        per_file = {e.path: self.ctx.project_root(e.start) for e in self.sel}
        distinct: dict[Path, str] = {}
        for hit in per_file.values():
            if hit is not None:
                distinct.setdefault(hit[0], hit[1])
        # The outermost per-file root that contains every selection file (an unrooted file inside it counts).
        containing = [r for r in distinct if all(_under(e.path, r) for e in self.sel)]
        if containing:
            project = min(containing, key=lambda r: len(r.parts))
            return self._fence_project(project)
        if len(distinct) >= 2:
            cands = list(distinct)
            enclosing = self._enclosing_suggestion()
            if enclosing is not None and enclosing not in cands:
                cands.append(enclosing)
            raise self._fail(
                "ambiguous",
                "Cannot derive working_dir: the files belong to different projects. Candidate project roots: "
                f"{_listed(cands)} (an enclosing root is only a suggestion). Pass working_dir.",
                cands[:_MAX_CANDIDATES],
            )
        unrooted = [str(e.path) for e in self.sel if per_file[e.path] is None]
        self.tried.append("files: no project root")
        self.warn(
            f"files gave no project root (no project marker within {self.ctx.max_depth} levels above: "
            f"{_listed(unrooted or [str(e.path) for e in self.sel])})"
        )
        return None

    def _enclosing_suggestion(self) -> Path | None:
        try:
            common = Path(os.path.commonpath([str(e.start) for e in self.sel]))
        except ValueError:
            return None
        hit = self.ctx.project_root(common)
        return hit[0] if hit else None

    def _fence_project(self, project: Path) -> Derivation:
        """Apply the roots fence to a files-derived project: a root inside the project clamps to the innermost
        root containing all files (never widened, never refused); a project inside a root stands."""
        if self.fence:
            inner = [
                r
                for r in self.fence
                if r != project
                and _under(r, project)
                and all(_under(e.path, r) for e in self.sel)
                and not self.ctx.broad_reason(r)
            ]
            if inner:
                chosen = max(inner, key=lambda r: len(r.parts))
                self.warn(
                    f"the project root {project} is wider than the client's declared root {chosen}; "
                    "using the declared root"
                )
                return self._ok(chosen, "roots")
            if not self._in_fence(project):
                raise self._fail(
                    "roots_conflict",
                    f"The project root {project} derived from the files is not inside any root the client declared "
                    f"{self._fence_text()} and no declared root contains all the files. Pass working_dir explicitly.",
                    [*self.fence, project],
                )
        return self._ok(project, "files")

    # ----- 4. roots (selector when no files were passed) -----------------------------------------

    def _roots_selector(self) -> Derivation | None:
        if self.sel:
            self.tried.append("roots: files passed, roots only fence")
            return None
        if not self.fence:
            self.tried.append("roots: unavailable")
            return None
        usable: list[Path] = []
        for r in self.fence:
            why = self.ctx.broad_reason(r)
            if why:
                self.warn(f"root {r} not used as working_dir: it is {why}")
            else:
                usable.append(r)
        if len(usable) == 1:
            return self._ok(usable[0], "roots")
        if len(usable) > 1:
            raise self._fail(
                "ambiguous",
                f"The client declared {len(usable)} roots and no files were passed to choose between them: "
                f"{_listed(usable)}. Pass working_dir.",
                usable[:_MAX_CANDIDATES],
            )
        self.tried.append("roots: no usable root")
        return None

    # ----- 7. server cwd (guarded last resort) ----------------------------------------------------

    def _server_cwd(self, policy: str) -> Derivation | None:
        if self.server_cwd is None or not str(self.server_cwd).strip():
            self.tried.append("server_cwd: not set")
            return None
        if self.server_cwd_substituted:
            self.tried.append("server_cwd: substituted")
            self.warn(
                f"server cwd {_short(self.server_cwd)} not used: the server was launched from a non-workspace "
                "directory and substituted its own install directory"
            )
            return None
        rp = self._declared(self.server_cwd, "server cwd")
        if rp is None:
            self.tried.append("server_cwd: skipped")
            return None
        if policy == "refuse":
            self.tried.append("server_cwd: refused by CLINK_WORKING_DIR_FALLBACK=refuse")
            raise self._fail(
                "fallback_refused",
                "working_dir was not given and could not be derived; refusing to fall back to the server's current "
                f"directory {rp} (CLINK_WORKING_DIR_FALLBACK=refuse). Pass working_dir.",
                [rp],
            )
        if policy == "warn":
            note = ""
            if not self.ctx.marker_at(rp):
                # The launch directory is often a subdirectory of the checkout, not its root.
                enclosing = self.ctx.project_root(rp)
                note = (
                    f" (it is not itself a project root; it lies inside the project at {enclosing[0]})"
                    if enclosing
                    else " (it has no project marker)"
                )
            self.warn(
                f"working_dir was not given and could not be derived; ran in the server's current directory {rp}{note}"
            )
        return self._ok(rp, "server_cwd")


def derive_working_dir(
    *,
    explicit: str | Path | None = None,
    roots: Sequence[str | Path] | None = None,
    file_paths: Sequence[str | Path] = (),
    continuation_dir: str | Path | None = None,
    continuation_source: str | None = None,
    manifest_dir: str | Path | None = None,
    env_default: str | Path | None = None,
    server_cwd: str | Path | None = None,
    server_cwd_substituted: bool = False,
    fallback_policy: str | None = "warn",
    home: Path | str | None = None,
    temp_roots: Sequence[Path | str] | None = None,
    max_depth: int = 12,
    deadline: float | None = None,
    extra_warnings: Sequence[str] = (),
    _ctx_out: list[Any] | None = None,
) -> Derivation:
    """Choose the directory a spawned CLI runs in, or raise ``WorkingDirError``.

    ``file_paths`` are only the absolute files passed on this call; ``roots`` are already-converted directory
    paths (None or empty means roots are unavailable); ``continuation_source`` is the source stored with the
    continuation directory (a stored ``explicit`` directory may be the home directory); ``fallback_policy`` is the
    raw ``CLINK_WORKING_DIR_FALLBACK`` value; ``deadline`` is an absolute ``time.monotonic()`` value checked before
    every filesystem probe; ``extra_warnings`` are prepended to the result's (or the error's) warnings;
    ``_ctx_out`` is a test hook that receives the internal context.
    """
    ctx = _make_ctx(home, temp_roots, max_depth, deadline)
    if _ctx_out is not None:
        _ctx_out.append(ctx)
    run = _Deriver(
        ctx,
        explicit=explicit,
        roots=roots,
        file_paths=file_paths,
        continuation_dir=continuation_dir,
        continuation_source=continuation_source,
        manifest_dir=manifest_dir,
        env_default=env_default,
        server_cwd=server_cwd,
        server_cwd_substituted=server_cwd_substituted,
        fallback_policy=fallback_policy,
        extra_warnings=extra_warnings,
    )
    try:
        return run.run()
    except _BudgetExhausted:
        raise run.timeout_error() from None


# --------------------------------------------------------------------------- roots: CLINK_MCP_ROOTS and URIs

# off: never request roots. legacy: a server request only, never the input-request exchange. auto: a server request
# where the connection can carry one, and the multi-round-trip input-request exchange only on a modern-era connection
# that cannot. The era decides how roots are requested; the mode only decides whether the exchange is allowed.
ROOTS_MODES = ("off", "legacy", "auto")
# What an unset or blank CLINK_MCP_ROOTS means (the single place the default lives).
DEFAULT_ROOTS_MODE = "auto"
# What an unrecognised CLINK_MCP_ROOTS value means: the conservative mode (never the exchange), a literal that does
# not follow DEFAULT_ROOTS_MODE, so a typo can never switch the exchange on.
FALLBACK_ROOTS_MODE = "legacy"

_SCHEME_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*):")
_ENCODED_SLASH_RE = re.compile(r"%2f", re.IGNORECASE)
_ENCODED_BACKSLASH_RE = re.compile(r"%5c", re.IGNORECASE)
_WIN_SEPARATORS_RE = re.compile(r"[\\/]")
_WIN_DRIVE_RE = re.compile(r"[A-Za-z]:")
_UNC_HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*", re.ASCII)
_WIN_RESERVED_CHARS = frozenset('<>:"|?*')  # not valid in a Windows name; ":" would also open an NTFS data stream


def _echo(raw: object) -> str:
    """A caller-supplied string for a message: control characters escaped, long values truncated."""
    text = _clean(str(raw))
    return text if len(text) <= _MAX_SHOWN else text[:_MAX_SHOWN] + "..."


def parse_roots_mode(raw: str | None) -> tuple[str, list[str]]:
    """Interpret the ``CLINK_MCP_ROOTS`` value: ``(mode, warnings)`` with the mode one of ``ROOTS_MODES``.

    The value is trimmed and case-insensitive; unset or blank means ``DEFAULT_ROOTS_MODE`` (``auto``) without a
    warning, and an unrecognised value means ``FALLBACK_ROOTS_MODE`` (``legacy``, the conservative mode, not the
    default) with a warning naming the value and saying it behaves as legacy.
    """
    if raw is None:
        return DEFAULT_ROOTS_MODE, []
    text = raw.strip().lower() if isinstance(raw, str) else str(raw)
    if not text:
        return DEFAULT_ROOTS_MODE, []
    if text in ROOTS_MODES:
        return text, []
    allowed = ", ".join(ROOTS_MODES)
    return FALLBACK_ROOTS_MODE, [
        f"CLINK_MCP_ROOTS={_short(str(raw).strip())} is not one of {allowed}; it behaves as {FALLBACK_ROOTS_MODE!r}"
    ]


def _windows_path(segments: list[str], unc_host: str | None) -> tuple[str | None, str | None]:
    """Native Windows path from decoded URI segments: ``C:\\a\\b`` (the first segment is the drive) or, with a UNC
    server, ``\\\\server\\share\\a``."""
    if unc_host is not None:
        if not segments:
            return None, "UNC path has no share name"
        head, rest = "\\\\" + unc_host + "\\", segments
    else:
        if not segments or _WIN_DRIVE_RE.fullmatch(segments[0]) is None:
            return None, "not an absolute Windows path (no drive letter)"
        head, rest = segments[0][0].upper() + ":\\", segments[1:]
    if any(_WIN_RESERVED_CHARS & set(seg) for seg in rest):
        return None, "a path segment holds a character that is not valid in a Windows path"
    return head + "\\".join(rest), None


def file_uri_to_dir(uri: str, *, windows: bool = False) -> tuple[str | None, str | None]:
    """Convert a ``file://`` root URI to a directory path: ``(path, None)``, or ``(None, reason)`` when rejected.

    Pure string work: it never touches the filesystem, never raises, and leaves existence, symlinks, ``..`` segments
    and broadness to ``derive_working_dir``. Accepted: scheme ``file`` (any case) followed by ``//``; an empty or
    ``localhost`` host (any case; with ``windows=True`` any other plain host names a UNC server, so
    ``file://server/share/x`` becomes ``\\\\server\\share\\x``, while on POSIX it is rejected); a path that
    percent-decodes as strict UTF-8. Rejected: other schemes, ``file:`` without ``//``, a user, port or encoded host,
    an encoded path separator (``%2F``, and ``%5C`` with ``windows=True``) found in the *raw* path before anything is
    decoded, a NUL byte (raw or ``%00``), a relative result, and the POSIX filesystem root. A literal ``?`` or ``#`` is
    rejected rather than stripped: stripping would silently turn ``/work/proj#1`` into ``/work/proj``, a different
    directory (an encoded ``%3F`` or ``%23`` is an ordinary character). A malformed escape such as ``%ZZ`` stays
    literal, like ``urllib.parse.unquote``. Redundant ``/`` and ``.`` segments and a trailing slash are dropped;
    with ``windows=True`` a raw ``\\`` separates segments, and ``file:///c%3A/x`` and ``file:///C:/x`` both give
    ``C:\\x``.
    """
    if not isinstance(uri, str):
        return None, "not a string"
    if not uri:
        return None, "empty URI"
    if "\x00" in uri:
        return None, "contains a NUL byte"
    scheme = _SCHEME_RE.match(uri)
    if scheme is None:
        return None, "not a URI (no scheme)"
    if scheme.group(1).lower() != "file":
        return None, f"scheme {_short(scheme.group(1).lower())} is not file"
    rest = uri[scheme.end() :]
    if not rest.startswith("//"):
        return None, "only file:// URIs are supported"
    after = rest[2:]
    end = next((i for i, c in enumerate(after) if c in "/?#"), len(after))  # the authority ends at / ? or #
    authority, raw_path = after[:end], after[end:]
    unc_host: str | None = None
    if authority.lower() not in ("", "localhost"):
        if not windows:
            return None, f"host {_short(authority)} is not this machine"
        if _UNC_HOST_RE.fullmatch(authority) is None:
            return None, f"host {_short(authority)} is not a valid UNC server name"
        unc_host = authority
    if "?" in raw_path or "#" in raw_path:
        return None, "a query or fragment in a file URI is not supported"
    if _ENCODED_SLASH_RE.search(raw_path):
        return None, "encoded path separator %2F"
    if windows and _ENCODED_BACKSLASH_RE.search(raw_path):
        return None, "encoded path separator %5C"
    if "%00" in raw_path:
        return None, "contains a NUL byte"
    try:
        path = unquote_to_bytes(raw_path).decode("utf-8")
    except ValueError:  # UnicodeDecodeError (%ff) and UnicodeEncodeError (a lone surrogate) are ValueErrors
        return None, "the percent-encoded path is not valid UTF-8"
    if "\x00" in path:
        return None, "contains a NUL byte"
    if not path.startswith("/"):
        return None, "not an absolute path"
    if windows:
        segments = [s for s in _WIN_SEPARATORS_RE.split(path) if s not in ("", ".")]
        return _windows_path(segments, unc_host)
    segments = [s for s in path.split("/") if s not in ("", ".")]
    if not segments:
        return None, "the filesystem root is not a project directory"
    return "/" + "/".join(segments), None


def roots_from_uris(uris: Sequence[str], *, windows: bool = False) -> tuple[list[str], list[str]]:
    """Convert every root URI: ``(directory paths in order without duplicates, warnings)``.

    Each rejected URI is skipped with the warning ``root <uri> ignored: <reason>`` (the URI echoed with control
    characters escaped and truncated). Existence is not checked: ``derive_working_dir`` does that, and drops broad
    roots. Duplicates are silent; with ``windows=True`` they are compared case-insensitively.
    """
    paths: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for uri in uris:
        path, reason = file_uri_to_dir(uri, windows=windows)
        if path is None:
            warnings.append(f"root {_echo(uri)} ignored: {reason}")
            continue
        key = path.casefold() if windows else path
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths, warnings
