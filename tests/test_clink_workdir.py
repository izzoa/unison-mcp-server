"""Tests for clink.workdir: the pure working-directory derivation (OpenSpec change clink-working-dir-derivation).

Real temporary directories and repositories are used throughout (no mocks of the filesystem). Every sandbox is
created under ``/tmp`` because the default macOS temp directory (``/private/var/folders/...``) is refused by the
broadness guard, and every call injects ``home=`` so nothing depends on the real home directory.

The table of 51 cases and the 6 extra tests are ported from the research prototype
(``openspec/changes/clink-working-dir-derivation/reference/``); the specs win where the prototype differs.
Deviations from the prototype, by case id:

* All cases: the function is keyword-only; roots are directory paths (URI parsing is a later task); the sandbox
  uses ``tempfile.mkdtemp(dir="/tmp")`` and ``temp_roots=[systmp]`` (the prototype's separate ``scratch_roots``
  and ``allow_unmarked`` parameters no longer exist). Warning texts changed, so substrings were adapted.
* 01-05, 51: the explicit tier returns the resolved (realpath) directory (identical here: the sandbox is resolved).
* 06: explicit is never fenced; the roots passed here produce no roots warning (asserted).
* 10, 15, 17, 22, 26: the "no project marker" substring now comes from the files-tier warning
  ("files gave no project root (no project marker within N levels ...").
* 16: rewritten. ``allow_unmarked`` is not in the specs ("no marker means no derivation from files"); the case now
  asserts that a marker-less file yields nothing from the files tier and a manifest directory containing it is used.
* 18: rewritten for the same reason: a dotfile in $HOME with an env default of $HOME is skipped as broad.
* 24: the server-cwd warning reads "current directory" (was "cwd").
* 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40: roots are directory paths, not ``file://`` URIs.
* 28: the ``%20`` URI conversion moved to task 5.2; the case now checks a directory with a space in its name.
* 31, 33, 36: rewritten, they encode roots-first. Files now select the project and roots only fence it (sources
  ``files``; the paths are repoB, mono and mono/sub).
* 34: no "not used as working_dir" warning when a broad root merely fences a project under it (the root is not
  ignored: it still fences); the warning appears only when a broad root would have been the selector (case 35).
* 38: the stored directory is now skipped with a warning naming it and the reason ("does not contain").
* 39, 43, 47, 49: the skip warning reads "<label> directory '<raw>' not used: ..." (was "<label> dir ... refused").
* 45: refused as before; 46 (allow) now asserts ``warnings == []`` for every cwd (the prototype treated an unknown
  policy as ``allow``; an unknown value now warns and behaves as ``warn``, see test_invalid_policy_*).
* 41, 42, 44: manifest, env and server cwd now also must contain the call's files and be the same checkout
  (no files are passed in these cases, so the expectations hold).
* test_precedence_is_total_and_ordered: the order is explicit, continuation, files, roots, manifest, env,
  server_cwd (roots moved after files).
* test_probe_cost_is_bounded: counters are split (search, per-file, guard setup) and the search probe count is
  asserted equal for 1 and for 200 files.
* test_parity_with_repo_security_config: only paths that exist on this machine are compared (guard entries that do
  not exist are skipped by design, so ``/root`` on macOS is not broad).
"""

from __future__ import annotations

import ast
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from clink import workdir
from clink.workdir import SOURCES, Derivation, WorkingDirError, broad_directory_reason, derive_working_dir

# --------------------------------------------------------------------------- sandbox


class Sandbox:
    """A throwaway tree under /tmp. ``home`` simulates $HOME and ``systmp`` the shared system temp directory."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="wd-", dir="/tmp")).resolve()
        self.home = self.mkdir("home")
        self.systmp = self.mkdir("systmp")

    def p(self, rel: str) -> Path:
        return self.root / rel

    def mkdir(self, rel: str) -> Path:
        d = self.p(rel)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def touch(self, rel: str, text: str = "x") -> Path:
        f = self.p(rel)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        return f

    def repo(self, rel: str, marker: str = ".git") -> Path:
        d = self.mkdir(rel)
        (d / marker).mkdir(exist_ok=True)
        return d

    def worktree(self, rel: str, main: str) -> Path:
        """git-worktree-style checkout: ``.git`` is a FILE pointing into the main repo's .git/worktrees/."""
        d = self.mkdir(rel)
        gitdir = self.p(main) / ".git" / "worktrees" / Path(rel).name
        gitdir.mkdir(parents=True, exist_ok=True)
        (d / ".git").write_text(f"gitdir: {gitdir}\n")
        return d

    def link(self, rel: str, target: Path | str) -> Path:
        lnk = self.p(rel)
        lnk.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, lnk)
        return lnk

    def base(self, **kw: Any) -> dict[str, Any]:
        """Default kwargs: nothing supplied, simulated home and temp dir, no leakage from the real machine."""
        args: dict[str, Any] = {"home": self.home, "temp_roots": [self.systmp]}
        args.update(kw)
        return args

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def sandbox() -> Any:
    sb = Sandbox()
    yield sb
    sb.cleanup()


def S(p: Path | str) -> str:
    return str(p)


def derive(kw: dict[str, Any]) -> Derivation:
    return derive_working_dir(**kw)


def refused(kw: dict[str, Any]) -> WorkingDirError:
    with pytest.raises(WorkingDirError) as ei:
        derive_working_dir(**kw)
    return ei.value


def blob(err: WorkingDirError) -> str:
    return err.message + " | " + " | ".join(err.warnings)


# --------------------------------------------------------------------------- the 51-case table
# Each case: (id, build, expect); build(sb) -> kwargs; expect -> ("ok", source, relpath, [warning substrings])
# or ("err", code, [message/warning substrings]). An "ok" case with no substrings must have no warnings at all.

CASES: list[tuple[str, Any, tuple[Any, ...]]] = []


def case(cid: str, expect: tuple[Any, ...]) -> Any:
    def deco(fn: Any) -> Any:
        CASES.append((cid, fn, expect))
        return fn

    return deco


# ---- explicit


@case("01 explicit dir, no marker anywhere", ("ok", "explicit", "plain", []))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(explicit=S(sb.mkdir("plain")))


@case("02 explicit relative path -> refused", ("err", "invalid_explicit", ["absolute"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(explicit="some/relative")


@case("03 explicit nonexistent -> refused", ("err", "invalid_explicit", ["does not exist"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(explicit=S(sb.p("nope")))


@case("04 explicit '/' -> refused (the old code accepts it)", ("err", "invalid_explicit", ["filesystem"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(explicit="/")


@case("05 explicit == home -> allowed but warned", ("ok", "explicit", "home", ["read and write everything"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(explicit=S(sb.home))


@case(
    "06 explicit beats every other source (warns on files outside, never fenced by roots)",
    ("ok", "explicit", "chosen", ["outside working_dir"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    chosen = sb.mkdir("chosen")
    r = sb.repo("repoA")
    f = sb.touch("repoA/a.py")
    return sb.base(
        explicit=S(chosen),
        roots=[S(r)],
        file_paths=[S(f)],
        continuation_dir=S(r),
        manifest_dir=S(r),
        env_default=S(r),
        server_cwd=S(r),
    )


# ---- files tier


@case("07 files: one file in a repo subdir -> repo root", ("ok", "files", "repoA", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    return sb.base(file_paths=[S(sb.touch("repoA/src/deep/a.py"))])


@case("08 files: two files in different subdirs of one repo -> repo root", ("ok", "files", "repoA", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    return sb.base(file_paths=[S(sb.touch("repoA/src/a.py")), S(sb.touch("repoA/tests/b.py"))])


@case("09 files: git WORKTREE (.git is a file) -> worktree top, not the main repo", ("ok", "files", "wt/feature", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("main")
    sb.worktree("wt/feature", "main")
    return sb.base(file_paths=[S(sb.touch("wt/feature/pkg/x.py"))])


@case("10 files: bogus .git FILE (not a gitdir pointer) is not a marker", ("err", "underivable", ["no project marker"]))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.mkdir("fake")
    sb.touch("fake/.git", "hello")
    return sb.base(file_paths=[S(sb.touch("fake/a.py"))])


@case("11 files: monorepo, nested package.json under a .git root -> VCS root wins", ("ok", "files", "mono", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("mono")
    sb.touch("mono/packages/foo/package.json", "{}")
    return sb.base(file_paths=[S(sb.touch("mono/packages/foo/src/a.ts"))])


@case("12 files: monorepo w/o VCS, files in two packages -> ambiguous", ("err", "ambiguous", ["different projects"]))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.touch("mono/package.json", "{}")
    sb.touch("mono/packages/a/package.json", "{}")
    sb.touch("mono/packages/b/package.json", "{}")
    return sb.base(file_paths=[S(sb.touch("mono/packages/a/x.ts")), S(sb.touch("mono/packages/b/y.ts"))])


@case("13 files: two unrelated repos -> ambiguous (never pick one)", ("err", "ambiguous", ["different projects"]))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    sb.repo("repoB")
    return sb.base(file_paths=[S(sb.touch("repoA/a.py")), S(sb.touch("repoB/b.py"))])


@case("14 files: outer repo + nested repo, files in both -> outer repo", ("ok", "files", "outer", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("outer")
    sb.repo("outer/vendor/inner")
    return sb.base(file_paths=[S(sb.touch("outer/x.py")), S(sb.touch("outer/vendor/inner/y.py"))])


@case("15 files: no project marker anywhere -> refused", ("err", "underivable", ["no project marker"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(file_paths=[S(sb.touch("loose/notes/a.txt"))])


@case(
    "16 files: no marker -> files tier yields nothing; a manifest dir containing the file is used",
    ("ok", "manifest", "loose", ["no project marker"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    f = sb.touch("loose/notes/a.txt")
    return sb.base(file_paths=[S(f)], manifest_dir=S(sb.p("loose")))


@case(
    "17 files: single dotfile directly in $HOME (even if ~/.git exists) -> refused",
    ("err", "underivable", ["no project marker"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    (sb.home / ".git").mkdir()  # a dotfiles repo in $HOME must NOT make $HOME a derivable root
    return sb.base(file_paths=[S(sb.touch("home/.zshrc"))])


@case(
    "18 files: dotfile in $HOME + env default $HOME -> env skipped as broad -> refused",
    ("err", "underivable", ["home directory"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(file_paths=[S(sb.touch("home/.zshrc"))], env_default=S(sb.home))


@case("19 files: repo under $HOME -> repo root found before the home ceiling", ("ok", "files", "home/dev/proj", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("home/dev/proj")
    return sb.base(file_paths=[S(sb.touch("home/dev/proj/m.py"))])


@case("20 files: symlinked alias dir -> resolved to the real repo root", ("ok", "files", "repoA", []))
def _(sb: Sandbox) -> dict[str, Any]:
    r = sb.repo("repoA")
    sb.touch("repoA/a.py")
    sb.link("alias", r)
    return sb.base(file_paths=[S(sb.p("alias/a.py"))])


@case("21 files: symlink inside repoA pointing into repoB -> ambiguous", ("err", "ambiguous", ["different projects"]))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    sb.repo("repoB")
    target = sb.touch("repoB/secret.py")
    real = sb.touch("repoA/real.py")
    sb.link("repoA/link.py", target)
    return sb.base(file_paths=[S(real), S(sb.p("repoA/link.py"))])


@case(
    "22 files: symlink to /etc/hosts + a repo file -> refused (resolved target is a system path)",
    ("err", "underivable", ["no project marker"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    real = sb.touch("repoA/real.py")
    sb.link("repoA/hosts", Path("/etc/hosts"))
    return sb.base(file_paths=[S(real), S(sb.p("repoA/hosts"))])


@case("23 files: a directory path (src/) -> repo root; the repo itself -> itself", ("ok", "files", "repoA", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    return sb.base(file_paths=[S(sb.mkdir("repoA/src")), S(sb.p("repoA"))])


@case(
    "24 files: only nonexistent paths -> ignored with a warning, falls through to server cwd",
    ("ok", "server_cwd", "repoA", ["nonexistent", "current directory"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    return sb.base(file_paths=["/definitely/not/here.py"], server_cwd=S(sb.p("repoA")))


@case(
    "25 files: repo file + a scratch file in the temp dir -> repo root, scratch reported",
    ("ok", "files", "repoA", ["temp dir"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    return sb.base(file_paths=[S(sb.touch("repoA/a.py")), S(sb.touch("systmp/prompt.txt"))])


@case("26 files: depth cap (15 levels deep, max_depth=12) -> not found", ("err", "underivable", ["no project marker"]))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("deep")
    return sb.base(file_paths=[S(sb.touch("deep/" + "/".join(f"d{i}" for i in range(15)) + "/f.py"))])


# ---- roots (directory paths)


@case("27 roots: one root, no files -> that root", ("ok", "roots", "repoA", []))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(roots=[S(sb.repo("repoA"))])


@case("28 roots: a root whose name contains a space", ("ok", "roots", "My Proj", []))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(roots=[S(sb.repo("My Proj"))])


@case(
    "29 roots: files outside every declared root -> terminal conflict (manifest NOT consulted)",
    ("err", "roots_conflict", ["outside every root"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    a, b = sb.repo("repoA"), sb.repo("repoB")
    return sb.base(roots=[S(a)], file_paths=[S(sb.touch("repoB/b.py"))], manifest_dir=S(b))


@case("30 roots: two roots, no files -> ambiguous", ("err", "ambiguous", ["2 roots"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(roots=[S(sb.repo("repoA")), S(sb.repo("repoB"))])


@case(
    "31 roots: two roots, files in one -> the files select that project, the roots only fence it",
    ("ok", "files", "repoB", []),
)
def _(sb: Sandbox) -> dict[str, Any]:
    a, b = sb.repo("repoA"), sb.repo("repoB")
    return sb.base(roots=[S(a), S(b)], file_paths=[S(sb.touch("repoB/x.py"))])


@case("32 roots: marker-less container root + files in a repo inside -> that repo", ("ok", "files", "work/repoA", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.mkdir("work")
    sb.repo("work/repoA")
    return sb.base(roots=[S(sb.p("work"))], file_paths=[S(sb.touch("work/repoA/a.py"))])


@case(
    "33 roots: root IS the project, files in a nested package -> files select the root (VCS outranks the package)",
    ("ok", "files", "mono", []),
)
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("mono")
    sb.touch("mono/packages/foo/package.json", "{}")
    return sb.base(roots=[S(sb.p("mono"))], file_paths=[S(sb.touch("mono/packages/foo/a.ts"))])


@case(
    "34 roots: only root is $HOME (broad) + files in a repo under it -> the root fences, files decide",
    ("ok", "files", "home/dev/proj", []),
)
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("home/dev/proj")
    return sb.base(roots=[S(sb.home)], file_paths=[S(sb.touch("home/dev/proj/a.py"))])


@case(
    "35 roots: only root is $HOME and no files -> nothing usable, no other source -> refused",
    ("err", "underivable", ["home"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(roots=[S(sb.home)])


@case(
    "36 roots: a nested repo inside the root project; files in it -> the files select the nested repo, the root fences",
    ("ok", "files", "mono/sub", []),
)
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("mono")
    sb.repo("mono/sub")
    return sb.base(roots=[S(sb.p("mono"))], file_paths=[S(sb.touch("mono/sub/a.py"))])


# ---- continuation


@case("37 continuation: valid prior dir beats roots and files (sticky)", ("ok", "continuation", "repoA/sub", []))
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    sub = sb.mkdir("repoA/sub")
    return sb.base(continuation_dir=S(sub), roots=[S(sb.p("repoA"))], file_paths=[S(sb.touch("repoA/sub/a.py"))])


@case(
    "38 continuation: new files outside the old dir -> stale, re-derived from files",
    ("ok", "files", "repoB", ["does not contain", "repoA"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    a = sb.repo("repoA")
    sb.repo("repoB")
    return sb.base(continuation_dir=S(a), file_paths=[S(sb.touch("repoB/b.py"))])


@case(
    "39 continuation: prior dir deleted -> skipped with a warning, falls through",
    ("ok", "manifest", "m", ["does not exist"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(continuation_dir=S(sb.p("gone")), manifest_dir=S(sb.mkdir("m")))


@case(
    "40 continuation: prior dir outside the (changed) roots -> not reused",
    ("ok", "roots", "repoB", ["outside the client's declared roots"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    a, b = sb.repo("repoA"), sb.repo("repoB")
    return sb.base(continuation_dir=S(a), roots=[S(b)])


# ---- defaults + server cwd


@case("41 manifest beats env beats server cwd", ("ok", "manifest", "m", []))
def _(sb: Sandbox) -> dict[str, Any]:
    m, e, c = sb.mkdir("m"), sb.mkdir("e"), sb.repo("c")
    return sb.base(manifest_dir=S(m), env_default=S(e), server_cwd=S(c))


@case("42 env default beats server cwd", ("ok", "env", "e", []))
def _(sb: Sandbox) -> dict[str, Any]:
    e, c = sb.mkdir("e"), sb.repo("c")
    return sb.base(env_default=S(e), server_cwd=S(c))


@case(
    "43 manifest is '/' -> skipped with a warning (not silently honoured)",
    ("ok", "env", "e", ["manifest directory '/' not used"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(manifest_dir="/", env_default=S(sb.mkdir("e")))


@case("44 server cwd is a project -> used, but WARNED", ("ok", "server_cwd", "proj", ["current directory"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(server_cwd=S(sb.repo("proj")))


@case("45 server cwd + fallback_policy=refuse -> refused", ("err", "fallback_refused", ["refusing to fall back"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(server_cwd=S(sb.repo("proj")), fallback_policy="refuse")


@case("46 server cwd + fallback_policy=allow -> no warning of any kind", ("ok", "server_cwd", "proj", []))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(server_cwd=S(sb.repo("proj")), fallback_policy="allow")


@case(
    "47 Claude-Desktop launch: cwd '/' -> unusable -> refused with guidance",
    ("err", "underivable", ["server cwd directory '/' not used"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(server_cwd="/")


@case("48 server cwd was SUBSTITUTED with the install dir -> not trusted", ("err", "underivable", ["substituted"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(server_cwd=S(sb.repo("unison-install")), server_cwd_substituted=True)


@case(
    "49 server cwd is $HOME (host launched from ~) -> refused",
    ("err", "underivable", ["server cwd directory", "home directory"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(server_cwd=S(sb.home))


@case(
    "50 server cwd does not contain the files and files are not derivable -> refused",
    ("err", "underivable", ["does not contain the call's files"]),
)
def _(sb: Sandbox) -> dict[str, Any]:
    sb.repo("repoA")
    return sb.base(server_cwd=S(sb.p("repoA")), file_paths=[S(sb.touch("loose/a.txt"))])


@case("51 explicit shared temp dir is refused as broad", ("err", "invalid_explicit", ["shared"]))
def _(sb: Sandbox) -> dict[str, Any]:
    return sb.base(explicit=S(sb.systmp))


def test_table_has_all_51_cases() -> None:
    assert len(CASES) == 51


@pytest.mark.parametrize("cid,build,expect", CASES, ids=[c[0] for c in CASES])
def test_case(sandbox: Sandbox, cid: str, build: Any, expect: tuple[Any, ...]) -> None:
    kw = build(sandbox)
    if expect[0] == "ok":
        _, source, rel, warn_subs = expect
        path, src, warnings = derive(kw)
        assert src == source, (path, src, warnings)
        assert path == sandbox.p(rel), (path, sandbox.p(rel))
        joined = " | ".join(warnings)
        for sub in warn_subs:
            assert sub in joined, (sub, warnings)
        if not warn_subs:
            assert warnings == [], warnings
        assert src in SOURCES
    else:
        _, code, msg_subs = expect
        err = refused(kw)
        assert err.code == code, (err.code, err.message)
        text = blob(err)
        for sub in msg_subs:
            assert sub in text, (sub, text)
        assert err.tried, "a refusal lists the tiers tried"


# --------------------------------------------------------------------------- the 6 extra prototype tests


def test_precedence_is_total_and_ordered(sandbox: Sandbox) -> None:
    """Peel sources off one at a time: each removal must fall to the next tier in the documented order."""
    r = sandbox.repo("repoA")
    f = sandbox.touch("repoA/a.py")
    d = {n: sandbox.mkdir(n) for n in ("c", "m", "e")}
    kw: dict[str, Any] = {
        "explicit": S(sandbox.mkdir("x")),
        "roots": [S(r)],
        "file_paths": [S(f)],
        "continuation_dir": S(r),
        "manifest_dir": S(d["m"]),
        "env_default": S(d["e"]),
        "server_cwd": S(d["c"]),
    }
    order = []
    for drop in ("explicit", "continuation_dir", "file_paths", "roots", "manifest_dir", "env_default"):
        order.append(derive_working_dir(**sandbox.base(**kw)).source)
        kw[drop] = None
    order.append(derive_working_dir(**sandbox.base(**kw)).source)
    assert order == list(SOURCES)


def test_probe_cost_is_bounded_no_directory_scan(sandbox: Sandbox) -> None:
    """200 files in one directory cost the same marker-search probes as one file, and never a tree walk."""
    sandbox.repo("big")
    files = [S(sandbox.touch(f"big/src/f{i}.py")) for i in range(200)]
    one: list[Any] = []
    many: list[Any] = []
    derive_working_dir(**sandbox.base(file_paths=files[:1]), _ctx_out=one)
    path, src, _ = derive_working_dir(**sandbox.base(file_paths=files), _ctx_out=many)
    assert src == "files" and path == sandbox.p("big")
    assert many[0].probes == one[0].probes, (many[0].probes, one[0].probes)
    per_dir = len(workdir.VCS_MARKERS) + len(workdir.PROJECT_MARKERS) + 3
    assert many[0].probes <= 3 * per_dir + 40, many[0].probes
    assert many[0].file_probes <= 2 * 200  # per-file normalisation is the only O(files) cost: realpath + stat


def test_marker_search_never_climbs_above_home_or_evaluates_it(sandbox: Sandbox) -> None:
    (sandbox.home / ".git").mkdir()
    f = sandbox.touch("home/notes/a.txt")
    out: list[Any] = []
    with pytest.raises(WorkingDirError):
        derive_working_dir(**sandbox.base(file_paths=[S(f)]), _ctx_out=out)
    assert sandbox.home not in out[0].marker_cache  # home was never probed for markers
    assert sandbox.root not in out[0].marker_cache  # nor an ancestor of home


def test_parity_with_repo_security_config_on_dangerous_paths() -> None:
    """The broadness guard must be at least as strict as utils.security_config.is_dangerous_path on real paths."""
    from utils.security_config import is_dangerous_path

    checked = 0
    for p in ("/", "/etc", "/usr/bin", "/var/log", "/root", "/proc", "/dev", "/home", "/private/var/folders"):
        if not os.path.exists(p):
            continue  # guard entries that do not exist on this machine are skipped by design
        rp = Path(os.path.realpath(p))
        if is_dangerous_path(rp):
            assert broad_directory_reason(rp, home="/nonexistent-home") is not None, p
            checked += 1
    assert checked >= 4


def test_worktree_gitdir_pointer_is_not_followed(sandbox: Sandbox) -> None:
    """The root must be the worktree top even though its .git points into another repo."""
    sandbox.repo("main")
    wt = sandbox.worktree("wt/feat", "main")
    path = derive_working_dir(**sandbox.base(file_paths=[S(sandbox.touch("wt/feat/a.py"))])).path
    assert path == wt and not str(path).startswith(str(sandbox.p("main")))


def test_case_count_meets_bar() -> None:
    assert len(CASES) >= 15


# --------------------------------------------------------------------------- task 1.2: nested checkouts


def test_worktree_nested_in_server_cwd_checkout_never_uses_the_cwd_tier(sandbox: Sandbox) -> None:
    main = sandbox.repo("main")
    wt = sandbox.worktree("main/.worktrees/feat", "main")
    f = sandbox.touch("main/.worktrees/feat/pkg/a.py")
    res = derive_working_dir(**sandbox.base(file_paths=[S(f)], server_cwd=S(main)))
    assert (res.path, res.source) == (wt, "files")
    assert res.warnings == []


def test_nested_worktree_makes_every_declared_candidate_a_different_checkout(sandbox: Sandbox) -> None:
    """Continuation, manifest, env and cwd naming the main checkout are skipped when the files are in a worktree
    nested inside it, even when the files tier cannot answer (the file is deeper than the search bound)."""
    main = sandbox.repo("main")
    sandbox.worktree("main/wt/feat", "main")
    deep = sandbox.touch("main/wt/feat/" + "/".join(f"d{i}" for i in range(13)) + "/f.py")
    for field in ("continuation_dir", "manifest_dir", "env_default", "server_cwd"):
        err = refused(sandbox.base(file_paths=[S(deep)], **{field: S(main)}))
        assert err.code == "underivable", field
        assert "different checkout" in blob(err), (field, blob(err))
        assert str(sandbox.p("main/wt/feat")) in blob(err)


def test_stored_main_checkout_with_worktree_files_rederives_from_files(sandbox: Sandbox) -> None:
    main = sandbox.repo("main")
    wt = sandbox.worktree("main/wt/feat", "main")
    res = derive_working_dir(
        **sandbox.base(continuation_dir=S(main), file_paths=[S(sandbox.touch("main/wt/feat/a.py"))])
    )
    assert (res.path, res.source) == (wt, "files")
    assert "different checkout" in " ".join(res.warnings) and str(main) in " ".join(res.warnings)


def test_vcs_root_deeper_than_the_bound_below_the_candidate_is_still_detected(sandbox: Sandbox) -> None:
    """The 'different checkout' walk from file to candidate is not limited by max_depth."""
    cand = sandbox.mkdir("cand")
    sandbox.repo("cand/vendor/inner")
    f = sandbox.touch("cand/vendor/inner/" + "/".join(f"d{i}" for i in range(20)) + "/f.py")
    err = refused(sandbox.base(file_paths=[S(f)], manifest_dir=S(cand), max_depth=2))
    assert "different checkout" in blob(err)
    # control: without the nested repo the same candidate is fine
    sandbox.p("cand/vendor/inner/.git").rmdir()
    res = derive_working_dir(**sandbox.base(file_paths=[S(f)], manifest_dir=S(cand), max_depth=2))
    assert (res.path, res.source) == (cand, "manifest")


@pytest.mark.parametrize("levels,found", [(12, True), (13, False)])
def test_search_is_bounded_to_12_ancestors(sandbox: Sandbox, levels: int, found: bool) -> None:
    sandbox.repo("r")
    f = sandbox.touch("r/" + "/".join(f"d{i}" for i in range(levels)) + "/f.py")
    kw = sandbox.base(file_paths=[S(f)])
    if found:
        assert derive(kw).path == sandbox.p("r")
    else:
        assert refused(kw).code == "underivable"


def test_max_depth_parameter_is_honoured(sandbox: Sandbox) -> None:
    sandbox.repo("r")
    f = sandbox.touch("r/a/b/c/f.py")
    assert derive(sandbox.base(file_paths=[S(f)], max_depth=3)).path == sandbox.p("r")
    assert refused(sandbox.base(file_paths=[S(f)], max_depth=2)).code == "underivable"


def test_probe_count_ceiling_for_a_markerless_walk(sandbox: Sandbox) -> None:
    f = sandbox.touch("deep/" + "/".join(f"d{i}" for i in range(20)) + "/f.py")
    out: list[Any] = []
    refused(sandbox.base(file_paths=[S(f)], _ctx_out=out))
    levels = 13  # the start directory plus 12 ancestors
    per_dir = len(workdir.VCS_MARKERS) + len(workdir.PROJECT_MARKERS) + 2
    assert out[0].probes <= levels * (per_dir + 2) + 40, out[0].probes


def test_git_pointer_file_is_not_followed(sandbox: Sandbox) -> None:
    """A worktree whose gitdir pointer targets a missing or unrelated place is still the root, and is not read."""
    wt = sandbox.mkdir("wt")
    sandbox.touch("wt/.git", "gitdir: /definitely/not/there\n")
    res = derive_working_dir(**sandbox.base(file_paths=[S(sandbox.touch("wt/src/a.py"))]))
    assert res.path == wt


@pytest.mark.parametrize("marker", [".hg", ".svn", ".jj"])
def test_other_vcs_roots(sandbox: Sandbox, marker: str) -> None:
    sandbox.repo("proj", marker)
    sandbox.touch("proj/sub/package.json", "{}")  # a nested package marker must not outrank the VCS root
    res = derive_working_dir(**sandbox.base(file_paths=[S(sandbox.touch("proj/sub/a.py"))]))
    assert (res.path, res.source) == (sandbox.p("proj"), "files")


def test_file_in_markerless_directory_yields_nothing(sandbox: Sandbox) -> None:
    err = refused(sandbox.base(file_paths=[S(sandbox.touch("loose/a.txt"))]))
    assert err.code == "underivable" and "files:" in " ".join(err.tried)


def test_relative_file_entry_is_ignored(sandbox: Sandbox) -> None:
    sandbox.repo("repoA")
    res = derive_working_dir(**sandbox.base(file_paths=["rel/a.py", S(sandbox.touch("repoA/a.py"))]))
    assert (res.path, res.source) == (sandbox.p("repoA"), "files")
    assert any("non-absolute" in w for w in res.warnings)


def test_nonexistent_file_is_named_in_the_warnings(sandbox: Sandbox) -> None:
    sandbox.repo("repoA")
    res = derive_working_dir(
        **sandbox.base(
            file_paths=[S(sandbox.touch("repoA/a.py")), "/no/such/file.py"], server_cwd=S(sandbox.p("repoA"))
        )
    )
    assert res.source == "files" and any("/no/such/file.py" in w for w in res.warnings)


def test_only_scratch_files_yield_nothing(sandbox: Sandbox) -> None:
    f = sandbox.touch("systmp/prompt.txt")
    err = refused(sandbox.base(file_paths=[S(f)]))
    assert err.code == "underivable" and "temp directory" in blob(err)
    # scratch files are no evidence for the lower tiers either: the manifest need not contain them
    res = derive_working_dir(**sandbox.base(file_paths=[S(f)], manifest_dir=S(sandbox.mkdir("m"))))
    assert (res.source, res.path) == ("manifest", sandbox.p("m"))


def test_scratch_file_takes_no_part_in_containment_or_selection(sandbox: Sandbox) -> None:
    sandbox.repo("repoA")
    res = derive_working_dir(
        **sandbox.base(
            file_paths=[S(sandbox.touch("repoA/a.py")), S(sandbox.touch("systmp/p.txt"))],
            continuation_dir=S(sandbox.p("repoA")),
        )
    )
    assert (res.source, res.path) == ("continuation", sandbox.p("repoA"))
    assert any("ignored for selection" in w for w in res.warnings)


def test_repository_under_a_temp_dir_is_a_project_not_scratch(sandbox: Sandbox) -> None:
    """With the real default temp roots (/tmp is one) a repo under /tmp is still a project."""
    sandbox.repo("repoT")
    res = derive_working_dir(home=sandbox.home, file_paths=[S(sandbox.touch("repoT/src/a.py"))])
    assert (res.path, res.source, res.warnings) == (sandbox.p("repoT"), "files", [])


def test_unrooted_file_under_a_temp_dir_is_scratch_with_default_temp_roots(sandbox: Sandbox) -> None:
    err = refused({"home": sandbox.home, "file_paths": [S(sandbox.touch("loose/a.txt"))]})
    assert "temp directory" in blob(err)


def test_unrooted_file_blocks_candidates_that_do_not_contain_it(sandbox: Sandbox) -> None:
    """Environment default and files without a project root."""
    sandbox.repo("proj")
    f = sandbox.touch("loose/a.txt")
    err = refused(sandbox.base(file_paths=[S(f)], env_default=S(sandbox.p("proj"))))
    assert err.code == "underivable" and "does not contain the call's files" in blob(err)
    ok = derive_working_dir(**sandbox.base(file_paths=[S(f)], env_default=S(sandbox.p("loose"))))
    assert (ok.source, ok.path) == ("env", sandbox.p("loose"))


def test_manifest_not_containing_files_is_skipped_with_warning(sandbox: Sandbox) -> None:
    sandbox.repo("proj")
    other = sandbox.mkdir("other")
    f = sandbox.touch("loose/a.txt")
    err = refused(sandbox.base(file_paths=[S(f)], manifest_dir=S(other)))
    assert "manifest directory" in blob(err) and "does not contain" in blob(err)


def test_files_in_two_repositories_name_both_and_explicit_resolves(sandbox: Sandbox) -> None:
    a, b = sandbox.repo("repoA"), sandbox.repo("repoB")
    files = [S(sandbox.touch("repoA/a.py")), S(sandbox.touch("repoB/b.py"))]
    err = refused(sandbox.base(file_paths=files, manifest_dir=S(a), server_cwd=S(b)))
    assert err.code == "ambiguous" and {Path(c) for c in err.candidates} >= {a, b}
    assert str(a) in err.message and str(b) in err.message
    res = derive_working_dir(**sandbox.base(file_paths=files, explicit=S(sandbox.mkdir("chosen"))))
    assert (res.source, res.path) == ("explicit", sandbox.p("chosen"))


def test_ambiguity_suggests_the_enclosing_marker_root_without_picking_it(sandbox: Sandbox) -> None:
    sandbox.repo("umbrella")
    sandbox.repo("umbrella/a")
    sandbox.repo("umbrella/b")
    files = [S(sandbox.touch("umbrella/a/x.py")), S(sandbox.touch("umbrella/b/y.py"))]
    err = refused(sandbox.base(file_paths=files))
    assert err.code == "ambiguous"
    assert err.candidates == [sandbox.p("umbrella/a"), sandbox.p("umbrella/b"), sandbox.p("umbrella")]
    assert "suggestion" in err.message


def test_ambiguity_with_an_unrooted_third_file_is_still_ambiguous(sandbox: Sandbox) -> None:
    sandbox.repo("repoA")
    sandbox.repo("repoB")
    files = [S(sandbox.touch("repoA/a.py")), S(sandbox.touch("repoB/b.py")), S(sandbox.touch("loose/c.txt"))]
    assert refused(sandbox.base(file_paths=files)).code == "ambiguous"


# --------------------------------------------------------------------------- explicit tier


def _existing_containers() -> list[str]:
    return [
        p
        for p in (
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
            "/private",
            "/tmp",
            "/var/tmp",
        )
        if os.path.isdir(p)
    ]


@pytest.mark.parametrize("path", _existing_containers())
def test_explicit_shared_containers_are_refused(sandbox: Sandbox, path: str) -> None:
    err = refused(sandbox.base(explicit=path))
    assert err.code == "invalid_explicit", path


@pytest.mark.parametrize(
    "path",
    [
        p
        for p in ("/etc", "/usr", "/usr/bin", "/bin", "/var", "/dev", "/sbin", "/lib", "/boot", "/proc", "/sys", "/run")
        if os.path.isdir(p)
    ],
)
def test_explicit_system_directories_are_refused(sandbox: Sandbox, path: str) -> None:
    err = refused(sandbox.base(explicit=path))
    assert err.code == "invalid_explicit" and "system directory" in err.message, path


def test_explicit_file_path_is_refused(sandbox: Sandbox) -> None:
    f = sandbox.touch("plain/a.txt")
    err = refused(sandbox.base(explicit=S(f)))
    assert err.code == "invalid_explicit" and "not a directory" in err.message


def test_explicit_unreadable_directory_is_refused(sandbox: Sandbox) -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can read a mode-000 directory")
    d = sandbox.mkdir("locked")
    d.chmod(0)
    try:
        err = refused(sandbox.base(explicit=S(d)))
    finally:
        d.chmod(stat.S_IRWXU)
    assert err.code == "invalid_explicit" and "accessible" in err.message


def test_explicit_nul_byte_is_a_clean_refusal(sandbox: Sandbox) -> None:
    err = refused(sandbox.base(explicit=S(sandbox.p("a")) + "\0b"))
    assert err.code == "invalid_explicit"
    res = derive_working_dir(**sandbox.base(file_paths=["/tmp/x\0y"], server_cwd=S(sandbox.repo("proj"))))
    assert res.source == "server_cwd" and any("nonexistent" in w for w in res.warnings)


def test_explicit_with_dotdot_and_symlink_returns_the_resolved_path(sandbox: Sandbox) -> None:
    real = sandbox.mkdir("a/b")
    dotdot = f"{sandbox.p('a')}/b/../b"
    res = derive_working_dir(**sandbox.base(explicit=dotdot))
    assert (res.path, res.source, res.warnings) == (real, "explicit", [])
    link = sandbox.link("lnk", real)
    res = derive_working_dir(**sandbox.base(explicit=S(link)))
    assert res.path == real and str(res.path) != str(link)


def test_explicit_outside_roots_is_not_fenced_and_has_no_roots_warning(sandbox: Sandbox) -> None:
    root = sandbox.repo("repoA")
    chosen = sandbox.mkdir("chosen")
    res = derive_working_dir(**sandbox.base(explicit=S(chosen), roots=[S(root)]))
    assert (res.path, res.source, res.warnings) == (chosen, "explicit", [])


def test_explicit_lists_files_outside_it(sandbox: Sandbox) -> None:
    sandbox.repo("repoB")
    res = derive_working_dir(
        **sandbox.base(explicit=S(sandbox.mkdir("chosen")), file_paths=[S(sandbox.touch("repoB/b.py"))])
    )
    assert any("outside working_dir" in w and "b.py" in w for w in res.warnings)


def test_tmp_child_is_allowed_but_var_tmp_and_var_folders_are_refused(sandbox: Sandbox) -> None:
    proj = sandbox.mkdir("proj")
    res = derive_working_dir(home=sandbox.home, explicit=S(proj))  # default temp roots: /tmp and friends
    assert res.path == proj
    assert refused({"home": sandbox.home, "explicit": "/tmp"}).code == "invalid_explicit"
    if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK):
        d = tempfile.mkdtemp(dir="/var/tmp")
        try:
            err = refused({"home": sandbox.home, "explicit": d})
            assert err.code == "invalid_explicit" and "/var" in err.message
        finally:
            shutil.rmtree(d, ignore_errors=True)
    user_tmp = ""
    if sys.platform == "darwin":
        try:
            user_tmp = os.confstr(65537)  # _CS_DARWIN_USER_TEMP_DIR
        except (ValueError, OSError):  # pragma: no cover
            user_tmp = ""
    if user_tmp and os.path.isdir(user_tmp):  # /var/folders/xx/yyyy/T/ on macOS (reached via /private/var/folders)
        d = tempfile.mkdtemp(dir=user_tmp)
        try:
            err = refused({"home": sandbox.home, "explicit": d})
            assert err.code == "invalid_explicit" and "/var" in err.message
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_shared_temp_dir_itself_is_refused_for_the_platform_temp_dir(sandbox: Sandbox) -> None:
    real = os.path.realpath(tempfile.gettempdir())
    reason = broad_directory_reason(real, home=sandbox.home)
    assert reason is not None


@pytest.mark.skipif(
    not os.path.isdir("/users") or not os.path.isdir("/Users"), reason="no case-insensitive /Users alias"
)
def test_case_variant_alias_of_a_broad_directory_is_refused(sandbox: Sandbox) -> None:
    assert refused(sandbox.base(explicit="/users")).code == "invalid_explicit"
    assert refused(sandbox.base(manifest_dir="/users", env_default="/USERS")).code == "underivable"


@pytest.mark.skipif(
    not os.path.isdir("/System/Volumes/Data/Users"), reason="no /System/Volumes/Data firmlink (not macOS)"
)
def test_system_volumes_data_alias_is_refused(sandbox: Sandbox) -> None:
    err = refused(sandbox.base(explicit="/System/Volumes/Data/Users"))
    assert err.code == "invalid_explicit" and "shared container" in err.message


def test_guard_compares_filesystem_identity_not_path_text(sandbox: Sandbox) -> None:
    """A symlink whose text is nowhere near /etc is the same directory as /etc."""
    if not os.path.isdir("/etc"):
        pytest.skip("no /etc")
    lnk = sandbox.link("innocent", "/etc")
    ctx = workdir._make_ctx(sandbox.home, [sandbox.systmp], 12, None)
    assert ctx.broad_reason(lnk) is not None and "etc" in (ctx.broad_reason(lnk) or "")
    assert ctx.broad_reason(sandbox.mkdir("plain")) is None


def test_home_is_matched_by_filesystem_identity_not_path_text(sandbox: Sandbox) -> None:
    """An alias of $HOME whose text differs (a symlink here, /System/Volumes/Data/... or a case variant elsewhere)
    is still the home directory: the user's-own-tree exemption and the home refusal both follow identity."""
    alias = sandbox.link("homealias", sandbox.home)
    ctx = workdir._make_ctx(sandbox.home, [sandbox.systmp], 12, None)
    assert ctx.broad_reason(alias) == workdir.HOME_REASON
    assert ctx.broad_reason(sandbox.mkdir("plain")) is None


def test_home_directory_is_broad_for_every_candidate_except_explicit(sandbox: Sandbox) -> None:
    assert refused(sandbox.base(manifest_dir=S(sandbox.home))).code == "underivable"
    assert refused(sandbox.base(env_default=S(sandbox.home))).code == "underivable"
    assert refused(sandbox.base(continuation_dir=S(sandbox.home))).code == "underivable"
    assert derive_working_dir(**sandbox.base(explicit=S(sandbox.home))).source == "explicit"


def test_ancestors_of_home_are_broad(sandbox: Sandbox) -> None:
    assert refused(sandbox.base(explicit=S(sandbox.root))).code == "invalid_explicit"
    assert "ancestor" in refused(sandbox.base(explicit=S(sandbox.root))).message


def test_stored_explicit_home_is_reused_with_the_warning_only_for_explicit_source(sandbox: Sandbox) -> None:
    res = derive_working_dir(**sandbox.base(continuation_dir=S(sandbox.home), continuation_source="explicit"))
    assert (res.path, res.source) == (sandbox.home, "continuation")
    assert any("read and write everything" in w for w in res.warnings)
    err = refused(sandbox.base(continuation_dir=S(sandbox.home), continuation_source="files"))
    assert "home directory" in blob(err)


def test_continuation_is_reused_without_files_and_returns_resolved_path(sandbox: Sandbox) -> None:
    real = sandbox.repo("repoA")
    link = sandbox.link("alias", real)
    res = derive_working_dir(**sandbox.base(continuation_dir=S(link)))
    assert (res.path, res.source, res.warnings) == (real, "continuation", [])


def test_declared_directories_are_returned_resolved(sandbox: Sandbox) -> None:
    real = sandbox.mkdir("m")
    link = sandbox.link("lnk", real)
    assert derive_working_dir(**sandbox.base(manifest_dir=S(link))).path == real


def test_user_home_beneath_a_system_directory(monkeypatch: pytest.MonkeyPatch, sandbox: Sandbox) -> None:
    """HOME=/root with /root/project: a directory strictly beneath the home is never a 'system directory'."""
    fakeroot = sandbox.mkdir("fakeroot")
    project = sandbox.mkdir("fakeroot/project")
    other = sandbox.mkdir("fakeroot2/project")
    monkeypatch.setattr(workdir, "_POSIX_SYSTEM_DIRS", (S(fakeroot), S(sandbox.p("fakeroot2"))))
    res = derive_working_dir(home=fakeroot, temp_roots=[sandbox.systmp], explicit=S(project))
    assert (res.path, res.warnings) == (project, [])
    # the home itself: allowed explicitly with the warning, broad for every other tier
    assert any(
        "everything" in w for w in derive_working_dir(home=fakeroot, temp_roots=[], explicit=S(fakeroot)).warnings
    )
    # without that home the same directory is beneath a system directory
    err = refused({"home": sandbox.home, "temp_roots": [sandbox.systmp], "explicit": S(project)})
    assert "system directory" in err.message
    assert refused({"home": fakeroot, "temp_roots": [sandbox.systmp], "explicit": S(other)}).code == "invalid_explicit"
    # files tier: the project under the fake root is found with that home
    sandbox.repo("fakeroot/project")
    res = derive_working_dir(
        home=fakeroot, temp_roots=[sandbox.systmp], file_paths=[S(sandbox.touch("fakeroot/project/a.py"))]
    )
    assert (res.path, res.source) == (project, "files")


def test_windows_system_directories_through_the_testable_seam(
    monkeypatch: pytest.MonkeyPatch, sandbox: Sandbox
) -> None:
    posix = workdir._system_dir_names(False)
    win = workdir._system_dir_names(True)
    assert set(posix) < set(win)
    assert {"C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)", "C:\\ProgramData"} <= set(win) - set(posix)
    winlike = sandbox.mkdir("winlike")
    child = sandbox.mkdir("winlike/inner")
    monkeypatch.setattr(workdir, "_WINDOWS_SYSTEM_DIRS", (S(winlike),))
    ctx_nt = workdir._make_ctx(sandbox.home, [sandbox.systmp], 12, None, nt=True)
    assert "system directory" in (ctx_nt.broad_reason(child) or "")
    ctx_posix = workdir._make_ctx(sandbox.home, [sandbox.systmp], 12, None, nt=False)
    assert ctx_posix.broad_reason(child) is None


def test_guard_skips_entries_that_do_not_exist(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workdir, "_POSIX_SYSTEM_DIRS", (S(sandbox.p("not-here")),))
    assert broad_directory_reason(sandbox.mkdir("fine"), home=sandbox.home, temp_roots=[]) is None


# --------------------------------------------------------------------------- fallback policy


def test_invalid_policy_warns_and_behaves_as_warn(sandbox: Sandbox) -> None:
    cwd = sandbox.repo("proj")
    res = derive_working_dir(**sandbox.base(server_cwd=S(cwd), fallback_policy="sometimes"))
    assert (res.path, res.source) == (cwd, "server_cwd")
    text = " | ".join(res.warnings)
    assert "'sometimes'" in text and "current directory" in text


def test_policy_is_trimmed_and_case_insensitive_and_blank_is_default(sandbox: Sandbox) -> None:
    cwd = S(sandbox.repo("proj"))
    assert refused(sandbox.base(server_cwd=cwd, fallback_policy="  ReFuSe ")).code == "fallback_refused"
    assert derive_working_dir(**sandbox.base(server_cwd=cwd, fallback_policy=" ALLOW ")).warnings == []
    for raw in (None, "", "   ", "warn"):
        res = derive_working_dir(**sandbox.base(server_cwd=cwd, fallback_policy=raw))
        assert len(res.warnings) == 1 and "current directory" in res.warnings[0]


def test_allow_policy_is_quiet_for_any_cwd_but_not_looser(sandbox: Sandbox) -> None:
    plain = sandbox.mkdir("plain")  # no project marker
    assert derive_working_dir(**sandbox.base(server_cwd=S(plain), fallback_policy="allow")).warnings == []
    for bad in ("/", S(sandbox.home), S(sandbox.systmp)):
        assert refused(sandbox.base(server_cwd=bad, fallback_policy="allow")).code == "underivable", bad
    err = refused(sandbox.base(server_cwd=S(plain), fallback_policy="allow", server_cwd_substituted=True))
    assert err.code == "underivable"


def test_substituted_cwd_asks_for_working_dir_or_the_env_default(sandbox: Sandbox) -> None:
    err = refused(sandbox.base(server_cwd=S(sandbox.repo("install")), server_cwd_substituted=True))
    assert "working_dir" in err.message and "CLINK_DEFAULT_WORKING_DIR" in err.message
    # a substituted cwd never wins, but a real default still does
    res = derive_working_dir(
        **sandbox.base(
            server_cwd=S(sandbox.repo("install")), server_cwd_substituted=True, env_default=S(sandbox.mkdir("e"))
        )
    )
    assert res.source == "env"


def test_refuse_policy_only_applies_when_nothing_else_derives(sandbox: Sandbox) -> None:
    repo = sandbox.repo("repoA")
    res = derive_working_dir(
        **sandbox.base(server_cwd=S(repo), fallback_policy="refuse", file_paths=[S(sandbox.touch("repoA/a.py"))])
    )
    assert res.source == "files"
    err = refused(sandbox.base(server_cwd=S(repo), fallback_policy="refuse"))
    assert err.code == "fallback_refused" and err.candidates == [repo]


def test_nothing_derivable_names_the_tiers_tried(sandbox: Sandbox) -> None:
    err = refused(sandbox.base(manifest_dir=S(sandbox.p("missing"))))
    assert err.code == "underivable"
    for tier in ("explicit", "continuation", "files", "roots", "manifest", "env", "server_cwd"):
        assert any(t.startswith(tier) for t in err.tried), (tier, err.tried)
        assert tier in err.message
    assert "working_dir" in err.message
    order = [t.split(":")[0] for t in err.tried]
    assert order == ["explicit", "continuation", "files", "roots", "manifest", "env", "server_cwd"]


def test_invalid_operator_defaults_are_skipped_with_warnings(sandbox: Sandbox) -> None:
    f = sandbox.touch("afile")
    for bad in (S(sandbox.p("missing")), S(f), "/", "relative/dir"):
        res = derive_working_dir(**sandbox.base(manifest_dir=bad, env_default=S(sandbox.mkdir("e"))))
        assert res.source == "env" and any("manifest directory" in w for w in res.warnings), bad


def test_manifest_created_after_the_first_call_is_honoured(sandbox: Sandbox) -> None:
    later = sandbox.p("later")
    assert refused(sandbox.base(manifest_dir=S(later))).code == "underivable"
    later.mkdir()
    assert derive_working_dir(**sandbox.base(manifest_dir=S(later))).path == later


def test_extra_warnings_are_prepended(sandbox: Sandbox) -> None:
    res = derive_working_dir(**sandbox.base(server_cwd=S(sandbox.repo("p")), extra_warnings=["root URI x ignored"]))
    assert res.warnings[0] == "root URI x ignored" and len(res.warnings) == 2
    err = refused(sandbox.base(extra_warnings=["root URI x ignored"]))
    assert err.warnings[0] == "root URI x ignored"
    res = derive_working_dir(**sandbox.base(explicit=S(sandbox.mkdir("e")), extra_warnings=["w"]))
    assert res.warnings == ["w"]


# --------------------------------------------------------------------------- roots as fence and selector


def test_root_inside_the_project_clamps_to_the_root(sandbox: Sandbox) -> None:
    sandbox.repo("repoA")
    sub = sandbox.mkdir("repoA/sub")
    f = sandbox.touch("repoA/sub/x/a.py")
    res = derive_working_dir(**sandbox.base(roots=[S(sub)], file_paths=[S(f)]))
    assert (res.path, res.source) == (sub, "roots")
    assert any(str(sandbox.p("repoA")) in w for w in res.warnings)


def test_innermost_root_containing_all_files_wins_the_clamp(sandbox: Sandbox) -> None:
    sandbox.repo("repoA")
    outer, inner = sandbox.mkdir("repoA/pkg"), sandbox.mkdir("repoA/pkg/lib")
    f = sandbox.touch("repoA/pkg/lib/a.py")
    res = derive_working_dir(**sandbox.base(roots=[S(outer), S(inner)], file_paths=[S(f)]))
    assert (res.path, res.source) == (inner, "roots")


def test_root_that_does_not_contain_all_files_does_not_clamp(sandbox: Sandbox) -> None:
    """A root inside the project that holds only some of the files cannot narrow the directory; here b.py lies
    outside every root, which is a roots_conflict."""
    sandbox.repo("repoA")
    sub = sandbox.mkdir("repoA/sub")
    files = [S(sandbox.touch("repoA/sub/a.py")), S(sandbox.touch("repoA/other/b.py"))]
    err = refused(sandbox.base(roots=[S(sub)], file_paths=files))
    assert err.code == "roots_conflict"  # b.py is outside every root


def test_project_root_inside_a_root_stands(sandbox: Sandbox) -> None:
    sandbox.mkdir("ws")
    sandbox.repo("ws/a")
    sandbox.repo("ws/b")
    res = derive_working_dir(**sandbox.base(roots=[S(sandbox.p("ws"))], file_paths=[S(sandbox.touch("ws/b/x.py"))]))
    assert (res.path, res.source, res.warnings) == (sandbox.p("ws/b"), "files", [])


def test_worktree_nested_under_a_root_stands(sandbox: Sandbox) -> None:
    root = sandbox.repo("ws")
    wt = sandbox.worktree("ws/.wt/feat", "ws")
    res = derive_working_dir(**sandbox.base(roots=[S(root)], file_paths=[S(sandbox.touch("ws/.wt/feat/a.py"))]))
    assert (res.path, res.source) == (wt, "files")


def test_files_in_two_different_roots_are_refused_not_widened(sandbox: Sandbox) -> None:
    """Every file is inside some root, but no single root holds them all and the project is wider than each root:
    refused (roots_conflict) rather than widened. The roots spec does not define this case; fail-safe direction."""
    sandbox.repo("umbrella")
    a, b = sandbox.mkdir("umbrella/a"), sandbox.mkdir("umbrella/b")
    files = [S(sandbox.touch("umbrella/a/x.py")), S(sandbox.touch("umbrella/b/y.py"))]
    assert refused(sandbox.base(roots=[S(a), S(b)], file_paths=files)).code == "roots_conflict"


def test_roots_conflict_is_terminal_and_names_the_roots(sandbox: Sandbox) -> None:
    a, b = sandbox.repo("repoA"), sandbox.repo("repoB")
    err = refused(
        sandbox.base(
            roots=[S(a)],
            file_paths=[S(sandbox.touch("repoB/b.py"))],
            manifest_dir=S(b),
            env_default=S(b),
            server_cwd=S(b),
        )
    )
    assert err.code == "roots_conflict" and str(a) in err.message and err.candidates == [a]


@pytest.mark.parametrize("field", ["manifest_dir", "env_default", "server_cwd"])
def test_lower_tier_candidates_are_fenced_by_roots(sandbox: Sandbox, field: str) -> None:
    root = sandbox.mkdir("root")
    outside = sandbox.mkdir("outside")
    f = sandbox.touch("root/loose/a.txt")  # no project root: the files tier yields nothing and roots only fence
    err = refused(sandbox.base(roots=[S(root)], file_paths=[S(f)], **{field: S(outside)}))
    assert err.code == "underivable"
    assert any("outside the client's declared roots" in w and str(root) in w for w in err.warnings)
    # a broad root cannot select but still fences
    err = refused(sandbox.base(roots=[S(sandbox.home)], **{field: S(outside)}))
    assert any("outside the client's declared roots" in w for w in err.warnings)


@pytest.mark.parametrize("field", ["manifest_dir", "env_default", "server_cwd"])
def test_lower_tier_candidate_inside_a_root_is_used_with_files_unrooted(sandbox: Sandbox, field: str) -> None:
    root = sandbox.mkdir("root")
    inside = sandbox.mkdir("root/inside")
    f = sandbox.touch("root/inside/a.txt")
    res = derive_working_dir(**sandbox.base(roots=[S(root)], file_paths=[S(f)], **{field: S(inside)}))
    assert res.path == inside


def test_roots_do_not_select_when_files_were_passed(sandbox: Sandbox) -> None:
    """Files that yield no project fall through (roots select only when no usable files were passed)."""
    root = sandbox.mkdir("root")
    f = sandbox.touch("root/loose/a.txt")
    err = refused(sandbox.base(roots=[S(root)], file_paths=[S(f)]))
    assert err.code == "underivable"
    assert any(t.startswith("roots") for t in err.tried)


def test_single_root_without_files_selects_and_several_are_ambiguous(sandbox: Sandbox) -> None:
    a, b = sandbox.repo("repoA"), sandbox.repo("repoB")
    assert derive_working_dir(**sandbox.base(roots=[S(a)])).source == "roots"
    err = refused(sandbox.base(roots=[S(a), S(b)]))
    assert err.code == "ambiguous" and set(err.candidates) == {a, b}


def test_broad_roots_are_dropped_with_a_warning_and_usable_ones_still_select(sandbox: Sandbox) -> None:
    a = sandbox.repo("repoA")
    res = derive_working_dir(**sandbox.base(roots=[S(sandbox.home), S(a)]))
    assert (res.path, res.source) == (a, "roots")
    assert any("home directory" in w and str(sandbox.home) in w for w in res.warnings)


def test_invalid_roots_are_ignored_with_warnings(sandbox: Sandbox) -> None:
    a = sandbox.repo("repoA")
    res = derive_working_dir(**sandbox.base(roots=["relative/root", S(sandbox.p("gone")), S(a), S(a)]))
    assert (res.path, res.source) == (a, "roots")
    assert len([w for w in res.warnings if w.startswith("ignored root")]) == 2


def test_roots_empty_or_none_mean_unavailable_and_silent(sandbox: Sandbox) -> None:
    cwd = S(sandbox.repo("proj"))
    for roots in (None, []):
        res = derive_working_dir(**sandbox.base(roots=roots, server_cwd=cwd, fallback_policy="allow"))
        assert res.warnings == []


def test_root_fence_is_by_realpath(sandbox: Sandbox) -> None:
    real = sandbox.repo("repoA")
    link = sandbox.link("alias", real)
    res = derive_working_dir(**sandbox.base(roots=[S(link)], file_paths=[S(sandbox.touch("repoA/a.py"))]))
    assert (res.path, res.source) == (real, "files")


# --------------------------------------------------------------------------- the time budget


class _Clock:
    """A fake time module: every reading advances by one tick."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        self.now += 1.0
        return self.now


def test_expired_deadline_is_terminal_and_never_reaches_lower_tiers(sandbox: Sandbox) -> None:
    sandbox.repo("repoA")
    out: list[Any] = []
    kw = sandbox.base(
        file_paths=[S(sandbox.touch("repoA/a.py"))],
        manifest_dir=S(sandbox.mkdir("m")),
        env_default=S(sandbox.mkdir("e")),
        server_cwd=S(sandbox.repo("c")),
        deadline=time.monotonic() - 1.0,
        _ctx_out=out,
    )
    err = refused(kw)
    assert err.code == "underivable" and "timed out" in err.message and "working_dir" in err.message
    assert out[0].total_probes == 0  # not a single filesystem probe was issued
    # without a deadline the same call succeeds, so the refusal was the budget
    del kw["deadline"], kw["_ctx_out"]
    assert derive(kw).source == "files"


def test_deadline_applies_to_every_tier(sandbox: Sandbox) -> None:
    past = time.monotonic() - 1.0
    d = S(sandbox.mkdir("d"))
    for field in ("explicit", "continuation_dir", "manifest_dir", "env_default", "server_cwd"):
        err = refused(sandbox.base(deadline=past, **{field: d}))
        assert err.code == "underivable" and "timed out" in err.message, field
    err = refused(sandbox.base(deadline=past, roots=[d]))
    assert err.code == "underivable" and "timed out" in err.message


def test_deadline_expiring_mid_search_never_falls_through(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever probe the budget runs out at, the answer is the files-tier project or a timeout, never a
    lower tier."""
    sandbox.repo("repoA")
    f = S(sandbox.touch("repoA/src/a.py"))
    manifest, env, cwd = S(sandbox.mkdir("m")), S(sandbox.mkdir("e")), S(sandbox.repo("c"))
    timeouts = successes = 0
    for budget in range(1, 80):
        clock = _Clock()
        monkeypatch.setattr(workdir, "time", clock)
        try:
            res = derive_working_dir(
                **sandbox.base(
                    file_paths=[f], manifest_dir=manifest, env_default=env, server_cwd=cwd, deadline=float(budget)
                )
            )
        except WorkingDirError as err:
            assert err.code == "underivable" and "timed out" in err.message, (budget, err.code)
            timeouts += 1
        else:
            assert (res.source, res.path) == ("files", sandbox.p("repoA")), budget
            successes += 1
    assert timeouts > 5 and successes > 0


def test_deadline_in_the_marker_walk_of_a_declared_candidate(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a continuation candidate and no way to answer, an exhausted budget is terminal for any budget."""
    cand = sandbox.mkdir("cand")
    f = sandbox.touch("cand/a/b/c/f.py")
    for budget in range(1, 60):
        monkeypatch.setattr(workdir, "time", _Clock())
        try:
            res = derive_working_dir(
                **sandbox.base(file_paths=[S(f)], continuation_dir=S(cand), deadline=float(budget))
            )
        except WorkingDirError as err:
            assert "timed out" in err.message, budget
        else:
            assert (res.source, res.path) == ("continuation", cand)


# --------------------------------------------------------------------------- misc contract


def test_every_result_source_is_a_known_source(sandbox: Sandbox) -> None:
    for _cid, build, expect in CASES:
        sb = Sandbox()
        try:
            kw = build(sb)
            if expect[0] == "ok":
                assert derive(kw).source in SOURCES
        finally:
            sb.cleanup()


def test_result_and_error_shapes(sandbox: Sandbox) -> None:
    res = derive_working_dir(**sandbox.base(server_cwd=S(sandbox.repo("p"))))
    assert isinstance(res, Derivation) and isinstance(res.path, Path) and isinstance(res.warnings, list)
    path, source, warnings = res  # NamedTuple unpacking
    assert (path, source, warnings) == tuple(res)
    err = refused(sandbox.base())
    assert isinstance(err, Exception) and isinstance(err.candidates, list) and isinstance(err.tried, list)
    assert err.message == str(err)


def test_public_helpers(sandbox: Sandbox) -> None:
    repo = sandbox.repo("repoA")
    f = sandbox.touch("repoA/src/a.py")
    assert workdir.find_project_root(f, home=sandbox.home, temp_roots=[sandbox.systmp]) == repo
    assert workdir.find_project_root(sandbox.p("nope"), home=sandbox.home) is None
    assert broad_directory_reason("/", home=sandbox.home) is not None
    assert broad_directory_reason(sandbox.home, home=sandbox.home) == workdir.HOME_REASON


def test_module_is_pure_no_mcp_asyncio_or_raw_path_probes() -> None:
    tree = ast.parse(Path(workdir.__file__).read_text())
    imported: set[str] = set()
    raw_probes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"is_dir", "exists", "resolve", "is_file", "iterdir", "glob", "rglob", "read_text"}:
                raw_probes.append(f"line {node.lineno}: .{node.func.attr}()")
    assert not imported & {"asyncio", "mcp", "utils", "tools", "server", "config"}, imported
    assert not raw_probes, raw_probes


# --------------------------------------------------------------------------- audit regressions


def _set_home(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HOME", value)
    monkeypatch.setenv("USERPROFILE", value)


@pytest.mark.parametrize("home_value", ["", "/"])
def test_degenerate_home_does_not_switch_off_the_system_guard(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, home_value: str
) -> None:
    """HOME= / HOME=/ makes Path.home() the filesystem root; every directory is then 'beneath home'."""
    _set_home(monkeypatch, home_value)
    assert Path.home() == Path("/")
    for path in [p for p in ("/etc", "/usr", "/usr/bin", "/var", "/dev") if os.path.isdir(p)]:
        assert refused({"explicit": path, "temp_roots": [sandbox.systmp]}).code == "invalid_explicit", path
        err = refused({"manifest_dir": path, "server_cwd": path, "temp_roots": [sandbox.systmp]})
        assert err.code == "underivable", path
    assert broad_directory_reason("/etc", temp_roots=[sandbox.systmp]) is not None


@pytest.mark.parametrize("home_dir", ["/", "/tmp", "/private", "/Users"])
def test_home_that_is_a_root_or_shared_container_is_discarded(sandbox: Sandbox, home_dir: str) -> None:
    if not os.path.isdir(home_dir):
        pytest.skip(f"no {home_dir}")
    for path in [p for p in ("/etc", "/usr/bin") if os.path.isdir(p)]:
        assert refused(sandbox.base(home=home_dir, explicit=path)).code == "invalid_explicit", (home_dir, path)
    if home_dir != "/":  # the container itself stays refused, not accepted as "the home directory"
        assert refused(sandbox.base(home=home_dir, explicit=home_dir)).code == "invalid_explicit"


def test_system_directory_beneath_a_real_home_is_still_refused(sandbox: Sandbox) -> None:
    """The exemption covers the home and what is above it, not a system directory below it."""
    home = sandbox.mkdir("fakeroot/root")
    sandbox.mkdir("fakeroot/root/project")
    ctx = workdir._make_ctx(home, [sandbox.systmp], 12, None)
    assert ctx.broad_reason(sandbox.p("fakeroot/root/project")) is None
    ctx._ensure_guard()
    ctx._guard.append(workdir._GuardEntry("system", sandbox.p("fakeroot/root/project"), ctx.ident(sandbox.p("fakeroot/root/project"))))  # type: ignore[union-attr]
    ctx._broad_cache.clear()
    assert "system directory" in (ctx.broad_reason(sandbox.p("fakeroot/root/project")) or "")


def test_many_scratch_files_do_not_cost_a_quadratic_list_scan(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    original = workdir._Ev.__eq__

    def counting(self: Any, other: object) -> bool:
        calls["n"] += 1
        return bool(original(self, other))

    monkeypatch.setattr(workdir._Ev, "__eq__", counting)
    files = [S(sandbox.touch(f"systmp/scr/f{i}.txt")) for i in range(150)]
    err = refused(sandbox.base(file_paths=files))
    assert err.code == "underivable" and "150 file(s)" in blob(err)
    assert calls["n"] < 150, calls["n"]  # 11k+ comparisons with `e not in scratch`


def test_overlong_paths_are_not_resolved_and_not_echoed_whole(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    long_path = S(sandbox.root) + "/" + "a/" * 30000  # ~60KB: realpath is quadratic and uninterruptible
    seen: list[int] = []
    real_realpath = os.path.realpath

    def spy(p: Any, *a: Any, **k: Any) -> str:
        seen.append(len(str(p)))
        return real_realpath(p, *a, **k)

    monkeypatch.setattr(os.path, "realpath", spy)
    err = refused(sandbox.base(explicit=long_path))
    assert err.code == "invalid_explicit" and len(err.message) < 1000
    ok = sandbox.repo("proj")
    res = derive_working_dir(**sandbox.base(file_paths=[long_path], manifest_dir=long_path, server_cwd=S(ok)))
    assert res.source == "server_cwd"
    assert all(len(w) < 1000 for w in res.warnings), [len(w) for w in res.warnings]
    assert max(seen, default=0) < 5000, max(seen)


def test_out_of_contract_internal_parameters_degrade_instead_of_crashing(sandbox: Sandbox) -> None:
    proj = sandbox.repo("proj")
    f = S(sandbox.touch("proj/a.py"))
    assert derive_working_dir(explicit=S(proj), home=5).path == proj  # type: ignore[arg-type]
    for depth in (None, "12", -3, True):
        res = derive_working_dir(**sandbox.base(file_paths=[f], max_depth=depth))  # type: ignore[arg-type]
        assert (res.path, res.source) == (proj, "files"), depth
    assert derive_working_dir(**sandbox.base(explicit=S(proj), temp_roots=[5])).path == proj  # type: ignore[list-item]


def test_nan_deadline_is_expired_not_infinite(sandbox: Sandbox) -> None:
    err = refused(sandbox.base(explicit=S(sandbox.mkdir("d")), deadline=float("nan")))
    assert err.code == "underivable" and "timed out" in err.message


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read a mode-000 directory")
def test_unreadable_declared_directory_is_skipped_with_a_warning(sandbox: Sandbox) -> None:
    locked = sandbox.mkdir("locked")
    fallback = sandbox.repo("fallback")
    locked.chmod(0)
    try:
        res = derive_working_dir(**sandbox.base(manifest_dir=S(locked), env_default=S(fallback)))
    finally:
        locked.chmod(stat.S_IRWXU)
    assert (res.path, res.source) == (fallback, "env")
    assert any("not accessible" in w and "manifest" in w for w in res.warnings)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can read a mode-000 file")
def test_unreadable_worktree_pointer_still_counts_as_a_checkout(sandbox: Sandbox) -> None:
    main = sandbox.repo("main")
    wt = sandbox.worktree("main/.wt/feat", "main")
    f = sandbox.touch("main/.wt/feat/a.py")
    pointer = wt / ".git"
    pointer.chmod(0)
    try:
        res = derive_working_dir(**sandbox.base(file_paths=[S(f)], server_cwd=S(main)))
    finally:
        pointer.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert (res.path, res.source) == (wt, "files")


def test_broad_directory_reason_fails_closed_on_unusable_paths(sandbox: Sandbox) -> None:
    assert broad_directory_reason("", home=sandbox.home) is not None
    assert broad_directory_reason("relative/dir", home=sandbox.home) is not None
    assert broad_directory_reason("/tmp/\x00x", home=sandbox.home) is not None
    assert broad_directory_reason(sandbox.mkdir("fine"), home=sandbox.home) is None


def test_control_characters_in_directory_names_cannot_forge_message_lines(sandbox: Sandbox) -> None:
    evil = sandbox.mkdir("evil\nINJECTED: ok")
    res = derive_working_dir(**sandbox.base(server_cwd=S(evil)))
    assert res.source == "server_cwd"
    assert all("\n" not in w and "\r" not in w for w in res.warnings)
    assert any("evil\\nINJECTED" in w for w in res.warnings)


def test_ambiguity_message_and_candidates_are_bounded(sandbox: Sandbox) -> None:
    files = []
    for i in range(30):
        sandbox.repo(f"r{i}")
        files.append(S(sandbox.touch(f"r{i}/a.py")))
    err = refused(sandbox.base(file_paths=files))
    assert err.code == "ambiguous"
    assert len(err.candidates) <= 20 and "more" in err.message and len(err.message) < 2000


# ---- scenarios the first audit found implemented but unpinned


def test_scratch_file_outside_every_root_does_not_cause_roots_conflict(sandbox: Sandbox) -> None:
    repo = sandbox.repo("repoA")
    files = [S(sandbox.touch("repoA/a.py")), S(sandbox.touch("systmp/scr.txt"))]
    res = derive_working_dir(**sandbox.base(roots=[S(repo)], file_paths=files))
    assert (res.path, res.source) == (repo, "files")
    assert any("ignored for selection" in w for w in res.warnings)


def test_only_scratch_files_with_a_root_let_the_root_select(sandbox: Sandbox) -> None:
    repo = sandbox.repo("repoA")
    res = derive_working_dir(**sandbox.base(roots=[S(repo)], file_paths=[S(sandbox.touch("systmp/scr.txt"))]))
    assert (res.path, res.source) == (repo, "roots")
    res = derive_working_dir(**sandbox.base(roots=[S(repo)], file_paths=[S(sandbox.p("nope.py"))]))
    assert (res.path, res.source) == (repo, "roots")


def test_inner_root_wins_over_an_outer_root_containing_the_project(sandbox: Sandbox) -> None:
    sandbox.mkdir("W")
    project = sandbox.repo("W/P")
    inner = sandbox.mkdir("W/P/sub")
    f = sandbox.touch("W/P/sub/x.py")
    res = derive_working_dir(**sandbox.base(roots=[S(sandbox.p("W")), S(inner)], file_paths=[S(f)]))
    assert (res.path, res.source) == (inner, "roots")
    assert any(str(project) in w for w in res.warnings)


def test_package_marker_only_project_and_nearest_marker_precedence(sandbox: Sandbox) -> None:
    sandbox.touch("pkg/pyproject.toml")
    sandbox.touch("pkg/sub/package.json")
    f = S(sandbox.touch("pkg/sub/deep/a.py"))
    res = derive_working_dir(**sandbox.base(file_paths=[f]))
    assert (res.path, res.source) == (sandbox.p("pkg/sub"), "files")
    sandbox.mkdir("pkg/.git")  # a VCS root above outranks the nearer package marker
    res = derive_working_dir(**sandbox.base(file_paths=[f]))
    assert (res.path, res.source) == (sandbox.p("pkg"), "files")


def test_stored_continuation_directory_that_is_broad_is_not_reused(sandbox: Sandbox) -> None:
    fallback = sandbox.repo("proj")
    for path in [p for p in ("/", "/etc", "/usr/bin") if os.path.isdir(p)]:
        res = derive_working_dir(**sandbox.base(continuation_dir=path, server_cwd=S(fallback)))
        assert (res.source, res.path) == ("server_cwd", fallback), path
        assert any("stored continuation" in w and "not used" in w for w in res.warnings), path


def test_a_project_inside_opt_is_not_broad(sandbox: Sandbox) -> None:
    children = [os.path.join("/opt", n) for n in sorted(os.listdir("/opt"))] if os.path.isdir("/opt") else []
    real = [c for c in children if os.path.isdir(c) and os.access(c, os.R_OK | os.X_OK)]
    if not real:
        pytest.skip("no readable directory under /opt")
    assert broad_directory_reason(real[0], home=sandbox.home) is None
    assert broad_directory_reason("/opt", home=sandbox.home) is not None


# --------------------------------------------------------------------------- review regressions (core)


def test_broad_directory_name_sets_match_the_spec_on_every_os() -> None:
    """The parametrised refusal tests skip names absent from the host (macOS has no /boot or /srv); pin the lists."""
    assert set(workdir._POSIX_SYSTEM_DIRS) == {
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
    }
    assert set(workdir._WINDOWS_SYSTEM_DIRS) == {
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\ProgramData",
    }
    assert set(workdir._SHARED_CONTAINERS) >= {
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
        "/private",
        "/tmp",
        "/var/tmp",
    }


def test_many_bad_file_paths_give_a_bounded_warning_and_message(sandbox: Sandbox) -> None:
    bad = [f"/nonexistent/{i}" for i in range(3000)] + [f"rel/{i}" for i in range(3000)]
    err = refused(sandbox.base(file_paths=bad, deadline=time.monotonic() + 30))
    assert len(err.warnings) <= 5 and len(err.message) < 5000
    text = " | ".join(err.warnings)
    assert "/nonexistent/0" in text and "3000 nonexistent" in text and "and 2995 more" in text
    assert "3000 non-absolute" in text
    ok = derive_working_dir(**sandbox.base(file_paths=bad, server_cwd=S(sandbox.repo("proj"))))
    assert ok.source == "server_cwd" and len(ok.warnings) <= 5


def test_unreadable_parent_is_reported_as_nonexistent_or_inaccessible(sandbox: Sandbox) -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can read a mode-000 directory")
    sandbox.repo("unr")
    f = S(sandbox.touch("unr/f.py"))
    sandbox.p("unr").chmod(0)
    try:
        res = derive_working_dir(**sandbox.base(file_paths=[f], server_cwd=S(sandbox.repo("proj"))))
    finally:
        sandbox.p("unr").chmod(stat.S_IRWXU)
    assert any("inaccessible" in w and f in w for w in res.warnings)


def test_outermost_per_file_root_wins_when_two_roots_contain_every_file(sandbox: Sandbox) -> None:
    """Design D4: file A resolves to the VCS root R, file B (beyond the depth bound from R) to the package dir Q
    inside R. Both contain both files; the outermost (R) is chosen."""
    sandbox.repo("r")
    sandbox.touch("r/q/pyproject.toml", "")
    a = S(sandbox.touch("r/q/a.py"))
    b = S(sandbox.touch("r/q/x/y/b.py"))
    assert derive(sandbox.base(file_paths=[a], max_depth=2)).path == sandbox.p("r")
    assert derive(sandbox.base(file_paths=[b], max_depth=2)).path == sandbox.p("r/q")
    res = derive(sandbox.base(file_paths=[a, b], max_depth=2))
    assert (res.path, res.source) == (sandbox.p("r"), "files")


def test_warn_policy_notes_a_server_cwd_without_a_project_marker(sandbox: Sandbox) -> None:
    plain = sandbox.mkdir("plain")
    res = derive_working_dir(**sandbox.base(server_cwd=S(plain), fallback_policy="warn"))
    assert len(res.warnings) == 1 and res.warnings[0].endswith("(it has no project marker)")
    marked = derive_working_dir(**sandbox.base(server_cwd=S(sandbox.repo("proj"))))
    assert "no project marker" not in marked.warnings[0] and "not itself" not in marked.warnings[0]


def test_server_cwd_inside_a_project_subdirectory_is_not_called_markerless(sandbox: Sandbox) -> None:
    sandbox.repo("repoA")
    sub = sandbox.mkdir("repoA/sub")
    res = derive_working_dir(**sandbox.base(server_cwd=S(sub)))
    assert (res.source, res.path) == ("server_cwd", sub)
    text = " ".join(res.warnings)
    assert "no project marker" not in text and "not itself a project root" in text and S(sandbox.p("repoA")) in text
