"""Tests for the roots helpers of clink.workdir (OpenSpec change clink-working-dir-derivation, task 5.2).

``parse_roots_mode``, ``file_uri_to_dir`` and ``roots_from_uris`` are pure string functions, so most tests are
tables; the last section feeds their output to ``derive_working_dir`` with real temporary directories (under
``/tmp``: the default macOS temp directory is refused by the broadness guard).

The roots semantics of ``derive_working_dir`` (fence, selector, clamp) are covered in ``tests/test_clink_workdir.py``;
task 5.2 lists them and each one has a test there:

* fence conflict only for files outside all roots: case 29, ``test_roots_conflict_is_terminal_and_names_the_roots``
  (conflict) with cases 31 and 34, ``test_project_root_inside_a_root_stands`` (no conflict inside a root)
* a root inside the project clamps with source ``roots``: ``test_root_inside_the_project_clamps_to_the_root``
* a nested worktree under a root: ``test_worktree_nested_under_a_root_stands``
* a marker-less root narrowed to the project: case 32
* one root and no files: case 27; several roots are ambiguous: case 30
* a broad root is dropped: ``test_broad_roots_are_dropped_with_a_warning_and_usable_ones_still_select``, case 35
* manifest, env and server-cwd candidates are fenced: ``test_lower_tier_candidates_are_fenced_by_roots``

Nothing from that list was missing, so none is repeated here; the end-to-end tests below only prove that URIs
converted by ``roots_from_uris`` reach the same behavior.

Deviations from ``reference/exp_uri.py``, the research prototype (the specs win): ``%2F`` is rejected, a literal
``?`` or ``#`` is rejected (the prototype stripped it), the drive letter is upper-cased, and a UNC share is required.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from clink import workdir
from clink.workdir import (
    DEFAULT_ROOTS_MODE,
    FALLBACK_ROOTS_MODE,
    ROOTS_MODES,
    WorkingDirError,
    derive_working_dir,
    file_uri_to_dir,
    parse_roots_mode,
    roots_from_uris,
)

# --------------------------------------------------------------------------- parse_roots_mode


def test_roots_modes_and_the_default_and_fallback_constants() -> None:
    assert ROOTS_MODES == ("off", "legacy", "auto")
    assert DEFAULT_ROOTS_MODE == "auto"
    assert FALLBACK_ROOTS_MODE == "legacy"
    assert DEFAULT_ROOTS_MODE in ROOTS_MODES and FALLBACK_ROOTS_MODE in ROOTS_MODES


@pytest.mark.parametrize("raw", ["off", "legacy", "auto"])
def test_every_mode_is_accepted_without_a_warning(raw: str) -> None:
    assert parse_roots_mode(raw) == (raw, [])


@pytest.mark.parametrize(
    "raw,mode",
    [
        ("OFF", "off"),
        ("Legacy", "legacy"),
        ("LEGACY", "legacy"),
        ("AUTO", "auto"),
        ("Auto", "auto"),
        ("  auto  ", "auto"),
        ("\toff\n", "off"),
        (" LeGaCy ", "legacy"),
    ],
)
def test_mode_is_trimmed_and_case_insensitive(raw: str, mode: str) -> None:
    assert parse_roots_mode(raw) == (mode, [])


@pytest.mark.parametrize("raw", [None, "", " ", "   ", "\t\n"])
def test_unset_or_blank_means_auto_without_a_warning(raw: str | None) -> None:
    assert parse_roots_mode(raw) == ("auto", [])
    assert parse_roots_mode(raw) == (DEFAULT_ROOTS_MODE, [])


@pytest.mark.parametrize("raw", ["garbage", "on", "true", "1", "automatic", "off legacy", "au to", "none"])
def test_unrecognised_value_behaves_as_legacy_and_warns_naming_it(raw: str) -> None:
    mode, warnings = parse_roots_mode(raw)
    assert mode == "legacy"
    assert len(warnings) == 1
    assert "CLINK_MCP_ROOTS" in warnings[0] and repr(raw) in warnings[0]
    assert "behaves as 'legacy'" in warnings[0]
    assert all(m in warnings[0] for m in ROOTS_MODES)


def test_unrecognised_value_is_echoed_trimmed_and_bounded() -> None:
    assert "'bogus'" in parse_roots_mode("  bogus  ")[1][0]
    huge = parse_roots_mode("x" * 10_000)[1][0]
    assert len(huge) < 500
    assert "\n" not in parse_roots_mode("bad\nvalue")[1][0]  # repr escapes it: no forged extra line


def test_default_and_fallback_are_separate_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unset value follows DEFAULT_ROOTS_MODE; an unrecognised one follows FALLBACK_ROOTS_MODE and never the default,
    # so a typo cannot switch the exchange on whatever the default is.
    monkeypatch.setattr(workdir, "DEFAULT_ROOTS_MODE", "off")
    assert parse_roots_mode(None) == ("off", [])
    assert parse_roots_mode("bogus")[0] == "legacy"
    monkeypatch.setattr(workdir, "DEFAULT_ROOTS_MODE", "auto")
    monkeypatch.setattr(workdir, "FALLBACK_ROOTS_MODE", "off")
    assert parse_roots_mode(None) == ("auto", [])
    assert parse_roots_mode("bogus")[0] == "off"


# --------------------------------------------------------------------------- file_uri_to_dir: POSIX, accepted

POSIX_OK = [
    # encoded space and non-ASCII (the example of the spec)
    ("file:///Users/a/My%20Project/caf%C3%A9", "/Users/a/My Project/café"),
    ("file:///Users/a/My%20Project", "/Users/a/My Project"),
    ("file:///Users/a/caf%c3%a9", "/Users/a/café"),  # lowercase hex digits
    ("file:///Users/a/%E6%97%A5%E6%9C%AC", "/Users/a/日本"),  # 3-byte UTF-8
    ("file:///Users/a/%F0%9F%90%8D", "/Users/a/\U0001f40d"),  # 4-byte UTF-8
    ("file:///Users/a/café", "/Users/a/café"),  # raw non-ASCII characters
    ("file:///Users/a/My Project", "/Users/a/My Project"),  # a host that forgot to encode the space
    # hosts
    ("file:///Users/a/proj", "/Users/a/proj"),
    ("file://localhost/Users/a/proj", "/Users/a/proj"),
    ("file://LOCALHOST/Users/a/proj", "/Users/a/proj"),
    ("file://LocalHost/Users/a/proj", "/Users/a/proj"),
    # scheme case
    ("FILE:///Users/a/proj", "/Users/a/proj"),
    ("File:///Users/a/proj", "/Users/a/proj"),
    # trailing slash, redundant slashes and "." segments are dropped; ".." is left for the realpath in derive
    ("file:///Users/a/proj/", "/Users/a/proj"),
    ("file:///Users/a/proj//", "/Users/a/proj"),
    ("file:///Users/a//proj", "/Users/a/proj"),
    ("file:////Users/a/proj", "/Users/a/proj"),
    ("file:///Users/a/./proj/.", "/Users/a/proj"),
    ("file:///Users/a/proj/sub/..", "/Users/a/proj/sub/.."),
    ("file:///Users/a/proj/%2e%2e/x", "/Users/a/proj/../x"),
    # characters that are ordinary on POSIX
    ("file:///Users/a/50%25", "/Users/a/50%"),
    ("file:///Users/a/bad%ZZ", "/Users/a/bad%ZZ"),  # a malformed escape stays literal (urllib.unquote semantics)
    ("file:///Users/a/100%", "/Users/a/100%"),
    ("file:///Users/a/%252F", "/Users/a/%2F"),  # %25 decodes to "%": not an encoded separator
    ("file:///Users/a/what%3F", "/Users/a/what?"),  # an encoded ? or # is an ordinary character
    ("file:///Users/a/c%23", "/Users/a/c#"),
    ("file:///Users/a/a%5Cb", "/Users/a/a\\b"),  # %5C is only a separator on Windows
    ("file:///Users/a/a\\b", "/Users/a/a\\b"),
    ("file:///C:/Users/me/proj", "/C:/Users/me/proj"),  # a drive-letter URI is just an odd POSIX path
    ("file:///c%3A/Users/me/proj", "/c:/Users/me/proj"),
    ("file:///tmp/a%0Ab", "/tmp/a\nb"),  # a control character other than NUL is the directory's business
]


@pytest.mark.parametrize("uri,path", POSIX_OK, ids=[u for u, _ in POSIX_OK])
def test_posix_conversion(uri: str, path: str) -> None:
    assert file_uri_to_dir(uri) == (path, None)


def test_spec_scenario_encoded_characters() -> None:
    assert file_uri_to_dir("file:///Users/a/My%20Project/caf%C3%A9") == ("/Users/a/My Project/café", None)


# --------------------------------------------------------------------------- file_uri_to_dir: POSIX, rejected

POSIX_REJECTED = [
    # non-local hosts (UNC form on POSIX), a user, a port, an encoded or look-alike host
    ("file://fileserver/share/proj", "host 'fileserver'"),
    ("file://server/share/proj", "host 'server'"),
    ("file://localhost.example.com/x", "not this machine"),
    ("file://localhost./x", "not this machine"),
    ("file://localhost:8080/x", "not this machine"),
    ("file://user@localhost/x", "not this machine"),
    ("file://%6Cocalhost/x", "not this machine"),
    ("file://127.0.0.1/x", "not this machine"),
    ("file://[::1]/x", "not this machine"),
    ("file://relative/dir", "not this machine"),
    # encoded separators, upper and lower case, checked on the raw path
    ("file:///Users/me/a%2Fb", "%2F"),
    ("file:///Users/me/a%2fb", "%2F"),
    ("file:///Users/me/%2F", "%2F"),
    ("file:///%2Fetc", "%2F"),
    ("file:///Users/me/proj%2F..%2F..%2Fetc", "%2F"),
    ("file:///Users/me/%2%2F", "%2F"),
    # NUL
    ("file:///Users/me/pro%00ject", "NUL"),
    ("file:///Users/me/pro\x00ject", "NUL"),
    ("file:///Users/me/proj%00", "NUL"),
    ("\x00file:///Users/me/proj", "NUL"),
    # relative or empty results
    ("file://", "absolute"),
    ("file://localhost", "absolute"),
    # not a file:// URI
    ("file:relative/dir", "file://"),
    ("file:/Users/me/proj", "file://"),
    ("file:", "file://"),
    ("https://example.com/proj", "'https'"),
    ("http://localhost/proj", "'http'"),
    ("vscode-remote://ssh-remote+host/home/me/proj", "'vscode-remote'"),
    ("ftp://host/x", "'ftp'"),
    ("/Users/me/proj", "no scheme"),
    ("Users/me/proj", "no scheme"),
    ("garbage", "no scheme"),
    ("not a uri", "no scheme"),
    ("://", "no scheme"),
    ("%66ile:///x", "no scheme"),
    (" file:///Users/me/proj", "no scheme"),
    ("", "empty"),
    # invalid UTF-8 and a lone surrogate
    ("file:///Users/me/x%FF%FE", "UTF-8"),
    ("file:///Users/me/%ff", "UTF-8"),
    ("file:///Users/me/%C3", "UTF-8"),  # truncated sequence
    ("file:///Users/me/%C0%AF", "UTF-8"),  # overlong encoding of "/"
    ("file:///Users/me/\ud800", "UTF-8"),
    # query and fragment (a literal ? or # is rejected, not stripped)
    ("file:///Users/me/proj?x=1", "query or fragment"),
    ("file:///Users/me/proj#frag", "query or fragment"),
    ("file:///Users/me/proj?x=1#frag", "query or fragment"),
    ("file:///Users/me/proj#1", "query or fragment"),
    ("file://?x", "query or fragment"),
    ("file://#x", "query or fragment"),
    # the filesystem root is not a project directory
    ("file:///", "root"),
    ("file:////", "root"),
    ("file:///.", "root"),
    ("file://localhost/", "root"),
]


@pytest.mark.parametrize("uri,reason", POSIX_REJECTED, ids=[repr(u) for u, _ in POSIX_REJECTED])
def test_posix_rejection_names_a_reason(uri: str, reason: str) -> None:
    path, why = file_uri_to_dir(uri)
    assert path is None
    assert why is not None and reason in why, why


@pytest.mark.parametrize("bad", [None, 5, b"file:///x", ["file:///x"]])
def test_a_non_string_is_rejected_not_raised(bad: Any) -> None:
    assert file_uri_to_dir(bad) == (None, "not a string")
    assert file_uri_to_dir(bad, windows=True) == (None, "not a string")


def test_encoded_separator_is_checked_on_the_raw_path_before_decoding() -> None:
    # decoding first would give "/a/b" (a valid path that hides a separator inside one segment)
    assert file_uri_to_dir("file:///a%2Fb")[0] is None
    # a double-encoded separator is a literal "%2F" in a name, not a separator
    assert file_uri_to_dir("file:///a%252Fb") == ("/a%2Fb", None)
    # mixed-case escape
    assert file_uri_to_dir("file:///a%2fb")[0] is None and file_uri_to_dir("file:///a%2Fb")[0] is None


# --------------------------------------------------------------------------- file_uri_to_dir: Windows

WINDOWS_OK = [
    ("file:///c%3A/x", "C:\\x"),
    ("file:///C:/x", "C:\\x"),
    ("file:///c:/x", "C:\\x"),
    ("file:///C%3A/x", "C:\\x"),
    ("file:///c%3a/Users/me/proj", "C:\\Users\\me\\proj"),
    ("file:///C:/Users/me/proj", "C:\\Users\\me\\proj"),
    ("file:///C:/Users/me/My%20Project/", "C:\\Users\\me\\My Project"),
    ("file:///C:/Users/me/caf%C3%A9", "C:\\Users\\me\\café"),
    ("file://localhost/C:/x", "C:\\x"),
    ("file://LOCALHOST/C:/x", "C:\\x"),
    ("file:///C:", "C:\\"),  # a drive root: the broadness guard in derive_working_dir refuses it
    ("file:///C:/", "C:\\"),
    ("file:///C:\\Users\\me\\proj", "C:\\Users\\me\\proj"),  # raw backslashes separate segments (as the prototype)
    ("file:///C:/Users\\me/proj", "C:\\Users\\me\\proj"),
    ("file:///C:/a//b/./c", "C:\\a\\b\\c"),
    ("file:///C:/a/b/..", "C:\\a\\b\\.."),  # ".." is left for the realpath in derive_working_dir
    ("file:///D:/data/50%25", "D:\\data\\50%"),
    # UNC
    ("file://server/share/dir", "\\\\server\\share\\dir"),
    ("file://fileserver/share/proj", "\\\\fileserver\\share\\proj"),
    ("file://SERVER/Share/Dir/", "\\\\SERVER\\Share\\Dir"),
    ("file://server/share", "\\\\server\\share"),
    ("file://server.corp.example.com/share/My%20Project", "\\\\server.corp.example.com\\share\\My Project"),
    ("file://10.0.0.5/share/x", "\\\\10.0.0.5\\share\\x"),
    ("file://server/c$/Users", "\\\\server\\c$\\Users"),
    ("file://server/share/caf%C3%A9", "\\\\server\\share\\café"),
]


@pytest.mark.parametrize("uri,path", WINDOWS_OK, ids=[u for u, _ in WINDOWS_OK])
def test_windows_conversion(uri: str, path: str) -> None:
    assert file_uri_to_dir(uri, windows=True) == (path, None)


WINDOWS_REJECTED = [
    ("file:///c%3A/Users/a%5Cb", "%5C"),
    ("file:///C:/Users/a%5cb", "%5C"),
    ("file:///C:/Users/%5C%5Cserver%5Cshare", "%5C"),
    ("file:///C:/Users/a%2Fb", "%2F"),
    ("file:///C:/Users/a%2fb", "%2F"),
    ("file:///Users/me/proj", "drive"),  # no drive letter: a rooted path is not absolute on Windows
    ("file:///", "drive"),
    ("file://localhost/Users/me", "drive"),
    ("file:////server/share/x", "drive"),
    ("file:///1:/x", "drive"),
    ("file:///CC:/x", "drive"),
    ("file:///C|/x", "drive"),
    ("file://server", "absolute"),  # no path at all
    ("file://server/", "share"),
    ("file://server/./", "share"),
    ("file://c:/x", "UNC server name"),  # a drive in the host slot
    ("file://server:445/share/x", "UNC server name"),
    ("file://user@server/share/x", "UNC server name"),
    ("file://-server/share/x", "UNC server name"),
    ("file://.server/share/x", "UNC server name"),
    ("file://./pipe/x", "UNC server name"),  # the \\.\ device namespace
    ("file://%73erver/share/x", "UNC server name"),
    ("file://[::1]/share/x", "UNC server name"),
    ("file://server/c:/x", "not valid in a Windows path"),
    ("file:///C:/dir/file%3Astream", "not valid in a Windows path"),  # an NTFS data stream
    ("file:///C:/a%3Cb", "not valid in a Windows path"),
    ("file:///C:/a%22b", "not valid in a Windows path"),
    ("file:///C:/a%7Cb", "not valid in a Windows path"),
    ("file:///C:/a%2Ab", "not valid in a Windows path"),
    ("file:///C:/x?y", "query or fragment"),
    ("file:///C:/x#y", "query or fragment"),
    ("file:///C:/x%00", "NUL"),
    ("file:///C:/x%FF", "UTF-8"),
    ("https://example.com/C:/x", "'https'"),
]


@pytest.mark.parametrize("uri,reason", WINDOWS_REJECTED, ids=[repr(u) for u, _ in WINDOWS_REJECTED])
def test_windows_rejection_names_a_reason(uri: str, reason: str) -> None:
    path, why = file_uri_to_dir(uri, windows=True)
    assert path is None
    assert why is not None and reason in why, why


def test_windows_flag_changes_only_the_host_and_separator_rules() -> None:
    # the same UNC URI: rejected on POSIX, mapped on Windows
    assert file_uri_to_dir("file://server/share/x")[0] is None
    assert file_uri_to_dir("file://server/share/x", windows=True) == ("\\\\server\\share\\x", None)
    # %5C: a literal backslash on POSIX, a rejected separator on Windows
    assert file_uri_to_dir("file:///a/b%5Cc") == ("/a/b\\c", None)
    assert file_uri_to_dir("file:///C:/a/b%5Cc", windows=True)[0] is None


# --------------------------------------------------------------------------- reference/exp_uri.py

# The prototype's cases, with `HERE` replaced by /w. Expected: (POSIX result, Windows result). The prototype's own
# output is reproduced except where the specs differ (marked): %2F (rejected), a literal ? or # (rejected, the
# prototype stripped it) and the drive letter (upper-cased, the prototype kept "c").
EXP_URI = [
    ("file:///w/fx/My%20Project/", ("/w/fx/My Project", None)),
    ("file:///w/fx/caf%C3%A9", ("/w/fx/café", None)),
    ("file:///w/fx/My%20Project/sub/..", ("/w/fx/My Project/sub/..", None)),
    ("file://localhost/w/fx", ("/w/fx", None)),
    ("file://fileserver/share/proj", (None, "fileserver")),
    ("file:///C:/Users/me/proj", ("/C:/Users/me/proj", "C:\\Users\\me\\proj")),
    ("file:///c%3A/Users/me/proj", ("/c:/Users/me/proj", "C:\\Users\\me\\proj")),  # prototype: "c:\\Users\\me\\proj"
    ("file:///C:\\Users\\me\\proj", ("/C:\\Users\\me\\proj", "C:\\Users\\me\\proj")),
    ("file:///Users/me/a%2Fb", (None, "%2F")),  # DIFFERS: the prototype returned /Users/me/a/b
    ("file:///Users/me/pro%00ject", (None, "NUL")),
    ("file:///Users/me/bad%ZZ", ("/Users/me/bad%ZZ", None)),
    ("file:///Users/me/x%FF%FE", (None, "UTF-8")),
    ("file:///Users/me/proj?x=1#frag", (None, "query or fragment")),  # DIFFERS: the prototype stripped it
    ("file:///", (None, "root")),
    ("file:relative/dir", (None, "file://")),  # the prototype's reason was "not absolute"
    ("https://example.com/proj", (None, "'https'")),
    ("vscode-remote://ssh-remote+host/home/me/proj", (None, "'vscode-remote'")),
    ("FILE:///Users/me/UpperScheme", ("/Users/me/UpperScheme", None)),
]


@pytest.mark.parametrize("uri,expected", EXP_URI, ids=[u for u, _ in EXP_URI])
def test_reference_prototype_outputs_are_reproduced_on_posix(uri: str, expected: tuple[str | None, str | None]) -> None:
    path, reason = file_uri_to_dir(uri)
    want, note = expected
    if want is not None:
        assert (path, reason) == (want, None)
    else:
        assert path is None and reason is not None and note in reason


def test_reference_prototype_windows_outputs() -> None:
    for uri in ("file:///C:/Users/me/proj", "file:///c%3A/Users/me/proj", "file:///C:\\Users\\me\\proj"):
        assert file_uri_to_dir(uri, windows=True) == ("C:\\Users\\me\\proj", None)
    assert file_uri_to_dir("file://fileserver/share/proj", windows=True) == ("\\\\fileserver\\share\\proj", None)


# --------------------------------------------------------------------------- purity and robustness


def test_helpers_never_touch_the_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the filesystem was touched")

    with monkeypatch.context() as m:
        for name in ("stat", "lstat", "scandir", "listdir", "getcwd", "access", "readlink"):
            m.setattr(os, name, boom)
        for name in ("exists", "isdir", "isfile", "islink", "realpath", "lexists"):
            m.setattr(os.path, name, boom)
        for windows in (False, True):
            for uri, _ in POSIX_OK + POSIX_REJECTED + WINDOWS_OK + WINDOWS_REJECTED:
                file_uri_to_dir(uri, windows=windows)
            roots_from_uris([u for u, _ in POSIX_OK], windows=windows)
        parse_roots_mode("auto")


_TOKENS = [
    "file:",
    "FILE:",
    "//",
    "/",
    "/",
    "\\",
    "%2F",
    "%2f",
    "%5C",
    "%00",
    "%ff",
    "%41",
    "%25",
    "%",
    "%Z",
    "C:",
    "c%3A",
    "localhost",
    "server",
    "share",
    "..",
    ".",
    "?",
    "#",
    "a",
    "B",
    " ",
    "é",
    "\x00",
    "\ud800",
    "@",
    ":",
    "[",
    "]",
    "\n",
    "|",
    "*",
]


def test_fuzz_never_raises_and_results_are_well_formed() -> None:
    rng = random.Random(20260429)
    seen_ok = 0
    for _ in range(6000):
        uri = "".join(rng.choice(_TOKENS) for _ in range(rng.randint(0, 9)))
        if rng.random() < 0.6:
            uri = rng.choice(["file://", "file:///", "file://server/", "file://localhost/"]) + uri
        for windows in (False, True):
            path, reason = file_uri_to_dir(uri, windows=windows)
            assert (path is None) != (reason is None), (uri, path, reason)
            if path is None:
                continue
            seen_ok += 1
            assert "\x00" not in path, (uri, path)
            if windows:
                assert re.match(r"^([A-Z]:\\|\\\\[A-Za-z0-9])", path), (uri, path)
                assert "/" not in path and not any(c in path[2:] for c in '<>"|?*'), (uri, path)
            else:
                assert path.startswith("/") and path != "/" and "//" not in path and not path.endswith("/"), (uri, path)
                assert "/./" not in path + "/" and "?" not in uri.replace("%3F", ""), (uri, path)
    assert seen_ok > 300  # the generator reaches the accepting branches, not only the rejections


# --------------------------------------------------------------------------- roots_from_uris


def test_roots_from_uris_converts_in_order() -> None:
    uris = ["file:///w/b", "file:///w/a/", "file://localhost/w/c%20d"]
    assert roots_from_uris(uris) == (["/w/b", "/w/a", "/w/c d"], [])


def test_roots_from_uris_removes_duplicates_silently_keeping_the_first() -> None:
    uris = ["file:///w/a", "file:///w/b", "file:///w/a/", "file://localhost/w/a", "FILE:///w//a", "file:///w/a/."]
    assert roots_from_uris(uris) == (["/w/a", "/w/b"], [])


def test_roots_from_uris_compares_windows_duplicates_case_insensitively() -> None:
    uris = ["file:///C:/Work/Proj", "file:///c%3A/work/proj", "file:///D:/Work/Proj"]
    assert roots_from_uris(uris, windows=True) == (["C:\\Work\\Proj", "D:\\Work\\Proj"], [])
    assert roots_from_uris(uris[:2]) == (["/C:/Work/Proj", "/c:/work/proj"], [])  # POSIX paths are case-sensitive


def test_roots_from_uris_empty_input_is_silent() -> None:
    assert roots_from_uris([]) == ([], [])
    assert roots_from_uris(()) == ([], [])


def test_each_rejected_uri_yields_a_warning_and_the_rest_still_convert() -> None:
    uris = [
        "https://example.com/x",
        "file:///w/ok",
        "file://server/share/x",
        "file:///w/a%2Fb",
        "",
        "garbage",
        "file:///w/also-ok",
    ]
    paths, warnings = roots_from_uris(uris)
    assert paths == ["/w/ok", "/w/also-ok"]
    assert len(warnings) == 5
    assert warnings[0] == "root https://example.com/x ignored: scheme 'https' is not file"
    assert warnings[1].startswith("root file://server/share/x ignored: host 'server'")
    assert warnings[2] == "root file:///w/a%2Fb ignored: encoded path separator %2F"
    assert warnings[3] == "root  ignored: empty URI"
    assert warnings[4] == "root garbage ignored: not a URI (no scheme)"
    assert all(re.fullmatch(r"root .* ignored: .+", w) for w in warnings)


def test_non_file_scheme_is_a_warning_not_an_exception() -> None:
    paths, warnings = roots_from_uris(["vscode-remote://ssh-remote+h/home/me/p"])
    assert paths == [] and len(warnings) == 1 and "ignored" in warnings[0] and "vscode-remote" in warnings[0]


def test_unc_is_converted_only_with_the_windows_flag() -> None:
    uri = "file://server/share/p"
    assert roots_from_uris([uri]) == ([], [f"root {uri} ignored: host 'server' is not this machine"])
    assert roots_from_uris([uri], windows=True) == (["\\\\server\\share\\p"], [])


def test_warning_echo_escapes_control_characters_and_truncates() -> None:
    _, warnings = roots_from_uris(["https://x/\nroot file:///etc ignored: forged"])
    assert len(warnings) == 1 and "\n" not in warnings[0]
    _, warnings = roots_from_uris(["https://x/" + "a" * 10_000])
    assert len(warnings) == 1 and len(warnings[0]) < 500 and warnings[0].endswith("is not file")
    _, warnings = roots_from_uris([None])  # type: ignore[list-item]  # out of contract: still a warning
    assert warnings == ["root None ignored: not a string"]


# --------------------------------------------------------------------------- end to end with derive_working_dir


class _Tree:
    """A throwaway tree under /tmp: ``home`` simulates $HOME, ``systmp`` the shared temp directory."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="wdr-", dir="/tmp")).resolve()
        self.home = self.mkdir("home")
        self.systmp = self.mkdir("systmp")

    def mkdir(self, rel: str) -> Path:
        d = self.root / rel
        d.mkdir(parents=True, exist_ok=True)
        return d

    def repo(self, rel: str) -> Path:
        d = self.mkdir(rel)
        (d / ".git").mkdir(exist_ok=True)
        return d

    def touch(self, rel: str) -> Path:
        f = self.root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
        return f

    def derive(self, uris: list[str], **kw: Any) -> Any:
        roots, _warnings = roots_from_uris(uris)
        return derive_working_dir(home=self.home, temp_roots=[self.systmp], roots=roots, **kw)


@pytest.fixture
def tree() -> Iterator[_Tree]:
    t = _Tree()
    yield t
    shutil.rmtree(t.root, ignore_errors=True)


def test_a_percent_encoded_root_with_a_space_and_non_ascii_selects_that_directory(tree: _Tree) -> None:
    repo = tree.repo("My Project/café")
    uri = f"file://{str(repo).replace(' ', '%20').replace('é', '%C3%A9')}"
    assert "%20" in uri and "%C3%A9" in uri
    res = tree.derive([uri])
    assert (res.path, res.source, res.warnings) == (repo, "roots", [])


def test_path_as_uri_round_trips_through_the_converter(tree: _Tree) -> None:
    repo = tree.repo("a b/ünï#c?d")  # pathlib encodes space, # and ? and the non-ASCII characters
    uri = repo.as_uri()
    assert file_uri_to_dir(uri) == (str(repo), None)
    assert tree.derive([uri]).path == repo


def test_the_shape_claude_code_sends_one_root_the_percent_encoded_launch_directory(tree: _Tree) -> None:
    repo = tree.repo("projB dir")
    uri = f"file://{str(repo).replace(' ', '%20')}"
    assert uri.endswith("projB%20dir")
    res = tree.derive([uri])
    assert (res.path, res.source) == (repo, "roots")
    res = tree.derive([uri], file_paths=[str(tree.touch("projB dir/src/a.py"))])
    assert (res.path, res.source) == (repo, "files")  # files select, the root only fences


def test_files_outside_every_converted_root_conflict(tree: _Tree) -> None:
    a, b = tree.repo("repoA"), tree.repo("repoB")
    with pytest.raises(WorkingDirError) as ei:
        tree.derive([a.as_uri()], file_paths=[str(tree.touch("repoB/b.py"))])
    assert ei.value.code == "roots_conflict" and ei.value.candidates == [a]
    assert tree.derive([a.as_uri(), b.as_uri()], file_paths=[str(tree.touch("repoB/b.py"))]).path == b


def test_a_converted_root_inside_the_project_clamps_with_source_roots(tree: _Tree) -> None:
    tree.repo("repoA")
    sub = tree.mkdir("repoA/my sub")
    res = tree.derive([sub.as_uri()], file_paths=[str(tree.touch("repoA/my sub/x/a.py"))])
    assert (res.path, res.source) == (sub, "roots")
    assert any(str(tree.root / "repoA") in w for w in res.warnings)


def test_several_converted_roots_without_files_are_ambiguous(tree: _Tree) -> None:
    a, b = tree.repo("repoA"), tree.repo("repoB")
    with pytest.raises(WorkingDirError) as ei:
        tree.derive([a.as_uri(), b.as_uri()])
    assert ei.value.code == "ambiguous" and set(ei.value.candidates) == {a, b}


def test_a_broad_converted_root_is_dropped_and_a_rejected_one_never_reaches_derive(tree: _Tree) -> None:
    a = tree.repo("repoA")
    uris = [tree.home.as_uri(), "https://example.com/p", a.as_uri(), "file:///", a.as_uri()]
    roots, warnings = roots_from_uris(uris)
    assert roots == [str(tree.home), str(a)]
    assert len(warnings) == 2  # https and the filesystem root
    res = derive_working_dir(home=tree.home, temp_roots=[tree.systmp], roots=roots)
    assert (res.path, res.source) == (a, "roots")
    assert any("home directory" in w for w in res.warnings)


def test_a_converted_root_that_does_not_exist_is_ignored_by_derive_not_by_the_converter(tree: _Tree) -> None:
    gone = tree.root / "gone"
    a = tree.repo("repoA")
    roots, warnings = roots_from_uris([gone.as_uri(), a.as_uri()])
    assert roots == [str(gone), str(a)] and warnings == []  # existence is derive's check
    res = derive_working_dir(home=tree.home, temp_roots=[tree.systmp], roots=roots)
    assert (res.path, res.source) == (a, "roots")
    assert any("ignored root" in w and "does not exist" in w for w in res.warnings)
