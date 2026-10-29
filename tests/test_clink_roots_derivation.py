"""Tool-layer tests for clink consuming the host's roots (clink-working-dir-derivation, task 5.5).

The MCP handler resolves the host's roots in the foreground, before any state changes, and hands them to the tool as
the private argument ``_mcp_roots = {"uris": [...], "warnings": [...]}`` (the key is absent when roots were not
requested). ``CLinkTool._execute_blocking`` pops it before building the request model, converts the URIs with
``clink.workdir.roots_from_uris`` and passes the paths (``roots``) and every warning (``extra_warnings``) to
``derive_working_dir``, which owns the fence/selector/clamp rules (``tests/test_clink_workdir.py``) and the URI rules
(``tests/test_clink_workdir_roots.py``). These tests drive ``CLinkTool`` end to end with a fake agent (never a real
CLI) and prove the wiring: which directory reaches ``agent.run`` and the response metadata, the refusal payloads, the
warnings, that an explicit ``working_dir`` is neither fenced nor warned about, that a background job derives the same
directory from the roots it was started with (and never asks the host itself), and that nothing is cached between calls.

Everything except the hybrid-job section runs twice, in ``blocking`` and in ``hybrid`` execution mode, because the
roots reach a hybrid job through its ``run_args`` rather than through the foreground call.

Trust boundary: the tool treats whatever ``_mcp_roots`` it receives as the roots source. Keeping a client from
supplying that key itself is the handler's job (it strips it from the wire arguments before resolving roots), so it
cannot be tested here; ``test_the_tool_trusts_the_private_argument`` only documents that the boundary is the handler's.

Real directories are built under ``/tmp`` (``tempfile.mkdtemp(dir="/tmp")``): the broadness guard refuses the default
macOS temp directory, and a file under ``/tmp`` with no project marker is scratch (ignored for selection), so every file
here lives in a repository (a directory with ``.git``).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clink.agents import AgentOutput
from clink.parsers.base import ParsedCLIResponse
from tools.clink import CLinkRequest, CLinkTool
from tools.clink_jobs import reset_clink_job_manager_for_tests
from tools.shared.exceptions import ToolExecutionError
from utils.launch_info import record_cwd_substitution
from utils.mcp_context import get_current_request_context, reset_current_request_context, set_current_request_context

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No operator defaults from the developer's environment; fresh job manager; a generous foreground wait."""
    for name in ("CLINK_DEFAULT_WORKING_DIR", "CLINK_WORKING_DIR_FALLBACK", "CLINK_EXECUTION_MODE", "CLINK_MCP_ROOTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(CLinkTool, "_DEFAULT_JOB_WAIT_SECONDS", 10.0)
    monkeypatch.setattr(CLinkTool, "_MAX_JOB_WAIT_SECONDS", 10.0)
    reset_clink_job_manager_for_tests()
    yield
    reset_clink_job_manager_for_tests()


@pytest.fixture(params=["blocking", "hybrid"])
def both_modes(request, monkeypatch) -> str:
    """Run the test once through the blocking path and once through a background job's ``run_args``."""
    monkeypatch.setenv("CLINK_EXECUTION_MODE", request.param)
    return request.param


@pytest.fixture
def sandbox():
    base = Path(tempfile.mkdtemp(dir="/tmp")).resolve()
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def home(sandbox, monkeypatch) -> Path:
    """A fake home directory inside the sandbox (``Path.home()`` honours ``$HOME``), so a home root is testable."""
    path = sandbox / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


def make_repo(base: Path, name: str, *, file_name: str = "src/a.py") -> tuple[Path, str]:
    """A directory with a ``.git`` marker and one file; returns (root, file path)."""
    root = base / name
    (root / ".git").mkdir(parents=True)
    target = root / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1\n")
    return root, str(target)


class FakeAgent:
    """Records every ``run`` and optionally sleeps; the agent the tool gets from ``create_agent``."""

    fs_violation_ignore_patterns: tuple = ()

    def __init__(self, runs: list[dict[str, Any]], *, delay: float = 0.0) -> None:
        self.runs = runs
        self.delay = delay

    def get_read_only_args(self):
        return []

    async def run(self, **kwargs):
        self.runs.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        return AgentOutput(
            parsed=ParsedCLIResponse(content="ok", metadata={"model_used": "gemini-2.5-flash"}),
            sanitized_command=["gemini"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.1,
            parser_name="gemini_json",
            output_file_content=None,
        )


class Runs(list):
    """Every ``agent.run`` kwargs, in order; ``options`` are the FakeAgent settings for agents created later."""

    def __init__(self) -> None:
        super().__init__()
        self.options: dict[str, Any] = {}


@pytest.fixture
def runs(monkeypatch) -> Runs:
    recorded = Runs()
    monkeypatch.setattr("tools.clink.create_agent", lambda client: FakeAgent(recorded, **recorded.options))
    return recorded


@pytest.fixture
def derive_calls(monkeypatch) -> list[dict[str, Any]]:
    """The keyword arguments of every ``derive_working_dir`` call the tool makes (the real one still runs)."""
    import tools.clink as clink_module

    calls: list[dict[str, Any]] = []
    real = clink_module.derive_working_dir

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr("tools.clink.derive_working_dir", spy)
    return calls


def manifest_working_dir(monkeypatch, value: Path | None) -> None:
    """Make every configured CLI's manifest ``working_dir`` resolve to ``value`` (registry stays untouched)."""
    from clink.registry import ClinkRegistry

    original = ClinkRegistry.get_client

    def patched(self, cli_name):
        return original(self, cli_name).model_copy(update={"working_dir": value})

    monkeypatch.setattr(ClinkRegistry, "get_client", patched)


def uri(path: Path | str) -> str:
    return Path(path).as_uri()


def mcp_roots(*roots: Path | str, warnings: tuple[str, ...] = ()) -> dict[str, Any]:
    """The handler's private argument for these roots (paths are turned into ``file://`` URIs; strings that
    already look like a URI are passed through)."""
    return {"uris": [str(r) if "://" in str(r) else uri(r) for r in roots], "warnings": list(warnings)}


def args(**overrides) -> dict[str, Any]:
    base: dict[str, Any] = {"prompt": "hi", "cli_name": "gemini", "absolute_file_paths": [], "images": []}
    base.update(overrides)
    return base


async def call(tool: CLinkTool | None = None, **overrides) -> dict[str, Any]:
    """Run one call and return the parsed ToolOutput JSON."""
    result = await (tool or CLinkTool()).execute(args(**overrides))
    return json.loads(result[0].text)


async def call_error(**overrides) -> dict[str, Any]:
    with pytest.raises(ToolExecutionError) as exc_info:
        await CLinkTool().execute(args(**overrides))
    return json.loads(exc_info.value.payload)


class SpySession:
    """A stand-in MCP session that records any attempt to ask the host for something (roots, sampling, ...)."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    async def send_request(self, *a, **k):
        self.requests.append("send_request")
        raise AssertionError("clink must not send requests to the host; the handler resolves roots")

    async def list_roots(self, *a, **k):
        self.requests.append("list_roots")
        raise AssertionError("clink must not ask the host for roots; the handler resolves them")

    async def report_progress(self, *a, **k):  # the foreground heartbeat is allowed
        return None


@pytest.fixture
def host_session():
    """Bind a request context whose session fails the test if the tool (or a job it started) queries the host."""
    session = SpySession()
    token = set_current_request_context(SimpleNamespace(session=session))
    try:
        yield session
    finally:
        reset_current_request_context(token)


# ---------------------------------------------------------------------------
# Roots select the directory when no files are passed
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("both_modes")
class TestRootsSelect:
    @pytest.mark.asyncio
    async def test_one_root_and_no_files_selects_it_with_source_roots(self, sandbox, runs, host_session, monkeypatch):
        root, _ = make_repo(sandbox, "proj")
        manifest, _ = make_repo(sandbox, "manifest")
        env_dir, _ = make_repo(sandbox, "envdir")
        server_dir, _ = make_repo(sandbox, "server")
        manifest_working_dir(monkeypatch, manifest)
        monkeypatch.setenv("CLINK_DEFAULT_WORKING_DIR", str(env_dir))
        monkeypatch.chdir(server_dir)

        payload = await call(_mcp_roots=mcp_roots(root))

        assert runs[0]["working_dir"] == root
        metadata = payload["metadata"]
        assert metadata["working_dir"] == str(root)
        assert metadata["working_dir_source"] == "roots"  # the label flows through unchanged
        assert metadata["working_dir_warnings"] == []
        assert host_session.requests == []

    @pytest.mark.asyncio
    async def test_two_roots_and_no_files_are_ambiguous(self, sandbox, runs):
        root_a, _ = make_repo(sandbox, "a")
        root_b, _ = make_repo(sandbox, "b")

        payload = await call_error(_mcp_roots=mcp_roots(root_a, root_b))

        metadata = payload["metadata"]
        assert payload["status"] == "error"
        assert metadata["error_code"] == "ambiguous"
        assert set(metadata["candidates"]) == {str(root_a), str(root_b)}
        assert isinstance(metadata["tried"], list) and metadata["tried"]
        assert runs == []

    @pytest.mark.asyncio
    async def test_a_root_named_by_a_path_with_encoded_characters_is_decoded(self, sandbox, runs):
        root, _ = make_repo(sandbox, "My Project")  # as_uri() percent-encodes the space

        payload = await call(_mcp_roots=mcp_roots(root))

        assert " " in str(root) and "%20" in uri(root)
        assert runs[0]["working_dir"] == root
        assert payload["metadata"]["working_dir_source"] == "roots"

    @pytest.mark.asyncio
    async def test_whitespace_working_dir_counts_as_omitted(self, sandbox, runs):
        root, _ = make_repo(sandbox, "proj")

        payload = await call(working_dir="   ", _mcp_roots=mcp_roots(root))

        assert runs[0]["working_dir"] == root
        assert payload["metadata"]["working_dir_source"] == "roots"

    @pytest.mark.asyncio
    async def test_a_root_that_does_not_exist_is_ignored_with_a_warning(self, sandbox, runs, monkeypatch):
        repo, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(repo)
        gone = sandbox / "gone"

        payload = await call(_mcp_roots=mcp_roots(gone))

        metadata = payload["metadata"]
        assert metadata["working_dir_source"] == "server_cwd"
        assert any(str(gone) in warning and "ignored root" in warning for warning in metadata["working_dir_warnings"])

    @pytest.mark.asyncio
    async def test_no_roots_argument_behaves_as_before(self, sandbox, runs, monkeypatch):
        repo, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(repo)

        payload = await call()

        assert payload["metadata"]["working_dir"] == str(repo)
        assert payload["metadata"]["working_dir_source"] == "server_cwd"


# ---------------------------------------------------------------------------
# Files select the project and roots fence it
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("both_modes")
class TestFilesSelectRootsFence:
    @pytest.mark.asyncio
    async def test_files_outside_every_root_fail_with_roots_conflict(self, sandbox, runs):
        _, file_path = make_repo(sandbox, "proj")
        root, _ = make_repo(sandbox, "elsewhere")

        payload = await call_error(absolute_file_paths=[file_path], _mcp_roots=mcp_roots(root))

        metadata = payload["metadata"]
        assert payload["status"] == "error"
        assert metadata["error_code"] == "roots_conflict"
        assert metadata["candidates"] == [str(root)]  # the roots, so the caller can pick one
        assert str(root) in payload["content"] and "working_dir" in payload["content"]
        assert isinstance(metadata["tried"], list) and isinstance(metadata["warnings"], list)
        assert runs == []

    @pytest.mark.asyncio
    async def test_files_spread_across_roots_narrower_than_the_project_are_a_conflict(self, sandbox, runs):
        project, file_a = make_repo(sandbox, "proj", file_name="a/one.py")
        file_b = project / "b" / "two.py"
        file_b.parent.mkdir()
        file_b.write_text("y = 2\n")

        payload = await call_error(
            absolute_file_paths=[file_a, str(file_b)], _mcp_roots=mcp_roots(project / "a", project / "b")
        )

        metadata = payload["metadata"]
        assert metadata["error_code"] == "roots_conflict"
        assert {str(project / "a"), str(project / "b")} <= set(metadata["candidates"])
        assert runs == []  # never widened to the project root, never narrowed to one root

    @pytest.mark.asyncio
    async def test_files_in_a_repository_inside_a_markerless_root_select_the_repository(self, sandbox, runs):
        workspace = sandbox / "workspace"  # a folder of repositories: no marker of its own
        workspace.mkdir()
        repo, file_path = make_repo(workspace, "proj")

        payload = await call(absolute_file_paths=[file_path], _mcp_roots=mcp_roots(workspace))

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir_source"] == "files"
        assert payload["metadata"]["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_a_root_inside_the_project_clamps_to_the_root_and_names_the_project(self, sandbox, runs):
        project, file_path = make_repo(sandbox, "proj", file_name="pkg/a.py")
        root = project / "pkg"

        payload = await call(absolute_file_paths=[file_path], _mcp_roots=mcp_roots(root))

        assert runs[0]["working_dir"] == root
        metadata = payload["metadata"]
        assert metadata["working_dir"] == str(root)
        assert metadata["working_dir_source"] == "roots"
        assert any(
            str(project) in warning and "declared root" in warning for warning in metadata["working_dir_warnings"]
        )

    @pytest.mark.asyncio
    async def test_a_worktree_nested_under_a_root_is_selected_by_the_files(self, sandbox, runs):
        main, _ = make_repo(sandbox, "main")
        worktree = main / "wt"
        (worktree / "src").mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
        file_path = worktree / "src" / "a.py"
        file_path.write_text("z = 3\n")

        payload = await call(absolute_file_paths=[str(file_path)], _mcp_roots=mcp_roots(main))

        assert runs[0]["working_dir"] == worktree
        assert payload["metadata"]["working_dir_source"] == "files"

    @pytest.mark.asyncio
    async def test_a_home_root_does_not_make_a_repository_under_home_look_outside(self, sandbox, home, runs):
        repo, file_path = make_repo(home, "proj")

        payload = await call(absolute_file_paths=[file_path], _mcp_roots=mcp_roots(home))

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir_source"] == "files"
        assert payload["metadata"]["working_dir_warnings"] == []


# ---------------------------------------------------------------------------
# Lower tiers are fenced; broad roots never select
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("both_modes")
class TestFencedCandidatesAndBroadRoots:
    @pytest.mark.asyncio
    async def test_manifest_env_and_cwd_candidates_outside_the_roots_are_skipped_with_warnings(
        self, sandbox, home, runs, monkeypatch
    ):
        # A home root cannot select (it is broad) but still fences every lower-tier candidate.
        manifest, _ = make_repo(sandbox, "manifest")
        env_dir, _ = make_repo(sandbox, "envdir")
        server_dir, _ = make_repo(sandbox, "server")
        manifest_working_dir(monkeypatch, manifest)
        monkeypatch.setenv("CLINK_DEFAULT_WORKING_DIR", str(env_dir))
        monkeypatch.chdir(server_dir)

        payload = await call_error(_mcp_roots=mcp_roots(home))

        metadata = payload["metadata"]
        assert metadata["error_code"] == "underivable"
        fenced = [w for w in metadata["warnings"] if "outside the client's declared roots" in w]
        assert len(fenced) == 3 and all(str(home) in w for w in fenced)
        for candidate in (manifest, env_dir, server_dir):
            assert any(str(candidate) in w for w in fenced)
        assert runs == []

    @pytest.mark.asyncio
    async def test_a_manifest_directory_inside_the_roots_is_still_used(self, sandbox, home, runs, monkeypatch):
        manifest, _ = make_repo(home, "manifest")
        manifest_working_dir(monkeypatch, manifest)

        payload = await call(_mcp_roots=mcp_roots(home))

        assert runs[0]["working_dir"] == manifest
        assert payload["metadata"]["working_dir_source"] == "manifest"
        # The home root was dropped as a selector, and the response says so.
        assert any("home directory" in w and str(home) in w for w in payload["metadata"]["working_dir_warnings"])

    @pytest.mark.asyncio
    async def test_a_broad_root_is_dropped_with_a_warning_and_a_usable_root_still_selects(self, sandbox, home, runs):
        proj, _ = make_repo(home, "proj")

        payload = await call(_mcp_roots=mcp_roots(home, proj))

        assert runs[0]["working_dir"] == proj
        metadata = payload["metadata"]
        assert metadata["working_dir_source"] == "roots"
        assert any("home directory" in w and str(home) in w for w in metadata["working_dir_warnings"])


# ---------------------------------------------------------------------------
# Warnings: the handler's, the URI conversion's, and the derivation's all reach the response
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("both_modes")
class TestRootsWarnings:
    @pytest.mark.asyncio
    async def test_handler_and_uri_conversion_warnings_appear_in_the_response(self, sandbox, runs):
        root, _ = make_repo(sandbox, "proj")
        handler_warning = "roots/list did not answer within 3s"
        roots = mcp_roots(root, "file://build-host/work/proj", "https://example.com/proj", warnings=(handler_warning,))

        payload = await call(_mcp_roots=roots)

        assert runs[0]["working_dir"] == root
        metadata = payload["metadata"]
        assert metadata["working_dir_source"] == "roots"
        warnings = metadata["working_dir_warnings"]
        assert warnings[0] == handler_warning  # the handler's warnings come first
        assert any("build-host" in w and "ignored" in w for w in warnings)  # a non-local host
        assert any("https://example.com/proj" in w and "ignored" in w for w in warnings)  # a non-file root
        assert len(warnings) == 3

    @pytest.mark.asyncio
    async def test_a_handler_warning_alone_is_reported_and_roots_are_unavailable(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")

        payload = await call(
            absolute_file_paths=[file_path], _mcp_roots={"uris": [], "warnings": ["roots request failed: boom"]}
        )

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir_source"] == "files"
        assert payload["metadata"]["working_dir_warnings"] == ["roots request failed: boom"]

    @pytest.mark.asyncio
    async def test_an_empty_reply_is_silent(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")

        payload = await call(absolute_file_paths=[file_path], _mcp_roots={"uris": [], "warnings": []})

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_only_unusable_uris_leave_roots_unavailable_with_a_warning_each(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")
        bad = ["https://example.com/a", "file://remote/b", "file:///x/%2F/y"]

        payload = await call(absolute_file_paths=[file_path], _mcp_roots=mcp_roots(*bad))

        assert runs[0]["working_dir"] == repo  # no fence: the call is not refused over unusable roots
        warnings = payload["metadata"]["working_dir_warnings"]
        assert payload["metadata"]["working_dir_source"] == "files"
        assert len(warnings) == 3 and all("ignored" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_the_warnings_are_logged(self, sandbox, runs, caplog):
        root, _ = make_repo(sandbox, "proj")

        with caplog.at_level("WARNING", logger="tools.clink"):
            payload = await call(_mcp_roots=mcp_roots(root, "https://example.com/p", warnings=("handler said so",)))

        warnings = payload["metadata"]["working_dir_warnings"]
        assert "handler said so" in warnings
        for warning in warnings:
            assert any(warning in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_refusal_carries_the_handler_and_conversion_warnings(self, sandbox, runs, monkeypatch):
        repo, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(repo)
        record_cwd_substitution("/", str(repo))  # nothing else can be derived: underivable

        payload = await call_error(
            _mcp_roots=mcp_roots("https://example.com/p", warnings=("roots request failed: boom",))
        )

        metadata = payload["metadata"]
        assert metadata["error_code"] == "underivable"
        assert "roots request failed: boom" in metadata["warnings"]
        assert any("https://example.com/p" in w and "ignored" in w for w in metadata["warnings"])
        assert runs == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [None, "file:///x", ["file:///x"], 7, {"uris": "file:///x"}, {"warnings": "w"}, {}])
    async def test_a_malformed_private_argument_means_no_roots(self, sandbox, runs, bad):
        repo, file_path = make_repo(sandbox, "proj")

        payload = await call(absolute_file_paths=[file_path], _mcp_roots=bad)

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir_source"] == "files"
        assert payload["metadata"]["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_non_string_entries_are_skipped_with_a_warning(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")

        payload = await call(
            absolute_file_paths=[file_path],
            _mcp_roots={"uris": [123, None, uri(repo)], "warnings": ["ok", 5, None]},
        )

        warnings = payload["metadata"]["working_dir_warnings"]
        assert warnings[0] == "ok"
        assert sum("ignored: not a string" in w for w in warnings) == 2
        assert payload["metadata"]["working_dir_source"] == "files"  # the one valid root fences, the project stands


# ---------------------------------------------------------------------------
# An explicit working_dir needs no roots
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("both_modes")
class TestExplicitWorkingDir:
    @pytest.mark.asyncio
    async def test_an_explicit_directory_outside_the_roots_proceeds_without_a_roots_warning(
        self, sandbox, runs, derive_calls
    ):
        repo, _ = make_repo(sandbox, "proj")
        root, _ = make_repo(sandbox, "root")

        payload = await call(
            working_dir=str(repo),
            _mcp_roots=mcp_roots(root, "https://example.com/x", warnings=("roots request failed",)),
        )

        assert runs[0]["working_dir"] == repo
        metadata = payload["metadata"]
        assert metadata["working_dir"] == str(repo)
        assert metadata["working_dir_source"] == "explicit"
        assert metadata["working_dir_warnings"] == []  # no fence, no conflict, no roots warning of any kind
        assert derive_calls[0]["roots"] is None and list(derive_calls[0]["extra_warnings"]) == []

    @pytest.mark.asyncio
    async def test_an_explicit_directory_still_warns_about_files_outside_it_but_not_about_roots(self, sandbox, runs):
        repo, _ = make_repo(sandbox, "proj")
        _, foreign_file = make_repo(sandbox, "foreign")
        root, _ = make_repo(sandbox, "root")

        payload = await call(working_dir=str(repo), absolute_file_paths=[foreign_file], _mcp_roots=mcp_roots(root))

        warnings = payload["metadata"]["working_dir_warnings"]
        assert len(warnings) == 1 and foreign_file in warnings[0]

    @pytest.mark.asyncio
    async def test_an_invalid_explicit_directory_is_still_refused(self, sandbox, runs):
        root, _ = make_repo(sandbox, "root")

        payload = await call_error(working_dir="/", _mcp_roots=mcp_roots(root))

        assert payload["metadata"]["error_code"] == "invalid_explicit"
        assert runs == []


# ---------------------------------------------------------------------------
# The shape of the call into derive_working_dir, and the private key's lifetime
# ---------------------------------------------------------------------------


class TestPrivateArgument:
    @pytest.mark.asyncio
    async def test_roots_and_warnings_are_handed_to_the_derivation(self, sandbox, runs, derive_calls):
        root, _ = make_repo(sandbox, "proj")

        await call(_mcp_roots=mcp_roots(root, root, "https://example.com/x", warnings=("w1", "w2")))

        seen = derive_calls[0]
        assert seen["roots"] == [str(root)]  # converted to paths, duplicates collapsed
        extra = list(seen["extra_warnings"])
        assert extra[:2] == ["w1", "w2"]  # handler warnings first, then the conversion warning
        assert len(extra) == 3 and "https://example.com/x" in extra[2]

    @pytest.mark.asyncio
    async def test_no_usable_uri_and_no_key_mean_roots_none(self, sandbox, runs, derive_calls, monkeypatch):
        repo, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(repo)

        await call()
        await call(_mcp_roots={"uris": [], "warnings": []})
        await call(_mcp_roots=mcp_roots("https://example.com/x"))

        assert [c["roots"] for c in derive_calls] == [None, None, None]
        assert [list(c["extra_warnings"]) for c in derive_calls][:2] == [[], []]

    @pytest.mark.asyncio
    async def test_the_key_never_reaches_the_request_model_or_the_cli(self, sandbox, runs, monkeypatch):
        root, file_path = make_repo(sandbox, "proj")
        built: list[set[str]] = []

        class RecordingRequest(CLinkRequest):
            def __init__(self, **data: Any) -> None:
                built.append(set(data))
                super().__init__(**data)

        monkeypatch.setattr(CLinkTool, "get_request_model", lambda self: RecordingRequest)
        roots = mcp_roots(root)

        for extra in ({"absolute_file_paths": [file_path]}, {"working_dir": str(root)}):  # explicit pops it too
            tool = CLinkTool()
            payload = json.loads((await tool._execute_blocking(args(_mcp_roots=roots, **extra)))[0].text)

            assert "_mcp_roots" not in built[-1]
            assert "_mcp_roots" not in tool._current_arguments
            assert "_mcp_roots" not in runs[-1]
            assert uri(root) not in repr(runs[-1])  # not in the prompt, the files, or any other CLI input
            assert "_mcp_roots" not in json.dumps(payload) and uri(root) not in json.dumps(payload)

    def test_the_key_is_not_a_request_field(self):
        assert "_mcp_roots" not in CLinkRequest.model_fields
        assert "_mcp_roots" not in CLinkTool().get_input_schema()["properties"]

    @pytest.mark.asyncio
    async def test_the_tool_trusts_the_private_argument(self, sandbox, runs):
        """Whatever reaches the tool as ``_mcp_roots`` is the roots source. Stopping a client from sending the key
        itself is the handler's responsibility (it removes it from the wire arguments before resolving roots); this
        only documents that the tool is not the boundary."""
        _, file_path = make_repo(sandbox, "proj")
        forged, _ = make_repo(sandbox, "forged")

        payload = await call_error(absolute_file_paths=[file_path], _mcp_roots=mcp_roots(forged))

        assert payload["metadata"]["error_code"] == "roots_conflict"
        assert runs == []


# ---------------------------------------------------------------------------
# Roots are fresh on every call
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("both_modes")
class TestFreshRoots:
    @pytest.mark.asyncio
    async def test_roots_changing_between_two_calls_are_honoured(self, sandbox, runs, monkeypatch):
        root_a, _ = make_repo(sandbox, "a")
        root_b, _ = make_repo(sandbox, "b")
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        tool = CLinkTool()  # one instance: nothing may be remembered on it

        first = await call(tool, _mcp_roots=mcp_roots(root_a))
        second = await call(tool, _mcp_roots=mcp_roots(root_b))
        third = await call(tool)  # the host no longer declares roots (the key is absent)
        fourth = await call(tool, _mcp_roots=mcp_roots(root_a))

        assert [r["working_dir"] for r in runs] == [root_a, root_b, server_dir, root_a]
        assert [p["metadata"]["working_dir_source"] for p in (first, second, third, fourth)] == [
            "roots",
            "roots",
            "server_cwd",
            "roots",
        ]

    @pytest.mark.asyncio
    async def test_changed_roots_fence_the_files_of_the_next_call(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")
        elsewhere, _ = make_repo(sandbox, "elsewhere")
        tool = CLinkTool()

        ok = await call(tool, absolute_file_paths=[file_path], _mcp_roots=mcp_roots(repo))
        assert ok["metadata"]["working_dir"] == str(repo)

        refused = await call_error(absolute_file_paths=[file_path], _mcp_roots=mcp_roots(elsewhere))
        assert refused["metadata"]["error_code"] == "roots_conflict"

    @pytest.mark.asyncio
    async def test_roots_changing_between_continuation_turns_are_honoured(self, sandbox, runs):
        root_a, _ = make_repo(sandbox, "a")
        root_b, _ = make_repo(sandbox, "b")

        first = await call(_mcp_roots=mcp_roots(root_a))
        thread_id = first["continuation_offer"]["continuation_id"]
        assert first["metadata"]["working_dir"] == str(root_a)

        # The stored directory (a) is outside the host's current roots: not reused, warned about, replaced.
        second = await call(
            continuation_id=thread_id, prompt="more", _passed_file_paths=[], _mcp_roots=mcp_roots(root_b)
        )
        metadata = second["metadata"]
        assert metadata["working_dir"] == str(root_b) and metadata["working_dir_source"] == "roots"
        assert any(str(root_a) in w and "declared roots" in w for w in metadata["working_dir_warnings"])

        # The re-derived directory is inside the current roots, so the next turn reuses it.
        third = await call(
            continuation_id=thread_id, prompt="again", _passed_file_paths=[], _mcp_roots=mcp_roots(root_b)
        )
        assert third["metadata"]["working_dir"] == str(root_b)
        assert third["metadata"]["working_dir_source"] == "continuation"
        assert third["metadata"]["working_dir_warnings"] == []


# ---------------------------------------------------------------------------
# Hybrid jobs: roots travel in run_args, the job never queries the host
# ---------------------------------------------------------------------------


def _short_wait(monkeypatch) -> None:
    monkeypatch.setattr(CLinkTool, "_DEFAULT_JOB_WAIT_SECONDS", 0.05)


class TestHybridJobs:
    @pytest.mark.asyncio
    async def test_a_job_started_with_roots_derives_the_same_directory(self, sandbox, runs, monkeypatch, host_session):
        _short_wait(monkeypatch)
        assert get_current_request_context().session is host_session  # the spy really is the live context
        root, _ = make_repo(sandbox, "proj")
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)  # what a job without the roots would have used
        runs.options["delay"] = 0.3
        roots = mcp_roots(root, "https://example.com/x", warnings=("handler warning",))

        received: list[dict[str, Any]] = []
        original = CLinkTool._execute_blocking

        async def spy(self, arguments):
            received.append(dict(arguments))
            return await original(self, arguments)

        monkeypatch.setattr(CLinkTool, "_execute_blocking", spy)

        running = json.loads(json.loads((await CLinkTool().execute(args(_mcp_roots=roots)))[0].text)["content"])
        assert running["status"] == "running"
        final = json.loads((await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 5}))[0].text)

        # The job got the roots resolved by its originating request through run_args, exactly once.
        assert len(received) == 1 and received[0]["_mcp_roots"] == roots
        assert runs[0]["working_dir"] == root
        metadata = final["metadata"]
        assert metadata["working_dir"] == str(root)
        assert metadata["working_dir_source"] == "roots"
        assert metadata["working_dir_warnings"][0] == "handler warning"
        # Neither the foreground call, nor the job, nor the polling call asked the host for anything.
        assert host_session.requests == []

    @pytest.mark.asyncio
    async def test_a_job_derives_what_a_blocking_call_derives(self, sandbox, runs, monkeypatch):
        root_a, _ = make_repo(sandbox, "a")
        _, file_path = make_repo(sandbox, "b")
        cases = [
            {"_mcp_roots": mcp_roots(root_a)},
            {"absolute_file_paths": [file_path], "_mcp_roots": mcp_roots(root_a)},
        ]
        outcomes: dict[str, list[Any]] = {"blocking": [], "hybrid": []}

        for mode in outcomes:
            monkeypatch.setenv("CLINK_EXECUTION_MODE", mode)
            reset_clink_job_manager_for_tests()
            for case in cases:
                try:
                    payload = await call(**case)
                    outcomes[mode].append(
                        (payload["metadata"]["working_dir"], payload["metadata"]["working_dir_source"])
                    )
                except ToolExecutionError as exc:
                    metadata = json.loads(exc.payload)["metadata"]
                    outcomes[mode].append((metadata["error_code"], metadata["candidates"]))

        assert outcomes["blocking"] == outcomes["hybrid"]
        assert outcomes["blocking"][0] == (str(root_a), "roots")
        assert outcomes["blocking"][1] == ("roots_conflict", [str(root_a)])

    @pytest.mark.asyncio
    async def test_a_job_refusal_from_roots_carries_the_error_payload(self, sandbox, runs):
        root_a, _ = make_repo(sandbox, "a")
        root_b, _ = make_repo(sandbox, "b")

        with pytest.raises(ToolExecutionError) as exc_info:
            await CLinkTool().execute(args(_mcp_roots=mcp_roots(root_a, root_b), wait_seconds=5))

        metadata = json.loads(exc_info.value.payload)["metadata"]
        assert metadata["error_code"] == "ambiguous"
        assert set(metadata["candidates"]) == {str(root_a), str(root_b)}
        assert runs == []

    @pytest.mark.asyncio
    async def test_polling_a_job_does_not_need_or_use_roots(self, sandbox, runs, monkeypatch, derive_calls):
        _short_wait(monkeypatch)
        root, _ = make_repo(sandbox, "proj")
        runs.options["delay"] = 0.3

        running = json.loads(
            json.loads((await CLinkTool().execute(args(_mcp_roots=mcp_roots(root))))[0].text)["content"]
        )
        # The continuation carries only job control keys; derivation ran once, in the job, from the stored roots.
        final = json.loads((await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 5}))[0].text)

        assert len(derive_calls) == 1 and derive_calls[0]["roots"] == [str(root)]
        assert final["metadata"]["working_dir_source"] == "roots"
