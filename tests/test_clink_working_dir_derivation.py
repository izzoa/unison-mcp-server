"""Tool-layer tests for clink's working-directory derivation (clink-working-dir-derivation, tasks 3.1-3.3, 4.1-4.2).

``clink/workdir.py`` owns the derivation rules and has its own table tests (``tests/test_clink_workdir.py``); these
tests drive ``CLinkTool`` end to end with a fake agent (never a real CLI): which directory reaches ``agent.run``, the
read-only snapshot and the response metadata, the refusal payloads, provenance on CLI failures and job payloads,
operator defaults, and reuse of the resolved directory across a conversation (including the real handler path that
replays the first turn's files).

Real directories are built under ``/tmp`` (``tempfile.mkdtemp(dir="/tmp")``): the broadness guard refuses the default
macOS temp directory.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from clink.agents import AgentOutput, CLIAgentError
from clink.parsers.base import ParsedCLIResponse
from clink.workdir import Derivation, WorkingDirError
from tools.clink import MAX_RESPONSE_CHARS, CLinkRequest, CLinkTool
from tools.clink_jobs import reset_clink_job_manager_for_tests
from tools.shared.exceptions import ToolExecutionError
from utils.launch_info import record_cwd_substitution

PROVENANCE_KEYS = ("working_dir", "working_dir_source", "working_dir_warnings")
HOME_WARNING = "user's home directory"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No operator defaults from the developer's environment; fresh job manager.

    The foreground wait budget is generous so a call that finishes quickly always returns its final payload, even on a
    loaded machine; tests that want a ``running`` payload shrink it themselves (``_short_wait``).
    """
    for name in ("CLINK_DEFAULT_WORKING_DIR", "CLINK_WORKING_DIR_FALLBACK", "CLINK_MCP_ROOTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(CLinkTool, "_DEFAULT_JOB_WAIT_SECONDS", 10.0)
    monkeypatch.setattr(CLinkTool, "_MAX_JOB_WAIT_SECONDS", 10.0)
    reset_clink_job_manager_for_tests()
    yield
    reset_clink_job_manager_for_tests()


@pytest.fixture
def sandbox():
    base = Path(tempfile.mkdtemp(dir="/tmp")).resolve()
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def make_repo(base: Path, name: str, *, file_name: str = "src/a.py") -> tuple[Path, str]:
    """A directory with a ``.git`` marker and one file; returns (root, file path)."""
    root = base / name
    (root / ".git").mkdir(parents=True)
    target = root / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1\n")
    return root, str(target)


class FakeAgent:
    """Records every ``run`` and optionally fails or sleeps; the agent class the tool gets from ``create_agent``."""

    fs_violation_ignore_patterns: tuple = ()

    def __init__(
        self,
        runs: list[dict[str, Any]],
        *,
        content: str = "ok",
        delay: float = 0.0,
        error: CLIAgentError | None = None,
        write_file: Path | None = None,
    ) -> None:
        self.runs = runs
        self.content = content
        self.delay = delay
        self.error = error
        self.write_file = write_file

    def get_read_only_args(self):
        return []

    async def run(self, **kwargs):
        self.runs.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.write_file is not None:
            self.write_file.write_text("written by the CLI")
        if self.error is not None:
            raise self.error
        return AgentOutput(
            parsed=ParsedCLIResponse(content=self.content, metadata={"model_used": "gemini-2.5-flash"}),
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


def agent_options(runs: Runs) -> dict[str, Any]:
    return runs.options


def manifest_working_dir(monkeypatch, value: Path | None) -> None:
    """Make every configured CLI's manifest ``working_dir`` resolve to ``value`` (registry stays untouched)."""
    from clink.registry import ClinkRegistry

    original = ClinkRegistry.get_client

    def patched(self, cli_name):
        return original(self, cli_name).model_copy(update={"working_dir": value})

    monkeypatch.setattr(ClinkRegistry, "get_client", patched)


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


# ---------------------------------------------------------------------------
# Derivation from files, explicit values, and the single directory
# ---------------------------------------------------------------------------


class TestFilesTier:
    @pytest.mark.asyncio
    async def test_omitted_working_dir_derives_the_repo_root_from_files(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")

        payload = await call(absolute_file_paths=[file_path])

        assert runs[0]["working_dir"] == repo
        metadata = payload["metadata"]
        assert metadata["working_dir"] == str(repo)
        assert metadata["working_dir_source"] == "files"
        assert metadata["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_agent_is_always_given_the_directory(self, sandbox, runs, monkeypatch):
        repo, _ = make_repo(sandbox, "proj")
        monkeypatch.chdir(repo)

        await call()

        assert runs[0]["working_dir"] == repo

    @pytest.mark.asyncio
    async def test_files_outrank_the_manifest_directory(self, sandbox, runs, monkeypatch):
        repo, file_path = make_repo(sandbox, "proj")
        other, _ = make_repo(sandbox, "other")
        manifest_working_dir(monkeypatch, other)

        payload = await call(absolute_file_paths=[file_path])

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir_source"] == "files"

    @pytest.mark.asyncio
    async def test_explicit_project_directory_behaves_as_before(self, sandbox, runs):
        repo, _ = make_repo(sandbox, "proj")

        payload = await call(working_dir=str(repo))

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir"] == str(repo)
        assert payload["metadata"]["working_dir_source"] == "explicit"
        assert payload["metadata"]["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_explicit_working_dir_is_reported_resolved(self, sandbox, runs):
        repo, _ = make_repo(sandbox, "proj")
        link = sandbox / "link"
        link.symlink_to(repo)

        payload = await call(working_dir=f"{link}/src/..")

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir"] == str(repo)

    @pytest.mark.asyncio
    async def test_explicit_symlinked_working_dir_is_stored_resolved_on_the_conversation(self, sandbox, runs):
        repo, _ = make_repo(sandbox, "proj")
        link = sandbox / "link"
        link.symlink_to(repo)

        payload = await call(working_dir=f"{link}/src/..")

        assert payload["metadata"]["working_dir"] == str(repo)
        assert _stored(_thread_id(payload)) == (str(repo), "explicit")

    @pytest.mark.asyncio
    async def test_explicit_working_dir_overrides_files_and_warns_about_them(self, sandbox, runs):
        repo, _ = make_repo(sandbox, "proj")
        _, foreign_file = make_repo(sandbox, "foreign")

        payload = await call(working_dir=str(repo), absolute_file_paths=[foreign_file])

        assert runs[0]["working_dir"] == repo
        assert payload["metadata"]["working_dir_source"] == "explicit"
        assert any(foreign_file in warning for warning in payload["metadata"]["working_dir_warnings"])


class TestReadOnlySnapshot:
    @pytest.mark.asyncio
    async def test_read_only_call_snapshots_the_derived_directory_not_the_manifest_one(
        self, sandbox, runs, monkeypatch
    ):
        repo, file_path = make_repo(sandbox, "proj")
        other, _ = make_repo(sandbox, "other")
        manifest_working_dir(monkeypatch, other)
        agent_options(runs)["write_file"] = repo / "written.txt"

        snapshots: list[str] = []
        import tools.clink as clink_module

        real_capture = clink_module.capture_snapshot

        def recording_capture(directory, **kwargs):
            snapshots.append(str(directory))
            return real_capture(directory, **kwargs)

        monkeypatch.setattr("tools.clink.capture_snapshot", recording_capture)

        payload = await call(absolute_file_paths=[file_path], read_only=True)

        assert snapshots == [str(repo), str(repo)]
        assert runs[0]["working_dir"] == repo
        # A write into the derived directory is reported (it would be invisible if the manifest dir were covered).
        assert "written.txt" in payload["metadata"]["read_only_violations"]["by_model"]["created"]


# ---------------------------------------------------------------------------
# The shape of the derivation call
# ---------------------------------------------------------------------------


class TestDerivationCall:
    @pytest.mark.asyncio
    async def test_inputs_handed_to_derive_working_dir(self, sandbox, runs, monkeypatch):
        repo, file_path = make_repo(sandbox, "proj")
        manifest, _ = make_repo(sandbox, "manifest")
        env_dir, _ = make_repo(sandbox, "envdir")
        manifest_working_dir(monkeypatch, manifest)
        monkeypatch.setenv("CLINK_DEFAULT_WORKING_DIR", f"  {env_dir}  ")
        monkeypatch.setenv("CLINK_WORKING_DIR_FALLBACK", "  GARBAGE ")
        monkeypatch.chdir(repo)
        seen: dict[str, Any] = {}

        def stub(**kwargs):
            seen.update(kwargs)
            return Derivation(repo, "files", [])

        monkeypatch.setattr("tools.clink.derive_working_dir", stub)

        await call(absolute_file_paths=[file_path], working_dir="  ")
        started = time.monotonic()

        assert seen["explicit"] is None  # whitespace counts as omitted
        assert seen["file_paths"] == [file_path]
        assert seen["manifest_dir"] == manifest
        assert seen["env_default"] == str(env_dir)
        assert seen["fallback_policy"] == "  GARBAGE "  # raw: derive_working_dir owns the parsing and the warning
        assert seen["server_cwd"] == str(repo)
        assert seen["server_cwd_substituted"] is False
        assert seen["continuation_dir"] is None and seen["continuation_source"] is None
        assert 0 < seen["deadline"] - started <= 2.0

    @pytest.mark.asyncio
    async def test_deleted_server_cwd_is_passed_as_none(self, sandbox, runs, monkeypatch):
        repo, _ = make_repo(sandbox, "proj")
        seen: dict[str, Any] = {}

        def stub(**kwargs):
            seen.update(kwargs)
            return Derivation(repo, "explicit", [])

        def gone():
            raise FileNotFoundError

        monkeypatch.setattr("tools.clink.derive_working_dir", stub)
        monkeypatch.setattr("tools.clink.os.getcwd", gone)

        await call()

        assert seen["server_cwd"] is None

    @pytest.mark.asyncio
    async def test_backstop_turns_a_blocked_probe_into_underivable(self, sandbox, runs, monkeypatch):
        import tools.clink as clink_module

        monkeypatch.setenv("CLINK_EXECUTION_MODE", "blocking")
        monkeypatch.setattr(clink_module, "_WORKDIR_WAIT_SECONDS", 0.1)
        monkeypatch.setattr(clink_module, "_WORKDIR_DEADLINE_SECONDS", 0.05)

        def blocked(**kwargs):
            time.sleep(1.0)  # a probe stuck in the kernel: the cooperative deadline cannot fire
            return Derivation(sandbox, "files", [])

        monkeypatch.setattr("tools.clink.derive_working_dir", blocked)

        started = time.monotonic()
        payload = await call_error()

        assert time.monotonic() - started < 0.9  # returned at the backstop, not when the probe finished
        assert payload["metadata"]["error_code"] == "underivable"
        assert "timed out" in payload["content"] and "working_dir" in payload["content"]
        assert runs == []

    def test_budget_constants(self):
        import tools.clink as clink_module

        assert clink_module._WORKDIR_DEADLINE_SECONDS == 2.0
        assert clink_module._WORKDIR_WAIT_SECONDS == 2.5


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.asyncio
    async def test_ambiguous_lists_candidates_and_never_runs(self, sandbox, runs):
        repo_a, file_a = make_repo(sandbox, "a")
        repo_b, file_b = make_repo(sandbox, "b")

        payload = await call_error(absolute_file_paths=[file_a, file_b])

        metadata = payload["metadata"]
        assert payload["status"] == "error"
        assert metadata["error_code"] == "ambiguous"
        assert {str(repo_a), str(repo_b)} <= set(metadata["candidates"])
        assert all(isinstance(candidate, str) for candidate in metadata["candidates"])
        assert isinstance(metadata["tried"], list) and metadata["tried"]
        assert isinstance(metadata["warnings"], list)
        assert "working_dir" in payload["content"]
        assert runs == []

    @pytest.mark.asyncio
    async def test_roots_conflict_payload_from_a_patched_derive(self, sandbox, runs, monkeypatch):
        def refuse(**kwargs):
            raise WorkingDirError(
                "roots_conflict", "files are outside every root", [sandbox / "x"], ["roots", "files"], ["w1"]
            )

        monkeypatch.setattr("tools.clink.derive_working_dir", refuse)

        payload = await call_error()

        assert payload["metadata"] == {
            "error_code": "roots_conflict",
            "candidates": [str(sandbox / "x")],
            "tried": ["roots", "files"],
            "warnings": ["w1"],
        }
        assert payload["content"] == "files are outside every root"
        assert runs == []

    @pytest.mark.asyncio
    async def test_a_non_utf8_candidate_still_yields_a_serializable_refusal(self, sandbox, runs, monkeypatch):
        # A surrogateescape'd directory name (a Latin-1 name on Linux) must not break JSON serialization.
        bad = Path("/a/caf\udce9")

        def refuse(**kwargs):
            raise WorkingDirError("ambiguous", "two projects", [bad], ["files"], ["warn \udce9"])

        monkeypatch.setattr("tools.clink.derive_working_dir", refuse)

        payload = await call_error()

        assert payload["metadata"]["error_code"] == "ambiguous"
        assert payload["metadata"]["candidates"] == ["/a/caf\\udce9"]
        assert payload["metadata"]["warnings"] == ["warn \\udce9"]
        assert runs == []

    @pytest.mark.asyncio
    async def test_a_non_utf8_chosen_directory_still_yields_a_serializable_response(self, sandbox, runs, monkeypatch):
        bad = Path("/a/caf\udce9")

        def derive(**kwargs):
            return Derivation(bad, "server_cwd", ["note \udce9"])

        monkeypatch.setattr("tools.clink.derive_working_dir", derive)

        payload = await call()

        assert payload["metadata"]["working_dir"] == "/a/caf\\udce9"
        assert payload["metadata"]["working_dir_warnings"] == ["note \\udce9"]

    @pytest.mark.asyncio
    async def test_refusal_warnings_are_logged(self, sandbox, runs, monkeypatch, caplog):
        def refuse(**kwargs):
            raise WorkingDirError("ambiguous", "two projects", [sandbox / "x"], ["files"], ["watch out one", "two"])

        monkeypatch.setattr("tools.clink.derive_working_dir", refuse)

        with caplog.at_level("WARNING", logger="tools.clink"):
            await call_error()

        logged = [record.getMessage() for record in caplog.records]
        for warning in ("watch out one", "two"):
            assert any(warning in line for line in logged)

    @pytest.mark.asyncio
    async def test_prompt_preparation_failure_carries_the_provenance(self, sandbox, runs, monkeypatch):
        repo, file_path = make_repo(sandbox, "proj")

        async def boom(self, *a, **k):
            raise RuntimeError("cannot read role prompt")

        monkeypatch.setattr(CLinkTool, "_prepare_prompt_for_role", boom)

        payload = await call_error(absolute_file_paths=[file_path])

        assert "Failed to prepare prompt" in payload["content"]
        metadata = payload["metadata"]
        assert metadata["working_dir"] == str(repo)
        assert metadata["working_dir_source"] == "files"
        assert metadata["working_dir_warnings"] == []
        assert runs == []

    @pytest.mark.asyncio
    async def test_fallback_refused(self, sandbox, runs, monkeypatch):
        repo, _ = make_repo(sandbox, "proj")
        monkeypatch.chdir(repo)
        monkeypatch.setenv("CLINK_WORKING_DIR_FALLBACK", "refuse")

        payload = await call_error()

        assert payload["metadata"]["error_code"] == "fallback_refused"
        assert runs == []

    @pytest.mark.asyncio
    async def test_underivable_when_the_server_cwd_was_substituted(self, sandbox, runs, monkeypatch):
        repo, _ = make_repo(sandbox, "proj")
        monkeypatch.chdir(repo)  # a perfectly good project, but the server put us here itself
        record_cwd_substitution("/", str(repo))

        payload = await call_error()

        assert payload["metadata"]["error_code"] == "underivable"
        assert payload["metadata"]["tried"]
        assert runs == []

    @pytest.mark.asyncio
    async def test_substituted_cwd_still_allows_files_to_select(self, sandbox, runs, monkeypatch):
        repo, file_path = make_repo(sandbox, "proj")
        record_cwd_substitution("/", "/somewhere")

        payload = await call(absolute_file_paths=[file_path])

        assert payload["metadata"]["working_dir_source"] == "files"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", ["/", "/etc", "/tmp", "relative/path", "missing"])
    async def test_invalid_explicit_is_refused_before_any_cli_spawns(self, sandbox, runs, bad):
        value = str(sandbox / "does-not-exist") if bad == "missing" else bad

        payload = await call_error(working_dir=value)

        assert payload["metadata"]["error_code"] == "invalid_explicit"
        assert isinstance(payload["metadata"]["candidates"], list)
        assert isinstance(payload["metadata"]["warnings"], list)
        assert runs == []

    @pytest.mark.asyncio
    async def test_relative_file_path_fails_validation_before_derivation(self, sandbox, runs, monkeypatch):
        def must_not_run(**kwargs):
            raise AssertionError("derivation ran before file-path validation")

        monkeypatch.setattr("tools.clink.derive_working_dir", must_not_run)

        payload = await call_error(absolute_file_paths=["relative/a.py"])

        assert "absolute" in payload["content"]
        assert "error_code" not in (payload.get("metadata") or {})
        assert runs == []


# ---------------------------------------------------------------------------
# Provenance in every response
# ---------------------------------------------------------------------------


class TestProvenanceMetadata:
    @pytest.mark.asyncio
    async def test_survives_output_truncation(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")
        agent_options(runs)["content"] = "x" * (MAX_RESPONSE_CHARS + 500)

        payload = await call(absolute_file_paths=[file_path])

        metadata = payload["metadata"]
        assert metadata["output_truncated"] is True
        assert metadata["working_dir"] == str(repo)
        assert metadata["working_dir_source"] == "files"
        assert metadata["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_survives_summary_pruning(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")
        agent_options(runs)["content"] = "<SUMMARY>short</SUMMARY>" + "x" * (MAX_RESPONSE_CHARS + 500)

        payload = await call(absolute_file_paths=[file_path])

        metadata = payload["metadata"]
        assert metadata["output_summarized"] is True
        assert payload["content"] == "short"
        assert metadata["working_dir"] == str(repo)
        assert metadata["working_dir_source"] == "files"
        assert metadata["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_warnings_are_reported_and_logged(self, sandbox, runs, monkeypatch, caplog):
        repo, _ = make_repo(sandbox, "proj")
        monkeypatch.chdir(repo)

        with caplog.at_level("WARNING", logger="tools.clink"):
            payload = await call()

        warnings = payload["metadata"]["working_dir_warnings"]
        assert warnings
        for warning in warnings:
            assert any(warning in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_cli_timeout_carries_the_provenance(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")
        agent_options(runs)["error"] = CLIAgentError("CLI 'gemini' timed out after 30s", stderr="slow")

        payload = await call_error(absolute_file_paths=[file_path])

        metadata = payload["metadata"]
        assert "timed out" in payload["content"]
        assert metadata["stderr"] == "slow"
        assert metadata["working_dir"] == str(repo)
        assert metadata["working_dir_source"] == "files"
        assert metadata["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_non_zero_exit_carries_the_provenance(self, sandbox, runs, monkeypatch):
        repo, _ = make_repo(sandbox, "proj")
        monkeypatch.chdir(repo)
        agent_options(runs)["error"] = CLIAgentError("exited with status 2", returncode=2, stdout="o", stderr="boom")

        payload = await call_error()

        metadata = payload["metadata"]
        assert metadata["return_code"] == 2
        assert metadata["stderr"] == "boom"
        assert metadata["working_dir"] == str(repo)
        assert metadata["working_dir_source"] == "server_cwd"
        assert metadata["working_dir_warnings"]  # the server-cwd fallback warning

    def test_build_error_metadata_without_provenance_is_unchanged(self):
        tool = CLinkTool()
        client = tool._registry.get_client("gemini")

        metadata = tool._build_error_metadata(client, CLIAgentError("x", returncode=1, stderr="e"))

        assert metadata == {"cli_name": "gemini", "return_code": 1, "stderr": "e"}


def _short_wait(monkeypatch) -> None:
    monkeypatch.setattr(CLinkTool, "_DEFAULT_JOB_WAIT_SECONDS", 0.05)


class TestHybridJobs:
    @pytest.mark.asyncio
    async def test_job_result_carries_the_provenance(self, sandbox, runs, monkeypatch):
        _short_wait(monkeypatch)
        repo, file_path = make_repo(sandbox, "proj")
        agent_options(runs)["delay"] = 0.3

        running = json.loads(
            json.loads((await CLinkTool().execute(args(absolute_file_paths=[file_path])))[0].text)["content"]
        )
        assert running["status"] == "running"

        final = json.loads((await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 5}))[0].text)
        assert final["metadata"]["working_dir"] == str(repo)
        assert final["metadata"]["working_dir_source"] == "files"
        assert final["metadata"]["working_dir_warnings"] == []
        assert runs[0]["working_dir"] == repo

    @pytest.mark.asyncio
    async def test_job_error_payload_carries_the_provenance(self, sandbox, runs, monkeypatch):
        _short_wait(monkeypatch)
        repo, file_path = make_repo(sandbox, "proj")
        agent_options(runs).update(delay=0.3, error=CLIAgentError("exited with status 3", returncode=3))

        running = json.loads(
            json.loads((await CLinkTool().execute(args(absolute_file_paths=[file_path])))[0].text)["content"]
        )
        assert running["status"] == "running"

        with pytest.raises(ToolExecutionError) as exc_info:
            await CLinkTool().execute({"job_id": running["job_id"], "wait_seconds": 5})
        metadata = json.loads(exc_info.value.payload)["metadata"]
        assert metadata["return_code"] == 3
        assert metadata["working_dir"] == str(repo)
        assert metadata["working_dir_source"] == "files"
        assert metadata["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_job_refusal_carries_the_error_payload(self, sandbox, runs):
        _, file_a = make_repo(sandbox, "a")
        _, file_b = make_repo(sandbox, "b")

        with pytest.raises(ToolExecutionError) as exc_info:
            await CLinkTool().execute(args(absolute_file_paths=[file_a, file_b], wait_seconds=5))

        assert json.loads(exc_info.value.payload)["metadata"]["error_code"] == "ambiguous"

    @pytest.mark.asyncio
    async def test_private_passed_files_travel_to_the_job_unchanged(self, sandbox, runs):
        repo_a, file_a = make_repo(sandbox, "a")
        _, file_b = make_repo(sandbox, "b")

        # The merged list is ambiguous; only the files passed on this call (the private key) select repo a.
        payload = await call(absolute_file_paths=[file_a, file_b], _passed_file_paths=[file_a])

        assert runs[0]["working_dir"] == repo_a
        assert payload["metadata"]["working_dir_source"] == "files"

    @pytest.mark.asyncio
    async def test_blocking_mode_derives_the_same_directory(self, sandbox, runs, monkeypatch):
        monkeypatch.setenv("CLINK_EXECUTION_MODE", "blocking")
        repo, file_path = make_repo(sandbox, "proj")

        payload = await call(absolute_file_paths=[file_path])

        assert payload["metadata"]["working_dir"] == str(repo)


# ---------------------------------------------------------------------------
# Operator defaults: manifest, environment, server cwd
# ---------------------------------------------------------------------------


class TestManifestDirectory:
    @pytest.mark.asyncio
    async def test_valid_manifest_directory_is_used_with_source_manifest(self, sandbox, runs, monkeypatch):
        manifest, _ = make_repo(sandbox, "manifest")
        manifest_working_dir(monkeypatch, manifest)

        payload = await call()

        assert runs[0]["working_dir"] == manifest
        assert payload["metadata"]["working_dir_source"] == "manifest"
        assert payload["metadata"]["working_dir_warnings"] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["missing", "file", "broad"])
    async def test_invalid_manifest_directory_is_skipped_with_a_warning(self, sandbox, runs, monkeypatch, kind):
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        if kind == "missing":
            bad = sandbox / "nope"
        elif kind == "file":
            bad = sandbox / "plain.txt"
            bad.write_text("not a directory")
        else:
            bad = Path("/etc")
        manifest_working_dir(monkeypatch, bad)

        payload = await call()

        metadata = payload["metadata"]
        assert metadata["working_dir"] == str(server_dir)
        assert metadata["working_dir_source"] == "server_cwd"
        assert any(str(bad) in warning for warning in metadata["working_dir_warnings"])

    @pytest.mark.asyncio
    async def test_a_directory_created_after_startup_is_honoured(self, sandbox, runs, monkeypatch):
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        late = sandbox / "late"
        manifest_working_dir(monkeypatch, late)

        first = await call()
        assert first["metadata"]["working_dir_source"] == "server_cwd"

        late.mkdir()
        second = await call()
        assert second["metadata"]["working_dir"] == str(late)
        assert second["metadata"]["working_dir_source"] == "manifest"


class TestEnvironmentDefaults:
    @pytest.mark.asyncio
    async def test_default_working_dir_env_is_used_with_source_env(self, sandbox, runs, monkeypatch):
        env_dir, _ = make_repo(sandbox, "envdir")
        monkeypatch.setenv("CLINK_DEFAULT_WORKING_DIR", str(env_dir))

        payload = await call()

        assert runs[0]["working_dir"] == env_dir
        assert payload["metadata"]["working_dir_source"] == "env"

    @pytest.mark.asyncio
    async def test_manifest_outranks_the_env_default(self, sandbox, runs, monkeypatch):
        env_dir, _ = make_repo(sandbox, "envdir")
        manifest, _ = make_repo(sandbox, "manifest")
        monkeypatch.setenv("CLINK_DEFAULT_WORKING_DIR", str(env_dir))
        manifest_working_dir(monkeypatch, manifest)

        payload = await call()

        assert payload["metadata"]["working_dir_source"] == "manifest"

    @pytest.mark.asyncio
    async def test_invalid_env_default_is_skipped_with_a_warning(self, sandbox, runs, monkeypatch):
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        missing = sandbox / "missing"
        monkeypatch.setenv("CLINK_DEFAULT_WORKING_DIR", str(missing))

        payload = await call()

        metadata = payload["metadata"]
        assert metadata["working_dir_source"] == "server_cwd"
        assert any(str(missing) in warning for warning in metadata["working_dir_warnings"])

    @pytest.mark.asyncio
    async def test_fallback_warn_is_the_default(self, sandbox, runs, monkeypatch):
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)

        payload = await call()

        assert payload["metadata"]["working_dir_source"] == "server_cwd"
        assert any("working_dir was not given" in w for w in payload["metadata"]["working_dir_warnings"])

    @pytest.mark.asyncio
    async def test_fallback_allow_is_quiet(self, sandbox, runs, monkeypatch):
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        monkeypatch.setenv("CLINK_WORKING_DIR_FALLBACK", "allow")

        payload = await call()

        assert payload["metadata"]["working_dir"] == str(server_dir)
        assert payload["metadata"]["working_dir_source"] == "server_cwd"
        assert payload["metadata"]["working_dir_warnings"] == []

    @pytest.mark.asyncio
    async def test_fallback_allow_does_not_loosen_the_guard(self, sandbox, runs, monkeypatch):
        record_cwd_substitution("/", str(sandbox))
        monkeypatch.chdir(sandbox)
        monkeypatch.setenv("CLINK_WORKING_DIR_FALLBACK", "allow")

        payload = await call_error()

        assert payload["metadata"]["error_code"] == "underivable"

    @pytest.mark.asyncio
    async def test_fallback_refuse(self, sandbox, runs, monkeypatch):
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        monkeypatch.setenv("CLINK_WORKING_DIR_FALLBACK", "Refuse")

        payload = await call_error()

        assert payload["metadata"]["error_code"] == "fallback_refused"

    @pytest.mark.asyncio
    async def test_fallback_garbage_behaves_as_warn_and_names_the_value(self, sandbox, runs, monkeypatch):
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        monkeypatch.setenv("CLINK_WORKING_DIR_FALLBACK", "bananas")

        payload = await call()

        warnings = payload["metadata"]["working_dir_warnings"]
        assert payload["metadata"]["working_dir_source"] == "server_cwd"
        assert any("bananas" in warning for warning in warnings)
        assert any("working_dir was not given" in warning for warning in warnings)


# ---------------------------------------------------------------------------
# Continuation: the resolved directory is stored on the thread and reused
# ---------------------------------------------------------------------------


def _handler_call(runs: Runs):
    """Drive turns through the real handler path (initial-context merge, ``_passed_file_paths`` capture)."""
    from unittest.mock import MagicMock, patch

    import server

    class _Driver:
        async def __call__(self, **overrides) -> dict[str, Any]:
            with patch("utils.model_context.ModelContext.calculate_token_allocation") as mock_calc:
                mock_calc.return_value = MagicMock(
                    total_tokens=200000,
                    content_tokens=160000,
                    response_tokens=40000,
                    file_tokens=64000,
                    history_tokens=64000,
                )
                with patch("utils.conversation_memory.build_conversation_history") as mock_build:
                    mock_build.return_value = ("=== CONVERSATION HISTORY ===\n", 1000)
                    arguments = args(**overrides)
                    if "absolute_file_paths" not in overrides:
                        # A real client simply omits the key; the handler then replays the first turn's files.
                        del arguments["absolute_file_paths"]
                    result = await server.handle_call_tool("clink", arguments)
            return json.loads(result[0].text)

        async def error(self, **overrides) -> dict[str, Any]:
            with pytest.raises(ToolExecutionError) as exc_info:
                await self(**overrides)
            return json.loads(exc_info.value.payload)

    return _Driver()


def _thread_id(payload: dict[str, Any]) -> str:
    return payload["continuation_offer"]["continuation_id"]


def _stored(thread_id: str) -> tuple[str | None, str | None]:
    from utils import conversation_store

    thread = conversation_store.get_thread(thread_id)
    assert thread is not None
    return thread.working_dir, thread.working_dir_source


class TestContinuation:
    @pytest.mark.asyncio
    async def test_first_call_stores_the_directory_and_source_on_the_new_thread(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")

        payload = await call(absolute_file_paths=[file_path])

        assert _stored(_thread_id(payload)) == (str(repo), "files")

    @pytest.mark.asyncio
    async def test_continuation_reuses_the_directory_without_files(self, sandbox, runs):
        repo, file_path = make_repo(sandbox, "proj")
        drive = _handler_call(runs)

        first = await drive(absolute_file_paths=[file_path])
        assert first["metadata"]["working_dir_source"] == "files"
        second = await drive(continuation_id=_thread_id(first), prompt="more")

        assert runs[1]["working_dir"] == repo
        assert second["metadata"]["working_dir"] == str(repo)
        assert second["metadata"]["working_dir_source"] == "continuation"
        assert second["metadata"]["working_dir_warnings"] == []
        # Reusing the directory leaves the stored origin alone.
        assert _stored(_thread_id(first)) == (str(repo), "files")

    @pytest.mark.asyncio
    async def test_replayed_first_turn_files_do_not_drive_derivation(self, sandbox, runs):
        repo_a, file_a = make_repo(sandbox, "a")
        repo_b, _ = make_repo(sandbox, "b")
        drive = _handler_call(runs)

        # Turn one: files in a, but an explicit directory b. If the replayed file were used on turn two, the stored
        # directory b would not contain it and clink would re-derive to a.
        first = await drive(absolute_file_paths=[file_a], working_dir=str(repo_b))
        second = await drive(continuation_id=_thread_id(first), prompt="more")

        assert second["metadata"]["working_dir"] == str(repo_b)
        assert second["metadata"]["working_dir_source"] == "continuation"
        assert second["metadata"]["working_dir_warnings"] == []
        assert runs[1]["working_dir"] == repo_b

    @pytest.mark.asyncio
    async def test_an_explicit_directory_from_turn_one_reports_continuation_on_turn_two(self, sandbox, runs):
        repo, _ = make_repo(sandbox, "proj")
        drive = _handler_call(runs)

        first = await drive(working_dir=str(repo))
        assert first["metadata"]["working_dir_source"] == "explicit"
        second = await drive(continuation_id=_thread_id(first), prompt="more")

        assert second["metadata"]["working_dir_source"] == "continuation"
        assert second["metadata"]["working_dir"] == str(repo)

    @pytest.mark.asyncio
    async def test_removed_directory_re_derives_warns_and_is_stored(self, sandbox, runs, monkeypatch):
        gone, _ = make_repo(sandbox, "gone")
        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        drive = _handler_call(runs)

        first = await drive(working_dir=str(gone))
        shutil.rmtree(gone)
        second = await drive(continuation_id=_thread_id(first), prompt="more")

        metadata = second["metadata"]
        assert metadata["working_dir"] == str(server_dir)
        assert metadata["working_dir_source"] == "server_cwd"
        assert any(str(gone) in warning for warning in metadata["working_dir_warnings"])
        assert _stored(_thread_id(first)) == (str(server_dir), "server_cwd")

        # A re-derived directory is what a later call without files reuses.
        third = await drive(continuation_id=_thread_id(first), prompt="and more")
        assert third["metadata"]["working_dir"] == str(server_dir)
        assert third["metadata"]["working_dir_source"] == "continuation"
        assert runs[2]["working_dir"] == server_dir

    @pytest.mark.asyncio
    async def test_new_files_outside_the_stored_directory_re_derive(self, sandbox, runs):
        repo_a, file_a = make_repo(sandbox, "a")
        repo_b, file_b = make_repo(sandbox, "b")
        drive = _handler_call(runs)

        first = await drive(absolute_file_paths=[file_a])
        second = await drive(continuation_id=_thread_id(first), prompt="more", absolute_file_paths=[file_b])

        metadata = second["metadata"]
        assert metadata["working_dir"] == str(repo_b)
        assert metadata["working_dir_source"] == "files"
        assert any(str(repo_a) in warning for warning in metadata["working_dir_warnings"])

        third = await drive(continuation_id=_thread_id(first), prompt="and more")
        assert third["metadata"]["working_dir"] == str(repo_b)
        assert third["metadata"]["working_dir_source"] == "continuation"

    @pytest.mark.asyncio
    async def test_stored_main_checkout_with_worktree_files_re_derives_to_the_worktree(self, sandbox, runs):
        main, main_file = make_repo(sandbox, "main")
        worktree = main / "wt"
        worktree.mkdir()
        (worktree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
        (worktree / "x.py").write_text("y = 2\n")
        drive = _handler_call(runs)

        first = await drive(absolute_file_paths=[main_file])
        assert first["metadata"]["working_dir"] == str(main)
        second = await drive(
            continuation_id=_thread_id(first), prompt="more", absolute_file_paths=[str(worktree / "x.py")]
        )

        metadata = second["metadata"]
        assert metadata["working_dir"] == str(worktree)
        assert metadata["working_dir_source"] == "files"
        assert any("checkout" in warning for warning in metadata["working_dir_warnings"])
        assert _stored(_thread_id(first)) == (str(worktree), "files")

    @pytest.mark.asyncio
    async def test_stored_explicit_home_is_reused_with_the_warning_and_stays_reusable(self, sandbox, runs, monkeypatch):
        home = sandbox / "home" / "me"
        home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        drive = _handler_call(runs)

        first = await drive(working_dir=str(home))
        assert first["metadata"]["working_dir_source"] == "explicit"
        assert any(HOME_WARNING in w for w in first["metadata"]["working_dir_warnings"])

        second = await drive(continuation_id=_thread_id(first), prompt="more")
        third = await drive(continuation_id=_thread_id(first), prompt="again")

        for later in (second, third):
            assert later["metadata"]["working_dir"] == str(home)
            assert later["metadata"]["working_dir_source"] == "continuation"
            assert any(HOME_WARNING in w for w in later["metadata"]["working_dir_warnings"])
        # Reuse must not rewrite the stored source, or turn three would see a non-explicit home and refuse it.
        assert _stored(_thread_id(first)) == (str(home), "explicit")

    @pytest.mark.asyncio
    async def test_a_later_explicit_override_replaces_the_stored_directory(self, sandbox, runs):
        repo_a, file_a = make_repo(sandbox, "a")
        repo_b, _ = make_repo(sandbox, "b")
        drive = _handler_call(runs)

        first = await drive(absolute_file_paths=[file_a])
        second = await drive(continuation_id=_thread_id(first), prompt="more", working_dir=str(repo_b))
        third = await drive(continuation_id=_thread_id(first), prompt="and more")

        assert second["metadata"]["working_dir"] == str(repo_b)
        assert second["metadata"]["working_dir_source"] == "explicit"
        assert third["metadata"]["working_dir"] == str(repo_b)
        assert third["metadata"]["working_dir_source"] == "continuation"
        assert _stored(_thread_id(first)) == (str(repo_b), "explicit")

    @pytest.mark.asyncio
    async def test_cross_tool_thread_without_a_stored_directory_derives_and_stores(self, sandbox, runs):
        from utils import conversation_store

        repo, file_path = make_repo(sandbox, "proj")
        thread_id = conversation_store.create_thread("chat", {"prompt": "earlier"})
        conversation_store.add_turn(thread_id, "user", "earlier", tool_name="chat")
        conversation_store.add_turn(thread_id, "assistant", "reply", tool_name="chat", model_name="gemini-2.5-flash")
        drive = _handler_call(runs)

        payload = await drive(continuation_id=thread_id, absolute_file_paths=[file_path])

        assert payload["metadata"]["working_dir"] == str(repo)
        assert payload["metadata"]["working_dir_source"] == "files"
        assert _stored(thread_id) == (str(repo), "files")

    @pytest.mark.asyncio
    async def test_legacy_thread_without_stored_fields_derives_afresh(self, sandbox, runs, monkeypatch):
        from utils import conversation_store

        server_dir, _ = make_repo(sandbox, "server")
        monkeypatch.chdir(server_dir)
        thread_id = conversation_store.create_thread("clink", {"prompt": "earlier"})
        conversation_store.add_turn(thread_id, "user", "earlier", tool_name="clink")
        conversation_store.add_turn(thread_id, "assistant", "reply", tool_name="clink", model_name="gemini-2.5-flash")
        drive = _handler_call(runs)

        payload = await drive(continuation_id=thread_id)

        assert payload["metadata"]["working_dir_source"] == "server_cwd"

    @pytest.mark.asyncio
    async def test_direct_invocation_with_empty_passed_files_ignores_replayed_request_files(self, sandbox, runs):
        from utils import conversation_store

        repo_a, file_a = make_repo(sandbox, "a")
        repo_b, _ = make_repo(sandbox, "b")
        thread_id = conversation_store.create_thread("clink", {"prompt": "earlier"})
        conversation_store.add_turn(thread_id, "user", "earlier", tool_name="clink")
        conversation_store.add_turn(thread_id, "assistant", "reply", tool_name="clink", model_name="gemini-2.5-flash")
        assert conversation_store.set_thread_working_dir(thread_id, str(repo_b), "explicit")

        # The request still lists a's file (as the replayed initial context would); the private key says none was
        # passed on this call, so the stored directory is reused.
        payload = await call(continuation_id=thread_id, absolute_file_paths=[file_a], _passed_file_paths=[])
        assert payload["metadata"]["working_dir"] == str(repo_b)
        assert payload["metadata"]["working_dir_source"] == "continuation"

        # Without the private key (a direct call) the request's own files are the files passed on this call.
        payload = await call(continuation_id=thread_id, absolute_file_paths=[file_a])
        assert payload["metadata"]["working_dir"] == str(repo_a)
        assert payload["metadata"]["working_dir_source"] == "files"

    @pytest.mark.asyncio
    async def test_storage_failure_never_fails_the_call(self, sandbox, runs, monkeypatch):
        repo, file_path = make_repo(sandbox, "proj")
        from utils import conversation_store

        def boom(*_args, **_kwargs):
            raise RuntimeError("store is down")

        monkeypatch.setattr(conversation_store, "set_thread_working_dir", boom)

        payload = await call(absolute_file_paths=[file_path])

        assert payload["metadata"]["working_dir"] == str(repo)

    @pytest.mark.asyncio
    async def test_private_key_is_not_a_request_field(self, sandbox, runs):
        assert "_passed_file_paths" not in CLinkRequest.model_fields


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchemaDescriptions:
    def test_both_descriptions_say_the_parameter_may_be_omitted(self):
        schema_text = CLinkTool().get_input_schema()["properties"]["working_dir"]["description"]
        model_text = CLinkRequest.model_fields["working_dir"].description or ""

        for text in (schema_text, model_text):
            assert "may be omitted" in text
            assert "derives" in text
            assert "files" in text and "conversation" in text
            assert "roots" in text
            assert "PWD" in text
            assert "override" in text

    def test_working_dir_stays_optional_in_the_schema(self):
        schema = CLinkTool().get_input_schema()
        assert "working_dir" not in schema["required"]
        assert schema["properties"]["working_dir"]["type"] == "string"
