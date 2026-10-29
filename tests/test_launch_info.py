"""The server records when it substituted its launch directory (clink-working-dir-derivation, D6)."""

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import server as server_module
from utils.launch_info import cwd_substitution, record_cwd_substitution, reset_launch_info_for_tests

SERVER_ROOT = Path(server_module.__file__).resolve().parent


@pytest.fixture
def launch(monkeypatch):
    """Return ``launch(directory)``: start the process in ``directory`` and call ``server.run()`` safely.

    Nothing really starts (``main`` and ``asyncio.run`` are stubbed). ``os.chdir`` stays real and is only
    observed, so a test can prove the original directory was captured *before* the switch;
    ``monkeypatch.chdir`` restores the caller's cwd on teardown. ``launch`` returns the directories
    ``run()`` asked ``os.chdir`` to switch into.
    """
    monkeypatch.delenv("SystemRoot", raising=False)
    monkeypatch.delenv("windir", raising=False)

    chdir_calls: list[str] = []
    real_chdir = os.chdir

    def spy_chdir(path):
        chdir_calls.append(str(path))
        real_chdir(path)

    monkeypatch.setattr(os, "chdir", spy_chdir)
    monkeypatch.setattr(server_module, "main", MagicMock(return_value=None))
    # Replace the module's asyncio reference rather than asyncio.run itself, so nothing global is patched.
    monkeypatch.setattr(server_module, "asyncio", SimpleNamespace(run=MagicMock()))

    def _launch(directory) -> list[str]:
        monkeypatch.chdir(directory)
        chdir_calls.clear()  # drop the test's own setup chdir; keep only what run() does
        server_module.run()
        return list(chdir_calls)

    return _launch


class TestLaunchInfoModule:
    def test_nothing_recorded_by_default(self):
        assert cwd_substitution() is None

    def test_record_then_read_back_as_strings(self, tmp_path):
        record_cwd_substitution(Path("/"), tmp_path)
        assert cwd_substitution() == (str(Path("/")), str(tmp_path))

    def test_accepts_plain_strings(self):
        record_cwd_substitution("/original", "/substituted")
        assert cwd_substitution() == ("/original", "/substituted")

    def test_reset_clears_the_record(self):
        record_cwd_substitution("/original", "/substituted")
        reset_launch_info_for_tests()
        assert cwd_substitution() is None

    def test_record_is_visible_across_threads(self):
        worker = threading.Thread(target=record_cwd_substitution, args=("/a", "/b"))
        worker.start()
        worker.join()
        assert cwd_substitution() == ("/a", "/b")


class TestRunRecordsSubstitution:
    def test_resolver_returns_target_without_recording(self):
        """``_resolve_pathological_cwd`` only decides; recording is run()'s job."""
        assert server_module._resolve_pathological_cwd(Path("/")) == SERVER_ROOT
        assert cwd_substitution() is None

    def test_launch_from_filesystem_root_records_original_and_target(self, launch):
        root = Path("/").resolve()
        assert server_module._resolve_pathological_cwd(root) == SERVER_ROOT

        chdir_targets = launch(root)

        assert chdir_targets == [str(SERVER_ROOT)]
        assert cwd_substitution() == (str(root), str(SERVER_ROOT))

    def test_original_cwd_is_captured_before_the_chdir(self, launch):
        """The chdir is real here, so a record taken after it would name the server root, not the launch dir."""
        root = Path("/").resolve()

        launch(root)

        assert Path.cwd().resolve() == SERVER_ROOT  # the switch really happened
        original, substituted_to = cwd_substitution()
        assert original == str(root)
        assert original != substituted_to

    def test_windows_system_directory_launch_records_substitution(self, tmp_path, monkeypatch, launch):
        fake_root = (tmp_path / "Windows").resolve()
        system32 = fake_root / "System32"
        system32.mkdir(parents=True)
        monkeypatch.setenv("SystemRoot", str(fake_root))

        launch(system32)

        assert cwd_substitution() == (str(system32), str(SERVER_ROOT))

    def test_project_directory_launch_records_nothing(self, tmp_path, launch):
        project = tmp_path / "project"
        project.mkdir()

        chdir_targets = launch(project)

        assert chdir_targets == []
        assert cwd_substitution() is None
        assert Path.cwd() == project.resolve()

    def test_launch_from_server_root_records_nothing(self, launch):
        """Developing Unison itself is a deliberate workspace, not a substitution."""
        chdir_targets = launch(SERVER_ROOT)

        assert chdir_targets == []
        assert cwd_substitution() is None


class TestStateDoesNotLeakBetweenTests:
    """Order-independent pair: whichever runs second would fail if the autouse reset did not run."""

    def test_first_of_pair(self):
        assert cwd_substitution() is None
        record_cwd_substitution("/leaky-one", SERVER_ROOT)

    def test_second_of_pair(self):
        assert cwd_substitution() is None
        record_cwd_substitution("/leaky-two", SERVER_ROOT)

    def test_run_path_state_does_not_leak_either(self, launch):
        assert cwd_substitution() is None
        launch("/")
        assert cwd_substitution() is not None

    def test_run_path_state_is_gone_afterwards(self):
        assert cwd_substitution() is None
