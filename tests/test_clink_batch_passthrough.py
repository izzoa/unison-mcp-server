"""clink forwards its model string verbatim and never batches (add-batch-model-support, section 10).

Specs: clink-runtime-model-selection. Unison spawns a CLI subprocess and never sees its HTTP request, so
a suffixed model string such as ``openrouter/openai/gpt-6-astra-pro:batch`` is the CLI's business: it must
reach the CLI command byte-for-byte, and the ``supported_models`` allowlist (exact match) must neither
interpret nor strip the suffix. These tests drive the real ``CLinkTool`` and the real ``OpencodeAgent``;
only the operating-system process launch is replaced, so the argv the CLI would have received is observable.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import clink.registry as clink_registry
from clink import get_registry
from tools.clink import CLinkTool
from tools.shared.exceptions import ToolExecutionError

SUFFIXED = "openrouter/openai/gpt-6-astra-pro:batch"
UNSUFFIXED = "openrouter/openai/gpt-6-astra-pro"

OPENCODE_STDOUT = json.dumps({"type": "text", "text": "ok"}).encode("utf-8")


class _Stdin:
    def __init__(self) -> None:
        self.data = b""

    def write(self, chunk: bytes) -> None:
        self.data += chunk

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


class _StubProcess:
    """Enough of ``asyncio.subprocess.Process`` for both the ``communicate`` and the live-pump transports."""

    returncode = 0
    pid = 0

    def __init__(self) -> None:
        self.stdin = _Stdin()
        self.stdout = _reader(OPENCODE_STDOUT)
        self.stderr = _reader(b"")

    async def communicate(self, stdin_data: bytes | None = None) -> tuple[bytes, bytes]:
        return OPENCODE_STDOUT, b""

    async def wait(self) -> int:
        return 0

    def kill(self) -> None:
        pass


class _Launches:
    """Replaces ``asyncio.create_subprocess_exec`` and records every argv that would have been spawned."""

    def __init__(self) -> None:
        self.argv: list[list[str]] = []

    async def __call__(self, *args: str, **_kwargs: Any) -> _StubProcess:
        self.argv.append(list(args))
        return _StubProcess()


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> _Launches:
    launched = _Launches()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launched)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    return launched


def _opencode_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, supported_models: list[str]) -> None:
    """Replace the opencode preset with one that declares ``supported_models`` (and drop the cached registry)."""

    manifest = {
        "name": "opencode",
        "command": "opencode run",
        "additional_args": [],
        "env": {},
        "roles": {"default": {"prompt_path": "systemprompts/clink/default.txt"}},
        "supported_models": supported_models,
    }
    (tmp_path / "opencode.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("CLI_CLIENTS_CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(clink_registry, "_REGISTRY", None)


def _arguments(model: str, **extra: Any) -> dict[str, Any]:
    return {"prompt": "hi", "cli_name": "opencode", "model": model, "absolute_file_paths": [], "images": [], **extra}


def _has_pair(argv: list[str], flag: str, value: str) -> bool:
    return any(argv[i] == flag and argv[i + 1] == value for i in range(len(argv) - 1))


class TestSuffixedModelReachesTheCliCommandUnmodified:
    @pytest.mark.asyncio
    async def test_the_assembled_opencode_command_carries_the_suffixed_model_after_dash_m(self, launches):
        """10.2: ``["-m", "openrouter/openai/gpt-6-astra-pro:batch"]`` is in the command opencode is launched with."""

        result = await CLinkTool().execute(_arguments(SUFFIXED))

        assert len(launches.argv) == 1
        argv = launches.argv[0]
        assert _has_pair(argv, "-m", SUFFIXED), argv
        # Exactly one model flag and exactly the string supplied: nothing split on the colon, nothing dropped.
        assert argv.count("-m") == 1
        assert SUFFIXED in argv
        assert not any(a in ("openrouter/openai/gpt-6-astra-pro", "batch", ":batch") for a in argv)
        payload = json.loads(result[0].text)
        assert payload["metadata"]["model_requested"] == SUFFIXED

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model",
        [
            "openrouter/openai/gpt-6-astra-pro:free",
            "openrouter/openai/gpt-6-astra-pro:online",
            "openrouter/openai/gpt-6-astra-pro:thinking",
            "openrouter/openai/gpt-6-astra-pro:batch:batch",
            "openrouter/openai/gpt-6-astra-pro:batch#high",
        ],
    )
    async def test_every_provider_variant_segment_is_forwarded_the_same_way(self, launches, model):
        await CLinkTool().execute(_arguments(model))

        assert _has_pair(launches.argv[0], "-m", model), launches.argv[0]

    @pytest.mark.asyncio
    async def test_the_model_is_forwarded_without_any_batch_submission_by_the_server(self, launches, monkeypatch):
        """The server submits no provider batch for a CLI bridge call, whatever the model string says."""

        import tools.shared.batch_support as batch_support

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("a batch was touched for a clink call")

        monkeypatch.setattr(batch_support.BatchResumableToolMixin, "execute_batch_start", boom, raising=False)
        monkeypatch.setattr(batch_support.BatchResumableToolMixin, "admit_batch_request", boom, raising=False)

        await CLinkTool().execute(_arguments(SUFFIXED))

        assert _has_pair(launches.argv[0], "-m", SUFFIXED)

    @pytest.mark.asyncio
    async def test_a_suffix_is_never_a_reason_to_reject_a_string_the_cli_may_understand(self, launches):
        """No manifest allowlist, no pre-emptive rejection on the basis of a suffix."""

        assert get_registry().get_client("opencode").supported_models == []

        await CLinkTool().execute(_arguments("some/unknown-model:madeup-suffix"))

        assert _has_pair(launches.argv[0], "-m", "some/unknown-model:madeup-suffix")


class _FailingProcess(_StubProcess):
    """A CLI that does not recognise the model it was given and exits non-zero."""

    returncode = 1
    STDERR = b"ProviderModelNotFoundError: model 'gpt-6-astra-pro:madeup' not found"

    def __init__(self) -> None:
        super().__init__()
        self.stdout = _reader(b"")
        self.stderr = _reader(self.STDERR)

    async def communicate(self, stdin_data: bytes | None = None) -> tuple[bytes, bytes]:
        return b"", self.STDERR

    async def wait(self) -> int:
        return 1


class TestUnrecognisedSuffixIsACliLevelError:
    """Without an allowlist, a suffix the CLI rejects is the CLI's error, reported as such (not a Unison rejection)."""

    @pytest.mark.asyncio
    async def test_the_cli_error_is_reported_with_its_own_output_and_the_value_was_still_forwarded(self, monkeypatch):
        argv: list[list[str]] = []

        async def failing_exec(*args: str, **_kwargs: Any) -> _FailingProcess:
            argv.append(list(args))
            return _FailingProcess()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", failing_exec)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        model = "openrouter/openai/gpt-6-astra-pro:madeup"

        with pytest.raises(ToolExecutionError) as excinfo:
            await CLinkTool().execute(_arguments(model))

        assert _has_pair(argv[0], "-m", model)  # Unison did not pre-judge the suffix: the CLI was run with it
        payload = json.loads(excinfo.value.payload)
        assert payload["metadata"]["return_code"] == 1
        assert "ProviderModelNotFoundError" in payload["metadata"]["stderr"]  # the CLI's own error is what surfaces
        assert "supported_models" not in payload["content"] and "allowed" not in payload["content"].lower()


class TestCliClientsGuidanceDocument:
    """10.3: the exact-match allowlist trap is documented in ``conf/cli_clients/`` and the document is inert."""

    README = Path(clink_registry.__file__).resolve().parent.parent / "conf" / "cli_clients" / "README.md"

    def test_the_guidance_states_the_exact_match_trap_and_how_to_permit_a_suffixed_variant(self):
        text = self.README.read_text(encoding="utf-8")

        assert "supported_models" in text
        assert "exact string equality" in text
        flat = " ".join(text.split())
        assert f'`model: "{SUFFIXED}"` is rejected before any subprocess is launched' in flat
        assert UNSUFFIXED in text and SUFFIXED in text
        assert "list the suffixed form explicitly" in text.lower()

    def test_the_guidance_states_verbatim_forwarding_and_that_batch_does_not_apply(self):
        text = self.README.read_text(encoding="utf-8")

        assert "forwarded verbatim" in text
        assert "does not apply to clink" in text

    def test_the_clink_docs_do_not_contradict_themselves_about_allowlist_rejection(self):
        """A suffix is a CLI-level error *unless* an allowlist is declared, in which case Unison rejects it first."""

        docs = (self.README.parent.parent.parent / "docs" / "tools" / "clink.md").read_text(encoding="utf-8")
        section = docs.split("### Model strings, batch pricing and the `supported_models` allowlist", 1)[1]
        section = " ".join(section.split("### opencode v2", 1)[0].split())

        assert "never as a Unison-side rejection" not in section
        assert "unless the manifest declares a non-empty `supported_models` allowlist" in section
        assert "is rejected before the CLI starts" in section

    def test_the_readme_is_never_loaded_as_a_manifest(self, monkeypatch):
        monkeypatch.setattr(clink_registry, "_REGISTRY", None)
        monkeypatch.delenv("CLI_CLIENTS_CONFIG_PATH", raising=False)
        registry = get_registry()

        loaded = list(registry._iter_config_files())

        assert loaded and all(path.suffix == ".json" for path in loaded)
        assert self.README.exists()
        assert self.README not in loaded
        assert "readme" not in [name.lower() for name in registry.list_clients()]


class TestAllowlistMatchesExactly:
    @pytest.mark.asyncio
    async def test_a_suffixed_value_does_not_match_an_unsuffixed_entry_and_is_rejected_before_any_subprocess(
        self, tmp_path, monkeypatch, launches
    ):
        """10.3: the call fails before the CLI is invoked, naming the model and listing the allowed values."""

        _opencode_manifest(tmp_path, monkeypatch, [UNSUFFIXED])
        created: list[Any] = []
        import tools.clink as clink_tool

        real_create_agent = clink_tool.create_agent
        monkeypatch.setattr(clink_tool, "create_agent", lambda c: created.append(c) or real_create_agent(c))

        with pytest.raises(ToolExecutionError) as excinfo:
            await CLinkTool().execute(_arguments(SUFFIXED))

        content = json.loads(excinfo.value.payload)["content"]
        assert SUFFIXED in content and UNSUFFIXED in content and "allowed" in content.lower()
        assert launches.argv == []  # no subprocess invocation
        assert created == []  # not even an agent was built

    @pytest.mark.asyncio
    async def test_the_unsuffixed_entry_itself_is_still_accepted(self, tmp_path, monkeypatch, launches):
        _opencode_manifest(tmp_path, monkeypatch, [UNSUFFIXED])

        await CLinkTool().execute(_arguments(UNSUFFIXED))

        assert _has_pair(launches.argv[0], "-m", UNSUFFIXED)

    @pytest.mark.asyncio
    async def test_an_explicitly_listed_suffixed_variant_is_accepted_with_its_suffix_intact(
        self, tmp_path, monkeypatch, launches
    ):
        _opencode_manifest(tmp_path, monkeypatch, [SUFFIXED])

        await CLinkTool().execute(_arguments(SUFFIXED))

        assert _has_pair(launches.argv[0], "-m", SUFFIXED)

    @pytest.mark.asyncio
    async def test_listing_the_suffixed_form_does_not_admit_the_unsuffixed_one_either(
        self, tmp_path, monkeypatch, launches
    ):
        _opencode_manifest(tmp_path, monkeypatch, [SUFFIXED])

        with pytest.raises(ToolExecutionError):
            await CLinkTool().execute(_arguments(UNSUFFIXED))

        assert launches.argv == []


class TestClinkSchemaDocumentsTheDistinction:
    """10.1: the published ``model`` description states verbatim forwarding and that batch does not apply."""

    def _description(self) -> str:
        return CLinkTool().get_input_schema()["properties"]["model"]["description"]

    def test_the_published_description_says_the_value_is_forwarded_verbatim(self):
        text = self._description()
        assert "forwarded to the external CLI verbatim" in text

    def test_the_published_description_says_server_side_batch_does_not_apply(self):
        text = self._description()
        assert "Server-side batch execution does not apply to clink" in text

    def test_the_published_description_states_the_exact_match_allowlist_rule(self):
        text = self._description()
        assert "supported_models" in text and "matches exactly" in text

    def test_the_request_model_field_carries_the_same_statements(self):
        from tools.clink import CLinkRequest

        text = CLinkRequest.model_fields["model"].description or ""
        assert "forwarded to the external CLI verbatim" in text
        assert "Server-side batch execution does not apply to clink" in text

    def test_the_description_does_not_tell_a_caller_to_add_an_execution_suffix(self):
        text = self._description().lower()
        assert "append" not in text and "add a suffix" not in text and "add :batch" not in text

    def test_clink_publishes_no_execution_mode_field(self):
        assert "execution_mode" not in CLinkTool().get_input_schema()["properties"]
