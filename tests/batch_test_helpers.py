"""In-process fakes shared by the batch wait/resume/cancel tests. No network, no API keys.

``FakeRemote`` is the provider's side of the wire: it survives "restarts" in a test, while a
``FakeProvider`` (the adapter object) is cheap to throw away and rebuild, exactly like a real
adapter that forgets everything it was told at submission time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from providers.shared import (
    BatchError,
    BatchFailed,
    BatchListing,
    BatchOutcome,
    BatchRequest,
    BatchState,
    BatchStatus,
    CancelOutcome,
    CancelState,
    ModelResponse,
    ProviderType,
    UnsupportedBatchOperation,
)
from utils.batch_coordinator import BatchCoordinator, ChildPlan
from utils.batch_execution import BatchExecutor, WaitPolicy
from utils.batch_store import BatchStore

MODEL = "anthropic/claude-opus-4.5"


class FakeClock:
    """A monotonic clock whose ``sleep`` just advances it, so waits cost no real time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class RemoteBatch:
    model: str
    custom_ids: list[str]
    #: Provider-side states returned by successive polls; the last one repeats.
    script: list[BatchState] = field(default_factory=lambda: [BatchState.VALIDATING])
    polls: int = 0
    results: dict[str, Any] = field(default_factory=dict)  # custom_id -> str content or BatchError
    fetch_error: BaseException | None = None
    fetches: int = 0


class FakeRemote:
    """The provider service: holds batches independently of any adapter object."""

    def __init__(self) -> None:
        self.batches: dict[str, RemoteBatch] = {}
        self.submissions = 0

    def finish(self, batch_id: str, *, results: dict[str, Any] | None = None) -> None:
        batch = self.batches[batch_id]
        batch.script = [BatchState.COMPLETED]
        batch.results = results if results is not None else {cid: f"answer:{cid}" for cid in batch.custom_ids}


class FakeProvider:
    """Duck-typed batch adapter over a :class:`FakeRemote`."""

    def __init__(
        self,
        remote: FakeRemote,
        name: str = "openrouter",
        *,
        cancel: CancelOutcome | BaseException | None = None,
        script: list[BatchState] | None = None,
    ) -> None:
        self.remote = remote
        self.name = name
        self.calls: list[str] = []
        #: The resolution each ``submit_batch`` was handed (``None`` when the plan carried none).
        self.resolutions: list[Any] = []
        self.fetch_expected_ids: list[tuple[str, ...] | None] = []
        self.cancelled: list[str] = []
        self.poll_errors: list[BaseException] = []
        self._cancel = cancel if cancel is not None else CancelOutcome(CancelState.UNSUPPORTED)
        self._default_script = script
        self.on_poll: Any = None
        #: Called at the start of every submit_batch (before the batch exists): a slow or hung submission.
        self.on_submit: Any = None
        self.submit_error: BaseException | None = None
        self.listing: BatchListing | None = None

    def get_provider_type(self) -> Any:
        return SimpleNamespace(value=self.name)

    def batch_submission_model(self, model_name: str) -> str:
        return model_name.removesuffix(":batch")

    def submit_batch(self, model_name: str, requests: list[BatchRequest], resolution: Any = None) -> str:
        self.calls.append("submit_batch")
        self.resolutions.append(resolution)
        if self.on_submit is not None:
            self.on_submit()
        if self.submit_error is not None:
            raise self.submit_error
        self.remote.submissions += 1
        batch_id = f"prov_batch_{self.remote.submissions}"
        batch = RemoteBatch(model=model_name, custom_ids=[r.custom_id for r in requests])
        if self._default_script is not None:
            batch.script = list(self._default_script)
        self.remote.batches[batch_id] = batch
        return batch_id

    def poll_batch(self, batch_id: str) -> BatchStatus:
        self.calls.append("poll_batch")
        if self.on_poll is not None:
            self.on_poll(batch_id)
        if self.poll_errors:
            raise self.poll_errors.pop(0)
        batch = self.remote.batches[batch_id]
        index = min(batch.polls, len(batch.script) - 1)
        batch.polls += 1
        return BatchStatus(state=batch.script[index], provider=self.name, total=len(batch.custom_ids))

    def fetch_batch(self, batch_id: str, expected_custom_ids: Any = None) -> list[BatchOutcome]:
        self.calls.append("fetch_batch")
        self.fetch_expected_ids.append(tuple(expected_custom_ids) if expected_custom_ids is not None else None)
        batch = self.remote.batches[batch_id]
        batch.fetches += 1
        if batch.fetch_error is not None:
            raise batch.fetch_error
        if not batch.results:
            raise BatchFailed("batch ended without results", state=BatchState.FAILED)
        outcomes: list[BatchOutcome] = []
        # Deliberately reversed: correlation must be by custom_id, never by order.
        for custom_id in reversed(batch.custom_ids):
            value = batch.results.get(custom_id)
            if isinstance(value, BatchError):
                outcomes.append(BatchOutcome.failed(custom_id, value))
            elif value is None:
                outcomes.append(BatchOutcome.failed(custom_id, BatchError("missing_result", code="missing_result")))
            else:
                outcomes.append(
                    BatchOutcome.succeeded(
                        custom_id,
                        ModelResponse(
                            content=str(value),
                            usage={"total_tokens": 7},
                            # What a restarted adapter would report: not the recorded model.
                            model_name="adapter-does-not-know",
                            provider=ProviderType.OPENROUTER,
                            metadata={"batch_execution": True},
                        ),
                    )
                )
        return outcomes

    def list_batches(self, limit: int = 100) -> BatchListing:
        self.calls.append("list_batches")
        if self.listing is None:
            raise UnsupportedBatchOperation(self.name, "list_batches")
        return self.listing

    def cancel_batch(self, batch_id: str) -> CancelOutcome:
        self.calls.append("cancel_batch")
        self.cancelled.append(batch_id)
        if isinstance(self._cancel, BaseException):
            raise self._cancel
        return self._cancel


def consensus_payload(*models: str, proposal: str = "Evaluate X") -> dict[str, Any]:
    """A batched consensus record's payload for ``models`` (one batched entry each, neutral stance)."""

    from utils.consensus_batch import build_payload, persisted_roster

    roster = persisted_roster([{"model": name, "stance": "neutral"} for name in models], ["batch"] * len(models))
    return build_payload(proposal, "my findings", roster, [])


def consensus_plans(provider: Any, *models: str) -> list[ChildPlan]:
    """One plan per model, each carrying the request id (``consensus-<position>``) its roster entry owns."""

    return [make_plan(provider, name, (f"consensus-{position}",)) for position, name in enumerate(models)]


def make_requests(*ids: str) -> list[BatchRequest]:
    return [BatchRequest(custom_id=i, prompt=f"prompt {i}") for i in ids]


def make_plan(provider: Any, model: str = MODEL, ids: tuple[str, ...] = ("chat",)) -> ChildPlan:
    return ChildPlan(provider=provider, model=model, requests=make_requests(*ids))


def make_executor(
    store: BatchStore,
    providers: dict[str, Any],
    clock: FakeClock,
    *,
    policy: WaitPolicy | None = None,
) -> BatchExecutor:
    coordinator = BatchCoordinator(store, provider_resolver=lambda name: providers.get(name))
    return BatchExecutor(coordinator, policy=policy or WaitPolicy(), clock=clock, sleep=clock.sleep)


class OpenRouterBatchService:
    """OpenRouter's side of the wire behind ``OpenRouterProvider._batch_send``, for many batches.

    Install it with ``provider._batch_send = service``. It outlives any adapter object (a "restart" builds
    a new provider over the same service), keeps what each ``POST /batches`` carried, and answers
    ``GET /batches/{id}`` from a per-batch state that a test advances with :meth:`finish` or :meth:`fail`.
    """

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self.states: dict[str, str] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.failure: dict[str, str] = {}
        self.models: dict[str, str] = {}
        self.custom_ids: dict[str, list[str]] = {}
        self.post_error: httpx.Response | None = None
        #: Called with the batch id at the start of every ``GET /batches/{id}``: lets a test complete a batch mid-wait.
        self.on_get: Any = None

    # -- scripting ---------------------------------------------------------------

    def finish(self, batch_id: str, results: dict[str, Any] | None = None) -> None:
        """Complete a batch. A result is text, or ``(error_type, message)`` for a request that failed."""

        self.states[batch_id] = "completed"
        self.results[batch_id] = (
            results if results is not None else {cid: f"verdict:{cid}" for cid in self.custom_ids[batch_id]}
        )

    def fail(self, batch_id: str, message: str = "the batch failed") -> None:
        self.states[batch_id] = "failed"
        self.failure[batch_id] = message

    def batch_for_model(self, model: str) -> str:
        matches = [batch_id for batch_id, name in self.models.items() if name == model]
        assert len(matches) == 1, f"expected exactly one batch for {model}, found {matches}"
        return matches[0]

    @property
    def submitted_models(self) -> list[str]:
        return list(self.models.values())

    # -- the wire -------------------------------------------------------------------

    def __call__(self, method: str, path: str, content: str | None = None) -> httpx.Response:
        from tests.test_openrouter_batch_adapter import _accepted, _completed, _err, _ok

        if method == "POST":
            if self.post_error is not None:
                return self.post_error
            envelope = json.loads(content or "{}")
            batch_id = f"or_batch_{len(self.posts) + 1}"
            ids = [entry["custom_id"] for entry in envelope["requests"]]
            self.posts.append({"id": batch_id, "envelope": envelope})
            self.states[batch_id] = "in_progress"
            self.models[batch_id] = envelope["model"]
            self.custom_ids[batch_id] = ids
            counts = {"total": len(ids), "completed": 0, "failed": 0}
            return httpx.Response(202, json=_accepted(batch_id, model=envelope["model"], request_counts=counts))

        assert method == "GET" and path.startswith("/batches/"), (method, path)
        batch_id = path.rsplit("/", 1)[1]
        self.gets.append(batch_id)
        if self.on_get is not None:
            self.on_get(batch_id)
        ids = self.custom_ids[batch_id]
        counts = {"total": len(ids), "completed": 0, "failed": 0}
        state = self.states[batch_id]
        if state == "completed":
            items = [
                _err(cid, value[0], value[1]) if isinstance(value, tuple) else _ok(cid, str(value))
                for cid, value in self.results[batch_id].items()
            ]
            return httpx.Response(200, json=_completed(items, id=batch_id, model=self.models[batch_id]))
        if state == "failed":
            body = _accepted(batch_id, status="failed", request_counts=counts, model=self.models[batch_id])
            body["error"] = {"message": self.failure[batch_id], "code": "batch_failed"}
            return httpx.Response(200, json=body)
        return httpx.Response(
            200, json=_accepted(batch_id, status="in_progress", request_counts=counts, model=self.models[batch_id])
        )


def batch_world_factory(tmp_path: Any, service: OpenRouterBatchService, *, name: str = "batches.db") -> Any:
    """Build ``boot()``: each call is a fresh "process" (new store, registry, adapter, executor and tools).

    The batch service, the database file and (optionally) the thread store outlive every call; nothing
    in-process does. ``boot()`` returns a namespace with ``chat``, ``consensus``, ``store``, ``executor``,
    ``provider`` and ``close()``. The real :class:`OpenRouterProvider` is used with ``_batch_send`` stubbed.
    """

    from providers.openrouter import OpenRouterProvider
    from providers.registry import ModelProviderRegistry, set_default_registry
    from providers.shared import ProviderType
    from tools.chat import ChatTool
    from tools.consensus import ConsensusTool

    db_path = str(tmp_path / "durable" / name)

    def boot() -> Any:
        store = BatchStore(db_path)
        registry = ModelProviderRegistry(config={"OPENROUTER_API_KEY": "test-key"})
        registry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        set_default_registry(registry)
        provider = registry.get_provider(ProviderType.OPENROUTER)
        provider._batch_send = service
        clock = FakeClock()
        executor = make_executor(store, {"openrouter": provider}, clock)
        chat, consensus = ChatTool(), ConsensusTool()
        chat._batch_executor = executor
        consensus._batch_executor = executor
        return SimpleNamespace(
            chat=chat,
            consensus=consensus,
            store=store,
            executor=executor,
            provider=provider,
            clock=clock,
            registry=registry,
            close=store.close,
        )

    return boot


# ----------------------------------------------------------------------
# OpenRouter batch catalog (offline; nothing here opens a socket)
# ----------------------------------------------------------------------

CATALOG_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "openrouter_models_catalog.json"


def catalog_fixture_payload() -> dict[str, Any]:
    """The trimmed copy of OpenRouter's public model list that every batch test resolves against."""

    return json.loads(CATALOG_FIXTURE_PATH.read_text(encoding="utf-8"))


def fixture_listings() -> dict[str, dict[str, Any]]:
    """The fixture's listings by id (``anthropic/claude-opus-5.5:batch``, ``anthropic/claude-opus-5.5`` ...)."""

    return {entry["id"]: entry for entry in catalog_fixture_payload()["data"]}


def payload_with(changes: Any = None, *, extra_batch: Any = (), drop: Any = ()) -> dict[str, Any]:
    """The fixture body with some listings changed (``changes`` maps an id to a function that edits a copy of
    its entry), extra ``<slug>:batch`` listings added (``extra_batch`` maps a slug to its entry fields) and some ids
    dropped. Never touches the network and never mutates the shared fixture."""

    import copy

    payload = copy.deepcopy(catalog_fixture_payload())
    data = [entry for entry in payload["data"] if entry["id"] not in set(drop)]
    for entry in data:
        edit = (changes or {}).get(entry["id"])
        if edit is not None:
            edit(entry)
    for slug, fields in dict(extra_batch).items():
        data.append({"id": f"{slug}:batch", **fields})
    payload["data"] = data
    return payload


def payload_for(batch_slugs: Any, *, extra_ids: Any = ()) -> dict[str, Any]:
    """A model-list body listing ``<slug>:batch`` for each of ``batch_slugs`` (plus plain ``extra_ids``)."""

    ids = [f"{slug}:batch" for slug in batch_slugs] + list(extra_ids)
    return {"data": [{"id": identifier} for identifier in ids]}


class RecordingFetcher:
    """A catalog fetcher that serves a fixed payload and records every call (the 'network')."""

    def __init__(self, payload: Any = None, *, error: BaseException | None = None) -> None:
        self.payload = catalog_fixture_payload() if payload is None else payload
        self.error = error
        self.calls: list[float] = []

    def __call__(self, timeout: float) -> Any:
        self.calls.append(timeout)
        if self.error is not None:
            raise self.error
        return self.payload


def install_batch_catalog(
    batch_slugs: Any = None, *, fetcher: RecordingFetcher | None = None, **source_kwargs: Any
) -> RecordingFetcher:
    """Make the process-wide batch catalog source serve a fixture-built catalog; returns its fetcher.

    ``batch_slugs`` replaces the fixture's batch listings with exactly those; without it the whole
    fixture is served.
    """

    from providers.shared.batch_catalog import BatchCatalogSource, set_batch_catalog_source

    fetch = fetcher or RecordingFetcher(payload_for(batch_slugs) if batch_slugs is not None else None)
    set_batch_catalog_source(BatchCatalogSource(fetcher=fetch, **source_kwargs))
    return fetch


def parameter_source(resolution) -> str | None:
    """Where a curated resolution's parameter list came from, asserting that nothing else but the published-cap source
    (``max_output_published``, added by D12) is recorded beside it."""

    extra = set(resolution.sources) - {"supported_parameters", "max_output_published"}
    assert not extra, f"unexpected resolution sources: {sorted(extra)}"
    return resolution.sources.get("supported_parameters")
