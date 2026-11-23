"""Consensus with batched consultations (add-batch-model-support, section 8).

In-process integration tests through the real ``ConsensusTool`` entry: a real ``BatchStore`` on disk, the
real ``OpenRouterProvider`` with only its HTTP edge (``_batch_send``) stubbed by an ``OpenRouterBatchService``,
the real executor, coordinator and registry. A "restart" discards every in-process object (store, registry,
adapter, executor and the tool instance) and keeps only the database file and the provider's service.
Synchronous consultations are stubbed at ``ConsensusTool._consult_model`` (the provider call a real run
makes), which is also how the tests count whether one was run again. No network, no API keys.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import utils.model_restrictions as model_restrictions
from providers.openrouter import OpenRouterProvider
from tests.batch_test_helpers import OpenRouterBatchService, batch_world_factory
from tools.consensus import ConsensusTool
from tools.shared.exceptions import ToolExecutionError
from utils import batch_store as bs
from utils.consensus_batch import PAYLOAD_KEYS

PROPOSAL = "Evaluate migrating the billing service to event sourcing. Consider cost, risk and timeline."
FINDINGS = "My own analysis: promising but the migration risk is high."
OPUS, SONNET, GPT5 = "opus", "sonnet", "gpt5"
# 14.13: under batch the family aliases resolve to the latest batch-listed member (fixture catalog), not the curated pins.
SLUG = {OPUS: "anthropic/claude-opus-5.5", SONNET: "anthropic/claude-sonnet-5.5", GPT5: "openai/gpt-5.5"}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("BATCH_STORE_PATH", "BATCH_RETENTION_DAYS", "OPENROUTER_ALLOWED_MODELS"):
        monkeypatch.delenv(name, raising=False)
    model_restrictions._restriction_service = None
    bs.reset_batch_store()
    yield
    bs.reset_batch_store()
    model_restrictions._restriction_service = None


@pytest.fixture(autouse=True)
def _no_synchronous_openrouter_call(monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("the synchronous provider path ran for a batched consultation")

    monkeypatch.setattr(OpenRouterProvider, "generate_content", boom)
    monkeypatch.setattr(OpenRouterProvider, "async_generate_content", boom, raising=False)


@pytest.fixture
def service() -> OpenRouterBatchService:
    return OpenRouterBatchService()


@pytest.fixture
def boot(tmp_path, service):
    started = []
    factory = batch_world_factory(tmp_path, service)

    def make():
        world = factory()
        started.append(world)
        return world

    yield make
    for world in started:
        try:
            world.close()
        except Exception:
            pass


class SyncCalls:
    """Stands in for ``ConsensusTool._consult_model`` and records which models were consulted."""

    def __init__(self, tool: ConsensusTool, monkeypatch: pytest.MonkeyPatch, *, fail: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail = fail or set()

        async def consult(this: ConsensusTool, config: dict, request: Any) -> dict:
            self.calls.append(config["model"])
            if config["model"] in self.fail:
                return {
                    "model": config["model"],
                    "stance": config.get("stance", "neutral"),
                    "status": "error",
                    "error": "boom",
                }
            return {
                "model": config["model"],
                "stance": config.get("stance", "neutral"),
                "status": "success",
                "verdict": f"sync verdict from {config['model']} ({this.original_proposal})",
                "metadata": {"provider": "test", "model_name": config["model"]},
            }

        monkeypatch.setattr(tool, "_consult_model", consult.__get__(tool))


def _args(models: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "step": PROPOSAL,
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": FINDINGS,
        "models": models,
        **extra,
    }


def _batched(model: str, stance: str = "neutral", **extra: Any) -> dict[str, Any]:
    return {"model": model, "stance": stance, "execution_mode": "batch", **extra}


def _running(blocks) -> dict[str, Any]:
    """The status payload of a non-terminal response (carried in a ToolOutput)."""

    outer = json.loads(blocks[0].text)
    assert outer["status"] == "success"
    body = json.loads(outer["content"])
    assert body["status"] == "running", body
    return body


def _final(blocks) -> dict[str, Any]:
    """The consensus response of a finished run (the same raw JSON a synchronous run returns)."""

    body = json.loads(blocks[0].text)
    assert body["status"] in ("consensus_workflow_complete", "all_providers_unavailable"), body
    return body


def _stored(world: Any, handle: str) -> dict[str, Any]:
    record = world.store.get(handle)
    assert record is not None
    return record.payload


# ----------------------------------------------------------------------
# 8.1 one submission per distinct batch-selected model
# ----------------------------------------------------------------------


class TestOneSubmissionPerDistinctModel:
    @pytest.mark.asyncio
    async def test_three_models_produce_three_submissions(self, boot, service):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against"), _batched(GPT5, "neutral")]

        body = _running(await world.consensus.execute(_args(models, wait_seconds=0)))

        assert len(service.posts) == 3
        assert sorted(service.submitted_models) == sorted(SLUG.values())
        for post in service.posts:
            assert len(post["envelope"]["requests"]) == 1
        assert body["batch_handle"].startswith("batch_")
        assert len({c.provider_batch_id for c in world.store.get(body["batch_handle"]).children}) == 3

    @pytest.mark.asyncio
    async def test_two_entries_for_one_model_with_different_stances_share_one_submission(self, boot, service):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "neutral"), _batched(OPUS, "against")]

        await world.consensus.execute(_args(models, wait_seconds=0))

        assert len(service.posts) == 2
        opus_post = next(p for p in service.posts if p["envelope"]["model"] == SLUG[OPUS])
        requests = opus_post["envelope"]["requests"]
        assert [r["custom_id"] for r in requests] == ["consensus-0", "consensus-2"]
        systems = [r["body"]["messages"][0]["content"] for r in requests]
        assert "SUPPORTIVE PERSPECTIVE" in systems[0] and "CRITICAL PERSPECTIVE" in systems[1]
        for r in requests:
            assert PROPOSAL in r["body"]["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_a_models_batch_is_the_providers_own_even_when_all_models_share_a_provider(self, boot, service):
        world = boot()
        models = [_batched(OPUS), _batched(SONNET), _batched("o3")]

        await world.consensus.execute(_args(models, wait_seconds=0))

        assert len(service.posts) == 3  # three models on one provider (OpenRouter): still three batches

    @pytest.mark.asyncio
    async def test_the_request_level_field_batches_every_entry_and_nothing_runs_synchronously(
        self, boot, service, monkeypatch
    ):
        world = boot()
        sync = SyncCalls(world.consensus, monkeypatch)
        models = [{"model": OPUS, "stance": "for"}, {"model": SONNET, "stance": "against"}]

        _running(await world.consensus.execute(_args(models, execution_mode="batch", wait_seconds=0)))

        assert len(service.posts) == 2 and sync.calls == []

    @pytest.mark.asyncio
    async def test_only_entries_that_selected_batch_are_batched(self, boot, service, monkeypatch):
        world = boot()
        sync = SyncCalls(world.consensus, monkeypatch)
        models = [_batched(OPUS, "for"), {"model": SONNET, "stance": "against"}, {"model": GPT5, "stance": "neutral"}]

        _running(await world.consensus.execute(_args(models, wait_seconds=0)))

        assert service.submitted_models == [SLUG[OPUS]]
        assert sorted(sync.calls) == sorted([SONNET, GPT5])

    @pytest.mark.asyncio
    async def test_a_whitespace_padded_identifier_shares_its_models_batch(self, boot, service):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(f"  {OPUS} ", "against")]

        await world.consensus.execute(_args(models, wait_seconds=0))

        assert len(service.posts) == 1 and len(service.posts[0]["envelope"]["requests"]) == 2


# ----------------------------------------------------------------------
# 8.2 / 8.8 the persisted payload
# ----------------------------------------------------------------------


class TestPersistedPayload:
    @pytest.mark.asyncio
    async def test_the_record_holds_the_proposal_findings_roster_and_accumulated_responses(
        self, boot, service, monkeypatch
    ):
        world = boot()
        SyncCalls(world.consensus, monkeypatch)
        models = [_batched(OPUS, "for"), {"model": SONNET, "stance": "against", "stance_prompt": "Be harsh."}]

        body = _running(await world.consensus.execute(_args(models, wait_seconds=0)))
        payload = _stored(world, body["batch_handle"])

        assert payload["original_proposal"] == PROPOSAL
        assert payload["findings"] == FINDINGS
        assert payload["models"] == [
            {"model": OPUS, "stance": "for", "execution_mode": "batch"},
            {"model": SONNET, "stance": "against", "stance_prompt": "Be harsh."},
        ]
        assert [r["model"] for r in payload["accumulated_responses"]] == [SONNET]  # the synchronous verdict
        assert payload["accumulated_responses"][0]["verdict"].startswith("sync verdict from sonnet")

    @pytest.mark.asyncio
    async def test_the_payload_is_exactly_those_four_fields_and_carries_no_work_history(
        self, boot, service, monkeypatch
    ):
        world = boot()
        SyncCalls(world.consensus, monkeypatch)
        world.consensus.work_history = [{"step": "an earlier step", "findings": "must never be persisted"}]
        models = [_batched(OPUS, "for"), _batched(SONNET, "against"), {"model": GPT5, "stance": "neutral"}]

        body = _running(await world.consensus.execute(_args(models, wait_seconds=0)))
        handle = body["batch_handle"]
        payload = _stored(world, handle)

        assert tuple(payload) == PAYLOAD_KEYS and set(payload) == {
            "original_proposal",
            "findings",
            "models",
            "accumulated_responses",
        }
        assert "work_history" not in json.dumps(payload) and "must never be persisted" not in json.dumps(payload)

        # ...and it stays those four fields as results are consumed, up to and including completion.
        for batch_id in list(service.states):
            service.finish(batch_id)
        _final(await world.consensus.execute({"batch_handle": handle}))
        assert tuple(_stored(world, handle)) == PAYLOAD_KEYS
        assert "work_history" not in json.dumps(_stored(world, handle))

    @pytest.mark.asyncio
    async def test_the_payload_does_not_grow_with_the_step_history_of_the_tool_instance(self, boot, service):
        world = boot()
        small = _running(await world.consensus.execute(_args([_batched(OPUS), _batched(SONNET)], wait_seconds=0)))
        baseline = len(json.dumps(_stored(world, small["batch_handle"])))

        world.consensus.work_history = [{"step": "x" * 50_000}]
        again = _running(await world.consensus.execute(_args([_batched(OPUS), _batched(SONNET)], wait_seconds=0)))

        assert len(json.dumps(_stored(world, again["batch_handle"]))) == baseline


# ----------------------------------------------------------------------
# 8.3 synthesis from the persisted record on a fresh instance (integration)
# ----------------------------------------------------------------------


class TestSynthesisFromTheDurableRecord:
    @pytest.mark.asyncio
    async def test_a_fresh_tool_instance_rebuilds_the_full_response_after_a_discarded_process(self, boot, service):
        first = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against"), _batched(OPUS, "neutral")]
        handle = _running(await first.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        first.close()
        del first  # the process is gone, taking the tool instance and every attribute on it
        opus_batch, sonnet_batch = service.batch_for_model(SLUG[OPUS]), service.batch_for_model(SLUG[SONNET])
        service.finish(opus_batch, {"consensus-0": "OPUS FOR: ship it", "consensus-2": "OPUS NEUTRAL: wait"})
        service.finish(sonnet_batch, {"consensus-1": "SONNET AGAINST: do not"})

        second = boot()  # new store, registry, adapter, executor and a ConsensusTool that never saw step 1
        assert second.consensus.original_proposal is None and second.consensus.initial_prompt is None
        assert second.consensus.accumulated_responses == [] and second.consensus.models_to_consult == []

        body = _final(await second.consensus.execute({"batch_handle": handle}))

        consensus = body["complete_consensus"]
        assert consensus["initial_prompt"] == PROPOSAL  # an empty one is what instance state would have given
        assert body["agent_analysis"] == {"initial_analysis": PROPOSAL, "findings": FINDINGS}
        assert [(r["model"], r["stance"], r["status"]) for r in body["accumulated_responses"]] == [
            (OPUS, "for", "success"),
            (SONNET, "against", "success"),
            (OPUS, "neutral", "success"),
        ]
        assert [r["verdict"] for r in body["accumulated_responses"]] == [
            "OPUS FOR: ship it",
            "SONNET AGAINST: do not",
            "OPUS NEUTRAL: wait",
        ]
        assert consensus["models_consulted"] == ["opus:for", "sonnet:against", "opus:neutral"]
        assert consensus["total_responses"] == 3 and consensus["consensus_confidence"] == "high"
        assert body["metadata"]["models_consulted"] == ["opus:for", "sonnet:against", "opus:neutral"]
        assert body["metadata"]["total_models"] == 3 and body["metadata"]["batch_handle"] == handle
        assert "Synthesize all perspectives" in body["next_steps"]
        assert len(service.posts) == 2  # nothing was resubmitted

    @pytest.mark.asyncio
    async def test_the_response_does_not_read_the_tool_instance(self, boot, service):
        """A different, hostile instance state must not leak into the rebuilt response."""

        world = boot()
        handle = _running(await world.consensus.execute(_args([_batched(OPUS), _batched(SONNET)], wait_seconds=0)))[
            "batch_handle"
        ]
        for batch_id in list(service.states):
            service.finish(batch_id)
        world.consensus.original_proposal = "STALE PROPOSAL FROM ANOTHER RUN"
        world.consensus.initial_prompt = "STALE PROMPT"
        world.consensus.models_to_consult = [{"model": "stale-model", "stance": "for"}]
        world.consensus.accumulated_responses = [{"model": "stale-model", "stance": "for", "status": "success"}]

        body = _final(await world.consensus.execute({"batch_handle": handle}))

        assert body["complete_consensus"]["initial_prompt"] == PROPOSAL
        assert "stale" not in json.dumps(body).lower()
        assert [r["model"] for r in body["accumulated_responses"]] == [OPUS, SONNET]

    @pytest.mark.asyncio
    async def test_a_run_that_finishes_inside_the_wait_is_answered_in_one_call(self, boot, service):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id)

        body = _final(await world.consensus.execute(_args([_batched(OPUS, "for"), _batched(SONNET, "against")])))

        assert body["status"] == "consensus_workflow_complete"
        assert [r["verdict"] for r in body["accumulated_responses"]] == ["verdict:consensus-0", "verdict:consensus-1"]
        assert body["complete_consensus"]["initial_prompt"] == PROPOSAL
        assert "continuation_offer" in body

    @pytest.mark.asyncio
    async def test_the_batched_response_has_the_structure_of_a_synchronous_one(self, boot, service, monkeypatch):
        """Same fields, from the same builder, for the same verdicts (8.9's unit-level companion)."""

        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id)
        batched = _final(await world.consensus.execute(_args([_batched(OPUS, "for"), _batched(SONNET, "against")])))

        sync_tool = ConsensusTool()
        SyncCalls(sync_tool, monkeypatch)
        monkeypatch.setattr(sync_tool, "get_model_provider", lambda name: None)
        synchronous = _final(
            await sync_tool.execute(_args([{"model": OPUS, "stance": "for"}, {"model": SONNET, "stance": "against"}]))
        )

        assert set(batched) - {"continuation_offer"} == set(synchronous) - {"continuation_offer"}
        assert set(batched["complete_consensus"]) == set(synchronous["complete_consensus"])
        assert set(batched["agent_analysis"]) == set(synchronous["agent_analysis"])
        assert batched["status"] == synchronous["status"] and batched["next_steps"] == synchronous["next_steps"]
        for left, right in zip(batched["accumulated_responses"], synchronous["accumulated_responses"], strict=True):
            assert set(left) == set(right)
            # the batched metadata is the synchronous one plus markers of how the model ran and was resolved
            assert set(right["metadata"]) <= set(left["metadata"])
            assert set(left["metadata"]) - set(right["metadata"]) <= {
                "execution_mode",
                "model_requested",
                "model_resolved",
            }


class TestThroughTheServerBoundary:
    @pytest.mark.asyncio
    async def test_a_batched_consensus_call_through_the_boundary_resumes_to_the_full_response(
        self, boot, service, monkeypatch
    ):
        from handlers.tool_handlers import build_handlers
        from server import tool_registry

        world = boot()
        consensus = tool_registry.get_tool_instance("consensus")
        monkeypatch.setattr(consensus, "_batch_executor", world.executor, raising=False)
        handlers = build_handlers(tool_registry)

        started = await handlers.handle_call_tool(
            "consensus", _args([_batched(OPUS, "for"), _batched(SONNET, "against")], wait_seconds=1)
        )
        handle = json.loads(json.loads(started[0].text)["content"])["batch_handle"]
        for batch_id in list(service.states):
            service.finish(batch_id)
        done = await handlers.handle_call_tool("consensus", {"batch_handle": handle})

        body = _final(done)
        assert body["complete_consensus"]["initial_prompt"] == PROPOSAL
        assert len(body["accumulated_responses"]) == 2 and len(service.posts) == 2


# ----------------------------------------------------------------------
# 8.4 mixed runs: finished consultations are durable before a handle is returned
# ----------------------------------------------------------------------


class TestMixedRuns:
    @pytest.mark.asyncio
    async def test_a_synchronous_verdict_is_persisted_before_the_handle_returns_and_not_rerun_on_resume(
        self, boot, service, monkeypatch
    ):
        first = boot()
        sync_one = SyncCalls(first.consensus, monkeypatch)
        seen_at_first_poll: list[list[str]] = []

        def at_first_poll(batch_id: str, world: Any = first) -> None:
            # The first provider poll happens only after the synchronous run: what is durable by then?
            handle = world.store.list_records()[0].handle
            seen_at_first_poll.append([r["model"] for r in _stored(world, handle)["accumulated_responses"]])

        service.on_get = at_first_poll
        models = [_batched(OPUS, "for"), {"model": SONNET, "stance": "against"}]

        body = _running(await first.consensus.execute(_args(models, wait_seconds=2)))
        handle = body["batch_handle"]

        assert sync_one.calls == [SONNET]
        assert seen_at_first_poll and seen_at_first_poll[0] == [SONNET]  # durable before anything waited
        assert [r["model"] for r in _stored(first, handle)["accumulated_responses"]] == [SONNET]
        assert body["consultations"]["completed_models"] == ["sonnet:against"]
        assert body["consultations"]["pending_models"] == ["opus:for"]

        first.close()
        del first, sync_one
        service.on_get = None
        service.finish(service.batch_for_model(SLUG[OPUS]), {"consensus-0": "OPUS: batched verdict"})

        second = boot()
        rerun = SyncCalls(second.consensus, monkeypatch)
        monkeypatch.setattr(
            second.consensus,
            "_consult_model",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("a completed consultation was executed again")),
        )
        final = _final(await second.consensus.execute({"batch_handle": handle}))

        assert rerun.calls == []
        verdicts = {r["model"]: r["verdict"] for r in final["accumulated_responses"]}
        assert verdicts[OPUS] == "OPUS: batched verdict"
        assert verdicts[SONNET].startswith("sync verdict from sonnet (")  # the one persisted before the restart
        assert [r["model"] for r in final["accumulated_responses"]] == [OPUS, SONNET]  # roster order, as synchronous
        assert len(service.posts) == 1  # nothing new was submitted

    @pytest.mark.asyncio
    async def test_a_mixed_run_waits_for_its_batches_only_after_the_synchronous_models_finish(
        self, boot, service, monkeypatch
    ):
        world = boot()
        order: list[str] = []
        SyncCalls(world.consensus, monkeypatch)
        original = world.consensus._consult_model

        async def tracked(config, request):
            order.append("sync")
            return await original(config, request)

        monkeypatch.setattr(world.consensus, "_consult_model", tracked)
        service.on_get = lambda batch_id: order.append("poll")

        _running(
            await world.consensus.execute(
                _args([_batched(OPUS), {"model": SONNET, "stance": "neutral"}], wait_seconds=2)
            )
        )

        assert service.posts and order[0] == "sync" and "poll" in order[1:]

    @pytest.mark.asyncio
    async def test_when_the_synchronous_verdicts_cannot_be_saved_no_handle_is_returned_and_they_are_not_lost(
        self, boot, service, monkeypatch
    ):
        world = boot()
        SyncCalls(world.consensus, monkeypatch)

        def refuse(handle: str, payload: Any) -> None:
            raise bs.BatchStoreError("disk is full")

        monkeypatch.setattr(world.store, "update_payload", refuse)

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.consensus.execute(_args([_batched(OPUS), {"model": SONNET, "stance": "neutral"}]))

        payload = json.loads(excinfo.value.payload)
        assert payload["metadata"]["batch_error"] == "persist_failed"
        assert "sync verdict from sonnet" in payload["content"]  # the finished verdict is in the error, not lost
        assert "no handle is returned" in payload["content"]

    @pytest.mark.asyncio
    async def test_a_batch_that_already_finished_is_persisted_before_a_running_handle_is_returned(self, boot, service):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against")]
        handle = _running(await world.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        service.finish(service.batch_for_model(SLUG[OPUS]), {"consensus-0": "OPUS finished first"})

        body = _running(await world.consensus.execute({"batch_handle": handle, "wait_seconds": 0}))

        stored = _stored(world, handle)["accumulated_responses"]
        assert [(r["model"], r["verdict"]) for r in stored] == [(OPUS, "OPUS finished first")]
        assert body["consultations"]["completed"] == 1 and body["consultations"]["pending_models"] == ["sonnet:against"]

    @pytest.mark.asyncio
    async def test_a_restart_between_completion_and_synthesis_loses_nothing(self, boot, service, monkeypatch):
        first = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against")]
        handle = _running(await first.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        for batch_id in list(service.states):
            service.finish(batch_id)

        def crash(self_, result):
            raise RuntimeError("the server died after recording the results, before synthesizing them")

        monkeypatch.setattr(ConsensusTool, "render_completed_batch", crash)
        with pytest.raises(Exception, match="died"):
            await first.consensus.execute({"batch_handle": handle})
        monkeypatch.undo()
        first.close()
        del first
        service.states.clear()  # the provider no longer answers for these batches at all
        service.on_get = lambda batch_id: (_ for _ in ()).throw(AssertionError("the provider was contacted"))

        gets_before = list(service.gets)
        second = boot()
        body = _final(await second.consensus.execute({"batch_handle": handle}))

        assert [r["verdict"] for r in body["accumulated_responses"]] == ["verdict:consensus-0", "verdict:consensus-1"]
        assert body["complete_consensus"]["initial_prompt"] == PROPOSAL
        assert service.gets == gets_before  # not one request reached the provider: the record alone sufficed

    @pytest.mark.asyncio
    async def test_a_synchronous_consultation_the_server_never_finished_is_reported_not_rerun(
        self, boot, service, monkeypatch
    ):
        """Resuming never submits work, so a sync entry with no recorded result becomes an error entry."""

        world = boot()
        monkeypatch.setattr(
            world.consensus,
            "_consult_models_concurrently",
            lambda configs, request: (_ for _ in ()).throw(RuntimeError("the server died while it ran")),
        )
        models = [_batched(OPUS, "for"), {"model": SONNET, "stance": "against"}]
        with pytest.raises(RuntimeError, match="died"):
            await world.consensus.execute(_args(models, wait_seconds=0))
        record = world.store.list_records()[0]
        service.finish(service.batch_for_model(SLUG[OPUS]), {"consensus-0": "OPUS ok"})

        body = _final(await world.consensus.execute({"batch_handle": record.handle}))

        by_model = {r["model"]: r for r in body["accumulated_responses"]}
        assert by_model[OPUS]["status"] == "success"
        assert by_model[SONNET]["status"] == "error" and "not re-run" in by_model[SONNET]["error"]
        assert len(service.posts) == 1


# ----------------------------------------------------------------------
# 8.5 several batches under one handle
# ----------------------------------------------------------------------


class TestSeveralBatchesOneHandle:
    @pytest.mark.asyncio
    async def test_the_run_reports_running_while_any_one_of_three_child_batches_is_processing(self, boot, service):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against"), _batched(GPT5, "neutral")]
        handle = _running(await world.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        service.finish(service.batch_for_model(SLUG[OPUS]), {"consensus-0": "a"})
        service.finish(service.batch_for_model(SLUG[SONNET]), {"consensus-1": "b"})
        # the third batch is still in progress at the provider

        body = _running(await world.consensus.execute({"batch_handle": handle, "wait_seconds": 1}))

        assert body["batch_handle"] == handle
        assert body["consultations"]["completed"] == 2 and body["consultations"]["pending_models"] == ["gpt5:neutral"]
        states = {c["model"]: c["state"] for c in body["children"]}
        assert states == {OPUS: "completed", SONNET: "completed", GPT5: "in_progress"}
        assert "NOT finished" in body["message"]

        service.finish(service.batch_for_model(SLUG[GPT5]), {"consensus-2": "c"})
        final = _final(await world.consensus.execute({"batch_handle": handle}))
        assert [r["verdict"] for r in final["accumulated_responses"]] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_a_resumption_polls_only_the_batches_still_outstanding_and_fetches_each_result_once(
        self, boot, service
    ):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against")]
        handle = _running(await world.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        opus_batch, sonnet_batch = service.batch_for_model(SLUG[OPUS]), service.batch_for_model(SLUG[SONNET])
        service.finish(opus_batch, {"consensus-0": "a"})
        _running(await world.consensus.execute({"batch_handle": handle, "wait_seconds": 1}))
        opus_gets_after_consumption = service.gets.count(opus_batch)

        for _ in range(3):
            _running(await world.consensus.execute({"batch_handle": handle, "wait_seconds": 1}))

        assert service.gets.count(opus_batch) == opus_gets_after_consumption  # a recorded verdict is never refetched
        assert service.gets.count(sonnet_batch) > 3  # the outstanding batch keeps being polled

    @pytest.mark.asyncio
    async def test_repeated_resumption_of_a_finished_run_returns_the_same_response_without_contacting_the_provider(
        self, boot, service
    ):
        world = boot()
        handle = _running(await world.consensus.execute(_args([_batched(OPUS), _batched(SONNET)], wait_seconds=0)))[
            "batch_handle"
        ]
        for batch_id in list(service.states):
            service.finish(batch_id)
        first = _final(await world.consensus.execute({"batch_handle": handle}))
        service.on_get = lambda batch_id: (_ for _ in ()).throw(AssertionError("provider contacted again"))

        again = _final(await world.consensus.execute({"batch_handle": handle}))

        assert again == first and len(service.posts) == 2


# ----------------------------------------------------------------------
# 8.6 per-model error isolation across batch failures
# ----------------------------------------------------------------------


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_one_failed_request_in_a_batch_leaves_its_siblings_intact(self, boot, service):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(OPUS, "against"), _batched(SONNET, "neutral")]
        handle = _running(await world.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        service.finish(
            service.batch_for_model(SLUG[OPUS]),
            {"consensus-0": "OPUS FOR fine", "consensus-1": ("invalid_request_error", "the prompt was refused")},
        )
        service.finish(service.batch_for_model(SLUG[SONNET]), {"consensus-2": "SONNET fine"})

        body = _final(await world.consensus.execute({"batch_handle": handle}))

        by_stance = {(r["model"], r["stance"]): r for r in body["accumulated_responses"]}
        assert by_stance[(OPUS, "for")]["status"] == "success" and by_stance[(SONNET, "neutral")]["status"] == "success"
        failed = by_stance[(OPUS, "against")]
        assert failed["status"] == "error" and "the prompt was refused" in failed["error"]
        assert body["status"] == "consensus_workflow_complete"
        assert body["complete_consensus"]["consensus_confidence"] == "reduced"
        assert body["complete_consensus"]["total_responses"] == 2
        assert "Synthesize from the available results only" in body["next_steps"]
        assert "opus:against" in body["next_steps"]

    @pytest.mark.asyncio
    async def test_a_batch_that_fails_as_a_whole_leaves_the_other_models_results(self, boot, service):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against")]
        handle = _running(await world.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        service.fail(service.batch_for_model(SLUG[OPUS]), "the provider could not process this batch")
        service.finish(service.batch_for_model(SLUG[SONNET]), {"consensus-1": "SONNET survived"})

        body = _final(await world.consensus.execute({"batch_handle": handle}))

        by_model = {r["model"]: r for r in body["accumulated_responses"]}
        assert by_model[SONNET]["status"] == "success" and by_model[SONNET]["verdict"] == "SONNET survived"
        assert by_model[OPUS]["status"] == "error" and by_model[OPUS]["error"]
        assert "or_batch_" not in json.dumps(body)  # a provider batch id never reaches the caller
        assert body["status"] == "consensus_workflow_complete"
        assert "Synthesize from the available results only" in body["next_steps"]

    @pytest.mark.asyncio
    async def test_when_every_batch_fails_the_response_has_the_all_failed_shape_a_synchronous_run_has(
        self, boot, service
    ):
        world = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against")]
        handle = _running(await world.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        for batch_id in list(service.states):
            service.fail(batch_id, "nothing worked")

        body = _final(await world.consensus.execute({"batch_handle": handle}))

        assert body["status"] == "all_providers_unavailable"
        assert [r["status"] for r in body["accumulated_responses"]] == ["error", "error"]
        assert body["consensus_complete"] is True

    @pytest.mark.asyncio
    async def test_a_failed_synchronous_model_does_not_stop_the_batched_ones(self, boot, service, monkeypatch):
        world = boot()
        SyncCalls(world.consensus, monkeypatch, fail={SONNET})
        service.on_get = lambda batch_id: service.finish(batch_id)

        body = _final(
            await world.consensus.execute(_args([_batched(OPUS, "for"), {"model": SONNET, "stance": "against"}]))
        )

        by_model = {r["model"]: r for r in body["accumulated_responses"]}
        assert by_model[OPUS]["status"] == "success" and by_model[SONNET]["status"] == "error"


# ----------------------------------------------------------------------
# 8.7 over-budget: a resumable response, nothing cancelled
# ----------------------------------------------------------------------


class TestOverBudgetWait:
    @pytest.mark.asyncio
    async def test_the_bounded_wait_returns_a_handle_and_never_cancels_the_submitted_batches(
        self, boot, service, monkeypatch
    ):
        world = boot()
        cancelled: list[str] = []
        monkeypatch.setattr(OpenRouterProvider, "cancel_batch", lambda self, batch_id: cancelled.append(batch_id))
        models = [_batched(OPUS, "for"), _batched(SONNET, "against")]

        body = _running(await world.consensus.execute(_args(models, wait_seconds=3)))

        assert body["batch_handle"].startswith("batch_") and body["resume_with"] == {
            "batch_handle": body["batch_handle"]
        }
        assert "NOT finished" in body["message"] and "cancel=true" in body["message"]
        assert cancelled == []
        record = world.store.get(body["batch_handle"])
        assert record.state.value == "running" and all(c.status.value != "cancelled" for c in record.children)
        assert len(service.posts) == 2  # both are still out there, and still the only submissions

        # waiting longer does not cancel either, and the work can still be collected
        _running(await world.consensus.execute({"batch_handle": body["batch_handle"], "wait_seconds": 5}))
        assert cancelled == []
        for batch_id in list(service.states):
            service.finish(batch_id)
        _final(await world.consensus.execute({"batch_handle": body["batch_handle"]}))

    @pytest.mark.asyncio
    async def test_an_explicit_cancel_is_local_and_says_the_provider_batches_continue(self, boot, service):
        world = boot()
        handle = _running(await world.consensus.execute(_args([_batched(OPUS), _batched(SONNET)], wait_seconds=0)))[
            "batch_handle"
        ]

        outer = json.loads((await world.consensus.execute({"batch_handle": handle, "cancel": True}))[0].text)
        body = json.loads(outer["content"])

        assert body["status"] == "cancelled" and body["provider_batches_continue"] is True
        assert "still be billed" in body["message"]


# ----------------------------------------------------------------------
# Admission and start-work rules specific to consensus
# ----------------------------------------------------------------------


class TestStartRules:
    @pytest.mark.asyncio
    async def test_a_batched_entry_for_a_model_that_cannot_batch_is_rejected_before_any_submission_or_sync_work(
        self, boot, service, monkeypatch
    ):
        world = boot()
        sync = SyncCalls(world.consensus, monkeypatch)
        models = [_batched(OPUS), _batched("mistral"), {"model": SONNET, "stance": "neutral"}]

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.consensus.execute(_args(models))

        assert "mistral" in json.loads(excinfo.value.payload)["content"]
        assert service.posts == [] and sync.calls == []

    @pytest.mark.asyncio
    async def test_a_refused_submission_runs_no_synchronous_consultation_and_returns_no_handle(
        self, boot, service, monkeypatch
    ):
        world = boot()
        sync = SyncCalls(world.consensus, monkeypatch)
        service.post_error = httpx.Response(402, json={"error": {"message": "insufficient credits"}})

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.consensus.execute(_args([_batched(OPUS), {"model": SONNET, "stance": "neutral"}]))

        assert "insufficient credits" in json.loads(excinfo.value.payload)["content"]
        assert sync.calls == [] and world.store.list_records() == []

    @pytest.mark.asyncio
    async def test_batch_applies_to_step_one_only(self, boot, service):
        world = boot()
        arguments = _args([_batched(OPUS), _batched(SONNET)], step_number=2, total_steps=2, next_step_required=True)

        with pytest.raises(ToolExecutionError) as excinfo:
            await world.consensus.execute(arguments)

        assert "step 1" in json.loads(excinfo.value.payload)["content"] and service.posts == []

    @pytest.mark.asyncio
    async def test_context_files_are_embedded_as_text_in_every_batched_request(self, boot, service, tmp_path):
        world = boot()
        notes = tmp_path / "design.md"
        notes.write_text("DESIGN DOC: the billing ledger is append-only")

        await world.consensus.execute(
            _args([_batched(OPUS, "for"), _batched(SONNET, "against")], relevant_files=[str(notes)], wait_seconds=0)
        )

        for post in service.posts:
            prompt = post["envelope"]["requests"][0]["body"]["messages"][-1]["content"]
            assert "=== CONTEXT FILES ===" in prompt and "the billing ledger is append-only" in prompt

    @pytest.mark.asyncio
    async def test_the_stance_comes_only_from_the_entrys_stance_field(self, boot, service):
        world = boot()
        models = [_batched("opus", "against"), _batched("sonnet", "for")]

        await world.consensus.execute(_args(models, wait_seconds=0))

        opus = next(p for p in service.posts if p["envelope"]["model"] == SLUG[OPUS])
        sonnet = next(p for p in service.posts if p["envelope"]["model"] == SLUG[SONNET])
        assert "CRITICAL PERSPECTIVE" in opus["envelope"]["requests"][0]["body"]["messages"][0]["content"]
        assert "SUPPORTIVE PERSPECTIVE" in sonnet["envelope"]["requests"][0]["body"]["messages"][0]["content"]


# ----------------------------------------------------------------------
# Review fixes: a stance that is null and "neutral" is one stance
# ----------------------------------------------------------------------


class TestNullAndNeutralStanceAreTheSameStance:
    @pytest.mark.parametrize("empty", [None, ""])
    def test_a_roster_naming_one_model_with_a_null_and_a_neutral_stance_is_a_duplicate(self, empty):
        from tools.consensus import ConsensusRequest

        with pytest.raises(ValueError, match="Duplicate model \\+ stance"):
            ConsensusRequest(**_args([{"model": OPUS, "stance": empty}, {"model": OPUS, "stance": "neutral"}]))
        with pytest.raises(ValueError, match="Duplicate model \\+ stance"):  # and a missing stance is neutral too
            ConsensusRequest(**_args([{"model": OPUS}, {"model": OPUS, "stance": empty}]))

    @pytest.mark.asyncio
    async def test_a_batched_run_never_collapses_two_paid_consultations_into_one(self, boot, service):
        """The reviewer's case: before the fix the second verdict was replaced by a copy of the first."""

        world = boot()
        with pytest.raises(ValueError, match="Duplicate model"):
            await world.consensus.execute(
                _args([_batched(OPUS, None), _batched(OPUS, "neutral")], wait_seconds=0)  # type: ignore[arg-type]
            )
        assert service.posts == [] and world.store.list_records() == []

    @pytest.mark.asyncio
    async def test_an_entry_without_a_stance_and_one_with_a_real_stance_each_keep_their_own_verdict(
        self, boot, service
    ):
        world = boot()
        models = [{"model": OPUS, "execution_mode": "batch"}, _batched(OPUS, "for")]
        handle = _running(await world.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        service.finish(service.batch_for_model(SLUG[OPUS]), {"consensus-0": "A no stance", "consensus-1": "B for"})

        body = _final(await world.consensus.execute({"batch_handle": handle}))

        assert [(r["stance"], r["verdict"]) for r in body["accumulated_responses"]] == [
            ("neutral", "A no stance"),
            ("for", "B for"),
        ]
        assert body["complete_consensus"]["total_responses"] == 2


# ----------------------------------------------------------------------
# Review fixes: money safety of a mixed run
# ----------------------------------------------------------------------


class GatedSync:
    """A synchronous consultation that finishes only when the test says so."""

    def __init__(self, tool: ConsensusTool, monkeypatch: pytest.MonkeyPatch) -> None:
        self.gate = asyncio.Event()
        self.started = asyncio.Event()
        self.calls: list[str] = []

        async def consult(config: dict, request: Any) -> dict:
            self.calls.append(config["model"])
            self.started.set()
            await self.gate.wait()
            return {
                "model": config["model"],
                "stance": config.get("stance", "neutral"),
                "status": "success",
                "verdict": f"SYNC verdict from {config['model']}",
                "metadata": {"provider": "test", "model_name": config["model"]},
            }

        monkeypatch.setattr(tool, "_consult_model", consult)


MIXED = [_batched(OPUS, "for"), {"model": SONNET, "stance": "against"}]


class TestMixedRunSurvivesACancelledCall:
    @pytest.mark.asyncio
    async def test_a_call_cancelled_during_the_synchronous_phase_loses_no_verdict_and_leaves_a_findable_handle(
        self, boot, service, monkeypatch, caplog
    ):
        world = boot()
        sync = GatedSync(world.consensus, monkeypatch)
        caplog.set_level(logging.INFO, logger="tools.consensus")

        call = asyncio.create_task(world.consensus.execute(_args(MIXED, wait_seconds=30)))
        await asyncio.wait_for(sync.started.wait(), 5)
        (record,) = world.store.list_records()
        handle = record.handle
        assert len(service.posts) == 1  # the batch is submitted and billed while the synchronous model runs

        call.cancel()  # the host gives up on the call
        with pytest.raises(asyncio.CancelledError):
            await call

        # The handle was logged at submission, so the paid batch is not stranded unfindable.
        assert any(handle in message and "resume" in message for message in caplog.messages)

        # The synchronous consultation was not abandoned with the call: it finishes and records its verdict.
        sync.gate.set()
        await asyncio.wait_for(ConsensusTool._live_synchronous[handle], 5)
        assert [r["verdict"] for r in _stored(world, handle)["accumulated_responses"]] == ["SYNC verdict from sonnet"]

        # And the run is resumable with nothing paid twice.
        service.finish(service.batch_for_model(SLUG[OPUS]), {"consensus-0": "OPUS batched"})
        final = _final(await world.consensus.execute({"batch_handle": handle}))
        assert {r["model"]: r["verdict"] for r in final["accumulated_responses"]} == {
            OPUS: "OPUS batched",
            SONNET: "SYNC verdict from sonnet",
        }
        assert sync.calls == [SONNET] and len(service.posts) == 1


class TestMixedRunReturnsItsHandleWithinTheWaitBudget:
    @pytest.mark.asyncio
    async def test_slow_synchronous_models_do_not_hold_the_call_past_its_wait(self, boot, service, monkeypatch):
        world = boot()
        sync = GatedSync(world.consensus, monkeypatch)

        loop = asyncio.get_running_loop()
        started = loop.time()
        body = _running(await world.consensus.execute(_args(MIXED, wait_seconds=0.3)))

        assert loop.time() - started < 5  # the synchronous model has not finished, and the call did not wait for it
        assert not sync.gate.is_set()
        assert body["consultations"]["pending_models"] == ["opus:for", "sonnet:against"]
        assert body["batch_handle"] in ConsensusTool._live_synchronous

        sync.gate.set()
        await asyncio.wait_for(ConsensusTool._live_synchronous[body["batch_handle"]], 5)
        again = _running(await world.consensus.execute({"batch_handle": body["batch_handle"], "wait_seconds": 0}))
        assert again["consultations"]["completed_models"] == ["sonnet:against"]

    @pytest.mark.asyncio
    async def test_a_resume_while_a_synchronous_model_still_runs_reports_running_not_an_interrupted_error(
        self, boot, service, monkeypatch
    ):
        world = boot()
        sync = GatedSync(world.consensus, monkeypatch)
        body = _running(await world.consensus.execute(_args(MIXED, wait_seconds=0.2)))
        handle = body["batch_handle"]
        service.finish(service.batch_for_model(SLUG[OPUS]), {"consensus-0": "OPUS batched"})

        # Every batch is done and recorded, but sonnet is still being consulted in this process.
        still = _running(await world.consensus.execute({"batch_handle": handle, "wait_seconds": 0}))
        assert still["consultations"]["pending_models"] == ["sonnet:against"]

        sync.gate.set()
        await asyncio.wait_for(ConsensusTool._live_synchronous[handle], 5)
        final = _final(await world.consensus.execute({"batch_handle": handle}))
        by_model = {r["model"]: r for r in final["accumulated_responses"]}
        assert by_model[SONNET]["status"] == "success" and by_model[SONNET]["verdict"] == "SYNC verdict from sonnet"
        assert sync.calls == [SONNET]

    @pytest.mark.asyncio
    async def test_when_a_detached_synchronous_task_cannot_save_its_verdicts_they_are_logged_not_lost(
        self, boot, service, monkeypatch, caplog
    ):
        world = boot()
        sync = GatedSync(world.consensus, monkeypatch)
        body = _running(await world.consensus.execute(_args(MIXED, wait_seconds=0.2)))
        handle = body["batch_handle"]

        def refuse(handle_: str, payload: Any) -> None:
            raise bs.BatchStoreError("disk is full")

        monkeypatch.setattr(world.store, "update_payload", refuse)
        caplog.set_level(logging.ERROR, logger="tools.consensus")
        sync.gate.set()
        await asyncio.wait_for(ConsensusTool._live_synchronous[handle], 5)

        assert any("SYNC verdict from sonnet" in message and handle in message for message in caplog.messages)


# ----------------------------------------------------------------------
# Review fixes: a cancelled run still hands over the verdicts it paid for
# ----------------------------------------------------------------------


class TestCancelledRunSurfacesPaidVerdicts:
    @pytest.mark.asyncio
    async def test_cancel_and_every_later_resume_return_the_synchronous_verdicts_already_recorded(
        self, boot, service, monkeypatch
    ):
        world = boot()
        SyncCalls(world.consensus, monkeypatch)
        handle = _running(await world.consensus.execute(_args(MIXED, wait_seconds=2)))["batch_handle"]

        cancelled = json.loads(
            json.loads((await world.consensus.execute({"batch_handle": handle, "cancel": True}))[0].text)["content"]
        )
        resumed = json.loads(
            json.loads((await world.consensus.execute({"batch_handle": handle, "wait_seconds": 0}))[0].text)["content"]
        )

        for body in (cancelled, resumed):
            assert body["status"] == "cancelled"
            (kept,) = body["completed_responses"]
            assert kept["model"] == SONNET and kept["status"] == "success"
            assert kept["verdict"].startswith("sync verdict from sonnet")
            assert body["consultations"]["completed_models"] == ["sonnet:against"]

    @pytest.mark.asyncio
    async def test_a_cancelled_run_with_nothing_recorded_carries_no_verdict_list(self, boot, service):
        world = boot()
        handle = _running(await world.consensus.execute(_args([_batched(OPUS), _batched(SONNET)], wait_seconds=0)))[
            "batch_handle"
        ]

        body = json.loads(
            json.loads((await world.consensus.execute({"batch_handle": handle, "cancel": True}))[0].text)["content"]
        )

        assert body["status"] == "cancelled" and "completed_responses" not in body


class TestCancelSaysWhatItDoesNotStop:
    @pytest.mark.asyncio
    async def test_cancelling_while_a_synchronous_model_still_runs_says_it_keeps_running_and_is_billed(
        self, boot, service, monkeypatch
    ):
        world = boot()
        sync = GatedSync(world.consensus, monkeypatch)
        body = _running(await world.consensus.execute(_args(MIXED, wait_seconds=0.2)))
        handle = body["batch_handle"]
        assert not sync.gate.is_set()

        cancelled = json.loads(
            json.loads((await world.consensus.execute({"batch_handle": handle, "cancel": True}))[0].text)["content"]
        )

        assert cancelled["status"] == "cancelled"
        assert cancelled["synchronous_consultations_continue"] == ["sonnet:against"]
        assert "NOT stopped by this cancel" in cancelled["message"] and "billed in full" in cancelled["message"]
        assert "message_note" not in cancelled  # the note is folded into the message, not leaked as a field

        sync.gate.set()
        await asyncio.wait_for(ConsensusTool._live_synchronous[handle], 5)
        later = json.loads(
            json.loads((await world.consensus.execute({"batch_handle": handle, "wait_seconds": 0}))[0].text)["content"]
        )
        assert "synchronous_consultations_continue" not in later  # once it has finished there is nothing to warn of

    @pytest.mark.asyncio
    async def test_a_cancel_with_no_synchronous_work_in_flight_carries_no_such_note(self, boot, service):
        world = boot()
        handle = _running(await world.consensus.execute(_args([_batched(OPUS), _batched(SONNET)], wait_seconds=0)))[
            "batch_handle"
        ]
        body = json.loads(
            json.loads((await world.consensus.execute({"batch_handle": handle, "cancel": True}))[0].text)["content"]
        )
        assert "synchronous_consultations_continue" not in body and "NOT stopped by this cancel" not in body["message"]


# ----------------------------------------------------------------------
# Review fixes: finished results are fetched under one shared budget
# ----------------------------------------------------------------------


class TestSettleFetchBudget:
    @pytest.mark.asyncio
    async def test_finished_children_share_one_fetch_budget_and_the_rest_wait_for_the_next_resume(
        self, boot, service, monkeypatch
    ):
        import tools.consensus as consensus_module

        world = boot()
        models = [_batched(OPUS, "for"), _batched(SONNET, "against"), _batched(GPT5, "neutral")]
        handle = _running(await world.consensus.execute(_args(models, wait_seconds=0)))["batch_handle"]
        for batch_id in list(service.states):
            service.finish(batch_id)

        now = [1000.0]
        monkeypatch.setattr(consensus_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
        budgets: list[float] = []
        real_fetch = world.executor.fetch_child

        def slow_fetch(child, *, budget_seconds=10.0):
            budgets.append(budget_seconds)
            now[0] += 3.0  # each finished child takes three seconds to fetch
            return real_fetch(child, budget_seconds=budget_seconds)

        monkeypatch.setattr(world.executor, "fetch_child", slow_fetch)

        first = _running(await world.consensus.execute({"batch_handle": handle, "wait_seconds": 0}))

        # one shared 5s budget (the floor): child 1 gets all of it, child 2 what is left, child 3 is not attempted
        assert budgets == [5.0, 2.0]
        assert first["consultations"]["completed"] == 2 and len(first["consultations"]["pending_models"]) == 1
        assert any("not fetched in this call" in (c.get("note") or "") for c in first["children"])

        final = _final(await world.consensus.execute({"batch_handle": handle, "wait_seconds": 0}))
        assert len(final["accumulated_responses"]) == 3
        assert len(budgets) == 3  # the next resume fetched only the one child still unrecorded


# ----------------------------------------------------------------------
# Review fixes: what the batched request carries, and how often results are fetched
# ----------------------------------------------------------------------


class TestBatchedRequestFidelity:
    @pytest.mark.asyncio
    async def test_a_custom_stance_prompt_reaches_the_batched_system_prompt_as_it_does_synchronously(
        self, boot, service
    ):
        world = boot()
        models = [_batched(OPUS, "for", stance_prompt="CUSTOM STANCE: argue as a tax auditor"), _batched(SONNET, "for")]

        await world.consensus.execute(_args(models, wait_seconds=0))

        opus = next(p for p in service.posts if p["envelope"]["model"] == SLUG[OPUS])
        sonnet = next(p for p in service.posts if p["envelope"]["model"] == SLUG[SONNET])
        system = opus["envelope"]["requests"][0]["body"]["messages"][0]["content"]
        assert system == world.consensus._get_stance_enhanced_prompt("for", "CUSTOM STANCE: argue as a tax auditor")
        assert "CUSTOM STANCE: argue as a tax auditor" in system and "SUPPORTIVE PERSPECTIVE" not in system
        # the sibling without a custom prompt still gets the built-in one
        assert "SUPPORTIVE PERSPECTIVE" in sonnet["envelope"]["requests"][0]["body"]["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_the_temperature_is_the_models_validated_temperature(self, boot, service, monkeypatch):
        # Section 15 (D8): ``opus``/``sonnet`` now resolve to models whose batch listings take no temperature, so the
        # entries that must carry one name the curated pins by full slug.
        opus, sonnet = "anthropic/claude-opus-4.5", "anthropic/claude-sonnet-4.5"
        world = boot()
        await world.consensus.execute(_args([_batched(opus, "for"), _batched(sonnet, "against")], wait_seconds=0))
        for post in service.posts:  # unpatched: the tool's default, as _consult_model sends it
            assert post["envelope"]["requests"][0]["body"]["temperature"] == world.consensus.get_default_temperature()

        seen: list[Any] = []

        def validated(requested: float, model_context: Any) -> tuple[float, list[str]]:
            seen.append((requested, model_context.model_name))
            return 0.61, []

        monkeypatch.setattr(world.consensus, "validate_and_correct_temperature", validated)
        before = len(service.posts)
        await world.consensus.execute(_args([_batched(opus, "for"), _batched(sonnet, "against")], wait_seconds=0))

        assert sorted(name for _, name in seen) == [opus, sonnet]  # validated once per model, against that model
        assert [post["envelope"]["requests"][0]["body"]["temperature"] for post in service.posts[before:]] == [
            0.61,
            0.61,
        ]


class TestEachFinishedBatchIsFetchedOnce:
    @pytest.mark.asyncio
    async def test_a_run_that_finishes_inside_the_start_call_fetches_each_child_exactly_once(self, boot, service):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id)
        fetched: list[str] = []
        real_fetch = world.provider.fetch_batch

        def counting(batch_id: str, **kwargs: Any) -> Any:
            fetched.append(batch_id)
            return real_fetch(batch_id, **kwargs)

        world.provider.fetch_batch = counting

        _final(await world.consensus.execute(_args([_batched(OPUS, "for"), _batched(SONNET, "against")])))

        assert sorted(fetched) == sorted(service.states)  # one fetch per provider batch, not two
        assert len(fetched) == 2


# ----------------------------------------------------------------------
# Review fixes: the response differs from a synchronous run only in documented keys
# ----------------------------------------------------------------------


def _key_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= _key_paths(item, path)
    elif isinstance(value, list):
        for item in value:
            paths |= _key_paths(item, f"{prefix}[]")
    return paths


class TestResponseStructureParity:
    @pytest.mark.asyncio
    async def test_every_nested_key_matches_a_synchronous_run_except_the_documented_batch_markers(
        self, boot, service, monkeypatch
    ):
        world = boot()
        service.on_get = lambda batch_id: service.finish(batch_id)
        batched = _final(await world.consensus.execute(_args([_batched(OPUS, "for"), _batched(SONNET, "against")])))

        sync_tool = ConsensusTool()
        SyncCalls(sync_tool, monkeypatch)
        monkeypatch.setattr(sync_tool, "get_model_provider", lambda name: None)
        synchronous = _final(
            await sync_tool.execute(_args([{"model": OPUS, "stance": "for"}, {"model": SONNET, "stance": "against"}]))
        )

        batch_paths, sync_paths = _key_paths(batched), _key_paths(synchronous)
        assert sync_paths - batch_paths == set()  # nothing a synchronous caller relies on is missing
        assert batch_paths - sync_paths == {  # the complete list of what batching adds
            "metadata.batch_handle",
            "metadata.batch_execution",
            "accumulated_responses[].metadata.execution_mode",
            "accumulated_responses[].metadata.model_requested",
            "accumulated_responses[].metadata.model_resolved",
        }
