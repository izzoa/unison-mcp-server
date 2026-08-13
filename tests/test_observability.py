"""Tests for utils/observability.py and utils/json_log_formatter.py.

Everything runs WITHOUT the OpenTelemetry packages: the default path must be
no-op, and recording/aggregation logic is exercised through the no-op
instruments plus the ContextVar plumbing, which is real either way.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from utils import observability as obs
from utils.json_log_formatter import JsonLogFormatter
from utils.logging_setup import LocalTimeFormatter, RedactingFilter, redact_text

SECRET = "sk-abcdefghijklmnop1234567890"  # matches the OpenAI-style pattern


# ---------------------------------------------------------------------------
# Redaction helper (shared surface)
# ---------------------------------------------------------------------------


class TestRedactText:
    def test_redacts_credential_shapes(self):
        assert "***REDACTED***" in redact_text(f"key is {SECRET}")
        assert SECRET not in redact_text(f"key is {SECRET}")

    def test_plain_text_untouched(self):
        assert redact_text("hello world") == "hello world"

    def test_filter_uses_shared_helper(self):
        record = logging.LogRecord("t", logging.INFO, "f", 1, f"leak {SECRET}", (), None)
        RedactingFilter().filter(record)
        assert SECRET not in record.getMessage()


# ---------------------------------------------------------------------------
# Text-mode exception redaction (the pre-existing leak, now fixed)
# ---------------------------------------------------------------------------


class TestTextModeExceptionRedaction:
    def test_exception_text_redacted_in_text_mode(self):
        formatter = LocalTimeFormatter("%(message)s")
        try:
            raise RuntimeError(f"boom with {SECRET}")
        except RuntimeError:
            import sys

            record = logging.LogRecord("t", logging.ERROR, "f", 1, "failed", (), sys.exc_info())
        out = formatter.format(record)
        assert SECRET not in out
        assert "***REDACTED***" in out


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


def _record(msg: str, *, extra: dict[str, Any] | None = None, exc: bool = False) -> logging.LogRecord:
    exc_info = None
    if exc:
        try:
            raise ValueError(f"bad {SECRET}")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
    record = logging.LogRecord("mcp_activity", logging.INFO, "f", 1, msg, (), exc_info)
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


class TestJsonLogFormatter:
    def test_required_fields_and_single_line(self):
        out = JsonLogFormatter().format(_record("hello"))
        assert "\n" not in out
        payload = json.loads(out)
        for key in ("timestamp", "level", "logger", "message", "schema_version"):
            assert key in payload
        assert payload["schema_version"] == "1.0"

    def test_tool_fields_lifted_and_absent_fields_omitted(self):
        out = json.loads(JsonLogFormatter().format(_record("x", extra={"tool_name": "chat", "latency_ms": 5})))
        assert out["tool_name"] == "chat"
        assert out["latency_ms"] == 5
        assert "model" not in out

    def test_unknown_extras_preserved_under_extra(self):
        out = json.loads(JsonLogFormatter().format(_record("x", extra={"custom_field": "v"})))
        assert out["extra"]["custom_field"] == "v"

    def test_message_redacted(self):
        out = JsonLogFormatter().format(_record(f"leak {SECRET}"))
        assert SECRET not in out

    def test_extra_fields_cannot_bypass_redaction(self):
        out = JsonLogFormatter().format(_record("x", extra={"api_response": f"token {SECRET}"}))
        assert SECRET not in out
        assert "***REDACTED***" in out

    def test_nested_extra_structures_redacted(self):
        out = JsonLogFormatter().format(_record("x", extra={"payload": {"inner": [f"k {SECRET}"]}}))
        assert SECRET not in out

    def test_exception_fields_redacted(self):
        payload = json.loads(JsonLogFormatter().format(_record("x", exc=True)))
        assert payload["exception_type"].endswith("ValueError")
        assert SECRET not in payload["exception_message"]


# ---------------------------------------------------------------------------
# No-op observability (default path)
# ---------------------------------------------------------------------------


class TestNoopObservability:
    def test_disabled_returns_noops_without_importing_otel(self, monkeypatch):
        import sys

        monkeypatch.delenv("UNISON_OTEL_ENABLED", raising=False)
        obs.init_observability()
        assert isinstance(obs.get_tracer(), obs._NoopTracer)
        assert isinstance(obs.get_meter(), obs._NoopMeter)
        # mcp 2.x hard-depends on opentelemetry-api, so the api package may be
        # imported by the SDK regardless of our flag. The invariant that
        # remains ours: the optional OTel SDK (the [observability] extra) is
        # never imported while observability is disabled.
        assert not any(m.startswith("opentelemetry.sdk") for m in sys.modules)

    def test_enabled_without_packages_degrades_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("UNISON_OTEL_ENABLED", "true")
        # OTel is not installed in this environment; if it ever is, force the
        # ImportError path so the test remains meaningful.
        import builtins

        real_import = builtins.__import__

        def deny_otel(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", deny_otel)
        with caplog.at_level(logging.WARNING):
            obs.init_observability()
        assert "observability stays no-op" in caplog.text
        assert isinstance(obs.get_tracer(), obs._NoopTracer)

    def test_span_and_metric_calls_are_cheap_noops(self):
        tracer = obs._NoopTracer()
        with tracer.start_as_current_span("x") as span:
            span.set_attribute("a", 1)
            span.record_exception(ValueError("x"))
        meter = obs._NoopMeter()
        meter.create_counter("c").add(1, {"k": "v"})
        meter.create_histogram("h").record(5)


# ---------------------------------------------------------------------------
# tool_span + provider aggregation via ContextVar
# ---------------------------------------------------------------------------


class _CapturingSpan(obs._NoopSpan):
    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}
        self.exceptions: list[BaseException] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(exception)


class _CapturingTracer:
    def __init__(self) -> None:
        self.span = _CapturingSpan()

    def start_as_current_span(self, name: str, **kwargs: Any):
        span = self.span

        class _Ctx:
            def __enter__(self_inner):
                return span

            def __exit__(self_inner, *exc):
                return None

        return _Ctx()


@pytest.fixture()
def capturing_tracer(monkeypatch):
    tracer = _CapturingTracer()
    monkeypatch.setattr(obs, "_tracer", tracer)
    return tracer


class TestToolSpan:
    def test_single_model_aggregation(self, capturing_tracer):
        with obs.tool_span("chat", {"prompt": "hi"}):
            obs.record_provider_call(
                provider_type="google",
                model="gemini-2.5-flash",
                duration_ms=10,
                usage={"input_tokens": 100, "output_tokens": 20},
            )
        attrs = capturing_tracer.span.attrs
        assert attrs["tool.name"] == "chat"
        assert attrs["tool.status"] == "success"
        assert attrs["tool.model"] == "gemini-2.5-flash"
        assert attrs["tool.tokens_in"] == 100
        assert attrs["tool.tokens_out"] == 20
        assert "tool.models" not in attrs

    def test_multi_model_uses_array_attribute(self, capturing_tracer):
        with obs.tool_span("consensus"):
            obs.record_provider_call(provider_type="google", model="gemini-pro", duration_ms=5, usage={})
            obs.record_provider_call(provider_type="openai", model="gpt-5", duration_ms=5, usage={})
        attrs = capturing_tracer.span.attrs
        assert attrs["tool.models"] == ["gemini-pro", "gpt-5"]
        assert "tool.model" not in attrs

    def test_error_attributes_redacted_and_exception_sanitized(self, capturing_tracer):
        with pytest.raises(RuntimeError):
            with obs.tool_span("chat"):
                raise RuntimeError(f"provider auth failed: {SECRET}")
        attrs = capturing_tracer.span.attrs
        assert attrs["tool.status"] == "error"
        assert SECRET not in attrs["tool.error_message"]
        assert all(SECRET not in str(e) for e in capturing_tracer.span.exceptions)

    def test_argument_values_never_exported(self, capturing_tracer):
        with obs.tool_span("chat", {"prompt": f"my key is {SECRET}", "model": "x"}):
            pass
        attrs = capturing_tracer.span.attrs
        assert attrs["tool.argument_keys"] == ["model", "prompt"]
        assert attrs["tool.argument_count"] == 2
        assert all(SECRET not in str(v) for v in attrs.values())

    def test_context_isolated_across_concurrent_tasks(self):
        """Consensus-shaped: concurrent provider calls attribute to their own ctx."""

        async def invocation(tool_name: str, model: str, tokens: int) -> tuple[int, list[str]]:
            with obs.tool_span(tool_name):
                await asyncio.sleep(0.01)
                obs.record_provider_call(provider_type="p", model=model, duration_ms=1, usage={"input_tokens": tokens})
                ctx = obs.current_tool_ctx.get()
                assert ctx is not None
                return ctx.tokens_in, list(ctx.models)

        async def main():
            return await asyncio.gather(
                invocation("a", "model-a", 11),
                invocation("b", "model-b", 22),
            )

        results = asyncio.run(main())
        assert results[0] == (11, ["model-a"])
        assert results[1] == (22, ["model-b"])

    def test_ctx_visible_inside_worker_thread_via_instrument_generate(self):
        """The sync wrapper copies the caller's context into the thread."""

        class FakeProvider:
            def get_provider_type(self):
                class T:
                    value = "fake"

                return T()

            def generate_content(self, prompt: str, model_name: str, **kwargs: Any) -> Any:
                class R:
                    usage = {"input_tokens": 7, "output_tokens": 3}

                return R()

        async def main():
            with obs.tool_span("chat"):
                wrapped = obs.instrument_generate(FakeProvider())
                await asyncio.to_thread(wrapped, prompt="hi", model_name="fake-model")
                ctx = obs.current_tool_ctx.get()
                assert ctx is not None
                return ctx.tokens_in, ctx.tokens_out, ctx.models, ctx.provider_calls

        tokens_in, tokens_out, models, calls = asyncio.run(main())
        assert (tokens_in, tokens_out) == (7, 3)
        assert models == ["fake-model"]
        assert calls == 1

    def test_provider_error_recorded_and_reraised(self):
        class FailingProvider:
            def get_provider_type(self):
                class T:
                    value = "fake"

                return T()

            def generate_content(self, **kwargs: Any) -> Any:
                raise ConnectionError("down")

        with obs.tool_span("chat"):
            wrapped = obs.instrument_generate(FailingProvider())
            with pytest.raises(ConnectionError):
                wrapped(prompt="hi", model_name="m")

    def test_instrument_stream_passthrough_and_error(self):
        class P:
            def get_provider_type(self):
                class T:
                    value = "fake"

                return T()

        chunks = iter([1, 2, 3])
        assert list(obs.instrument_stream(P(), chunks, model="m")) == [1, 2, 3]

        def failing():
            yield 1
            raise ValueError("mid-stream")

        with pytest.raises(ValueError):
            list(obs.instrument_stream(P(), failing(), model="m"))
