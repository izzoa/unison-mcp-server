"""Opt-in OpenTelemetry tracing and metrics for tool invocations.

Everything here is gated behind ``UNISON_OTEL_ENABLED=true``. When disabled
(the default), ``get_tracer()``/``get_meter()`` return no-op instances, no
``opentelemetry`` module is imported, and every recording call is a cheap
no-op — the OTel packages are an optional dependency (``pip install
unison-mcp-server[observability]``) and their absence is handled gracefully.

Architecture: one PARENT span per tool invocation, opened by the MCP handler
via :func:`tool_span`. ``ModelProvider.generate_content()`` is abstract and
overridden by every concrete provider (with native streaming paths besides),
so model/token attribution comes from :func:`record_provider_call`, invoked by
the shared call-site instrumentation. The active tool context travels in a
``contextvars.ContextVar`` — safe under consensus's concurrent provider calls
— and per-call data aggregates onto the parent span at exit.

Every exported string passes :func:`utils.logging_setup.redact_text`;
telemetry leaves the machine for external collectors by design, so an
unredacted credential in a span is exfiltration.
"""

from __future__ import annotations

import contextvars
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from utils.logging_setup import redact_text

logger = logging.getLogger(__name__)

_MAX_ATTR_LEN = 1000


# ---------------------------------------------------------------------------
# No-op instruments (always available; used whenever OTel is disabled/absent)
# ---------------------------------------------------------------------------


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D102
        pass

    def record_exception(self, exception: BaseException) -> None:  # noqa: D102
        pass

    def set_status(self, status: Any, description: str | None = None) -> None:  # noqa: D102
        pass


class _NoopSpanContext:
    def __enter__(self) -> _NoopSpan:
        return _NoopSpan()

    def __exit__(self, *exc: Any) -> None:
        return None


class _NoopTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoopSpanContext:  # noqa: D102
        return _NoopSpanContext()


class _NoopInstrument:
    def add(self, amount: float, attributes: dict[str, Any] | None = None) -> None:  # noqa: D102
        pass

    def record(self, amount: float, attributes: dict[str, Any] | None = None) -> None:  # noqa: D102
        pass


class _NoopMeter:
    def create_counter(self, name: str, **kwargs: Any) -> _NoopInstrument:  # noqa: D102
        return _NoopInstrument()

    def create_histogram(self, name: str, **kwargs: Any) -> _NoopInstrument:  # noqa: D102
        return _NoopInstrument()


_tracer: Any = _NoopTracer()
_meter: Any = _NoopMeter()
_initialized = False


def _enabled() -> bool:
    return os.getenv("UNISON_OTEL_ENABLED", "").strip().lower() == "true"


def init_observability() -> None:
    """Initialize OTel providers when enabled; fall back to no-ops otherwise.

    Missing OTel packages with the flag enabled logs a WARNING naming the
    extra to install and keeps the no-ops — never an exception.
    """
    global _tracer, _meter, _initialized
    _initialized = True
    if not _enabled():
        return

    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    except ImportError:
        logger.warning(
            "UNISON_OTEL_ENABLED=true but OpenTelemetry packages are not installed; "
            "observability stays no-op. Install with: pip install unison-mcp-server[observability]"
        )
        return

    try:
        sample_rate = float(os.getenv("UNISON_OTEL_SAMPLE_RATE", "1.0"))
    except ValueError:
        sample_rate = 1.0

    tracer_provider = TracerProvider(sampler=TraceIdRatioBased(max(0.0, min(1.0, sample_rate))))
    meter_provider = MeterProvider()

    exporter = os.getenv("UNISON_OTEL_EXPORTER", "otlp").strip().lower()
    try:
        if exporter == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

            tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        # exporter == "none": spans are created but not exported
    except Exception:
        logger.exception("Failed to configure OTel exporter '%s'; spans will not be exported", exporter)

    otel_trace.set_tracer_provider(tracer_provider)
    otel_metrics.set_meter_provider(meter_provider)
    _tracer = otel_trace.get_tracer("unison")
    _meter = otel_metrics.get_meter("unison")
    logger.info("Observability initialized (exporter=%s, sample_rate=%s)", exporter, sample_rate)


def get_tracer(name: str = "unison") -> Any:
    """Return the active tracer (no-op unless enabled and initialized)."""
    return _tracer


def get_meter(name: str = "unison") -> Any:
    """Return the active meter (no-op unless enabled and initialized)."""
    return _meter


# ---------------------------------------------------------------------------
# Tool-invocation context and parent span
# ---------------------------------------------------------------------------


@dataclass
class ToolCallContext:
    """Aggregation bucket for one tool invocation's provider activity."""

    tool_name: str
    models: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    provider_calls: int = 0


#: Active tool context; ContextVar isolates concurrent provider calls
#: (consensus fans out several at once) per asyncio task.
current_tool_ctx: contextvars.ContextVar[ToolCallContext | None] = contextvars.ContextVar(
    "unison_tool_ctx", default=None
)


def _redact_attr(value: str) -> str:
    redacted = redact_text(value)
    if len(redacted) > _MAX_ATTR_LEN:
        redacted = redacted[:_MAX_ATTR_LEN] + "[truncated]"
    return redacted


@contextmanager
def tool_span(tool_name: str, arguments: dict[str, Any] | None = None) -> Iterator[Any]:
    """Parent span for one tool invocation, with metrics and aggregation.

    Opens the ``tool.invoke`` span, installs a fresh :class:`ToolCallContext`,
    and on exit: sets aggregated model/token attributes (``tool.model`` scalar
    for a single model, ``tool.models`` array for multi-model runs like
    consensus), records the tool-call counter exactly once, and the latency
    histogram. Argument ATTRIBUTES are an allowlist of keys plus counts —
    never raw values. Error strings pass the shared redaction helper.
    """
    meter = get_meter()
    calls = meter.create_counter("unison.tool.calls")
    latency = meter.create_histogram("unison.tool.latency")

    ctx = ToolCallContext(tool_name=tool_name)
    token = current_tool_ctx.set(ctx)
    start = time.monotonic()
    status = "success"

    with get_tracer().start_as_current_span("tool.invoke") as span:
        span.set_attribute("tool.name", tool_name)
        if arguments:
            span.set_attribute("tool.argument_keys", sorted(_redact_attr(str(k)) for k in arguments))
            span.set_attribute("tool.argument_count", len(arguments))
        try:
            yield span
        except Exception as exc:
            status = "error"
            span.set_attribute("tool.status", "error")
            span.set_attribute("tool.error_type", type(exc).__name__)
            span.set_attribute("tool.error_message", _redact_attr(str(exc)))
            _record_exception_redacted(span, exc)
            raise
        else:
            span.set_attribute("tool.status", "success")
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            span.set_attribute("tool.latency_ms", elapsed_ms)
            span.set_attribute("tool.tokens_in", ctx.tokens_in)
            span.set_attribute("tool.tokens_out", ctx.tokens_out)
            unique_models = list(dict.fromkeys(ctx.models))
            if len(unique_models) == 1:
                span.set_attribute("tool.model", unique_models[0])
            elif unique_models:
                span.set_attribute("tool.models", unique_models)
            current_tool_ctx.reset(token)
            calls.add(1, {"tool_name": tool_name, "status": status})
            latency.record(elapsed_ms, {"tool_name": tool_name, "model": unique_models[0] if unique_models else ""})


def _record_exception_redacted(span: Any, exc: BaseException) -> None:
    """record_exception with a redacted stand-in, never the raw exception."""
    try:
        safe = RuntimeError(_redact_attr(f"{type(exc).__name__}: {exc}"))
        span.record_exception(safe)
    except Exception:  # pragma: no cover - defensive
        pass


# ---------------------------------------------------------------------------
# Shared provider-call instrumentation
# ---------------------------------------------------------------------------


def record_provider_call(
    *,
    provider_type: str,
    model: str,
    duration_ms: int,
    usage: dict[str, int] | None = None,
    error: BaseException | None = None,
) -> None:
    """Record one provider call: metrics plus parent-span aggregation.

    Invoked by the shared call-site wrappers around every provider execution
    path (sync-in-thread, native async, streaming). ``generate_content()`` is
    abstract on the base class, so this is the hook that actually executes.
    Retries and consensus fan-out land here as provider activity — the
    tool-call counter is untouched by design.
    """
    meter = get_meter()
    if error is not None:
        meter.create_counter("unison.provider.errors").add(
            1, {"provider_type": provider_type, "error_class": type(error).__name__}
        )
    meter.create_histogram("unison.provider.latency").record(
        duration_ms, {"provider_type": provider_type, "model": model}
    )

    ctx = current_tool_ctx.get()
    if ctx is not None:
        ctx.provider_calls += 1
        if model:
            ctx.models.append(model)
        if usage:
            tokens_in = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
            tokens_out = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
            if usage.get("total_tokens") and not (tokens_in or tokens_out):
                tokens_in = int(usage["total_tokens"])
            ctx.tokens_in += tokens_in
            ctx.tokens_out += tokens_out


def _provider_type_of(provider: Any) -> str:
    ptype = getattr(provider, "provider_type", None) or getattr(provider, "get_provider_type", None)
    try:
        if callable(ptype):
            ptype = ptype()
        return getattr(ptype, "value", None) or str(ptype or type(provider).__name__)
    except Exception:  # pragma: no cover - defensive
        return type(provider).__name__


def instrument_generate(provider: Any) -> Any:
    """Wrap ``provider.generate_content`` for sync (thread-offloaded) call sites.

    Drop-in for call sites that pass the bound method into
    ``asyncio.to_thread``: the wrapper preserves the signature and copies the
    caller's context so ``current_tool_ctx`` is visible inside the worker
    thread.
    """
    inner = provider.generate_content
    ptype = _provider_type_of(provider)
    caller_ctx = contextvars.copy_context()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model_name") or (args[1] if len(args) > 1 else ""))
        start = time.monotonic()
        try:
            response = caller_ctx.run(inner, *args, **kwargs)
        except Exception as exc:
            caller_ctx.run(
                record_provider_call,
                provider_type=ptype,
                model=model,
                duration_ms=int((time.monotonic() - start) * 1000),
                error=exc,
            )
            raise
        usage = getattr(response, "usage", None)
        caller_ctx.run(
            record_provider_call,
            provider_type=ptype,
            model=model,
            duration_ms=int((time.monotonic() - start) * 1000),
            usage=usage if isinstance(usage, dict) else None,
        )
        return response

    return wrapped


async def instrumented_async_generate(provider: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Await ``provider.async_generate_content`` with instrumentation."""
    ptype = _provider_type_of(provider)
    model = str(kwargs.get("model_name") or (args[1] if len(args) > 1 else ""))
    start = time.monotonic()
    try:
        response = await provider.async_generate_content(*args, **kwargs)
    except Exception as exc:
        record_provider_call(
            provider_type=ptype, model=model, duration_ms=int((time.monotonic() - start) * 1000), error=exc
        )
        raise
    usage = getattr(response, "usage", None)
    record_provider_call(
        provider_type=ptype,
        model=model,
        duration_ms=int((time.monotonic() - start) * 1000),
        usage=usage if isinstance(usage, dict) else None,
    )
    return response


def instrument_stream(provider: Any, chunks: Iterator[Any], *, model: str) -> Iterator[Any]:
    """Instrument a streaming provider iteration: duration and error capture.

    Token counts are recorded when the final assembled response reports usage
    elsewhere; the stream wrapper itself records latency and errors only.
    """
    ptype = _provider_type_of(provider)
    start = time.monotonic()
    try:
        yield from chunks
    except Exception as exc:
        record_provider_call(
            provider_type=ptype, model=model, duration_ms=int((time.monotonic() - start) * 1000), error=exc
        )
        raise
    record_provider_call(provider_type=ptype, model=model, duration_ms=int((time.monotonic() - start) * 1000))
