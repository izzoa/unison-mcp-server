# Batch Execution - Half-Price Runs for Calls Nobody Is Waiting On

**Run `chat` and `consensus` through OpenRouter's asynchronous Batch API at about 50% of the standard per-token price, and pick the result up later by handle**

Batch is not a separate tool. It is an optional `execution_mode="batch"` field on [`chat`](chat.md) and [`consensus`](consensus.md), plus `batch_handle`, `wait_seconds` and `cancel` fields on the same tools for resuming. It suits work where the answer is not needed this minute: a consensus run asks N frontier models the same question, nobody is waiting on any single answer, and it is the most expensive thing the server does.

## What you get, and what it costs you

- **Price**: about half the standard per-token rate for the model.
- **Latency**: minutes to hours. OpenRouter's completion window is 24 hours and is an expiry, not a promise. A call therefore never blocks on a batch (MCP hosts cancel long tool calls). It waits a bounded time and then hands back a handle.

## Limits

Read these before relying on batch:

- **OpenRouter only.** It needs `OPENROUTER_API_KEY`. Direct Anthropic, Gemini and OpenAI batch adapters are not implemented, so those providers cannot batch, even for the same model.
- **Text only.** A batch call carrying images, audio, video or attachments is refused before anything is submitted. Local files passed as `absolute_file_paths` / `files` are read into the prompt as text and are fine.
- **Only models OpenRouter lists as `:batch`.** OpenRouter lists a batch-priced copy of a model as a separate catalog entry (`anthropic/claude-opus-5.5:batch`). A model is batch-eligible when OpenRouter's public model list carries `<slug>:batch` for it, or (the offline fallback) when its entry in `conf/openrouter_models.json` has `supports_batch: true`. The curated file is therefore not the ceiling: a full slug such as `google/gemini-3.8-flash`, which the file does not contain, batches because OpenRouter lists `google/gemini-3.8-flash:batch`. Gemini models batch like any other. [`listmodels`](listmodels.md) tags curated models `batch` and shows what each family alias resolves to under batch. A model that is not batch-listed is refused by name with the instruction to repeat the call without `execution_mode`. It is never silently run synchronously at full price.
- **No provider-side cancel.** OpenRouter has no cancel endpoint. Cancelling is local only (see [Cancelling](#cancelling)); the provider batch keeps running and is still billed.
- **Results are kept 30 days by OpenRouter**, and Unison keeps its own record for `BATCH_RETENTION_DAYS` (default 30; see [The batch store](#the-batch-store)).
- **`thinking_mode` is not carried into a batch request**; the model runs with its provider defaults.
- **Not available elsewhere.** `execution_mode` is rejected on `clink` (Unison never sees the request a CLI subprocess makes, so it cannot batch on its behalf; see [clink](clink.md#model-strings-batch-pricing-and-the-supported_models-allowlist)) and on the multi-step workflow tools (`codereview`, `precommit`, `analyze`, `debug`, `thinkdeep`, ...). On `consensus` it applies to step 1, the step that consults the models.

## Asking for batch

Set `execution_mode` to `"batch"` (the only accepted value). Omit it, or pass `null`, for a normal immediate call. Any other value is an error that names the accepted values, so a typo cannot quietly cost double.

Batch is requested **only** through this field. Model names stay exactly as you type them: do not append `:batch` to a model to request batch. (`:batch` is an OpenRouter pricing listing, not a routing variant, and `model="vendor/model:batch"` without `execution_mode` runs synchronously.) The server also tells MCP clients this in its handshake instructions, so a natural-language request such as "run this chat as a batch" or "use unison consensus with opus and gpt5 batch models" maps onto the field.

### Which model does `execution_mode="batch"` run?

Under batch, provider selection only considers a provider that can batch the model, so an alias that a native provider and OpenRouter both serve (`opus` with `ANTHROPIC_API_KEY` configured, `gemini`, `flash`, `gpt5`) resolves to OpenRouter. What it resolves *to* depends on what you typed:

- **A family alias floats to the latest batch-listed model.** `opus`, `sonnet`, `haiku`, `gemini` (the Pro tier), `flash` (the Flash tier), `gpt5` (generation 5) and `grok`, with their other spellings (`claude-opus`, `gemini-pro`, `gemini-flash`, ...), resolve to the newest member of their family that OpenRouter lists as `:batch`. Against the 2026-11-22 catalog: `opus` is `anthropic/claude-opus-5.5`, `sonnet` is `anthropic/claude-sonnet-5.5`, `haiku` is `anthropic/claude-haiku-5.5`, `gemini` is `google/gemini-3.1-pro-preview`, `flash` is `google/gemini-3.8-flash` (never `-flash-lite` or `-pro`), `gpt5` is `openai/gpt-5.5` (never a `-pro`, `-mini` or codename model) and `grok` is `x-ai/grok-4.7`. No release needs a code change: a newer model is picked up as soon as OpenRouter lists it.
- **Version-pinned aliases and full slugs never float.** `opus4.5`, `sonnet4.5`, `o3`, `gpt5.2` and any `vendor/model` slug mean exactly that model.
- **Ordering.** Versions compare numerically (`5.5 > 5 > 4.8`, `4.10 > 4.9`). At an equal version a general-availability model beats its `-preview`; a newer preview still beats an older release. Codename models (`gemini-4-argon`, `gpt-6-astra`) have no version and are never picked by a family alias, but they are reported (next bullet).
- **Restrictions apply first.** `OPENROUTER_ALLOWED_MODELS` filters the candidates before the newest is chosen: a list naming only `anthropic/claude-opus-4.8` makes `opus` resolve to 4.8, and a list naming the alias `opus` admits the newest (so an allow-list of `opus` used as a cost cap lets batch `opus` reach a newer, possibly dearer model than a synchronous `opus` does; name the exact slugs you want to allow to keep a ceiling). A family with no batch-listed, allowed member is refused by name (alias and family), and a different family is never substituted.
- **A resolved model uses the limits of its own listing, not its family entry's.** (Limits are the context window and the output cap; request parameters follow the listing for curated slugs too, see the next bullet.) When an alias resolves to a slug `conf/openrouter_models.json` does not list (for example `opus` to `claude-opus-5.5`), or you type such a slug in full (`google/gemini-3.8-flash`), the context window, output limit, temperature support and the request parameters sent are those of that slug's own `:batch` listing in OpenRouter's model list: the live catalog's `:batch` listing, then the committed snapshot (which carries the same fields), then the plain listing of the same live body (the weakest evidence: its numbers can differ from the `:batch` listing's), then, only for a field none of them has, the curated entry with one warning in the server log. Each field falls back on its own; a value that is missing, malformed or absurd (a negative or non-numeric `context_length`, an output cap under 1,024 tokens, a `supported_parameters` that is not a list) is rejected and never repaired. `opus` under batch therefore has `anthropic/claude-opus-5.5`'s 1,000,000-token window and 128,000-token output, not the older `claude-opus-4.5` entry's 200,000 / 64,000, and an input larger than the resolved window is refused before anything is submitted, naming the slug and its limit. Only a listed window is enforced: when no listing gave the window (it is a stand-in from the curated family entry or the generic default) nothing is refused against it, and the provider is left to judge. The parameters follow the batch listing too: `temperature` is sent only where it is listed (the batch listing of `claude-opus-5.5`, `claude-haiku-5.5`, `gemini-3.8-flash` and `gpt-5.5` has none, even where the synchronous model takes one, so none is sent), a token cap is sent whenever `max_tokens` or `max_completion_tokens` is listed (with or without a temperature), and `top_p`, `seed`, `stop` and the like are each sent only when listed. Where the listing names `temperature`, it publishes no range, so the value is clamped to the range of the slug's vendor, chosen by the part before the `/`: `anthropic` 0 to 1, `openai`, `google` and `x-ai` 0 to 2, and **any other vendor 0 to 1** (the conservative range, so an unlisted vendor such as `qwen`, `z-ai` or `deepseek` is clamped harder than it needs to be, never softer; a provider-level request of 1.5 is sent as 1.0 to `anthropic/claude-opus-4.6` and as 1.5 to `openai/gpt-4.1`; `chat` and the workflow tools accept a `temperature` up to 2.0 and this per-model rule then clamps it, see "Published limits" below). A clamp is reported as a warning by the tools' temperature validator (`submit_batch` itself clamps silently, so a direct caller sees it only in the body that was sent); a model whose listing names no `temperature` is simply sent none (an info line, not a warning). A curated entry keeps its own reviewed range **narrowed by the same table**: the reviewed 0 to 2 of `claude-opus-4.5`, `claude-sonnet-4.5` and `claude-opus-4.1` is narrowed to Anthropic's 0 to 1, so 1.5 goes out as 1.0 to them as it does to the derived `claude-opus-4.6` (the narrowing is new, see "Published limits"). A temperature that is not a number (a `bool`, a string, NaN) is refused before anything is submitted. An entry LiteLLM merely auto-discovered is not a reviewed one and does not shadow the slug's own listing. Capability queries (`get_capabilities`) and synchronous calls never fetch; they apply the offline published limits described below.
- **A curated slug keeps its limits but sends only the parameters its `:batch` listing names.** A slug the reviewed curated file (`conf/openrouter_models.json`) lists (`openai/gpt-5`, `anthropic/claude-opus-4.5`, `openai/o3` ...) keeps its reviewed context window, and an input larger than that window is still refused; its output cap is the smallest of the reviewed one and every published one (see "Published limits"). Which optional parameters the request carries, though, is decided by the slug's own `:batch` listing (the live one, then the committed snapshot), parameter by parameter, exactly as for a resolved model, because the batch endpoint is what the listing describes and the curated entry describes the synchronous model. So batch calls to `openai/gpt-5`, `gpt-5-mini`, `gpt-5-nano` and `gpt-5.2` no longer send `temperature`, `top_p` or `stop` (their batch listings name only `seed` and `max_tokens`), `claude-opus-4.5` and `4.1` no longer send `seed` or `top_p`, and `claude-sonnet-4.5` no longer sends `seed`. In the other direction, `openai/o3`, `o3-mini`, `o4-mini` and `gpt-5.2-pro` (which take no temperature and used to receive an empty body beyond the messages) now carry your `max_output_tokens` as `max_tokens` and your `seed`, because their batch listings name them: a caller who set those got them silently dropped before. The token cap is bounded: a request's `max_output_tokens` above the entry's cap (the smallest of the reviewed cap and every published one) is reduced to it at submission, so `openai/o3` asked for 200,000 sends its 100,000; the tools never set it. A curated entry never gains a temperature its reviewed entry says the model lacks. If no listing exists at all for the slug (the live fetch is off or failing and the slug is not in the snapshot) the curated entry decides what is sent, with one warning in the server log. The batch record stores the parameter list and where it came from (`live`, `snapshot` or `curated`). The committed snapshot stores the reduction of each listing to the names the server can act on, so it is narrower than a live listing: the `gpt-5` family's live `:batch` listing also names `reasoning_effort`, `verbosity` and `tool_choice`, the Anthropic listings `tool_choice`, and `sonnet` `top_k`, none of which the server forwards, and a new forwardable name needs a snapshot refresh before an offline call sends it.
- **The call says what it ran.** When an alias resolves to a different model the response carries `model_requested` (`opus`) and `model_resolved` (`anthropic/claude-opus-5.5`), and a still-running status names `model_resolved` for each batch. `model_used` is the slug that was submitted.
- **A newer model the alias did not pick is reported.** The family rules are version-number rules, so OpenRouter's newer codename listings (`google/gemini-4-argon`, `-carbon`, `-hydrogen`, `-neon`; `openai/gpt-5.6-sol`, `-terra`, `-luna`, `gpt-6-astra`, ...) are never chosen by an alias, and that is kept (no manual order or tier list). What is added is visibility: when an alias resolves and the catalog lists a same-product-line model that is newer than the pick and that no family rule classifies, the response metadata that carries `model_requested` / `model_resolved` also carries `unclassified_newer_batch_models` (base slugs, newest version first then slug order, at most 10) and, only when the list was cut, `unclassified_newer_batch_models_omitted`; the server logs one warning per distinct alias, resolved slug and candidate set. In plain words, a slug is reported when it is batch-listed, starts with the alias's product line (`google/gemini-`, `openai/gpt-`, `anthropic/claude-opus-`, `x-ai/grok-`), is matched by no family rule, begins with a version strictly greater than the pick's, does not repeat a tier word a rule already covers (`flash` in a newer `-flash-lite`), has text-only output where the catalog says so (embeddings, image and audio listings are excluded) and is allowed by `OPENROUTER_ALLOWED_MODELS` *when named in full* (the note advises typing the full slug, so an allow-list naming only the alias `gpt5` admits the pick but advertises nothing). A newer slug with no tier word at all (`openai/gpt-6`) is reported, a `-pro` copy is folded into its plain sibling (one model, named once), and a slug with a colon or longer than 128 characters is never reported. Against the 2026-11-22 catalog `flash` and `gemini` report the four `gemini-4-*` codenames, `gpt5` reports ten models (`gpt-6.2-astra`, `gpt-6.2-sol`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-6-astra`, `gpt-6-luna`, `gpt-6-sol`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`) and nothing is omitted, and `opus`, `sonnet`, `haiku` and `grok` report nothing. The note is an alarm with known limits, not a ranking: the product-line narrowing means `gpt-oss-*`, the `o`-series and `claude-fable-*` are never reported for `gpt5` or `opus` (a vendor-wide test would report them on every call), a codename with no numeric lead is never reported (both are deliberate limits of a product-line-only note: `claude-fable-5.x` is not "a newer Opus", and a name like `gpt-astra` can be neither compared with the pick nor told from another product; type the full slug to use one), and a future newer `-flash-lite` with a tier word no rule covers is reported once. The note never changes which model runs, never appears on a synchronous call, is absent (not an empty list) when there is nothing to report, for a version-pinned alias and for a full slug, and is stored with the batch, so a resumed handle reports the same note. **To use a codename model, type its full slug** (`model="google/gemini-4-argon"`): a full slug is batch-eligible whenever OpenRouter lists it as `:batch`, is never floated, and gets the limits of its own listing.
- **Resolved once, then recorded.** The alias is resolved once, when the call is admitted, against one catalog for every model of the call; the slug, its limits and the note below are carried to submission (a catalog that changes in between changes none of them) and stored in the batch record. Resuming, cancelling or reading a result later never resolves or derives again and makes no catalog request, so a model released, or limits changed at OpenRouter, in between cannot change what a handle means; a still-running status shows the recorded `limits` for each batch.
- **Variants are refused, not stripped.** `openai/gpt-5:free`, `:online`, `:thinking` and the like have no batch listing and OpenRouter's batch API takes the model's normal slug; the call is refused naming the variant (stripping it would run a different model than the one named). A trailing `:batch` on a model still maps to the base slug. Without `execution_mode`, every identifier forwards verbatim, as before.
- **Synchronous calls are unchanged.** Without `execution_mode`, `opus` still resolves to the native Anthropic model (or the curated OpenRouter entry) exactly as before, and the batch catalog is never consulted.

When the exact model matters, pass the full slug and check `model_used` in the response.

#### Published limits (LiteLLM and OpenRouter), on every path

A model's output cap and temperature rules are checked against what is **published** about it, not only against the reviewed entry, and not only under batch: the same offline lookup (`providers/shared/published_limits.py`) is applied wherever a model's capabilities are resolved, so the synchronous and streaming paths of every provider (Gemini, OpenAI, xAI, Anthropic, Azure, DIAL, custom, OpenRouter and LiteLLM-discovered models) and `chat`, `consensus` and the workflow tools (`analyze`, `codereview`, `debug`, `thinkdeep`, `planner` ...) see the same limits as a batch call. `clink` makes no provider call and is not affected.

- **Output cap.** On the synchronous and streaming paths a model's effective `max_output_tokens` is the **smaller of the reviewed cap and what LiteLLM publishes** for the model: LiteLLM's figure is the only published source there. Under batch, for a model OpenRouter serves, OpenRouter's listing is a further source (the live `:batch` listing at batch admission, else the committed snapshot), and the **smallest valid figure** among them wins. The batch sources are consulted in that order (OpenRouter live, snapshot, LiteLLM, reviewed) and reported by the first that gives the smallest figure; when they disagree the smaller wins, so no published limit is ever exceeded under batch (on 13 of the 92 batch slugs the two sources disagree, for example `moonshotai/kimi-k3` 16,384 at OpenRouter against 1,048,576 at LiteLLM; the smaller is used under batch, which under-caps a model whose LiteLLM entry is stale). A published figure larger than the reviewed cap changes nothing. A request's `max_output_tokens` above the effective cap is reduced to it (one WARNING per model and cap per process); one at or below it, and one a model has no stated cap for, go out unchanged. Nothing is invented: a derived batch model whose cap is only a stand-in sends your figure as given, with one WARNING, and on the synchronous path the generic 32,768 cap of an uncurated OpenRouter slug is never enforced (only a published figure for that slug is). The committed OpenRouter snapshot does NOT bound a synchronous or streaming request: its figure is the `:batch` listing's, which can be far below what the plain listing accepts (for example `moonshotai/kimi-k3`: 16,384 against 943,718), and it stores no plain-listing figure, so it is used under batch only. On the synchronous path this lowers exactly two reviewed models today (`openai/gpt-5.2-pro` 272,000 to 128,000, `gemini-3-pro-preview` 65,536 to 65,535, both from LiteLLM and each on OpenRouter and on its native provider); an entry a provider auto-discovers from LiteLLM itself can also be lowered where LiteLLM's own keys disagree (Gemini `deep-research-pro-preview-12-2025`, `gemini-exp-1206`), which is not a reviewed entry. `anthropic/claude-opus-4.1` on OpenRouter keeps its reviewed 64,000 there (OpenRouter's listing publishes 32,000 and LiteLLM has no bare key for it, so a synchronous request above 32,000 is not caught locally); under batch it is bounded to 32,000. The batch record carries the source that bounded the cap as `sources.max_output_published`.
- **Temperature: what is and is not published.** LiteLLM publishes **no numeric temperature range, minimum, maximum or default** for any model (the word "temperature" appears in no entry of its model map), and OpenRouter's listing says only whether `temperature` is accepted. The one machine-readable datum is LiteLLM's boolean `supports_sampling_params`; when it is `false` on an Anthropic-family model the model accepts temperature 1 only, so none is sent and the constraint is fixed at 1.0 (LiteLLM's own code does the same; it is not applied to other vendors). Everything else is the labelled **fallback**: a vendor table kept in `VENDOR_TEMPERATURE_RANGES` (Anthropic 0 to 1; OpenAI, Google and xAI 0 to 2; any other vendor 0 to 1), which comes from vendor documentation and code knowledge, not from LiteLLM or OpenRouter data, and which could not be re-verified against the vendors' documentation when this was written. A model's rule is, in order: the sampling flag, its reviewed constraint where explicit (fixed, discrete, or none), then a reviewed range intersected with the range of a vendor NAMED in the table (Anthropic, OpenAI, Google, xAI; the vendor is taken from the provider, the OpenRouter slug prefix, or, for DIAL and custom models, an unambiguous model name such as `anthropic.claude-opus-4.1-...`). The "any other vendor 0 to 1" row is a guess, so it fills a gap (a derived batch entry, an uncurated OpenRouter slug's stand-in) and never narrows a reviewed range, and a model whose vendor cannot be told keeps its reviewed range. The intersection only narrows, and the narrowing is logged once per model at INFO with its provenance.
- **The tools accept 0 to 2.** `temperature` on `chat` and the workflow tools is accepted from 0 to 2 (it was 0 to 1) and the selected model's rule then corrects it with the existing message ("Temperature 1.5 invalid for ... Using 1.0 instead.") at WARNING: 1.5 is corrected to 1.0 for an Anthropic model (and Claude on OpenRouter or DIAL) and passes unchanged to `gpt-4.1`, a Gemini model, `grok-4` or a reviewed 0 to 2 range of another vendor (`deepseek/*`). A value from 0 to 1 produces the same request as before on every provider. This is a deliberate widening of the input contract. A direct caller of an Anthropic provider that skips the tools' correction and passes more than 1 now gets the provider's own `ValueError` (the reviewed 0 to 2 let it reach Anthropic's API, which rejects it).
- **Matching is conservative.** A model matches a LiteLLM key only exactly (provider-prefixed where LiteLLM names it so: `gemini/`, `vertex_ai/`, `xai/`, `azure/`, `openrouter/`, `mistral/`, `moonshot/`, `zai/`) and, for Anthropic names, the single dot-to-dash version rule (`claude-opus-4.5` to `claude-opus-4-5`); never a prefix or a guess, and a colon variant such as `:free` is not matched. Every tier is consulted (for an OpenRouter slug the `openrouter/<slug>` key and the vendor's own key) and all matching keys combine: the smaller figure is used. A garbled figure (zero, negative, NaN, a string, below 1,024 or above 10,000,000) is ignored, never clamped into a limit. No match means no check (one DEBUG line per model). Coverage against the 2026-11-22 data is 68 of the 92 batch slugs and 10 of the 11 reviewed batchable ones; unmatched, among others, are all DIAL entries, `llama3.2`, `grok-4` and `claude-opus-4.1` (which only batch admission checks, against OpenRouter's listing).
- **Offline.** The data is `conf/litellm_published_limits.json`, an extract of LiteLLM's model map written by `scripts/refresh_litellm_models.py` (the weekly LiteLLM refresh already runs it; one line per key, sorted), read once per process. Nothing is fetched at request time. A missing, oversized or corrupt file disables the LiteLLM part of the check with one WARNING (the vendor table still applies). Importing `litellm` itself still fetches its map at start-up unless `LITELLM_LOCAL_MODEL_COST_MAP=true`; the unit tests set it.
- **Not merged: the context window.** LiteLLM's `max_input_tokens` is the input limit while the reviewed `context_window` is the total window (`gpt-5` is 400,000 reviewed against 272,000 published input plus 128,000 output), so the window is not combined and no new refusal is added on the synchronous path. `listmodels` and auto-mode ranking read the reviewed entries and still show the reviewed limits.

#### Where the batch catalog comes from

OpenRouter's public model list (`GET https://openrouter.ai/api/v1/models`, no API key, nothing about your request is sent) is read **lazily**, the first time a batch call needs it, with a 5 second limit on the whole download and an 8 MB cap on the body (the real list is under 1 MB), and cached in memory for an hour. A slow lookup is spent from the call's `wait_seconds`, so it cannot push the call past its ceiling. It is never fetched at server start or by [`listmodels`](listmodels.md), which shows what is already loaded. If the fetch fails, times out or returns something unexpected (including a body that lists models but not one `:batch` entry, which would otherwise empty the catalog for an hour), the server logs one warning, uses the snapshot committed in `conf/openrouter_batch_catalog.json` (which carries each model's limits and parameter list as well as the slugs, so the offline fallback has the same capabilities; it also keeps any earlier good catalog), and does not retry for five minutes. If even the snapshot is unreadable it falls back to the `supports_batch` flags and the curated pins, which is how batch worked before aliases floated.

| Variable | Default | Effect |
|---|---|---|
| `OPENROUTER_BATCH_LIVE_CATALOG` | `true` | `false` disables the live fetch entirely (air-gapped or privacy-conscious installs); the snapshot and the curated flags remain. Any value other than a true/false word (`1`, `yes`, `on` ... or `0`, `no`, `off`, `disable` ...) is treated as `false` with a warning, so a typo never leaves the fetch running |
| `OPENROUTER_BATCH_CATALOG_TTL_SECONDS` | `3600` | How long a fetched catalog is trusted before the next batch call refreshes it; must be a positive, finite number (anything else logs a warning and uses 3600) |

To refresh the committed snapshot and see where things have drifted from OpenRouter's list (a `supports_batch` flag that disagrees, a family rule that matches nothing, a pin behind the family's latest, and any model whose temperature support or limits changed since the last snapshot):

```bash
python scripts/refresh_openrouter_batch_catalog.py --dry-run     # report only
python scripts/refresh_openrouter_batch_catalog.py               # rewrite conf/openrouter_batch_catalog.json
python scripts/refresh_openrouter_batch_catalog.py --allow-shrink # accept a list under half the snapshot's size, or one that drops the limits of most models
```

Exit status is 0 when the snapshot changed, 1 on an error (nothing is written) and 2 when it is unchanged. Family rules live on the curated entries as `batch_family` (see `_README.field_descriptions` in `conf/openrouter_models.json`).

## `chat` example

Start (waits up to `wait_seconds`, default 45, maximum 50; larger values are clamped):

```json
{
  "prompt": "Review the trade-offs between event sourcing and a plain audit table for our billing service.",
  "model": "openai/gpt-5",
  "working_directory_absolute_path": "/abs/path/to/project",
  "execution_mode": "batch"
}
```

If the batch finishes inside the wait, the normal chat reply comes back, with `batch_execution: true` and the `batch_handle` in its metadata. Otherwise:

```json
{
  "status": "running",
  "batch_handle": "<opaque handle>",
  "resume_with": {"batch_handle": "<opaque handle>"},
  "message": "The batch is still running at the provider; the work is NOT finished. Call chat again with ONLY batch_handle=..."
}
```

`running` means the work is not done. Do not resubmit the task (that starts, and bills, a second batch).

Resume by calling the same tool again with **only** the handle (a handle-only call needs no `prompt`, `model` or `working_directory_absolute_path`):

```json
{"batch_handle": "<opaque handle>"}
```

Add `"wait_seconds": 50` to wait longer per call. Resuming a finished handle returns the stored result again and submits nothing. A handle belongs to the tool that issued it; give a `chat` handle to `chat` and a `consensus` handle to `consensus`.

## `consensus` example

Batch every model at the request level:

```json
{
  "step": "Should we move our job queue from Redis to Postgres?",
  "step_number": 1,
  "total_steps": 2,
  "next_step_required": true,
  "findings": "Initial analysis of the proposal.",
  "models": [
    {"model": "anthropic/claude-opus-4.5", "stance": "for"},
    {"model": "openai/gpt-5", "stance": "against"}
  ],
  "execution_mode": "batch"
}
```

Or select it per model by putting `execution_mode` on individual `models[]` entries (omitted or `null` on an entry inherits the request-level value, and an entry cannot opt out of a request-level `"batch"`; to batch only some entries, leave the request-level field unset):

```json
"models": [
  {"model": "anthropic/claude-opus-4.5", "stance": "for", "execution_mode": "batch"},
  {"model": "openai/gpt-5", "stance": "against", "execution_mode": "batch"},
  {"model": "flash", "stance": "neutral"}
]
```

How it runs:

- One provider batch is submitted **per distinct batched model** (a batch carries one model; entries that differ only in stance share it), and all of them are joined under one `batch_handle`. The run reports `running` while any of them is unfinished.
- Entries that did not select batch run concurrently and synchronously as before, and their verdicts are written to the store before any handle is returned.
- If the models are slow, step 1 returns `running` with the handle. Resume with `{"batch_handle": "..."}`; the final response (`initial_prompt`, every verdict) is rebuilt from the stored record, so it works after a server restart, on a fresh server process, and a synchronous verdict is not run again.
- A failed request, or a wholly failed batch, does not lose its siblings: the other models' results are still returned.
- The completed response has the same structure as a synchronous consensus run.

## Cancelling

Call the same tool with the handle and `cancel`:

```json
{"batch_handle": "<opaque handle>", "cancel": true}
```

OpenRouter has no cancel endpoint, so this is a **local** cancel with honest reporting: the record is marked `cancelled`, polling stops and the results are ignored, but **the provider batch continues and is still billed**. The response says so (`provider_batches_continue: true` and a plain warning). Nothing is deleted from a running batch. Cancelling a handle that has already finished reports its existing state; `cancel` without a `batch_handle` is an error. If some children of a multi-batch run had already finished, the response still reports `cancelled` for the ones this call cancelled, and says that the finished children keep their results: call the tool again with only the handle to get them. The server log names the provider batch that was not stopped (`was NOT stopped at the provider`, with the handle and the provider's batch id), because callers are shown only the handle.

For a mixed `consensus` run, cancel stops polling the batches but cannot stop a synchronous consultation that is already running: it runs to completion, is billed in full and its verdict is stored. The response lists such consultations (`synchronous_consultations_continue`) and says so.

## Handle states

| Status | Meaning | What to do |
|---|---|---|
| `running` | Still processing, or part of the submission is not yet confirmed (the message then says the provider may or may not hold a batch for it, and nothing is shown as running) | Resume with the handle; do not resubmit |
| (completed) | The normal result of the tool | Nothing |
| `failed` | The request or the whole batch failed at the provider | Read the message; start again if appropriate |
| `cancelled` | Cancelled locally (the provider batch may still be billed) | Nothing |
| `expired` | OpenRouter's completion window lapsed before it was processed | Start again if still wanted (nothing is resubmitted automatically) |
| `results_expired` | OpenRouter no longer keeps the results (about 30 days; a poll or a fetch answering HTTP 410 is reported this way) | Start again |
| `handle_expired` | The Unison record was purged at its retention limit; only a tombstone remains | The handle was valid; start again |
| `unknown_handle` | Nothing was ever stored under that handle | Check the handle was copied exactly |
| `indeterminate`, `orphaned` | Needs an operator (below) | See below |

### `indeterminate` and `orphaned`

These are the two states where the server cannot say with certainty what the provider holds, and it deliberately does not guess or retry, because OpenRouter has no idempotency key and a retry could create a second, billed batch.

- **`indeterminate`**: a submission could not be confirmed. Submission is persisted before the provider is contacted, and a timeout, a 5xx or a crash between "accepted" and "recorded" leaves it unknown whether the provider holds a billable batch. The server tries to match the intent against OpenRouter's list of your batches (at startup and on every resume) and adopts one only when exactly one listed batch matches model, endpoint, request count and submission window. Zero matches is closed out as failed only after the window plus a 120 second safety margin. A listed batch whose model, endpoint and size match but whose creation time falls outside the window does not stop the close-out (a skewed clock between this host and OpenRouter would look exactly like that), but the recorded reason says so and the server log names the candidate batch ids (`possible clock skew`); check those in your OpenRouter account. Several matches, a possibly truncated list, a batch claimed by another record or a listing that differs from the intent only in model spelling stay `indeterminate`.
- **`orphaned`**: the provider accepted a batch that the server no longer tracks, for example its id could not be written to the store, or a later batch of a multi-model call failed to submit after an earlier one was accepted. It cannot be cancelled (no cancel endpoint), so it runs to completion and is billed.

What an operator should do:

1. Find the entry. The server logs the handle, provider, provider batch id (or `<unknown>`) and the reason: a WARNING at startup (`Batch <handle> needs operator attention ...`) and, for orphans, an ERROR at the time (`Orphaned <provider> batch <id> (handle <handle>) ...`). Search `logs/mcp_server.log` for `needs operator attention` and `Orphaned`. The error returned to the caller also names the provider and batch id of an orphan; that is the one place a provider batch id reaches a caller, deliberately, so the batch can be found at the provider. Retention can purge an `orphaned`, `indeterminate` or unresolved record while its batch may still be billed; it logs a WARNING when it does (`Retention is purging batch record ... while its ... child is still ...`).
2. Look at the batch in your OpenRouter account (dashboard, or `GET /api/v1/batches`) using the id, or for an `indeterminate` entry the model, request count and time. If it exists, it is billed and its results are retrievable from OpenRouter for about 30 days. If it does not exist, nothing was billed.
3. Do not resubmit blindly. Resume the handle once more first, since reconciliation re-runs on resume and some `indeterminate` records resolve themselves. Resubmit only after you have confirmed nothing is running at the provider, or accept paying for both.

## The batch store

A batch outlives the tool call and the server process, so every operation is recorded in a durable SQLite file the server creates for itself.

- **Works without configuration.** You do not set anything: the first batch call creates the store. It is **independent of `STORAGE_BACKEND`** (which defaults to `memory`) and of `STORAGE_SQLITE_PATH`; the conversation backend is never consulted, and batch does not share the conversation database.
- **Location.** `BATCH_STORE_PATH` when set, otherwise `<the server's working directory>/.unison/batches.db`, resolved when the store is first opened. Hosts that launch the server from `/` (Claude Desktop) or from a read-only directory should set an absolute `BATCH_STORE_PATH`; if the store cannot be opened, batch calls are refused with an error and never fall back to memory. An in-memory path (`:memory:`, `file::memory:`, `mode=memory`) is refused outright because it cannot survive a restart.
- **Protection at rest.** The file holds your prompts and the generated text in plaintext, protected by file permissions only (it is not encrypted). Its parent directory is set to `0700` and gets a `.gitignore` of `*`, and the database and its `-wal`/`-shm` files are `0600`. Because the parent directory's mode is changed, point `BATCH_STORE_PATH` at a directory dedicated to it. Handles are unguessable, server-generated identifiers and the only identifier a caller sees in a status or result (provider batch ids are replaced in provider text, and a reconciliation reason never names another operation's handle or a listed batch id; those go to the server log). The exception is the error for a batch that was accepted but could not be tracked (`orphaned`), which names the provider batch id for manual recovery. A mixed `consensus` run whose synchronous verdicts cannot be saved logs them at ERROR so a paid result is not lost; `logs/mcp_server.log` is plaintext with size-based rotation and no time-based retention. If `BATCH_STORE_PATH` names an existing file that is not a batch store (another SQLite database or any other file) the store refuses to open it, and leaves it and its directory untouched. A leading `~` in `BATCH_STORE_PATH` is expanded. `secure_delete` is on so purged text does not linger in free pages.
- **Retention.** Records are kept `BATCH_RETENTION_DAYS` (default `30`, matching OpenRouter's result retention) counted from creation, whether the batch finished or not. The prompts and any stored consensus state are retained until then. After that the content is deleted and only a tombstone remains (handle, creation time, expiry time), so an old handle reports `handle_expired` rather than `unknown_handle`. Retention is enforced for the life of the process on every read and write, not just at startup. Fractional days are allowed; an invalid, non-finite (`nan`, `inf`) or non-positive value (including `0`) logs a warning and uses 30.
- **What is stored.** The originating tool, the resumable payload (for consensus only `original_proposal`, `findings`, the model-and-stance list and the responses so far, never the step history), one row per provider batch, and outputs an operation had already consumed into its own state. Results the provider still holds are not copied in.
- **Startup.** The server reconciles unresolved submissions in the background at startup. It does nothing, and creates no file, when no batch store exists yet.

See `.env.example` for the two variables.

| Variable | Default | Meaning |
|---|---|---|
| `BATCH_STORE_PATH` | `<cwd>/.unison/batches.db` | Location of the batch SQLite file |
| `BATCH_RETENTION_DAYS` | `30` | Days a batch record is kept before its content is purged |

## Related

- [Chat](chat.md) and [Consensus](consensus.md): the tools that accept `execution_mode`
- [ListModels](listmodels.md): shows which models carry the `batch` tag
- [Configuration](../configuration.md#batch-store): the environment variables
- [clink](clink.md#model-strings-batch-pricing-and-the-supported_models-allowlist): why batch does not apply to CLI bridge tools
