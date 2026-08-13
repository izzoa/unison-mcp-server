# Changelog

All notable changes to the Unison MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **clink: `CLINK_TIMEOUT_SECONDS` runtime timeout override, and a higher default.** The per-CLI subprocess timeout was configurable only by editing `conf/cli_clients/*.json`; the new env var overrides it for every CLI at once (invalid or non-positive values are ignored with a warning), and the default ceiling doubled from 1800s to 3600s — safe now that host cancellation reaps the subprocess, since the host's own tool timeout usually fires first anyway. `docs/tools/clink.md` now documents all three timeout layers — Unison's, the MCP host's (e.g. Claude Code's `MCP_TOOL_TIMEOUT`, the one that actually cut the observed call), and the CLI's own.
- **clink: per-call `working_dir` parameter.** The spawned CLI previously always inherited the MCP server process's working directory (or a global manifest `working_dir`), so a server launched by Claude Desktop — or serving an agent working in a git worktree — ran CLIs somewhere the caller's files are invisible; Copilot's file tools are rooted at their cwd and refuse absolute paths outside it, so it would confidently answer about a repo it had never seen. Callers can now pass an absolute `working_dir` per call to run the CLI in their project or worktree root. Precedence: per-call value → manifest `working_dir` → server cwd. The read-only snapshot verifier now roots itself at the same effective directory (ending the false positives from scanning the server's own tree), and the effective directory is reported back as `metadata.working_dir` so callers never have to ask the CLI where it ran.
- **Native Anthropic provider.** `ANTHROPIC_API_KEY` now activates direct Claude access over the Messages API (official `anthropic` SDK) — previously the key activated nothing and Claude models were reachable only via OpenRouter. The curated catalog covers the latest two generations per family (six models: Fable 5, Opus 5, Opus 4.8, Sonnet 5, Sonnet 4.6, Haiku 4.5 — Fable is a single-generation family and Claude 3.5 Haiku is retired), deliberately curated-only (no LiteLLM auto-discovery, so the latest-two-generations policy holds), with `ANTHROPIC_ALLOWED_MODELS` restrictions, an `ANTHROPIC_API_URL` gateway override, and real extended-thinking budgets mapped from the server's thinking modes (the API's fixed-temperature-under-thinking constraint is handled provider-side). **Behavior note for dual-key setups:** `opus`/`sonnet`/`haiku`/`claude` aliases now resolve to the native provider ahead of OpenRouter (standing native→OpenRouter priority); OpenRouter-only configurations are unchanged.

- **Dependency locking.** `uv.lock` (universal — all platforms, Python ≥3.10) is now the installed source of truth, with committed pip-consumable exports `requirements.lock.txt` and `requirements-dev.lock.txt` for the pip-fallback path. Both setup scripts, CI, the semantic-release verification step, both quality-gate scripts, and Docker install the locked set instead of resolving open ranges — and `run-server.sh` no longer skips installation just because imports exist, so existing environments converge to the locked versions. Dev tooling moved from `requirements-dev.txt` (removed) into a `[dependency-groups] dev` group so it locks alongside runtime deps. A CI drift check (uv pinned at 0.8.12) fails when the lock or its exports fall out of sync with `pyproject.toml`. Exports are hash-free so the pip fallback works behind corporate TLS-intercepting proxies; the CI wheel smoke test deliberately keeps resolving open ranges as the canary for upstream breakage.

- **Third-party tool plugins.** `pip install` a package declaring a `BaseTool` subclass under `[project.entry-points."unison.tools"]` and it becomes a tool — the entry-point key is the tool name. The registry gains `register(name, cls)` (classes only, never instances — lazy instantiation is preserved), a `@register_tool("name")` decorator backed by a pending catalog with a startup freeze, and an opt-in import-only local scan (`UNISON_TOOL_AUTODISCOVERY=true`, default off) that excludes built-in modules *before* import so their lazy imports survive. Validation is two-stage: structural at registration (including detecting a non-overridden `BaseTool.execute`, whose inherited default only raises `NotImplementedError`); instance-level at first use, with failures **quarantining** the tool (excluded from availability, controlled error on direct access) instead of crashing the server. With no plugins and the scan flag unset, the registered tool set, schemas, dispatch, and import profile are unchanged. See `docs/plugins.md`.
- **Opt-in structured observability.** `UNISON_JSON_LOGS=true` switches `mcp_activity.log` to single-line, schema-versioned JSON (server log stays text); `UNISON_OTEL_ENABLED=true` adds one OpenTelemetry parent span per tool invocation plus tool/provider metrics, with OTel packages as an optional extra (`pip install unison-mcp-server[observability]`) and graceful no-op fallback when absent. Provider attribution comes from shared call-site instrumentation (sync-in-thread, native async, and streaming paths) with the active tool identity carried in a `ContextVar`, aggregating tokens and models onto the parent span — `tool.model` scalar for single-model calls, `tool.models` array for consensus fan-out; the tool-call counter increments exactly once per invocation, never per provider call or retry.

### Security

- **One public redaction helper now guards every export surface.** The 12.0.0 `RedactingFilter` scrubbed only `record.msg`, leaving bypass channels: formatted exception text (appended after filters run — leaking in TEXT mode today) and any field-level emission. `redact_text()` is now applied to text-mode exception output, every string the JSON formatter emits (message, tool fields, nested `extra` values, exception text), and all telemetry strings — span attributes, error messages, `record_exception` content. Tool-argument span attributes export key names and counts only, never values.

### Changed

- **BREAKING: migrated to the mcp 2.x SDK (`mcp>=2,<3`).** The low-level `Server` is now constructed with `on_*` callback adapters — 2.x removed decorator registration, `server.request_context`, automatic result wrapping, exception→`isError` conversion, and decorator-level input validation. The adapters in `handlers/` replicate all of it: request-context binding moves to a ContextVar (`utils/mcp_context.py`), jsonschema input validation keeps the 1.x error text (jsonschema is now a direct dependency), and handler composition changes from `register(server, …)` to `build_handlers(…)` + constructor callbacks with `server.py` still wiring-only. Wire-level equivalence was verified against captured 1.x fixtures for protocol `2024-11-05`: initialize, the complete tools/list schema set, unknown-tool, invalid-arguments, and prompts/list responses are value-identical (raw JSON differs only in key order). One reviewed deviation: `prompts/get` responses now carry the spec's top-level `description` and drop the nonstandard `prompt` object the 1.x extra-tolerant models leaked onto the wire. The server now also negotiates modern protocol revisions (up to `2025-11-25` over stdio) while remaining byte-compatible for legacy `2024-11-05` clients.
- **BREAKING: minimum supported Python is now 3.10.** The floor was declared three different ways — `requires-python = ">=3.9"`, setup scripts enforcing 3.10+, Black/Ruff targeting 3.9 while mypy targeted 3.10. It is now 3.10 everywhere: that is what the setup scripts already enforced and what the mcp 2.x SDK line requires. `requires-python`, Black, and Ruff targets are aligned, and the dead `importlib-resources; python_version<"3.9"` requirement marker was removed.
- **Annotations modernized for the py310 target.** Mechanical `ruff check --fix` sweep (383 fixes — PEP 604 unions and related pyupgrade rules) plus one explicit `zip(..., strict=False)` preserving existing behavior; no functional changes.
- **Claude Desktop is now detected on machines that haven't configured MCP yet — across install layouts.** `run-server.ps1` detected the app by the presence of `claude_desktop_config.json` — a file Claude Desktop only creates once MCP has been configured, i.e. the exact artifact the integration exists to write — so every fresh install reported "Claude Desktop not detected" (observed on a machine with the app installed). Detection now accepts any of the app's install footprints (`%APPDATA%\Claude`, `%LOCALAPPDATA%\Claude`, the `AnthropicClaude` updater directory, or an MSIX/Store package under `%LOCALAPPDATA%\Packages`), and the config path is install-aware: MSIX deployments read virtualized app data, so the config is written inside the package's `LocalCache` where the app actually looks. Client definitions gain multi-candidate `DetectionPaths` support; the dead `NeedsConfigDir` flag was removed.
- **The default `.env` template stopped advertising keys the server ignores.** `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` lines are removed (neither activates a provider — the Anthropic line sent a real onboarding down a dead end; Anthropic models route via OpenRouter), the previously-missing Azure OpenAI pair is included (commented), and secondary settings (`CUSTOM_*`, `DIAL_API_HOST`/`_VERSION`) are commented out with realistic examples instead of active placeholder values. A header now states the contract: only the listed variables enable providers, and placeholders count as unset.
- **`run-server.ps1` now declares `#Requires -Version 7.0`.** It previously declared 5.1 while using the PowerShell 7 ternary operator, and PowerShell parses a script in full before executing any of it — so on the Windows PowerShell 5.1 shipped with Windows 10/11 the script failed with an unexplained parse error rather than a version message. Windows users need PowerShell 7 (`winget install --id Microsoft.PowerShell`); they already did, they just weren't told. WSL remains unnecessary — the script is native PowerShell.

### Added

- **MCP host registration now covers the same hosts on every platform.** `run-server.sh` gains VS Code, VS Code Insiders, Cursor, Windsurf, and Trae registration (previously Windows-only); `run-server.ps1` gains Codex CLI registration (previously Unix-only). Both scripts now enumerate their full host set in one declarative structure — `MCP_HOST_REGISTRY` in bash, `$script:McpClientDefinitions` in PowerShell — so coverage can be compared row-for-row instead of by reading 5,000 lines of control flow. Editor registrations write the same entry shape on both platforms (`command`/`args`/`type: stdio`), clean the same legacy server names, and back up existing configs before writing.

### Fixed

- **clink nonzero-exit errors now lead with the CLI's own stderr.** A fast CLI failure surfaced as a bare `CLI 'copilot' exited with status 1` — the actual cause (e.g. GitHub's `Access denied by policy settings`) sat unread in `metadata.stderr`, and calling agents guessed "timed out" instead. The first lines of stderr (up to 400 chars) are now part of the error message itself, so policy denials, auth failures, and model rejections self-identify on the first line the caller reads.
- **clink no longer orphans the CLI subprocess when the MCP host cancels the call.** Process-tree reaping ran only on clink's own timeout; when the host cancelled first (its tool timeout, a user abort, a dropped connection — the common case, since hosts often time out long before clink's 30-minute default), the `CancelledError` path killed nothing and the spawned agentic CLI kept running headless, burning paid usage with nobody listening. Cancellation now terminates the whole process tree before propagating.
- **Anthropic provider audit: every enumeration site now includes — or deliberately, documentedly excludes — the native provider.** A four-way audit (Python source, setup/packaging/CI, docs, tests) swept every site that enumerates providers for ones that predate the Anthropic provider. Fixed: `listmodels` never rendered a native Anthropic section and undercounted configured providers (same missed-enumeration class as the `version` table); restriction typo-validation skipped `ANTHROPIC_ALLOWED_MODELS`; `run-server.sh` lagged `run-server.ps1` — key recognition falsely warned "No API keys found in .env!" for Anthropic-only setups, and the env→`.env` auto-fill and env-fallback arrays missed the key; the Docker healthcheck failed containers whose only key was `ANTHROPIC_API_KEY`; Qwen CLI registration didn't forward `ANTHROPIC_*` vars from process env; CI now blanks `ANTHROPIC_API_KEY` alongside the other guarded keys; `AnthropicModelRegistry` is exported from `providers.registries`; docs gained Anthropic across the configuration reference, getting-started, advanced-usage (auto-mode matrix, capabilities, vision), custom-models, docker-deployment, and listmodels pages plus all tool `model` enums, including a concrete note that `opus`/`sonnet`/`haiku` route to native Claude ahead of OpenRouter when the key is set. Tests gained listmodels-section, restriction-service, and priority-order coverage, and conftest now registers and sanitizes Anthropic like its siblings. Two exclusions were ruled deliberate and are now documented in place rather than silent: the weekly LiteLLM refresh and the LiteLLM discovery maps skip Anthropic because the catalog is hand-curated to the latest two generations per family — the earlier changelog claim of "LiteLLM auto-discovery enrichment" was corrected accordingly.
- **`version` and `listmodels` no longer contradict each other about Custom/Local — and `version` now shows all eight providers.** The startup placeholder guard lived only in provider registration, so a fresh `.env`'s `CUSTOM_API_URL=your_custom_api_url_here` left the provider unregistered (`version`: not configured) while `listmodels` read the raw env var and presented the placeholder as a configured endpoint complete with its 23-model catalog. Placeholder normalization is now a shared helper (`utils.env.get_custom_api_url`) applied by registration, the provider factory, and `listmodels`. Separately, `version`'s provider table was a hardcoded six-entry list predating the Anthropic and Azure providers; it now iterates the `ProviderType` enum itself (with display-name fallbacks), so a provider can never be silently missing from `version` output again.
- **The server no longer trusts a GUI host's working directory.** Claude Desktop spawns MCP servers from `C:\Windows\System32` (macOS: `/`), and everything downstream that resolves against the process cwd inherited it: clink subagents ran where the repo is invisible, clink's read-only snapshot verifier scanned the System32 tree and attributed Windows event-log writes to the model as `read_only_violations`, and `version` reported System32 as the installation path. At startup the server now detects non-workspace cwds (filesystem/drive roots and Windows system directories) and switches to its own directory; a deliberate workspace cwd — Claude Code launching from the project root — is preserved.
- **README + `.env.example` adversarial accuracy pass.** Every README claim was verified against the code (tool registry, clink manifests/read-only flags/recursion guard, provider implementations, infra env vars, and all 48 referenced files/assets). Fixed: the fork-adds intro said "four ways" above five bullets; the native Anthropic provider was missing from the fork-adds list, the PAL comparison table, and Key Features; the `#ide-clients` deep link resolved to nothing (real slug: `#ide-clients-cursor--vs-code`); the "Enabled by default" tool enumeration omitted `clink`; the recommended stack still said Sonnet 4.5. `.env.example` — where the README sends users for provider setup — never gained `ANTHROPIC_API_KEY` when the native provider shipped; it now carries the key plus commented `ANTHROPIC_API_URL`/`ANTHROPIC_ALLOWED_MODELS` entries and current Claude aliases in the `DEFAULT_MODEL` comment. All other verified claims were accurate.
- **MSIX Claude Desktop discovery no longer depends on the package's name.** The 14.1.2 dual-location fix looked for a package named `AnthropicPBC.Claude*`, but the MSIX installer's package is named `Claude_<publisherhash>` (observed live: `Claude_pzs8sxrjxfjjc`) — so on exactly the machines that fix targeted, discovery found nothing, the config was written only to the real `%APPDATA%\Claude`, and the app, whose virtualized copy shadows that file, never saw the server. Discovery is now content-based: any package under `%LOCALAPPDATA%\Packages` whose `LocalCache` contains the app's virtualized `Roaming\Claude` userData counts (most recently active wins when several match), with name-pattern fallbacks (`AnthropicPBC.Claude*`, `Claude_*`) for a package that has never been launched. The mirror step gained a guard that never overwrites an existing virtualized config with content merged from the classic side, and `Claude_*` joined the detection paths.
- **Claude Desktop registration is now visible across Store and classic install variants.** A machine where setup detected Claude Desktop and reported success could still show no server in the app: Store (MSIX) packages may or may not virtualize `%APPDATA%` depending on their manifest, so the app can read either the real `%APPDATA%\Claude\claude_desktop_config.json` or the package's `LocalCache` copy — and a config written to the wrong one is silently invisible. When a Store package is present, `run-server.ps1` now writes the config to **both** candidate locations (backing up before overwriting), and an existing virtualized copy takes precedence as the primary target since it shadows `%APPDATA%` for virtualized apps. Hardening in the same pass: all JSON host configs and `.env` are written through a helper that guarantees UTF-8 **without BOM** (a BOM makes Node hosts' `JSON.parse` reject the whole file; a config carrying one from an earlier run is repaired without prompting), every write is re-read through the JSON parser so an unparseable config fails loudly at setup time, the completion message now says to fully quit Claude Desktop from the system tray (closing the window doesn't restart it) and where the server appears, and a `Test-Path … -and` precedence bug that crashed the Qwen CLI backup step was fixed.
- **Empty provider content is now an error, not a silent success.** Reasoning models behind the OpenAI-compatible base class (X.AI, OpenRouter, custom/Ollama, Azure) can answer 200 OK with `content=""` and `finish_reason="length"` when reasoning consumes the whole output budget; that empty string was returned as a successful response, so tools completed having produced nothing with no error anywhere. Empty/None/whitespace content now raises a typed `EmptyContentError` naming `finish_reason` and `reasoning_tokens`, and the retry classifier matches on the type *before* any string inspection — the message embeds a token count, and digits like `8503` would otherwise string-match the `"503"` retry indicator. Budget exhaustion (`finish_reason="length"`) is deterministic and not retried; unexplained empty content is. Contributed by @Julzilla (#7).
- **uv installs work in OneDrive-synced checkouts.** uv hardlinks packages from its cache, which OneDrive cloud-files reject (`incompatible hardlinks`, os error 396) — observed the moment the TLS fix let uv succeed on a corporate machine. Setup now sets `UV_LINK_MODE=copy`.
- **The pip fallback survives uv-created venvs.** `uv venv` ships no pip, so when uv created the environment but the install fell back to pip, there was no `pip.exe` to fall back to. Both setup scripts now create venvs with `uv venv --seed`, and `run-server.ps1` additionally bootstraps via `ensurepip` if pip is still missing.
- **The simulator harness no longer races mcp 2.x shutdown.** `call_mcp_tool` wrote initialize + initialized + tools/call in one shot and closed stdin immediately (`subprocess.run(input=…)`); the 1.x SDK drained in-flight requests after EOF, but 2.x cancels them with a `-32000 Connection closed` error. The client now reads the tool response before closing stdin, with threaded stdout/stderr pumping and the same one-hour ceiling.
- **Wheel and uvx installs were missing `litellm` and `tiktoken`.** `pyproject.toml [project] dependencies` omitted both, even though `requirements.txt` declares them and runtime code imports them — packaged installs shipped without model-metadata discovery and token counting. The manifests are reconciled and the wheel smoke test now verifies both ship.
- **Fresh installs crashed at startup: `mcp` is now pinned below 2.** mcp 2.0.0 (released 2026-07-28) removed the 1.x low-level `Server` decorator API this server registers its handlers with, and the floor-only `mcp>=1.0.0` in `requirements.txt` and `pyproject.toml` let every new environment resolve it — failing at import with `AttributeError: 'Server' object has no attribute 'list_tools'`. Both manifests now pin `mcp>=1.0.0,<2` (fresh installs resolve 1.29.0), and the troubleshooting fallback `run-server.sh` prints quotes the same pin so copy-pasting it cannot reinstall 2.x. Existing environments are unaffected; adopting the 2.x API is tracked as separate work.
- **Windows setup no longer reports placeholder API keys as valid.** `Test-ApiKeys` compared values against `"your_${key.ToLower()}_here"` — a braced reference to a nonexistent variable that expands empty — so every template placeholder passed the check and a freshly created `.env` printed four "✓ Found valid" lines. The comparison now uses the `$($key.ToLower())` subexpression; validity is non-empty-and-not-placeholder (the stale per-provider format regexes were removed rather than enforced — they predate current `sk-proj-…` OpenAI keys); DIAL, Azure OpenAI (key + endpoint), and keyless custom endpoints such as Ollama now count as configured providers instead of triggering a false "No valid API keys found"; and the function's boolean no longer leaks into setup output as a stray `True`.
- **Windows dependency installs survive paths containing spaces.** The uv install went through `Start-Process -ArgumentList`, which joins arguments unquoted — a checkout under `OneDrive - <Company>` split into five tokens and aborted uv with a parse error on the bare `-`. Direct invocation now passes each argument intact, with an explicit `$LASTEXITCODE` check (`$ErrorActionPreference = "Stop"` does not cover native exit codes).
- **uv now trusts the OS certificate store on Windows setup.** Behind corporate TLS-intercepting proxies, uv's managed-Python download failed with `invalid peer certificate: UnknownIssuer` while pip succeeded, silently downgrading setup from the intended Python 3.12 to whatever system Python existed. The script now sets `UV_SYSTEM_CERTS=true` (plus the deprecated `UV_NATIVE_TLS=1` alias for older uv) for all uv operations, and a failed `uv venv` prints a warning naming the failure before falling back.
- **Windows setup verifies the server imports before configuring MCP clients.** Previously "Setup Complete" printed and every detected MCP client was configured before the server module had ever been imported, so a broken install surfaced as a crash at first client launch (exactly how the mcp 2.0.0 breakage was discovered). A post-install `import server` check — no API key required — now gates client configuration and exits nonzero with the underlying error.
- **pip self-upgrade no longer errors on every Windows setup run.** `pip.exe install --upgrade pip` cannot modify the running executable on Windows and always printed an ERROR; the upgrade now goes through `python -m pip`.
- **The documented execution-policy remedy was invalid PowerShell.** `Set-ExecutionPolicy -Scope Process -Bypass` fails to bind (`Bypass` is a value of `-ExecutionPolicy`, not a switch) — it appeared in the README and the script's own help text, and was the first thing to fail in a real Windows onboarding. Corrected to `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`, and the README now states the PowerShell 7 prerequisite *before* the command block it governs.
- **A fresh default `.env` no longer crashes the server at startup.** The `CUSTOM_API_URL` template placeholder was the only provider value in `providers/configure.py` without a placeholder check, so `your_custom_api_url_here` was registered as a real endpoint and provider instantiation died with `ValueError: Invalid URL scheme`. The placeholder is now treated as unconfigured — consistent with every other provider path — so an untouched `.env` produces the actionable "At least one API configuration is required" error instead, and the `CUSTOM_API_KEY`/`CUSTOM_MODEL_NAME` placeholders stop leaking into startup logs.
- **The PowerShell quality gate was weaker than the bash one.** `code_quality_checks.ps1` ran ruff, black, isort and pytest but skipped mypy on the strict allowlist, skipped the mockup-drift check, and did not enforce the coverage threshold, so a Windows contributor could see a green gate on a tree the bash gate rejects. All three are now enforced, mypy is included in the dev-tool install list, and each tool's resolution (venv vs `PATH`) is reported so a mixed toolchain is visible rather than silent.
- **`run_integration_tests.ps1` reported success after a failed simulator run.** It printed "Simulator tests failed!" immediately followed by "Integration tests completed successfully - you can proceed" and exited zero. It now exits non-zero. It also invoked the simulator through a bare `python`; both it and the integration suite now use the resolved virtual-environment interpreter.
- **VS Code MCP registration on Windows wrote an outdated configuration shape.** `run-server.ps1` wrote `settings.json` with an `mcp.servers` key; VS Code reads a user-profile `mcp.json` with a top-level `servers` key. VS Code Insiders in the same script was already correct — the format check is now keyed on the configured shape rather than on which client it is, so the two cannot drift apart again.
- **Claude CLI registration on Windows only printed instructions.** `run-server.ps1` displayed the `claude mcp add` command for the user to run while `run-server.sh` performed the registration. It now registers, falling back to printing the command only if that fails.

- **`code_quality_checks.sh` reformatted unrelated files on every run.** Tool paths were all gated on a single `.unison_venv/bin/ruff` existence check, so a venv holding some tools but not ruff sent every tool to whatever was on `PATH` — resolving `black` to a system copy two major versions behind the one CI installs. Each local run then reverted files CI considered correctly formatted, producing recurring churn in `utils/sqlite_storage.py` and three test modules. Each tool is now resolved independently.

### Added

- **GitHub Copilot CLI is now a clink target** (`cli_name="copilot"`). Invoked non-interactively by piping the prompt to stdin — no `-p` flag — which keeps prompts with embedded file contents off argv and away from the `ARG_MAX` ceiling. Output is parsed from `--output-format json` JSONL, selecting on `assistant.message` events and excluding messages from subagents spawned via Copilot's `task` tool (those carry `agentId` on the envelope and `parentToolCallId` in `data`). Supports runtime model selection via `--model` and native image/PDF attachments via repeated `--attachment` flags, including base64 blobs materialized to temporary files.
- **Copilot read-only mode is fail-closed.** `read_only=true` restricts the model's tool schema with `--available-tools view,grep,glob` rather than denylisting known-dangerous tools, so a tool added by a future Copilot release is excluded by default instead of silently permitted. Verified against CLI 1.0.78 that this reduces the schema from 23 tools to exactly 3 — and that it also removes MCP server tools (5 `github-mcp-server-*` tools disappeared alongside the native ones), so no separate MCP handling is needed. `--deny-tool write` / `--deny-tool shell` back it up as a second layer. Note this is application-level enforcement, not an OS sandbox like Codex's `--sandbox read-only`.
- **`InvocationPlan.extra_args`** lets a clink agent contribute per-request argv fragments regardless of transport kind. The `stdin` materializer previously contributed no argv at all, making per-request flags impossible for any stdin-transport CLI. Defaults to empty, so command construction for all seven existing agents is byte-identical.

### Changed

- **BREAKING: `clink` now requires `cli_name` when more than one CLI is configured.** The generated tool schema has always marked the field required in that case, but nothing enforced it — the Pydantic field was optional and an omitted value silently fell back to a hardcoded `gemini` preference. Requests that omit `cli_name` in a multi-CLI deployment now fail with an error enumerating the configured clients instead of dispatching somewhere the caller did not ask for. A single-CLI deployment may still omit the field. Note the previous fallback also contradicted the field's own documentation, which promised "the first configured CLI".

### Fixed

- **Every clink target was told it is Gemini.** `_agent_capabilities_guidance()` returned a hardcoded "You are operating through the Gemini CLI agent" string and was applied unconditionally, so all seven targets — aider, amp, claude, codex, crush, gemini, opencode — received false identity guidance on every invocation. The guidance is now derived from the resolved client, so a newly registered CLI is named correctly with no code change. Dates back to the original clink commit, when Gemini was the only target.
- **Guidance promised capabilities the system cannot verify.** The same text asserted every target could launch web searches and had a "full suite of CLI capabilities". No capability information is modelled per client and the claim is false for at least one target (aider has no web search), so the guidance now directs the CLI to use whatever tools are available to it.
- **The read-only prompt instruction named tools that no target has.** The injected `=== READ-ONLY MODE ===` block enumerated `EditFile`, `WriteFile`, `CreateFile`, `DeleteFile`, and `ReplaceInFile` — stale for all seven CLIs, including Gemini (which exposes `write_file` and `replace`). The prohibition is now expressed behaviorally, covers any filesystem-writing tool regardless of name, and explicitly includes shell redirection. This is layer 2 of the three-layer read-only model; layers 1 and 3 are unchanged.
- **`clink` prompt preparation could raise `AttributeError` on an omitted `cli_name`.** `prepare_prompt()` passed the raw value to `get_client()`, whose parameter is annotated `str` and immediately calls `.lower()`. Client resolution is now centralised in a single helper used by every entry point, which raises an actionable error instead.
- **The `clink` tool description advertised Qwen**, which has never been a configured target. It no longer names any CLI; the `cli_name` enum is the authoritative list.

## [12.0.0] - 2026-07-12

### Security

Remediation of an internal adversarial security & correctness audit. Highlights:

- **Path sandbox no longer exposes pseudo-filesystems (D-1).** `/proc`, `/sys`, `/dev`, and `/run` are now in `DANGEROUS_SYSTEM_PATHS`, closing a read-sandbox escape where `/proc/self/environ` (a zero-size, regular file on Linux) leaked the server's entire environment — every provider API key — into a model prompt. The `file_utils.py` "Security Model" docstring was corrected to describe the actual blocklist (it falsely claimed PROJECT_ROOT confinement).
- **clink `read_only=True` is now honest and enforced for Codex (A-1, A-4).** `CodexAgent` runs `codex exec --sandbox read-only` and strips the manifest's `--dangerously-bypass-approvals-and-sandbox` (which had disabled Codex's sandbox and approval prompt, allowing prompt-injected command execution). `tools/clink.py` no longer reports `read_only_enforced: true` unconditionally — it reflects whether a real layer-1 sandbox flag was applied and adds a `read_only_enforcement` breakdown and `read_only_verification_coverage` caveat.
- **Read-only filesystem verification closed several blind spots (A-2, A-3, A-7).** Snapshots now traverse the full tree (was depth-3), include gitignored/transient paths (so a write to `.env` or `*.log` can't evade detection), record symlinks, and key on `(mtime_ns, ctime_ns, size)` so a content edit that restores mtime is still detected. Entry-count-bounded with a logged warning instead of silent truncation.
- **clink model-flag injection blocked (A-6).** A `model` value beginning with `-` is rejected at the boundary, so a crafted value like `--yolo` can no longer corrupt the read-only command.
- **chat generated-code writer hardened (D-3, D-4).** Writes use `O_NOFOLLOW`/`O_TRUNC` and never proceed after a failed open (defeats a planted-symlink CWE-59/TOCTOU), and the target directory is now run through the same dangerous-path/home-root sandbox as reads.
- **Image validation no longer reads before checking (D-2).** `validate_image` stats the path, rejects non-regular files (`/dev/zero`, FIFOs, `/proc`) and oversize files before any read, and reads with a hard byte cap — preventing a memory-exhaustion/hang DoS.
- **Secrets kept out of logs (F-2, F-4).** Responses-API path logs metadata only (never model output/prompts); a redaction `logging.Filter` scrubs credential-shaped substrings from all handlers; tool name / continuation_id / clientInfo are sanitized before logging (log-injection, CWE-117); the shipped `LOG_LEVEL` default is now `INFO`.
- **Credential leaks narrowed (F-1/A-12, F-3, F-5).** Opt-in `UNISON_CLINK_STRIP_SECRETS` withholds Unison's provider keys from spawned CLIs; the DIAL client no longer follows redirects (its custom `Api-Key` header would otherwise leak cross-origin); `run-server.sh` masks API-key values in every echoed registration command.
- **SQLite conversation DB is now private (B-4).** Directory `0700`, DB + WAL/SHM `0600`, and a `.unison/.gitignore` (`*`) is dropped so plaintext history can't be accidentally committed or world-read.
- **Hardened clink output parsers (A-8, A-9, A-10, A-11) and non-string `continuation_id` (G-1)** against malformed/adversarial input (`RecursionError`, non-object JSON, non-dict event fields, non-string ids) so they fail cleanly instead of raising `AttributeError`.

### Fixed

- **Continuation history was double-recorded on every server-routed turn (B-1).** `SimpleTool.execute` checked for a `=== CONVERSATION HISTORY ===` marker that never matched the real `=== CONVERSATION HISTORY (CONTINUATION) ===` header, so each continuation persisted the full enhanced prompt as a duplicate turn and nested the history. Now detected via a shared header constant plus the server-injected execution context.
- **Continuation context was wiped at the MCP boundary (B-2).** `handle_call_tool` rebuilt `arguments["_context"]` from scratch, resetting `remaining_tokens`/`original_user_prompt` to defaults — causing spurious "prompt too large" errors on long conversations and collapsing the workflow expert-analysis file budget to ~1k tokens. It now preserves the reconstructed values via `dataclasses.replace`.
- **Workflow tool state leaked across sessions and concurrent calls (C-1, C-2).** Registry-cached workflow singletons now reset `work_history`/`consolidated_findings` on a fresh step-1 conversation and serialize execution with a per-instance `asyncio.Lock`, preventing stale findings from bleeding into new conversations or concurrent requests corrupting each other's summaries/persisted state.
- **Blocking provider calls no longer freeze the event loop (C-3).** The chat and non-streaming expert-analysis paths use `await provider.async_generate_content(...)` (offloaded via `asyncio.to_thread`) instead of the synchronous call with up to 17s of `time.sleep` retry backoff.
- **Streaming progress is delivered live (C-4).** Expert-analysis streaming bridges chunks to the event loop via an `asyncio.Queue` as they arrive (was: buffer everything, then burst), the Gemini provider no longer materializes the whole stream with `list()`, and the progress notifier enforces a wall-clock interval floor.
- **Circuit breaker robustness (C-6, C-7, C-8).** HALF_OPEN reclaims a leaked probe slot after the reset timeout (no permanent wedge); caller-fault 4xx errors no longer trip the breaker for a healthy provider; and the native streaming overrides now participate in the breaker.
- **Provider correctness (E-1…E-6).** Responses-API models: image inputs are mapped to `input_image` (vision was fully broken), `max_output_tokens` is used instead of the invalid `max_completion_tokens`, and token usage is read from the correct fields. `_is_error_retryable` classifies by structured status code before falling back to a "429" substring. CUSTOM/AZURE allowlists are honored by the restriction service (listing + auto-mode). `get_available_models` iterates in routing-priority order with `setdefault`.
- **Crash-hardening (G-2).** Malformed `issues_found` items (None/non-string severity, non-dict entries) are normalized by a field validator instead of crashing workflow summary builders.
- Concurrency: lazy OpenAI/Azure client construction is now guarded by a dedicated `_client_init_lock` (double-checked), distinct from DIAL's deployment-cache lock to avoid a non-reentrant deadlock (C-5).
- clink subprocess timeout now kills the whole process group and bounds the cleanup drain, so descendants aren't orphaned and a survivor holding the stdout pipe can't hang the request indefinitely (A-5).
- **Unknown/disabled tool calls with a `continuation_id` no longer persist an orphan user turn (B-5)** — the availability gate runs before thread reconstruction. `add_turn` is now atomic on SQLite via `BEGIN IMMEDIATE` (B-3).
- **Opencode read-only mode no longer returns plan-language stubs** ("I'll review...") instead of actual analysis. `OpencodeAgent.get_read_only_args()` now returns `[]` because opencode's `--agent plan` switches the agent persona (producing planning-language) rather than blocking writes — it was not a true read-only sandbox. Layer-2 (prompt instruction) and layer-3 (filesystem snapshot diff) provide read-only enforcement, both of which are CLI-agnostic. Corrects v11.8.0 behavior.
- **Opencode CLI bookkeeping no longer false-positives as read-only violations.** Opencode's first-run setup-cache files (`.opencode/.gitignore`, `.opencode/package.json`, `.opencode/package-lock.json`, `.opencode/node_modules/**`, `.git/opencode`) now classify as `read_only_violations.by_cli_bookkeeping` instead of mixing with genuine model writes. Pattern declaration is on `OpencodeAgent.fs_violation_ignore_patterns` and uses an explicit prefix DSL — `fnmatch` was rejected because stdlib `fnmatch` does not implement bash-style globstar and would have silently failed to match the actual bootstrap paths under Python 3.12. Tight enumeration (not a directory-wide `.opencode/**` glob) ensures model writes to `.opencode/skills/` or `.opencode/commands/` correctly classify as `by_model`.
- **BREAKING (read-only metadata shape):** `read_only_violations` in clink response metadata is now an object `{by_model: {created, modified, deleted}, by_cli_bookkeeping: {created, modified, deleted}}` instead of a flat `{created, modified, deleted}`. The empty-state representation also changes type: previously `[]` (falsy list), now always a dict with empty buckets (truthy). Callers that did `if metadata["read_only_violations"]:` should switch to inspecting the bucket lists explicitly (e.g. `if any(metadata["read_only_violations"]["by_model"].values()):`). Callers consuming `metadata["read_only_violations"]["created"]` should switch to `metadata["read_only_violations"]["by_model"]["created"]` for the same semantic content.
- **Installed wheel was missing the `handlers/` subpackage**, causing `ModuleNotFoundError: No module named 'handlers'` at server startup for all uvx / pip / Docker installs since 11.7.0. Manifested to users as MCP client handshake failures with "connection closed: initialize response" (e.g. Codex CLI via `uvx --from git+https://github.com/izzoa/unison-mcp-server.git`). Switched `[tool.setuptools.packages.find]` in `pyproject.toml` from an explicit allowlist to a denylist so future runtime packages are discovered automatically and cannot be silently dropped from the wheel
- Codex clink read-only sandbox: replaced invalid `--approval-mode suggest` flag (not supported by `codex exec`) with prompt-based enforcement
- Gemini clink read-only sandbox: replaced non-existent `--disallowedTools` flag with `--approval-mode plan` (Gemini CLI's actual read-only mode) and strip conflicting `--yolo`/`-y` flag when read-only is active
- `run-server.sh` now always refreshes MCP registrations with current `.env` values — previously, adding/removing tools via `DISABLED_TOOLS` (or changing any env var) had no effect because Claude Code, Claude Desktop, and Codex CLI registrations were skipped when the server path was unchanged

### Added

- **Amp clink integration (Phase 3 of `clink-aider-crush-amp-support`)**: adds `amp` as a seventh clink target. New manifest at `conf/cli_clients/amp.json`, agent class at `clink/agents/amp.py`, parser at `clink/parsers/amp.py`, plus registry wiring. Invocation via `amp --execute --stream-json` — Amp's documented non-interactive mode with parseable JSONL output. Authentication via `AMP_API_KEY` environment variable (non-interactive use; users run `amp login` for interactive first-time setup). Unison does NOT manage Amp's auth. **Model selection uses Amp's named modes** (`deep`, `large`, `rush`, `smart`) via `--mode` rather than arbitrary model strings — manifest declares a `supported_models` allowlist to enforce these values per `clink-runtime-model-selection`. **Image input** uses Phase 0's `stream_json` transport plan when `images` is non-empty; text-only invocations use stdin transport. Read-only mode is prompt-only + filesystem-snapshot fallback (Amp's permissions are managed via `amp permissions` config, not invocation flags). Parser extracts the canonical `result.result` field from Amp's JSONL event stream, with fallback to concatenated assistant messages when the result event is absent; captures `session_id` and `usage` in metadata. **Recursion guard:** Amp is MCP-aware via `amp mcp add` so wiring Unison as an MCP server in Amp's config + invoking `clink with cli_name="amp"` creates a loop risk — covered by Phase 0's cross-cutting recursion guard (no Amp-specific code needed). 26 new tests under `tests/test_clink_amp_smoke.py` cover parser against real amp JSONL fixtures, image-input plan switching, mode selection, and the recursion-guard scenario specifically for Amp loops
- **Crush clink integration (Phase 2 of `clink-aider-crush-amp-support`)**: adds `crush` as a sixth clink target. New manifest at `conf/cli_clients/crush.json`, agent class at `clink/agents/crush.py`, parser at `clink/parsers/crush.py`, plus registry wiring. Invocation via `crush run --quiet` — Crush's documented non-interactive mode (the bare `crush` invocation launches Charm's TUI). Multi-provider via `--model provider/model` syntax (same as opencode), so `clink-runtime-model-selection` mechanics apply directly. Read-only mode is prompt-only + filesystem-snapshot fallback (Crush has no native dry-run flag as of v0.70.0) — identical strategy to opencode. Crush's stored state under `.crush/` is classified as `by_cli_bookkeeping` in `read_only_violations` metadata. Parser is minimal — Crush's `--quiet` output is the model response alone (no preamble, no token footer). 20 new tests under `tests/test_clink_crush_smoke.py` with fixtures captured against crush v0.70.0 + gpt-4o-mini. Default + planner + codereviewer roles reuse shared prompts. `docs/tools/clink.md` Supported CLIs table updated to 6 rows (stability tier: `evolving`)
- **Aider clink integration (Phase 1 of `clink-aider-crush-amp-support`)**: adds `aider` as a fifth clink target alongside Claude, Codex, Gemini, and opencode. New manifest at `conf/cli_clients/aider.json`, agent class at `clink/agents/aider.py`, parser at `clink/parsers/aider.py`, plus registry wiring in `clink/constants.py`, `clink/agents/__init__.py`, and `clink/parsers/__init__.py`. Read-only mode uses Aider's documented `--dry-run` flag (not prompt-only). Prompt delivery uses Phase 0's `message_file` invocation plan because Aider has no stdin scripting mode — its non-interactive interface is `--message-file <path>`. Auto-commits are suppressed (`--no-auto-commits`) so clink-spawned Aider invocations never create surprising side-effect commits. Aider's bookkeeping files (`.aider.chat.history.md`, `.aider.input.history`, `.aider.tags.cache.v4/`) are classified as `by_cli_bookkeeping` rather than `by_model` writes in `read_only_violations` metadata. Runtime model selection via `--model`. 22 new unit + smoke tests under `tests/test_clink_aider_smoke.py` cover the parser against real Aider stdout fixtures captured during the implementation spike, agent class wiring, registry resolution, and end-to-end command-line construction. Default + planner + codereviewer role bundles use the shared prompts. `docs/tools/clink.md` Supported CLIs table updated to 5 rows with stability tiers and per-CLI read-only strategy. Aider users install via `pip install aider-chat`; uses standard provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
- **Cross-cutting clink infrastructure (Phase 0 of `clink-aider-crush-amp-support`)**: prerequisite plumbing for adding Aider, Crush, and Amp as clink targets in subsequent phases. Adds two foundational pieces, neither of which changes behavior for existing agents:
  - **`InvocationPlan` transport hook on `BaseCLIAgent`** (`clink/agents/base.py`): new virtual method `prepare_invocation(prompt, files, images) -> InvocationPlan` lets subclasses declare how to deliver the prompt — `stdin` (default), `argv`, `message_file`, or `stream_json`. Unblocks Aider's `--message-file` requirement and Amp's `--stream-json` image input, neither of which fit the prior stdin-only assumption in `BaseCLIAgent.run()`. Default returns `stdin` plan so existing Claude/Codex/Gemini/opencode agents are byte-identical.
  - **Cross-cutting recursion guard** (`tools/clink.py` + `clink/agents/base.py`): a clink-spawned CLI that itself wires Unison as an MCP server creates a context-window-exploding loop. The guard reads `UNISON_CLINK_DEPTH` at `CLinkTool.execute()` entry, raises `ToolExecutionError` with a clear remediation message when depth exceeds `CLINK_MAX_RECURSION_DEPTH` (default 1), and propagates the incremented depth via `BaseCLIAgent._build_environment()` so the guard fires at the child process boundary. Applies to ALL spawned CLIs, not just MCP-aware ones — Crush and Amp both expose MCP server config, and even Claude/Codex agents can be configured as MCP clients, so a per-CLI guard would have left gaps.
  - **Registry error-message improvement** (`clink/registry.py`): the "CLI 'X' is not supported" rejection now names the known CLIs and points contributors at the three required registry sites (`clink/constants.py` `INTERNAL_DEFAULTS`, `clink/agents/__init__.py` factory, `clink/parsers/__init__.py` parser registry). Without this it was easy to add a manifest + agent + parser but forget the registry wiring and get a silent failure
  - 34 new unit tests under `tests/test_clink_transport.py` and `tests/test_clink_recursion_guard.py` cover each plan kind, recursion-guard depth boundaries, env-var propagation across simulated nested invocation chains, and timeout-cleanup of `message_file` tempfiles
- **README terminal mockup system (`readme-terminal-mockups`)**: replaced the hero video and all five gallery video/image demos with text-source SVG terminal mockups generated from YAML scene files. Nine scenes total (hero, clink-subagent, gallery-chat, gallery-consensus, gallery-precommit, and before/after pairs for gallery-api-lookup and gallery-challenge) emit 18 SVGs (light + dark Catppuccin palette, ~77KB each). New scene-YAML schema documented at `docs/mockup-scenes/README.md` covers seven line kinds (prompt, output, tool_call, tree, status, blank, box) with semantic color tokens that swap palette per theme. Generator at `scripts/build_mockups.py` reads scenes, validates against the schema with file+field error messages, and renders through a shared Jinja2 template at `scripts/build_mockups_template.svg.j2`. JetBrains Mono Regular is checked in alongside its OFL license under `scripts/fonts/` and pre-subsetted into a deterministic `JetBrainsMono-subset.b64` cache; the generator embeds the cached subset directly via `@font-face` and validates that scene glyphs are covered before each run. Hero, clink section, and gallery references in `README.md` all switch to HTML `<picture>` blocks that pick light vs dark via `prefers-color-scheme`. Adds a regenerate-and-diff drift check to `code_quality_checks.sh` (Step 1c) and `.github/workflows/test.yml` so stale SVGs are caught at PR time. 31 new unit tests under `tests/test_build_mockups.py` cover schema validation, layout math, glyph collection, accessibility metadata placement, light/dark layout equality, and cross-process determinism. `docs/contributions.md` gains an "Updating README Mockups" section documenting the regeneration workflow

### Changed

- README restructured to stress the fork's differentiation from PAL. New "What this fork adds over PAL" top callout (after the elevator pitch, before the clink showcase) leads with four bolded categories: CLI-to-CLI orchestration, 75+ providers via opencode, 2000+ auto-discovered models, and production reliability. The former "Differences from PAL MCP" section is renamed to "PAL vs Unison: full comparison" and rewritten as a 9-row table that adds previously-omitted differentiators (clink orchestration, clink read-only mode, opencode provider reach, per-provider circuit breaker) and drops the cosmetic "Changelog format" and "Branding" rows. Migration-is-lossless reassurance carried in both places
- Applied black 26.3.1 formatting to 11 pre-existing files (`simulator_tests/*.py`, several `tests/*.py`, `utils/sqlite_storage.py`) — surfaced by CI once the wheel-packaging fix added `push: main` to the test workflow. Pure mechanical reformat (collapsed wrapped triple-quoted string arguments; removed stray blank line between module docstring and imports); no semantic changes

### Added

- **Clink integration for opencode CLI** (`clink/agents/opencode.py`, `clink/parsers/opencode.py`, `conf/cli_clients/opencode.json`): bridge Unison MCP to the open-source `opencode` AI coding CLI alongside the existing gemini/codex/claude integrations. JSONL event-stream parser, `--agent plan` for read-only enforcement, three role bundles (default, planner, codereviewer)
- **Runtime model selection on the clink tool**: new optional `model` field on `CLinkRequest` lets callers pick a model per call instead of editing JSON manifests. Each agent translates the value into its CLI's flag form (`-m` for opencode/codex, `--model` for claude/gemini). New `BaseCLIAgent.render_model_args()` and `model_flag_aliases` class attribute enable strip-then-append semantics — pre-existing model flags from the manifest are removed before the runtime override is appended, so behavior does not depend on per-CLI flag-precedence rules
- **Optional per-CLI `supported_models` allowlist** in `conf/cli_clients/*.json` manifests: when set, the runtime `model` value is validated against the list and rejected with a clear error if not allowed; when omitted, the model field is forwarded verbatim and CLI-level errors surface in response metadata
- **Wheel smoke test in CI** (`scripts/smoke_test_wheel.py`): builds the wheel, installs it into a fresh venv, spawns the `unison-mcp-server` entry point, and verifies the MCP `initialize` JSON-RPC handshake completes successfully. Runs on every Python version in the test matrix (3.10, 3.11, 3.12), on pull requests AND pushes to `main`. Designed to catch the class of bug that shipped 11.7.0–11.7.2 (installable wheel crashes at import) at CI time rather than at user runtime
- Streaming provider interface: `StreamChunk` dataclass and `ModelProvider.generate_content_stream()` method that yields response chunks incrementally. Default single-chunk wrapper calls `generate_content()` for backward compatibility — zero mandatory per-provider changes
- Native streaming for Gemini provider using `generate_content_stream()` with `stream=True` on the google-genai SDK
- Native streaming for OpenAI-compatible provider using `client.chat.completions.create(stream=True)` — inherited by OpenAI, Azure OpenAI, and xAI subclasses
- MCP progress notification bridge (`utils/streaming.py`): `StreamProgressNotifier` relays streaming chunks to clients via `notifications/progress` with rate limiting (100ms interval / 50-char minimum) and graceful no-op when client doesn't support progress
- Streaming opt-in for long-running workflow tools: `ThinkDeepTool`, `CodeReviewTool`, and `AnalyzeTool` now use `generate_content_stream()` for expert analysis, providing incremental feedback during generation. Per-tool opt-in via `supports_streaming = True` class attribute
- `_generate_stream()` method on `BaseWorkflowMixin` that calls the provider streaming interface, accumulates the full response, and relays chunks to MCP progress notifications
- Comprehensive streaming unit tests (`tests/test_streaming.py`): 14 tests covering default wrapper, native streaming, progress notifier, rate limiting, response assembly, and error handling
- Persistent conversation storage via SQLite backend (`utils/sqlite_storage.py`) — conversations survive server restarts with zero-config setup (sqlite3 is stdlib). Enable with `STORAGE_BACKEND=sqlite`. Features WAL mode for concurrent reads, lazy TTL expiry on read, periodic background sweep, schema migration support, and thread-safe writes. `InMemoryStorage` remains the default for backward compatibility
- Storage backend factory (`create_storage_backend()` in `utils/storage_backend.py`) — selects backend based on `STORAGE_BACKEND` env var with graceful fallback to in-memory on errors or unrecognised values
- Mypy static type checking in CI and local quality checks — strict enforcement on 15 modules across `utils/` and `providers/shared/` with gradual ratchet for expanding coverage. Configured in `pyproject.toml` with per-module overrides and `follow_imports = "silent"` for cross-boundary inference without non-strict noise
- Provider circuit breaker (`utils/circuit_breaker.py`) — three-state pattern (Closed/Open/Half-Open) that detects sustained provider failures and short-circuits requests, avoiding full retry×timeout waits when a provider is hard-down. Configurable via `CIRCUIT_FAILURE_THRESHOLD`, `CIRCUIT_RESET_TIMEOUT_SECONDS`, and `CIRCUIT_HALF_OPEN_MAX_CALLS` environment variables
- `ProviderUnavailable` exception for callers to distinguish circuit-open from transient API errors
- `ModelProvider.get_health_status()` and `ModelProviderRegistry.get_all_health_status()` for provider health diagnostics
- Consensus tool graceful degradation: skips providers with open circuit breakers and synthesizes from available results; returns clear error when all providers are unavailable
- Async provider interface: `ModelProvider.async_generate_content()` wraps the sync method via `asyncio.to_thread()` by default; providers may override with native async. Includes `_run_with_retries_async()` for async retry semantics
- Concurrent consensus dispatch: all model consultations now run in parallel via `asyncio.gather()` on step 1, reducing wall-clock time from sum-of-latencies to max-of-latencies. Per-model 120s timeout and error isolation ensure one slow/failed provider does not block others
- Clink read-only sandbox: `read_only` parameter on the clink tool enforces three-layer defence-in-depth — CLI-specific sandbox flags (`--disallowedTools` for Gemini, `--permission-mode plan` for Claude, `--approval-mode suggest` for Codex), prompt-level read-only instruction, and post-execution filesystem snapshot verification. Violations are reported in metadata (GitHub issues #389, #417)
- Filesystem snapshot utility (`utils/fs_snapshot.py`) for lightweight directory diffing with `.gitignore` and transient file filtering
- `ToolExecutionContext` dataclass (`utils/tool_execution_context.py`) replacing four ad-hoc underscore-prefixed keys in the tool arguments dict with a single typed `_context` object — makes the server-to-tool contract explicit and IDE-discoverable
- Coverage gate in CI and local quality checks via `pytest-cov` with 44% threshold — prevents silent test coverage regression
- Provider-aware token counting: `ModelProvider.count_tokens()` now uses a three-tier fallback (provider-specific tokenizer → litellm → content-aware heuristic) for accurate token budgeting
- `GeminiModelProvider.count_tokens()` override using `litellm.token_counter()`
- Tiktoken encoding cache in `OpenAICompatibleProvider` for faster repeated token counting
- `tiktoken>=0.7.0` as an explicit dependency

### Changed

- Decomposed `server.py` (1,003 → 140 lines) into focused modules: `tools/registry.py` (`ToolRegistry` class with lazy tool instantiation and DISABLED_TOOLS filtering), `handlers/tool_handlers.py` (`list_tools`/`call_tool` handlers + `reconstruct_thread_context`), and `handlers/prompt_handlers.py` (`list_prompts`/`get_prompt` handlers). Server is now a pure wiring module. All backward-compatible re-exports preserved.
- SQLite storage default path changed from `data/conversations.db` (relative to server install) to `.unison/conversations.db` (relative to working directory) — gives per-project conversation isolation so threads from different projects don't mix. Override with `STORAGE_SQLITE_PATH` for a shared/global database
- Decomposed `utils/conversation_memory.py` (1,108 lines) into three focused modules: `conversation_store.py` (thread lifecycle), `context_reconstructor.py` (history building), and `conversation_memory.py` (thin facade with re-exports)
- Eliminated circular dependency: `from server import TOOLS` in conversation memory replaced with `tool_formatter_fn` callback injected by `server.py`
- `ModelContext.estimate_tokens()` now delegates to the resolved provider's `count_tokens()` instead of using a fixed `len(text) // 3` heuristic
- Migrated all `estimate_tokens()` callers in `server.py`, `tools/simple/base.py`, and `utils/context_reconstructor.py` to provider-aware token counting

### Deprecated

- `utils.token_utils.estimate_tokens()` — use `ModelContext.estimate_tokens()` or `provider.count_tokens()` instead. Emits `DeprecationWarning` when called.

### Fixed

- SQLite storage segfault: `get()` and `keys()` now hold `_lock` during execution — the `sqlite3.Connection` object is not thread-safe for concurrent access even with `check_same_thread=False` and WAL mode, causing a CPython-level segfault when a reader and writer thread used the connection simultaneously
- Simulator test registry initialisation: `conversation_base_test.py` now creates and sets a default `ModelProviderRegistry` before calling `configure_providers()`, matching `server.py:main()` startup sequence
- Simulator test chat tool calls: auto-inject `working_directory_absolute_path` for chat tool invocations in both subprocess and in-process test paths
- `per_tool_deduplication` test: switched from subprocess (`call_mcp_tool`) to in-process (`call_mcp_tool_direct`) calls to preserve conversation state across tool invocations
- Quick mode test reporting: only track results for tests that actually run, preventing unrun tests from being reported as failures
- License inconsistency: SECURITY.md and Dockerfile now correctly reference AGPL-3.0 (was Apache 2.0); `pyproject.toml` gains a `license` field

### Changed (infrastructure)

- Renamed project from PAL to Unison
  ([`9304047`](https://github.com/izzoa/unison-mcp-server/commit/9304047))

## v10.0.0 (2026-04-03)

### Changed

- **refactor**: Decompose `server.py` from 1,526 to 962 lines — extract logging setup (`utils/logging_setup.py`), prompt templates (`conf/prompt_templates.py`), request helpers (`utils/request_helpers.py`), model resolution (`utils/model_resolution.py`), and provider configuration (`providers/configure.py`) into dedicated modules
- **refactor**: Decompose `tools/shared/base_tool.py` from 1,606 to 753 lines — extract `FileProcessor`, `ConversationHandler`, `ResponseFormatter`, and `ModelSchemaBuilder` as composed components under `tools/shared/`
- **refactor**: Replace 3x duplicated fallback model resolution pattern with single `resolve_fallback_model()` helper in `utils/model_resolution.py`
- **refactor**: Convert provider registration from two-pass key validation to single-pass data-driven approach with `ProviderSpec` in `providers/configure.py`
- **providers**: Replace string-only error classification with three-tier retry strategy: exception class hierarchy → numeric HTTP status code → string pattern fallback in `providers/base.py`
- **providers**: Gemini and OpenAI provider `_is_error_retryable()` overrides now delegate to `super()` for common fallback instead of duplicating string lists
- **performance**: Convert all f-string `logger.debug()` calls in `reconstruct_thread_context()` to `%s`-style lazy formatting with `isEnabledFor()` guards for clusters
- **testing**: Replace all 41 instances of `ModelProviderRegistry._instance = None` with `reset_for_testing()` across 13 test files

### Added

- `utils/logging_setup.py` — encapsulated logging configuration with `configure_logging()` function
- `conf/prompt_templates.py` — externalized prompt template definitions for all 18 tools
- `utils/request_helpers.py` — follow-up instruction generation utilities
- `utils/model_resolution.py` — `parse_model_option()` and `resolve_fallback_model()` helpers
- `providers/configure.py` — data-driven provider registration with `ProviderSpec` dataclass
- `tools/shared/file_processor.py` — file reading, deduplication, token-budget enforcement, image validation
- `tools/shared/conversation_handler.py` — conversation turn formatting and prompt-size checks
- `tools/shared/response_formatter.py` — response passthrough and parse hooks
- `tools/shared/model_schema_builder.py` — model field JSON schema generation and available models enumeration
- `StorageBackend` protocol in `utils/storage_backend.py` with `reset_storage_backend()` and injectable `get_storage_backend(backend=...)` for test isolation
- `ModelProviderRegistry.create_for_testing(config)` classmethod for config injection without environment variables
- Guarded `httpx`, `google.api_core.exceptions`, and `openai` SDK exception imports for structured error classification
- `_extract_status_code()` helper and `_RETRYABLE_STATUS_CODES`/`_NON_RETRYABLE_STATUS_CODES` class constants on `ModelProvider`
- 116 new unit tests across 5 new test files (`test_file_processor.py`, `test_conversation_handler.py`, `test_response_formatter.py`, `test_model_schema_builder.py`, `test_registry_testability.py`) plus 4 new tests in `test_rate_limit_patterns.py`

### Fixed

- Misleading comments in logging configuration (backupCount and maxBytes values now match actual code)

## v9.8.2 (2025-12-15)

### Fixed

- Allow home subdirectories through is_dangerous_path()
  ([`e5548ac`](https://github.com/izzoa/unison-mcp-server/commit/e5548acb984ca4f8b2ae8381f879a0285094257f))

- Path traversal vulnerability - use prefix matching in is_dangerous_path()
  ([`9ed15f4`](https://github.com/izzoa/unison-mcp-server/commit/9ed15f405a9462b4db7aa44ca2d989e092c008e4))

- Use Path.is_relative_to() for cross-platform dangerous path detection
  ([`91ffb51`](https://github.com/izzoa/unison-mcp-server/commit/91ffb51564e5655ec91111938039ed81e0d8e4c6))

- **security**: Handle macOS symlinked system dirs
  ([`ba08308`](https://github.com/izzoa/unison-mcp-server/commit/ba08308a23d1c1491099c5d0eae548077bd88f9f))

## v9.8.1 (2025-12-15)

### Fixed

- **providers**: Omit store parameter for OpenRouter responses endpoint
  ([`1f8b58d`](https://github.com/izzoa/unison-mcp-server/commit/1f8b58d607c2809b9fa78860718a69207cb66e32))

### Changed

- **tests**: Address code review feedback
  ([`0c3e63c`](https://github.com/izzoa/unison-mcp-server/commit/0c3e63c0c7f1556f4b6686f9c6f30e4bb4a48c7c))

- **tests**: Remove unused setUp method
  ([`b6a8d68`](https://github.com/izzoa/unison-mcp-server/commit/b6a8d682d920c2283724b588818bc1162a865d74))

## v9.8.0 (2025-12-15)

### Added

- Add Claude Opus 4.5 model via OpenRouter
  ([`813ce5c`](https://github.com/izzoa/unison-mcp-server/commit/813ce5c9f7db2910eb12d8c84d3d99f464c430ed))

### Changed

- Add comprehensive test coverage for Opus 4.5 aliases
  ([`cf63fd2`](https://github.com/izzoa/unison-mcp-server/commit/cf63fd25440d599f2ec006bb8cfda5b8a6f61524))

## v9.7.0 (2025-12-15)

### Added

- Re-enable web search for clink codex using correct --enable flag
  ([`e7b9f3a`](https://github.com/izzoa/unison-mcp-server/commit/e7b9f3a5d7e06c690c82b9fd13a93310bcf388ed))

## v9.6.0 (2025-12-15)

### Added

- Support native installed Claude CLI detection
  ([`adc6231`](https://github.com/izzoa/unison-mcp-server/commit/adc6231b98886f0bc35cb04d04d948eba2f0f058))

## v9.5.0 (2025-12-11)

### Fixed

- Grok test
  ([`39c7721`](https://github.com/izzoa/unison-mcp-server/commit/39c77215e5d6892269e523ff25b706dd5671c042))

### Changed

- Cleanup
  ([`74f26e8`](https://github.com/izzoa/unison-mcp-server/commit/74f26e82e7a9c8a0214deef1cb18a3b2fa074050))

- Cleanup
  ([`2b22174`](https://github.com/izzoa/unison-mcp-server/commit/2b221746fee6f7749d8aed8d07a85e428ac8e00f))

- Update subheading
  ([`591287c`](https://github.com/izzoa/unison-mcp-server/commit/591287cb2f442a1fa34cd1139e3a0ad887388e5b))

### Added

- GPT-5.2 support
  ([`8b16405`](https://github.com/izzoa/unison-mcp-server/commit/8b16405f0609e232ff808361dc2a4d8ec258b0f3))

- Grok-4.1 support https://github.com/izzoa/unison-mcp-server/issues/339
  ([`514c9c5`](https://github.com/izzoa/unison-mcp-server/commit/514c9c58fcc91933348d2188ed8c82bbe98132f2))

## v9.4.2 (2025-12-04)

### Fixed

- Rebranding, see [docs/name-change.md](docs/name-change.md) for details
  ([`b2dc849`](https://github.com/izzoa/unison-mcp-server/commit/b2dc84992d70839b29b611178b3871f4922b747f))

## v9.4.1 (2025-11-21)

### Fixed

- Regression https://github.com/izzoa/unison-mcp-server/issues/338
  ([`aceddb6`](https://github.com/izzoa/unison-mcp-server/commit/aceddb655fc36918108b3da1f926bdd4e94875a2))

## v9.4.0 (2025-11-18)

### Fixed

- Failing test for gemini 3.0 pro open router
  ([`19a2a89`](https://github.com/izzoa/unison-mcp-server/commit/19a2a89b12c5dec53aea21a4244aff7796a5e049))

### Added

- Gemini 3.0 Pro Preview for Open Router
  ([`bbfdfac`](https://github.com/izzoa/unison-mcp-server/commit/bbfdfac511668e8ae60f9b9b5d41eb9ab55d74cf))

### Changed

- Enable search on codex CLI
  ([`1579d9f`](https://github.com/izzoa/unison-mcp-server/commit/1579d9f806a653bb04c9c73ab304cdd0e78fbdfa))

## v9.2.2 (2025-11-18)

### Fixed

- **build**: Include clink resources in package
  ([`e9ac1ce`](https://github.com/izzoa/unison-mcp-server/commit/e9ac1ce3354fbb124a72190702618f94266b8459))

## v9.2.1 (2025-11-18)

### Fixed

- **server**: Iterate provider instances during shutdown
  ([`d40fc83`](https://github.com/izzoa/unison-mcp-server/commit/d40fc83d7549293372f3d20cc599a79ec355acef))

## v9.2.0 (2025-11-18)

### Changed

- Streamline advanced usage guide by reorganizing table of contents for improved navigation
  ([`698d391`](https://github.com/izzoa/unison-mcp-server/commit/698d391b26a0dd565eada8bfa6e67e549ce1dd20))

- Update .env.example to include new GPT-5.1 model options and clarify existing model descriptions
  ([`dbbfef2`](https://github.com/izzoa/unison-mcp-server/commit/dbbfef292c67ed54f90f7612c9c14d4095bd6c45))

- Update advanced usage and configuration to include new GPT-5.1 models and enhance tool parameters
  ([`807c9df`](https://github.com/izzoa/unison-mcp-server/commit/807c9df70e3b54031ec6beea10f3975455b36dfb))

### Added

- Add new GPT-5.1 models to configuration files and update model selection logic in OpenAI provider
  ([`8e9aa23`](https://github.com/izzoa/unison-mcp-server/commit/8e9aa2304d5e9ea9a9f8dc2a13a27a1ced6b1608))

- Enhance model support by adding GPT-5.1 to .gitignore and updating cassette maintenance
  documentation for dual-model testing
  ([`f713d8a`](https://github.com/izzoa/unison-mcp-server/commit/f713d8a354a37c32a806c98994e6f949ecd64237))

## v9.1.4 (2025-11-18)

### Fixed

- Replaced deprecated Codex web search configuration
  ([`2ec64ba`](https://github.com/izzoa/unison-mcp-server/commit/2ec64ba7489acc586846b25eedf94a4f05d5bd2d))

## v9.1.3 (2025-10-22)

### Fixed

- Reduced token usage, removed parameters from schema that CLIs never seem to use
  ([`3e27319`](https://github.com/izzoa/unison-mcp-server/commit/3e27319e60b0287df918856b58b2bbf042c948a8))

- Telemetry option no longer available in gemini 0.11
  ([`2a8dff0`](https://github.com/izzoa/unison-mcp-server/commit/2a8dff0cc8a3f33111533cdb971d654637ed0578))

### Changed

- Improved precommit system prompt
  ([`3efff60`](https://github.com/izzoa/unison-mcp-server/commit/3efff6056e322ee1531d7bed5601038c129a8b29))

## v9.1.2 (2025-10-21)

### Fixed

- Configure codex with a longer timeout
  ([`d2773f4`](https://github.com/izzoa/unison-mcp-server/commit/d2773f488af28986632846652874de9ff633049c))

- Handle claude's array style JSON https://github.com/izzoa/unison-mcp-server/issues/295
  ([`d5790a9`](https://github.com/izzoa/unison-mcp-server/commit/d5790a9bfef719f03d17f2d719f1882e55d13b3b))

## v9.1.1 (2025-10-17)

### Fixed

- Failing test
  ([`aed3e3e`](https://github.com/izzoa/unison-mcp-server/commit/aed3e3ee80c440ac8ab0d4abbf235b84df723d18))

- Handler for parsing multiple generated code blocks
  ([`f4c20d2`](https://github.com/izzoa/unison-mcp-server/commit/f4c20d2a20e1c57d8b10e8f508e07e2a8d72f94a))

- Improved error reporting; codex cli would at times fail to figure out how to handle plain-text /
  JSON errors
  ([`95e69a7`](https://github.com/izzoa/unison-mcp-server/commit/95e69a7cb234305dcd37dcdd2f22be715922e9a8))

## v9.1.0 (2025-10-17)

### Added

- Enhance review prompts to emphasize static analysis
  ([`36e66e2`](https://github.com/izzoa/unison-mcp-server/commit/36e66e2e9a44a73a466545d4d3477ecb2cb3e669))

## v9.0.3 (2025-10-16)

### Fixed

- Remove duplicate -o json flag in gemini CLI config
  ([`3b2eff5`](https://github.com/izzoa/unison-mcp-server/commit/3b2eff58ac0e2388045a7442c63f56ce259b54ba))

## v9.0.2 (2025-10-15)

### Fixed

- Update Claude CLI commands to new mcp syntax
  ([`a2189cb`](https://github.com/izzoa/unison-mcp-server/commit/a2189cb88a295ebad6268b9b08c893cd65bc1d89))

## v9.0.1 (2025-10-14)

### Fixed

- Add JSON output flag to gemini CLI configuration
  ([`eb3dff8`](https://github.com/izzoa/unison-mcp-server/commit/eb3dff845828f60ff2659586883af622b8b035eb))

## v9.0.0 (2025-10-08)

### Added

- Claude Code as a CLI agent now supported. Mix and match: spawn claude code from within claude
  code, or claude code from within codex.
  ([`4cfaa0b`](https://github.com/izzoa/unison-mcp-server/commit/4cfaa0b6060769adfbd785a072526a5368421a73))

## v8.0.2 (2025-10-08)

### Fixed

- Restore run-server quote trimming regex
  ([`1de4542`](https://github.com/izzoa/unison-mcp-server/commit/1de454224c105891137134e2a25c2ee4f00dba45))

## v8.0.1 (2025-10-08)

### Fixed

- Resolve executable path for cross-platform compatibility in CLI agent
  ([`f98046c`](https://github.com/izzoa/unison-mcp-server/commit/f98046c2fccaa7f9a24665a0d705a98006461da5))

### Changed

- Fix clink agent tests to mock shutil.which() for executable resolution
  ([`4370be3`](https://github.com/izzoa/unison-mcp-server/commit/4370be33b4b69a40456527213bcd62321a925a57))

## v7.8.1 (2025-10-07)

### Fixed

- Updated model description to fix test
  ([`04f7ce5`](https://github.com/izzoa/unison-mcp-server/commit/04f7ce5b03804564263f53a765931edba9c320cd))

### Changed

- Moved registries into a separate module and code cleanup
  ([`7c36b92`](https://github.com/izzoa/unison-mcp-server/commit/7c36b9255a13007a10af4fadefc21aadfce482b0))

## v7.8.0 (2025-10-07)

### Changed

- Consensus video
  ([`2352684`](https://github.com/izzoa/unison-mcp-server/commit/23526841922a73c68094e5205e19af04a1f6c8cc))

- Formatting
  ([`7d7c74b`](https://github.com/izzoa/unison-mcp-server/commit/7d7c74b5a38b7d1adf132b8e28034017df7aa852))

- Link to videos from main page
  ([`e8ef193`](https://github.com/izzoa/unison-mcp-server/commit/e8ef193daba393b55a3beaaba49721bb9182378a))

- Update README.md
  ([`7b13543`](https://github.com/izzoa/unison-mcp-server/commit/7b13543824fc0af729daf753ecdddba9ee7d9f1e))

### Added

- All native providers now read from catalog files like OpenRouter / Custom configs. Allows for
  greater control over the capabilities
  ([`2a706d5`](https://github.com/izzoa/unison-mcp-server/commit/2a706d5720c0bf97b71c3e0fc95c15f78015bedf))

- Provider cleanup
  ([`9268dda`](https://github.com/izzoa/unison-mcp-server/commit/9268ddad2a07306351765b47098134512739f49f))

- New base class for model registry / loading
  ([`02d13da`](https://github.com/izzoa/unison-mcp-server/commit/02d13da897016d7491b4a10a1195983385d66654))

## v7.7.0 (2025-10-07)

### Changed

- Video
  ([`ed5dda7`](https://github.com/izzoa/unison-mcp-server/commit/ed5dda7c5a9439c2835cc69d76e6377169ad048a))

### Added

- More aliases
  ([`5f0aaf5`](https://github.com/izzoa/unison-mcp-server/commit/5f0aaf5f69c9d188d817b5ffbf6738c61da40ec7))

## v7.6.0 (2025-10-07)

### Changed

- Info about AI client timeouts
  ([`3ddfed5`](https://github.com/izzoa/unison-mcp-server/commit/3ddfed5ef09000791e1c94b041c43dc273ed53a8))

### Added

- Add support for openai/gpt-5-pro model
  ([`abed075`](https://github.com/izzoa/unison-mcp-server/commit/abed075b2eaa99e9618202f47ff921094baae952))

## v7.5.2 (2025-10-06)

### Fixed

- Handle 429 response https://github.com/izzoa/unison-mcp-server/issues/273
  ([`cbe1d79`](https://github.com/izzoa/unison-mcp-server/commit/cbe1d7993276bd014b495cbd2d0ece1f5d7583d9))

## v7.5.0 (2025-10-06)

### Changed

- Video
  ([`775e4d5`](https://github.com/izzoa/unison-mcp-server/commit/775e4d50b826858095c5f2a61a07fc01c4a00816))

- Videos
  ([`bb2066c`](https://github.com/izzoa/unison-mcp-server/commit/bb2066c909f6581ba40fc5ddef3870954ae553ab))

### Added

- Support for GPT-5-Pro highest reasoning model
  https://github.com/izzoa/unison-mcp-server/issues/275
  ([`a65485a`](https://github.com/izzoa/unison-mcp-server/commit/a65485a1e52fc79739000426295a27d096f4c9d8))

## v7.4.0 (2025-10-06)

### Added

- Improved prompt
  ([`b1e9963`](https://github.com/izzoa/unison-mcp-server/commit/b1e9963991a41dff082ec1dce5691c318f105e6d))

## v7.3.0 (2025-10-06)

### Changed

- Fixed typo
  ([`3ab0aa8`](https://github.com/izzoa/unison-mcp-server/commit/3ab0aa8314ad5992bcb00de549a0fab2e522751d))

- Fixed typo
  ([`c17ce3c`](https://github.com/izzoa/unison-mcp-server/commit/c17ce3cf958d488b97fa7127942542ab514b58bd))

- Update apilookup.md
  ([`1918679`](https://github.com/izzoa/unison-mcp-server/commit/19186794edac4fce5523e671310aecff4cbfdc81))

- Update README.md
  ([`23c6c78`](https://github.com/izzoa/unison-mcp-server/commit/23c6c78bf152ede6e7b5f7b7770b12a8442845a3))

### Added

- Codex supports web-search natively but needs to be turned on, run-server script asks if the user
  would like this done
  ([`97ba7e4`](https://github.com/izzoa/unison-mcp-server/commit/97ba7e44ce7e3fd874759514ed2f0738033fc801))

## v7.2.0 (2025-10-06)

### Changed

- Updated
  ([`bb57f71`](https://github.com/izzoa/unison-mcp-server/commit/bb57f719666ab6a586d835688ff8086282a5a0dc))

### Added

- New tool to perform apilookup (latest APIs / SDKs / language features etc)
  https://github.com/izzoa/unison-mcp-server/issues/204
  ([`5bea595`](https://github.com/izzoa/unison-mcp-server/commit/5bea59540f58b3c45044828c10f131aed104dd1c))

- De-duplicate roles to avoid explosion when more CLIs get added
  ([`c42e9e9`](https://github.com/izzoa/unison-mcp-server/commit/c42e9e9c34d7ae4732e2e4fbed579b681a6d170d))

## v7.1.1 (2025-10-06)

### Fixed

- Clink missing in toml
  ([`1ff77fa`](https://github.com/izzoa/unison-mcp-server/commit/1ff77faa800ad6c2dde49cad98dfa72035fe1c81))

### Changed

- Example for codex cli
  ([`344c42b`](https://github.com/izzoa/unison-mcp-server/commit/344c42bcbfb543bfd05cbc27fd5b419c76b77954))

- Example for codex cli
  ([`c3044de`](https://github.com/izzoa/unison-mcp-server/commit/c3044de7424e638dde5c8ec49adb6c3c7c5a60b2))

- Update README.md
  ([`2e719ae`](https://github.com/izzoa/unison-mcp-server/commit/2e719ae35e7979f7b83bd910867e79863a7f9ceb))

## v7.1.0 (2025-10-05)

### Added

- Support for codex as external CLI
  ([`561e4aa`](https://github.com/izzoa/unison-mcp-server/commit/561e4aaaa8a89eb89c03985b9e7720cc98ef666c))

## v7.0.1 (2025-10-05)

### Fixed

- --yolo needed for running shell commands, documentation added
  ([`15ae3f2`](https://github.com/izzoa/unison-mcp-server/commit/15ae3f24babccf42f43be5028bf8c60c05a6beaf))

### Changed

- Updated intro
  ([`fb668c3`](https://github.com/izzoa/unison-mcp-server/commit/fb668c39b5f6e3dd37f7027f953f6004f258f2bf))

## v7.0.0 (2025-10-05)

### Changed

- Instructions for OpenCode
  ([`bd66622`](https://github.com/izzoa/unison-mcp-server/commit/bd666227c8f7557483f7e24fb8544fc0456600dc))

- Updated intro
  ([`615873c`](https://github.com/izzoa/unison-mcp-server/commit/615873c3db2ecf5ce6475caa3445e1da9a2517bd))

### Added

- Huge update - Link another CLI (such as `gemini` directly from with Claude Code / Codex).
  https://github.com/izzoa/unison-mcp-server/issues/208
  ([`a2ccb48`](https://github.com/izzoa/unison-mcp-server/commit/a2ccb48e9a5080a75dbfd483b5f09fc719c887e5))

- Fixed test
  ([`9c99b9b`](https://github.com/izzoa/unison-mcp-server/commit/9c99b9b35219f54db8d7be0958d4390a106631ae))

- Include file modification dates too
  ([`47973e9`](https://github.com/izzoa/unison-mcp-server/commit/47973e945efa2cdbdb8f3404d467d7f1abc62b0a))

## v6.1.0 (2025-10-04)

### Changed

- Updated intro
  ([`aa65394`](https://github.com/izzoa/unison-mcp-server/commit/aa6539472c4ddf1c3c1bac446fdee03e75e1cb50))

### Added

- Support for Qwen Code
  ([`fe9968b`](https://github.com/izzoa/unison-mcp-server/commit/fe9968b633d0312b82426e9ebddfe1d6515be3c5))

## v6.0.0 (2025-10-04)

### Changed

- Updated
  ([`e91ed2a`](https://github.com/izzoa/unison-mcp-server/commit/e91ed2a924b1702edf9e1417479ac0dee0ca1553))

### Added

- Azure OpenAI / Azure AI Foundry support. Models should be defined in conf/azure_models.json (or a
  custom path). See .env.example for environment variables or see readme.
  https://github.com/izzoa/unison-mcp-server/issues/265
  ([`ff9a07a`](https://github.com/izzoa/unison-mcp-server/commit/ff9a07a37adf7a24aa87c63b3ba9db88bdff467b))

- Breaking change - OpenRouter models are now read from conf/openrouter_models.json while Custom /
  Self-hosted models are read from conf/custom_models.json
  ([`ff9a07a`](https://github.com/izzoa/unison-mcp-server/commit/ff9a07a37adf7a24aa87c63b3ba9db88bdff467b))

- OpenAI/compatible models (such as Azure OpenAI) can declare if they use the response API instead
  via `use_openai_responses_api`
  ([`3824d13`](https://github.com/izzoa/unison-mcp-server/commit/3824d131618683572e9e8fffa6b25ccfabf4cf50))

- OpenRouter / Custom Models / Azure can separately also use custom config paths now (see
  .env.example )
  ([`ff9a07a`](https://github.com/izzoa/unison-mcp-server/commit/ff9a07a37adf7a24aa87c63b3ba9db88bdff467b))

- Breaking change: `is_custom` property has been removed from model_capabilities.py (and thus
  custom_models.json) given each models are now read from separate configuration files
  ([`ff9a07a`](https://github.com/izzoa/unison-mcp-server/commit/ff9a07a37adf7a24aa87c63b3ba9db88bdff467b))

- Model registry class made abstract, OpenRouter / Custom Provider / Azure OpenAI now subclass these
  ([`ff9a07a`](https://github.com/izzoa/unison-mcp-server/commit/ff9a07a37adf7a24aa87c63b3ba9db88bdff467b))

## v5.22.0 (2025-10-04)

### Fixed

- CI test
  ([`bc93b53`](https://github.com/izzoa/unison-mcp-server/commit/bc93b5343bbd8657b95ab47c00a2cb99a68a009f))

- Listmodels to always honor restricted models
  ([`4015e91`](https://github.com/izzoa/unison-mcp-server/commit/4015e917ed32ae374ec6493b74993fcb34f4a971))

### Added

- Centralized environment handling, ensures UNISON_MCP_FORCE_ENV_OVERRIDE is honored correctly
  ([`2c534ac`](https://github.com/izzoa/unison-mcp-server/commit/2c534ac06e4c6078b96781dfb55c5759b982afe8))

### Changed

- Don't retry on 429
  ([`d184024`](https://github.com/izzoa/unison-mcp-server/commit/d18402482087f52b7bd07755c9304ed00ed20592))

- Improved retry logic and moved core logic to base class
  ([`f955100`](https://github.com/izzoa/unison-mcp-server/commit/f955100f3a82973ccd987607e1d8a1bbe07828c8))

- Removed subclass override when the base class should be resolving the model name
  ([`06d7701`](https://github.com/izzoa/unison-mcp-server/commit/06d7701cc3ee09732ab713fa9c7c004199154483))

## v5.19.0 (2025-10-03)

### Fixed

- Add GPT-5-Codex to Responses API routing and simplify comments
  ([`82b021d`](https://github.com/izzoa/unison-mcp-server/commit/82b021d75acc791e68c7afb35f6492f68cf02bec))

### Changed

- Bumped defaults
  ([`95d98a9`](https://github.com/izzoa/unison-mcp-server/commit/95d98a9bc0a5bafadccb9f6d1e4eda97a0dd2ce7))

### Added

- Add GPT-5-Codex support with Responses API integration
  ([`f265342`](https://github.com/izzoa/unison-mcp-server/commit/f2653427ca829368e7145325d20a98df3ee6d6b4))

- Cross tool memory recall, testing continuation via cassette recording
  ([`88493bd`](https://github.com/izzoa/unison-mcp-server/commit/88493bd357c6a12477c3160813100dae1bc46493))

## v5.18.3 (2025-10-03)

### Fixed

- External model name now recorded properly in responses
  ([`d55130a`](https://github.com/izzoa/unison-mcp-server/commit/d55130a430401e106cd86f3e830b3d756472b7ff))

### Changed

- Updated docs
  ([`b4e5090`](https://github.com/izzoa/unison-mcp-server/commit/b4e50901ba60c88137a29d00ecf99718582856d3))

- Generic name for the CLI agent
  ([`e9b6947`](https://github.com/izzoa/unison-mcp-server/commit/e9b69476cd922c12931d62ccc3be9082bbbf6014))

- Generic name for the CLI agent
  ([`7a6fa0e`](https://github.com/izzoa/unison-mcp-server/commit/7a6fa0e77a8c4a682dc11c9bbb16bdaf86d9edf4))

- Generic name for the CLI agent
  ([`b692da2`](https://github.com/izzoa/unison-mcp-server/commit/b692da2a82facce7455b8f2ec0108e1db84c07c3))

- Generic name for the CLI agent
  ([`f76ebbf`](https://github.com/izzoa/unison-mcp-server/commit/f76ebbf280cc78ffcfe17cb4590aeaa231db8aa1))

- Generic name for the CLI agent
  ([`c05913a`](https://github.com/izzoa/unison-mcp-server/commit/c05913a09e53e195b9a108647c09c061ced19d17))

- Generic name for the CLI agent
  ([`0dfaa63`](https://github.com/izzoa/unison-mcp-server/commit/0dfaa6312ed95ac3d1ae0032334ae1286871b15e))

- Fixed integration tests, removed magicmock
  ([`87ccb6b`](https://github.com/izzoa/unison-mcp-server/commit/87ccb6b25ba32a3cb9c4cc64fc0e96294f492c04))

## v5.18.2 (2025-10-02)

### Fixed

- Https://github.com/izzoa/unison-mcp-server/issues/194
  ([`8b3a286`](https://github.com/izzoa/unison-mcp-server/commit/8b3a2867fb83eccb3a8e8467e7e3fc5b8ebe1d0c))

## v5.18.0 (2025-10-02)

### Added

- Added `intelligence_score` to the model capabilities schema; a 1-20 number that can be specified
  to influence the sort order of models presented to the CLI in `auto selection` mode
  ([`6cab9e5`](https://github.com/izzoa/unison-mcp-server/commit/6cab9e56fc5373da5c11d4545bcb85371d4803a4))

## v5.17.1 (2025-10-02)

### Fixed

- Baseclass should return MODEL_CAPABILITIES
  ([`82a03ce`](https://github.com/izzoa/unison-mcp-server/commit/82a03ce63f28fece17bfc1d70bdb75aadec4c6bb))

### Changed

- Document custom timeout values
  ([`218fbdf`](https://github.com/izzoa/unison-mcp-server/commit/218fbdf49cb90f2353f58bbaef567519dd876634))

- Clean temperature inference
  ([`9c11ecc`](https://github.com/izzoa/unison-mcp-server/commit/9c11ecc4bf37562aa08dc3ecfa70f380e0ead357))

- Cleanup
  ([`6ec2033`](https://github.com/izzoa/unison-mcp-server/commit/6ec2033f34c74ad139036de83a34cf6d374db77b))

- Cleanup provider base class; cleanup shared responsibilities; cleanup public contract
  ([`693b84d`](https://github.com/izzoa/unison-mcp-server/commit/693b84db2b87271ac809abcf02100eee7405720b))

- Cleanup token counting
  ([`7fe9fc4`](https://github.com/izzoa/unison-mcp-server/commit/7fe9fc49f8e3cd92be4c45a6645d5d4ab3014091))

- Code cleanup
  ([`bb138e2`](https://github.com/izzoa/unison-mcp-server/commit/bb138e2fb552f837b0f9f466027580e1feb26f7c))

- Code cleanup
  ([`182aa62`](https://github.com/izzoa/unison-mcp-server/commit/182aa627dfba6c578089f83444882cdd2635a7e3))

- Moved image related code out of base provider into a separate utility
  ([`14a35af`](https://github.com/izzoa/unison-mcp-server/commit/14a35afa1d25408e62b968d9846be7bffaede327))

- Moved temperature method from base provider to model capabilities
  ([`6d237d0`](https://github.com/izzoa/unison-mcp-server/commit/6d237d09709f757a042baf655f47eb4ddfc078ad))

- Moved temperature method from base provider to model capabilities
  ([`f461cb4`](https://github.com/izzoa/unison-mcp-server/commit/f461cb451953f882bbde096a9ecf0584deb1dde8))

- Removed hard coded checks, use model capabilities instead
  ([`250545e`](https://github.com/izzoa/unison-mcp-server/commit/250545e34f8d4f8026bfebb3171f3c2bc40f4692))

- Removed hook from base class, turned into helper static method instead
  ([`2b10adc`](https://github.com/izzoa/unison-mcp-server/commit/2b10adcaf2b8741f0da5de84cc3483eae742a014))

- Removed method from provider, should use model capabilities instead
  ([`a254ff2`](https://github.com/izzoa/unison-mcp-server/commit/a254ff2220ba00ec30f5110c69a4841419917382))

- Renaming to reflect underlying type
  ([`1dc25f6`](https://github.com/izzoa/unison-mcp-server/commit/1dc25f6c3d4cdbf01f041cc424e3b5235c23175b))

## v5.17.0 (2025-10-02)

### Fixed

- Use types.HttpOptions from module imports instead of local import
  ([`956e8a6`](https://github.com/izzoa/unison-mcp-server/commit/956e8a6927837f5c7f031a0db1dd0b0b5483c626))

### Changed

- Apply Black formatting to use double quotes
  ([`33ea896`](https://github.com/izzoa/unison-mcp-server/commit/33ea896c511764904bf2b6b22df823928f88a148))

### Added

- Add custom Gemini endpoint support
  ([`462bce0`](https://github.com/izzoa/unison-mcp-server/commit/462bce002e2141b342260969588e69f55f8bb46a))

- Simplify Gemini provider initialization using kwargs dict
  ([`023940b`](https://github.com/izzoa/unison-mcp-server/commit/023940be3e38a7eedbc8bf8404a4a5afc50f8398))

## v5.16.0 (2025-10-01)

### Fixed

- Resolve logging timing and import organization issues
  ([`d34c299`](https://github.com/izzoa/unison-mcp-server/commit/d34c299f02a233af4f17bdcc848219bf07799723))

### Changed

- Fix ruff import sorting issue
  ([`4493a69`](https://github.com/izzoa/unison-mcp-server/commit/4493a693332e0532d04ad3634de2a2f5b1249b64))

### Added

- Add configurable environment variable override system
  ([`93ce698`](https://github.com/izzoa/unison-mcp-server/commit/93ce6987b6e7d8678ffa5ac51f5106a7a21ce67b))

## v5.15.0 (2025-10-01)

### Added

- Depending on the number of tools in use, this change should save ~50% of overall tokens used.
  fixes https://github.com/izzoa/unison-mcp-server/issues/255 but also refactored
  individual tools to instead encourage the agent to use the listmodels tool if needed.
  ([`d9449c7`](https://github.com/izzoa/unison-mcp-server/commit/d9449c7bb607caff3f0454f210ddfc36256c738a))

### Changed

- Tweaks to schema descriptions, aiming to reduce token usage without performance degradation
  ([`cc8a4df`](https://github.com/izzoa/unison-mcp-server/commit/cc8a4dfd21b6f3dae4972a833b619e53c964693b))

- Trimmed some prompts
  ([`f69ff03`](https://github.com/izzoa/unison-mcp-server/commit/f69ff03c4d10e606a1dfed2a167f3ba2e2236ba8))

## v5.14.1 (2025-10-01)

### Fixed

- Https://github.com/izzoa/unison-mcp-server/issues/258
  ([`696b45f`](https://github.com/izzoa/unison-mcp-server/commit/696b45f25e80faccb67034254cf9a8fc4c643dbd))

## v5.14.0 (2025-10-01)

### Added

- Add Claude Sonnet 4.5 and update alias configuration
  ([`95c4822`](https://github.com/izzoa/unison-mcp-server/commit/95c4822af2dc55f59c0e4ed9454673d6ca964731))

### Changed

- Update tests to match new Claude Sonnet 4.5 alias configuration
  ([`7efb409`](https://github.com/izzoa/unison-mcp-server/commit/7efb4094d4eb7db006340d3d9240b9113ac25cd3))

## v5.13.0 (2025-10-01)

### Fixed

- Add sonnet alias for Claude Sonnet 4.1 to match opus/haiku pattern
  ([`dc96344`](https://github.com/izzoa/unison-mcp-server/commit/dc96344db043e087ee4f8bf264a79c51dc2e0b7a))

- Missing "optenai/" in name
  ([`7371ed6`](https://github.com/izzoa/unison-mcp-server/commit/7371ed6487b7d90a1b225a67dca2a38c1a52f2ad))

### Added

- Add comprehensive GPT-5 series model support
  ([`4930824`](https://github.com/izzoa/unison-mcp-server/commit/493082405237e66a2f033481a5f8bf8293b0d553))

## v5.12.1 (2025-10-01)

### Fixed

- Resolve consensus tool model_context parameter missing issue
  ([`9044b63`](https://github.com/izzoa/unison-mcp-server/commit/9044b63809113047fe678d659e4fcd175f58e87a))

### Changed

- Fix trailing whitespace in consensus.py
  ([`0760b31`](https://github.com/izzoa/unison-mcp-server/commit/0760b31f8a6d03c4bea3fd2a94dfbbfab0ad5079))

- Optimize ModelContext creation in consensus tool
  ([`30a8952`](https://github.com/izzoa/unison-mcp-server/commit/30a8952fbccd22bebebd14eb2c8005404b79bcd6))

## v5.12.0 (2025-10-01)

### Fixed

- Removed use_websearch; this parameter was confusing Codex. It started using this to prompt the
  external model to perform searches! web-search is enabled by Claude / Codex etc by default and the
  external agent can ask claude to search on its behalf.
  ([`cff6d89`](https://github.com/izzoa/unison-mcp-server/commit/cff6d8998f64b73265c4e31b2352462d6afe377f))

### Added

- Implement semantic cassette matching for o3 models
  ([`70fa088`](https://github.com/izzoa/unison-mcp-server/commit/70fa088c32ac4e6153d5e7b30a3e32022be2f908))

## v5.11.1 (2025-10-01)

### Fixed

- Remove duplicate OpenAI models from listmodels output
  ([`c29e762`](https://github.com/izzoa/unison-mcp-server/commit/c29e7623ace257eb45396cdf8c19e1659e29edb9))

### Changed

- Update OpenAI provider alias tests to match new format
  ([`d13700c`](https://github.com/izzoa/unison-mcp-server/commit/d13700c14c7ee3d092302837cb1726d17bab1ab8))

## v5.11.0 (2025-08-26)

### Added

- Codex CLI support
  ([`ce56d16`](https://github.com/izzoa/unison-mcp-server/commit/ce56d16240ddcc476145a512561efe5c66438f0d))

## v5.10.3 (2025-08-24)

### Fixed

- Address test failures and PR feedback
  ([`6bd9d67`](https://github.com/izzoa/unison-mcp-server/commit/6bd9d6709acfb584ab30a0a4d6891cabdb6d3ccf))

- Resolve temperature handling issues for O3/custom models
  ([#245](https://github.com/izzoa/unison-mcp-server/pull/245),
  [`3b4fd88`](https://github.com/izzoa/unison-mcp-server/commit/3b4fd88d7e9a3f09fea616a10cb3e9d6c1a0d63b))

## v5.10.2 (2025-08-24)

### Fixed

- Another fix for https://github.com/izzoa/unison-mcp-server/issues/251
  ([`a07036e`](https://github.com/izzoa/unison-mcp-server/commit/a07036e6805042895109c00f921c58a09caaa319))

## v5.10.0 (2025-08-22)

### Added

- Refactored and tweaked model descriptions / schema to use fewer tokens at launch (average
  reduction per field description: 60-80%) without sacrificing tool effectiveness
  ([`4b202f5`](https://github.com/izzoa/unison-mcp-server/commit/4b202f5d1d24cea1394adab26a976188f847bd09))

## v5.9.0 (2025-08-21)

### Changed

- Update instructions for precommit
  ([`90821b5`](https://github.com/izzoa/unison-mcp-server/commit/90821b51ff653475d9fb1bc70b57951d963e8841))

### Added

- Refactored and improved codereview in line with precommit. Reviews are now either external
  (default) or internal. Takes away anxiety and loss of tokens when Claude incorrectly decides to be
  'confident' about its own changes and bungle things up.
  ([`80d21e5`](https://github.com/izzoa/unison-mcp-server/commit/80d21e57c0246762c0a306ede5b93d6aeb2315d8))

- Minor prompt tweaks
  ([`d30c212`](https://github.com/izzoa/unison-mcp-server/commit/d30c212029c05b767d99b5391c1dd4cee78ef336))

## v5.8.6 (2025-08-20)

### Fixed

- Escape backslashes in TOML regex pattern
  ([`1c973af`](https://github.com/izzoa/unison-mcp-server/commit/1c973afb002650b9bbee8a831b756bef848915a1))

- Establish version 5.8.6 and add version sync automation
  ([`90a4195`](https://github.com/izzoa/unison-mcp-server/commit/90a419538128b54fbd30da4b8a8088ac59f8c691))

- Restore proper version 5.8.6
  ([`340b58f`](https://github.com/izzoa/unison-mcp-server/commit/340b58f2e790b84c3736aa96df7f6f5f2d6a13c9))

## v1.1.0 (2025-08-20)

### Added

- Improvements to precommit
  ([`2966dcf`](https://github.com/izzoa/unison-mcp-server/commit/2966dcf2682feb7eef4073738d0c225a44ce0533))

## v1.0.0 (2025-08-20)

- Initial Release
