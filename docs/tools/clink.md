# Clink Tool - CLI-to-CLI Bridge

**Spawn AI subagents, connect external CLIs, orchestrate isolated contexts – all without leaving your session**

The `clink` tool transforms your CLI into a multi-agent orchestrator. Launch isolated Codex instances from _within_ Codex, delegate to Gemini's 1M context, or run specialized Claude agents—all while preserving conversation continuity. Instead of context-switching or token bloat, spawn fresh subagents that handle complex tasks in isolation and return only the results you need.

> **CAUTION**: Clink launches real CLI agents with relaxed permission flags (Gemini ships with `--yolo`, Codex with `--dangerously-bypass-approvals-and-sandbox`, Claude with `--permission-mode acceptEdits`) so they can edit files and run tools autonomously via MCP. If that’s more access than you want, remove those flags—the CLI can still open/read files and report findings, it just won’t auto-apply edits. You can also tighten role prompts or system prompts with stop-words/guardrails, or disable clink entirely. Otherwise, keep the shipped presets confined to workspaces you fully trust.

## Why Use Clink (CLI + Link)?

### Codex-within-Codex: The Ultimate Context Management

**The Problem**: You're deep in a Codex session debugging authentication. Now you need a comprehensive security audit, but that'll consume 50K tokens of context you can't spare.

**The Solution**: Spawn a fresh Codex subagent in an isolated context:
```bash
clink with codex codereviewer to audit auth/ for OWASP Top 10 vulnerabilities
```

The subagent:
- Launches in a **pristine context** with full token budget
- Performs deep analysis using its own MCP tools and web search
- Returns **only the final security report** (not intermediate steps)
- Your main session stays **laser-focused** on debugging

**Works with any supported CLI**: Codex can spawn Codex / Claude Code / Gemini CLI subagents, or mix and match between different CLIs.

---

### Cross-CLI Orchestration

**Scenario 1**: You're in Codex and need Gemini's 1M context window to analyze a massive legacy codebase.

**Without clink**: Open new terminal → run `gemini` → lose conversation context → manually copy/paste findings → context mismatch hell.

**With clink**: `"clink with gemini to map dependencies across this 500-file monorepo"` – Gemini processes, returns insights, conversation flows seamlessly.

**Scenario 2**: Use [`consensus`](consensus.md) to debate features with multiple models, then hand off to Gemini for implementation.

```
"Use consensus with pro and gpt5 to decide whether to add dark mode or offline support next"
[consensus runs, models deliberate, recommendation emerges]

Use continuation with clink - implement the recommended feature
```

Gemini receives the full conversation context from `consensus` including the consensus prompt + replies, understands the chosen feature, technical constraints discussed, and can start implementation immediately. No re-explaining, no context loss - true conversation continuity across tools and models.

## Key Features

- **Stay in one CLI**: No switching between terminal sessions or losing context
- **Full conversation continuity**: Gemini's responses participate in the same conversation thread
- **Role-based prompts**: Pre-configured roles for planning, code review, or general questions
- **Full CLI capabilities**: Gemini can use its own web search, file tools, and latest features
- **Token efficiency**: File references (not full content) to conserve tokens
- **Cross-tool collaboration**: Combine with other Unison tools like `planner` → `clink` → `codereview`
- **Free tier available**: Gemini offers 1,000 requests/day free with a personal Google account - great for cost savings across tools

## Available Roles

**Default Role** - General questions, summaries, quick answers
```
Use clink to ask gemini about the latest React 19 features
```

**Planner Role** - Strategic planning with multi-phase approach
```
clink with gemini with planner role to map out our microservices migration strategy
```

**Code Reviewer Role** - Focused code analysis with severity levels
```
Use clink codereviewer role to review auth.py for security issues
```

You can make your own custom roles in `conf/cli_clients/` or tweak any of the shipped presets.

## Tool Parameters

- `prompt`: Your question or task for the external CLI (required)
- `cli_name`: Which CLI to use - `aider`, `amp`, `claude`, `codex`, `copilot`, `crush`, `gemini`, `opencode`, or add your own in `conf/cli_clients/`. **Required** when more than one CLI is configured; it may be omitted only in a single-CLI deployment, where that CLI is used. There is no vendor-preferred default.
- `role`: Preset role - `default`, `planner`, `codereviewer` (default: `default`)
- `model`: Optional model forwarded to the CLI (e.g. `sonnet` for claude, `provider/model` for opencode/crush, named modes for amp)
- `working_dir`: Optional **absolute path** the spawned CLI runs in. Pass your project or worktree root — some CLIs (Copilot) root their file tools at their cwd and refuse paths outside it. Defaults to the CLI manifest's `working_dir`, then the MCP server's own working directory. The CLI receives it as both its cwd and its `PWD` environment variable, so CLIs that root their session at `$PWD` (opencode) honor it too. The effective directory is reported back as `metadata.working_dir`
- `read_only`: Restrict the CLI to read-only operations (sandbox flags + prompt instruction + post-execution snapshot verification)
- `files`: Optional file paths for context (references only, CLI opens files itself)
- `images`: Optional image paths for visual context
- `continuation_id`: Continue previous clink conversations

## Usage Examples

**Architecture Planning:**
```
Use clink with gemini planner to design a 3-phase rollout plan for our feature flags system
```

**Code Review with Context:**
```
clink to gemini codereviewer: Review payment_service.py for race conditions and concurrency issues
```

**Codex Code Review:**
```
"clink with codex cli and perform a full code review using the codereview role"
```

**Quick Research Question:**
```
"Ask gemini via clink: What are the breaking changes in TypeScript 5.5?"
```

**Multi-Tool Workflow:**
```
"Use planner to outline the refactor, then clink gemini planner for validation,
then codereview to verify the implementation"
```

**Leveraging Gemini's Web Search:**
```
"Clink gemini to research current best practices for Kubernetes autoscaling in 2025"
```

## How Clink Works

1. **Your request** - You ask your current CLI to use `clink` with a specific CLI and role
2. **Background execution** - Unison spawns the configured CLI (e.g., `gemini --output-format json`)
3. **Context forwarding** - Your prompt, files (as references), and conversation history are sent as part of the prompt
4. **CLI processing** - Gemini (or other CLI) uses its own tools: web search, file access, thinking modes
5. **Seamless return** - Results flow back into your conversation with full context preserved
6. **Continuation support** - Future tools and models can reference Gemini's findings via [continuation support](../context-revival.md) within Unison.

## Best Practices

- **Pre-authenticate CLIs**: Install and configure Gemini CLI first (`npm install -g @google/gemini-cli`)
- **Choose appropriate roles**: Use `planner` for strategy, `codereviewer` for code, `default` for general questions
- **Leverage CLI strengths**: Gemini's 1M context for large codebases, web search for current docs
- **Combine with Unison tools**: Chain `clink` with `planner`, `codereview`, `debug` for powerful workflows
- **File efficiency**: Pass file paths, let the CLI decide what to read (saves tokens)

## Configuration

Clink configurations live in `conf/cli_clients/`. We ship presets for the supported CLIs:

| CLI | Command | Read-only mode | Stability |
|---|---|---|---|
| `gemini` | `gemini --telemetry false --yolo -o json` | `--approval-mode plan` | stable |
| `claude` | `claude --print --output-format json --permission-mode acceptEdits --model sonnet` | `--permission-mode plan` | stable |
| `codex` | `codex exec --json --dangerously-bypass-approvals-and-sandbox` | prompt-only | stable |
| `opencode` | `opencode run --format json` | prompt-only + filesystem snapshot | stable |
| `aider` | `aider --no-pretty --no-stream --no-auto-commits --yes-always` | `--dry-run` (native) | stable |
| `crush` | `crush run --quiet` | prompt-only + filesystem snapshot | evolving |
| `amp` | `amp --execute --stream-json` | prompt-only + filesystem snapshot | new |
| `copilot` | `copilot --output-format json --allow-all-tools --no-color --log-level none --no-auto-update` | `--available-tools view,grep,glob` + `--deny-tool write/shell` (application-level, not an OS sandbox) | new |

**Stability tiers:** `stable` = proven upstream, infrequent flag changes. `evolving` = active development, flags may rev. `new` = recently released, expect changes.

> **CAUTION**: These flags intentionally bypass each CLI's safety prompts so they can edit files or launch tools autonomously via MCP. Only enable them in trusted sandboxes and tailor role prompts or CLI configs if you need more guardrails.

> **Aider notes:** Auto-commits are disabled (`--no-auto-commits`) so clink-spawned Aider invocations never create git commits as a side effect. The prompt is delivered via `--message-file` (Aider has no stdin scripting mode). Bookkeeping files Aider creates (`.aider.chat.history.md`, `.aider.input.history`, `.aider.tags.cache.v4/`) are classified as `by_cli_bookkeeping` in `read_only_violations` metadata, not as model writes.

Each preset points to role-specific prompts in `systemprompts/clink/`. Duplicate those files to add more roles or adjust CLI flags.

### opencode v2

clink runs the first `opencode` on the Unison server's `PATH` and probes its version once per binary (cached by path, mtime and size, so an upgrade is picked up). The result is logged once, e.g. `opencode at /Users/you/.opencode/bin/opencode is v2.2.0` in `logs/mcp_server.log`; to check by hand, run `which opencode && opencode --version` in the environment that launches Unison. opencode 0.x/1.x are unchanged: their command line is exactly what the manifest says. Only an executable named `opencode` is probed: a wrapper such as `npx` or `bunx` answers `--version` with its own version, so it is treated as an unknown version and keeps the plain command line, as does a v2 beta/dev build that reports `0.0.0-…`. Everything below applies to **v2** only, except cancellation (below).

**Private server by default.** v2's `opencode run` is a thin client of a shared background service, and that service — not the process clink spawned — starts MCP servers, holds provider auth and runs the session. clink's environment (`UNISON_CLINK_DEPTH` for the recursion guard, `UNISON_CLINK_STRIP_SECRETS`, manifest `env`) therefore never reached them, and killing the spawned process did not stop the model. clink now runs `opencode run --format json [-m provider/model] --standalone --title clink:<role>`: a private server owned by the spawned process, which inherits clink's environment and ends with it. A call takes about 1.0–1.8 s here versus 0.2–0.7 s on the warm service (single calls of 8.7–11.4 s were seen with an MCP server configured). `--title` stops opencode making an extra paid title-generation request per run. Concurrent clink jobs each start their own private server against opencode's shared database: with an existing database 6 of 6 concurrent calls succeeded, but two servers racing to initialize a brand-new one can fail (1 of 2 trials: `Standalone server exited before reporting readiness`) — run one opencode command once before first heavy parallel use on a fresh machine.

**`CLINK_OPENCODE_MODE=service`** opts back into the shared service. You then lose: recursion-depth propagation to MCP servers (a nested Unison would not see the depth), `UNISON_CLINK_STRIP_SECRETS`, manifest `env` reaching provider auth, and reliable cancellation (the interrupt clink sends does reach the service, but nothing reclaims a session left behind by a force-kill). Worse, if no opencode service is running, the call auto-starts a detached one that outlives clink and keeps clink's environment — secrets included — for every later session that uses it, yours too. Start the service yourself (`opencode service start`) before opting in. clink logs a warning once per process when this mode (or a manifest `--server`) is in effect.

**MCP servers in a standalone run.** Upstream issue #51812 reproduces on v2.2.0. In a one-shot `opencode run --standalone --format json -m <provider>/<model>` run (prompt on stdin) the first model request carried only opencode's 12 built-in tools and nothing from a configured MCP server (6 of 6 runs, measured in a sandbox against a local mock provider; see `tests/opencode_support/`). opencode starts the MCP server at about the moment it sends that request, and the tool first appears on the **second** model step. A clink call whose first step needs an MCP tool will not have it; multi-step runs do (with Code Mode on, MCP tools are a catalog in the system prompt reachable through the `execute` tool rather than `tools[]` entries). The MCP server's environment is the full environment of the private server plus the config's `environment` map, so `UNISON_CLINK_DEPTH` reaches it (3 of 3 runs) — which is what keeps the recursion guard working. Under the shared service it would get the service's spawn-time environment instead, never your per-call one.

**Cancellation (every opencode version, POSIX).** When clink stops an opencode call — a timeout, `job_id` + `cancel: true`, server shutdown, or an MCP host cancelling a call made with `CLINK_EXECUTION_MODE=blocking` — it first sends SIGINT, which makes opencode interrupt the session and release it, waits up to 5 s, and only then kills the process group. In the default hybrid mode a host cancelling the waiting request does **not** stop the call; only `job_id` + `cancel: true` does. On Windows nothing can be signalled, so clink kills the process tree immediately. If v2 ignores the interrupt and is force-killed, its session can stay suspended, and your opencode service resumes suspended sessions the next time it starts (up to 10 times), spending with nobody listening. clink logs the session ID with the fix: `opencode session delete --standalone <id>`. The `--standalone` matters — without it the delete command boots the background service, which resumes the session it was meant to remove.

**Models.** `provider/model`, or `provider/model#variant` on v2 only (v1 takes plain `provider/model`). v2 renamed some provider IDs: `azure-cognitive-services` is now `azure`, `google-vertex-anthropic` is now `google-vertex`. See what your install can serve with `opencode models --standalone`. A run whose output is only an error event fails with opencode's own error type and message leading the text, whatever the exit status (v1 exits 0 on session errors).

**Context and tools.** v2 reads project instructions from `AGENTS.md` only (the `CLAUDE.md` fallback is gone), and its shell tool is called `shell`, not `bash` — adjust role prompts that name it.

**Read-only mode.** There is still no opencode flag for read-only-while-executing, so enforcement is the prompt instruction plus the filesystem snapshot diff. Two v2 limits: the snapshot covers only the `working_dir` subtree, and v2 allows `external_directory` by default, so a write outside the project is neither asked about nor detected. v2 writes no bookkeeping files into the project, so the read-only report shows none for it.

### Timeouts

Three layers can end a clink call — know which one fired:

1. **Unison's subprocess timeout** — default **3600s (60 min)** per CLI, deliberately generous. Override per CLI with `timeout_seconds` in `conf/cli_clients/<cli>.json`, or globally at runtime with the `CLINK_TIMEOUT_SECONDS` env var (no file edits needed). When this fires, clink kills the whole CLI process tree and returns a clear `timed out after N seconds` error.
2. **Your MCP host's tool-call timeout** — often much shorter than ours, and usually the one that actually fires. The setup scripts automatically seed `MCP_TOOL_TIMEOUT=900000` (15 min) into the top-level `env` block of `~/.claude/settings.json` when Claude is detected (never overwriting an existing value). To tune it manually for Claude Code, raise `MCP_TOOL_TIMEOUT` (milliseconds) in the environment where the `claude` CLI runs, or in that same top-level `env` block. Claude Desktop's local (embedded Claude Code) sessions honor that same top-level `env` block — or Desktop's local environment editor — after starting a new session or fully restarting Desktop. Do **not** put it under `mcpServers.<name>.env`; that environment goes to the spawned *server*, not the client enforcing the deadline. Desktop's plain chat has no documented knob. When a host cancels, clink reaps the CLI subprocess so it never keeps running orphaned, and while a CLI runs clink emits MCP progress heartbeats every ~10s — hosts that reset their timeout on progress keep long runs alive; hosts that sent no progress token get a guaranteed no-op.
3. **The CLI's own internal limits** — some CLIs abort long runs themselves and report it in their output; clink surfaces that as the CLI's error, not a clink timeout.

### Long runs under impatient hosts: built-in job control

When the host's tool timeout is shorter than a real CLI run (Claude Desktop's local-agent mode cancels around a minute; a serious Copilot review needs several), no single long MCP request can survive. clink therefore bounds **every** call itself — no separate tools needed:

- A normal `clink` call runs the CLI and waits up to `wait_seconds` (default 45s, max 50s — always under host deadlines). Runs that finish in time return **exactly** what they always did, byte-for-byte, continuation offer included.
- A run that needs longer returns `status: "running"` with a `job_id` and explicit instructions. **Call `clink` again with only that `job_id`** to keep waiting; each follow-up call waits another bounded window and returns the full original result the moment the CLI finishes (failures re-raise the original error).
- `job_id` + `cancel: true` aborts the job and stops the CLI's entire process tree: CLIs that keep session state outside the process (opencode) are interrupted first and given a few seconds to exit cleanly, then killed. The reply reports `status: "cancelled"`, or `"cancelling"` while the stop is still in progress (call clink again with the same `job_id` to confirm); it never reports `running` for a cancelled job.
- Optional extras: `wait_seconds` tunes the per-call wait; `idempotency_key` makes a retried start return the already-running job instead of launching duplicate paid CLI work.

Job state is in-memory (results retained 30 minutes; up to 3 concurrent jobs). If the MCP server restarts between calls, the follow-up says so explicitly. Each background run executes on an isolated tool instance; progress heartbeats come from whichever foreground call is currently waiting. Set `CLINK_EXECUTION_MODE=blocking` to restore the old fully-blocking behavior on hosts with generous timeouts.

**Read-only snapshot cost:** `read_only` verification walks the CLI's working directory before and after the run. The walk prunes bulk directories (`.git`, `node_modules`, virtualenvs, build outputs) and honors a wall-clock budget — `CLINK_SNAPSHOT_BUDGET_SECONDS`, default 30s per snapshot — so verification can never starve the CLI call itself of the host's tool-timeout budget (observed live: an unbounded walk of a large OneDrive repo took 60-90s per snapshot). When the budget or the 50,000-entry cap truncates the walk, `metadata.read_only_verification_coverage` reads `working_dir_subtree (partial)` and `read_only_verification_stats` carries entry counts and elapsed times.

> **Why `--yolo` for Gemini?** The Gemini CLI currently requires automatic approvals to execute its own tools (for example `run_shell_command`). Without the flag it errors with `Tool "run_shell_command" not found in registry`. See [issue #5382](https://github.com/google-gemini/gemini-cli/issues/5382) for more details.

**Adding new CLIs**: Drop a JSON config into `conf/cli_clients/`, create role prompts in `systemprompts/clink/`, and register a parser/agent if the CLI outputs a new format.

## When to Use Clink vs Other Tools

- **Use `clink`** for: Leveraging external CLI capabilities (Gemini's web search, 1M context), specialized CLI features, cross-CLI collaboration
- **Use `chat`** for: Direct model-to-model conversations within Unison
- **Use `planner`** for: Unison's native planning workflows with step validation
- **Use `codereview`** for: Unison's structured code review with severity levels

## Setup Requirements

Ensure the relevant CLI is installed and configured:

- [Claude Code](https://www.anthropic.com/claude-code)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- [Codex CLI](https://github.com/openai/codex)
- [opencode](https://opencode.ai)
- [Aider](https://aider.chat) — install with `pip install aider-chat` (or `pipx install aider-chat`). Aider uses standard provider API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) from the environment; clink does not manage Aider's auth.
- [Crush](https://github.com/charmbracelet/crush) — install with `brew install charmbracelet/tap/crush` (macOS) or see upstream install docs for other platforms. Crush is multi-provider; configure providers per Crush's own docs. Model selection at clink call time uses `provider/model` syntax (e.g., `anthropic/claude-sonnet-4-5`, `openai/gpt-4o`).
- [Amp](https://ampcode.com) — install per Sourcegraph's instructions; requires a Sourcegraph account. For non-interactive use (which clink always is), set `AMP_API_KEY` in your environment before launching Unison. For interactive first-time setup, run `amp login`. **Model selection on Amp uses named modes** (`deep`, `large`, `rush`, `smart`) via `--mode`, not arbitrary model strings — the manifest's `supported_models` allowlist enforces these values. **Recursion warning:** Amp is MCP-aware (`amp mcp add ...`). If you wire Unison as an MCP server in Amp's config AND invoke `clink with cli_name="amp"` from a Unison-aware CLI, the Phase 0 cross-cutting recursion guard will refuse the inner invocation to prevent a context-window-exploding loop. Remove Unison from Amp's MCP config or raise `CLINK_MAX_RECURSION_DEPTH` if the depth is intentional.

## Related Guides

- [Chat Tool](chat.md) - Direct model conversations
- [Planner Tool](planner.md) - Unison's native planning workflows
- [CodeReview Tool](codereview.md) - Structured code reviews
- [Context Revival](../context-revival.md) - Continuing conversations across tools
- [Advanced Usage](../advanced-usage.md) - Complex multi-tool workflows
