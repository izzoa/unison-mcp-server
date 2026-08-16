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
- `working_dir`: Optional **absolute path** the spawned CLI runs in. Pass your project or worktree root — some CLIs (Copilot) root their file tools at their cwd and refuse paths outside it. Defaults to the CLI manifest's `working_dir`, then the MCP server's own working directory. The effective directory is reported back as `metadata.working_dir`
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
| `opencode` | `opencode --format json` | prompt-only + filesystem snapshot | stable |
| `aider` | `aider --no-pretty --no-stream --no-auto-commits --yes-always` | `--dry-run` (native) | stable |
| `crush` | `crush run --quiet` | prompt-only + filesystem snapshot | evolving |
| `amp` | `amp --execute --stream-json` | prompt-only + filesystem snapshot | new |
| `copilot` | `copilot --output-format json --allow-all-tools --no-color --log-level none --no-auto-update` | `--available-tools view,grep,glob` + `--deny-tool write/shell` (application-level, not an OS sandbox) | new |

**Stability tiers:** `stable` = proven upstream, infrequent flag changes. `evolving` = active development, flags may rev. `new` = recently released, expect changes.

> **CAUTION**: These flags intentionally bypass each CLI's safety prompts so they can edit files or launch tools autonomously via MCP. Only enable them in trusted sandboxes and tailor role prompts or CLI configs if you need more guardrails.

> **Aider notes:** Auto-commits are disabled (`--no-auto-commits`) so clink-spawned Aider invocations never create git commits as a side effect. The prompt is delivered via `--message-file` (Aider has no stdin scripting mode). Bookkeeping files Aider creates (`.aider.chat.history.md`, `.aider.input.history`, `.aider.tags.cache.v4/`) are classified as `by_cli_bookkeeping` in `read_only_violations` metadata, not as model writes.

Each preset points to role-specific prompts in `systemprompts/clink/`. Duplicate those files to add more roles or adjust CLI flags.

### Timeouts

Three layers can end a clink call — know which one fired:

1. **Unison's subprocess timeout** — default **3600s (60 min)** per CLI, deliberately generous. Override per CLI with `timeout_seconds` in `conf/cli_clients/<cli>.json`, or globally at runtime with the `CLINK_TIMEOUT_SECONDS` env var (no file edits needed). When this fires, clink kills the whole CLI process tree and returns a clear `timed out after N seconds` error.
2. **Your MCP host's tool-call timeout** — often much shorter than ours, and usually the one that actually fires. For Claude Code, raise `MCP_TOOL_TIMEOUT` (milliseconds) in the environment where the `claude` CLI runs, or in the top-level `env` block of `~/.claude/settings.json`. Claude Desktop's local (embedded Claude Code) sessions honor that same top-level `env` block — or Desktop's local environment editor — after starting a new session or fully restarting Desktop. Do **not** put it under `mcpServers.<name>.env`; that environment goes to the spawned *server*, not the client enforcing the deadline. Desktop's plain chat has no documented knob. When a host cancels, clink reaps the CLI subprocess so it never keeps running orphaned, and while a CLI runs clink emits MCP progress heartbeats every ~10s — hosts that reset their timeout on progress keep long runs alive; hosts that sent no progress token get a guaranteed no-op.
3. **The CLI's own internal limits** — some CLIs abort long runs themselves and report it in their output; clink surfaces that as the CLI's error, not a clink timeout.

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
