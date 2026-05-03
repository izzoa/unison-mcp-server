<div align="center">

# Unison - Providers Together

  <em>Unison - Providers Together</em><br />
  <sub><a href="docs/name-change.md">Formerly known as Zen MCP</a></sub>

  [Unison in action](https://github.com/user-attachments/assets/0d26061e-5f21-4ab1-b7d0-f883ddc2c3da)

👉 **[Watch more examples](#-watch-tools-in-action)**

### Your CLI + Multiple Models = Your AI Dev Team

**Use the 🤖 CLI you love:**  
[Claude Code](https://www.anthropic.com/claude-code) · [Gemini CLI](https://github.com/google-gemini/gemini-cli) · [Codex CLI](https://github.com/openai/codex) · [Qwen Code CLI](https://qwenlm.github.io/qwen-code-docs/) · [Cursor](https://cursor.com) · _and more_

**With multiple models within a single prompt:**  
Gemini · OpenAI · Anthropic · Grok · Azure · Ollama · OpenRouter · DIAL · On-Device Model

</div>

---

## 🆕 Now with CLI-to-CLI Bridge

The new **[`clink`](docs/tools/clink.md)** (CLI + Link) tool connects external AI CLIs directly into your workflow:

- **Connect external CLIs** like [Gemini CLI](https://github.com/google-gemini/gemini-cli), [Codex CLI](https://github.com/openai/codex), [Claude Code](https://www.anthropic.com/claude-code), and [opencode](https://opencode.ai) directly into your workflow
- **CLI Subagents** - Launch isolated CLI instances from _within_ your current CLI! Claude Code can spawn Codex subagents, Codex can spawn Gemini CLI subagents, etc. Offload heavy tasks (code reviews, bug hunting) to fresh contexts while your main session's context window remains unpolluted. Each subagent returns only final results.
- **Context Isolation** - Run separate investigations without polluting your primary workspace
- **Role Specialization** - Spawn `planner`, `codereviewer`, or custom role agents with specialized system prompts
- **Per-call Model Selection** - Pick a model at call time via the `model` parameter. Opencode uses `provider/model` (e.g. `anthropic/claude-sonnet-4-5`, `openai/gpt-5`, `ollama/llama3.2`) and unlocks ~75 providers through a single integration. Manifests can declare an optional `supported_models` allowlist to curate available models per CLI
- **Full CLI Capabilities** - Web search, file inspection, MCP tool access, latest documentation lookups
- **Seamless Continuity** - Sub-CLIs participate as first-class members with full conversation context between tools

```bash
# Codex spawns Codex subagent for isolated code review in fresh context
clink with codex codereviewer to audit auth module for security issues
# Subagent reviews in isolation, returns final report without cluttering your context as codex reads each file and walks the directory structure

# Per-call model selection through opencode (anthropic, openai, ollama, openrouter, ...)
clink with opencode using anthropic/claude-sonnet-4-5 to refactor the payment flow

# Consensus from different AI models → Implementation handoff with full context preservation between tools
Use consensus with gpt-5 and gemini-pro to decide: dark mode or offline support next
Continue with clink gemini - implement the recommended feature
# Gemini receives full debate context and starts coding immediately
```

👉 **[Learn more about clink](docs/tools/clink.md)**

---

## How to clink

The `clink` tool takes four parameters that matter day-to-day: `cli_name`, `role`, `model`, and `read_only`. Everything else (file attachments, continuation IDs) works the same as any other Unison tool.

### Basic invocation

```bash
# Minimum: just pick a CLI and prompt it
clink with gemini to summarize the auth module

# Pick a role preset (default | planner | codereviewer)
clink with codex codereviewer to audit auth/ for OWASP top 10 issues
clink with claude planner to draft a phased migration off the legacy queue

# Pick a per-call model (especially powerful with opencode)
clink with opencode using anthropic/claude-sonnet-4-5 to refactor payments.py
clink with opencode using openai/gpt-5 to write a property-based test suite

# Read-only mode (analysis without file writes)
clink with claude codereviewer in read-only mode to review PR #482
```

### Supported CLIs

| CLI | Strengths | Read-only flag |
|---|---|---|
| `gemini` | Long-context analysis, free tier, web search | `--approval-mode plan` |
| `codex` | OpenAI-hosted reasoning models, web search via tool | prompt-only |
| `claude` | Anthropic models with strong agentic behavior | `--permission-mode plan` |
| `opencode` | **75+ providers** behind one CLI (OpenAI, Anthropic, Google, Ollama, OpenRouter, xAI, Mistral, Groq, …) via `provider/model` syntax | _(none — see note)_ |

> **Note on opencode read-only mode:** opencode has no CLI flag for read-only-while-still-executing semantics. Its `--agent plan` flag switches the agent persona (producing planning-language instead of executing the requested task), so it is not a true read-only sandbox. Read-only enforcement for opencode relies on the prompt-level instruction and the post-execution filesystem snapshot diff — both CLI-agnostic. CLI bookkeeping that opencode creates on first-run (`.opencode/...` and `.git/opencode`) is classified separately under `read_only_violations.by_cli_bookkeeping` so it doesn't drown out genuine model-write detection.

### Where to find valid `model` values

`opencode` is the heavy hitter — one integration unlocks the entire models.dev catalog. To see what your installation can serve:

```bash
opencode models                          # all models from authenticated providers
opencode models | grep ^anthropic/        # filter by provider
opencode models | wc -l                   # count (typically 300+ once you've authed a few providers)
opencode providers list                   # check which providers are authenticated
```

| CLI | How to discover valid models | Format |
|---|---|---|
| `opencode` | `opencode models` (local) · [models.dev](https://models.dev) (catalog) | `provider/model` (e.g. `anthropic/claude-sonnet-4-5`, `openai/gpt-5`, `ollama/llama3.2`) |
| `claude` | `claude --help` · [Claude Code docs](https://docs.claude.com/en/docs/claude-code) | aliases (`sonnet`, `opus`, `haiku`) or full IDs |
| `gemini` | `gemini --help` · [Gemini CLI docs](https://github.com/google-gemini/gemini-cli) | model IDs (e.g. `gemini-2.5-pro`) |
| `codex` | `codex --help` · [Codex CLI docs](https://github.com/openai/codex) | OpenAI model IDs (e.g. `o3-mini`) |

**Validation behavior:** the `model` field is free-text — invalid values come back as a CLI-level error in response metadata, not a schema rejection. Authentication for opencode providers is opencode's own concern (`opencode auth`); Unison doesn't manage credentials.

### Curating models per CLI (optional)

Add a `supported_models` allowlist to any `conf/cli_clients/*.json` manifest. When set, requests are validated against the list before the CLI is invoked — out-of-list values are rejected with a clear error naming the allowed values:

```json
{
  "name": "opencode",
  "command": "opencode run",
  "supported_models": [
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-opus-4-5",
    "openai/gpt-5"
  ],
  "roles": { "default": { "prompt_path": "systemprompts/clink/default.txt" } }
}
```

When `supported_models` is omitted (the default), any model string is forwarded verbatim.

### When `model` is omitted

Each manifest's `additional_args` carries the CLI's default — claude defaults to `sonnet`, gemini and codex use whatever default their CLI picks. Calls that don't specify `model` produce byte-for-byte identical commands to before this feature shipped, so existing workflows keep working unchanged.

---

## Why Unison MCP?

**Why rely on one AI model when you can orchestrate them all?**

A Model Context Protocol server that supercharges tools like [Claude Code](https://www.anthropic.com/claude-code), [Codex CLI](https://developers.openai.com/codex/cli), and IDE clients such
as [Cursor](https://cursor.com) or the [Claude Dev VS Code extension](https://marketplace.visualstudio.com/items?itemName=Anthropic.claude-vscode). **Unison MCP connects your favorite AI tool
to multiple AI models** for enhanced code analysis, problem-solving, and collaborative development.

### True AI Collaboration with Conversation Continuity

Unison supports **conversation threading** so your CLI can **discuss ideas with multiple AI models, exchange reasoning, get second opinions, and even run collaborative debates between models** to help you reach deeper insights and better solutions.

Your CLI always stays in control but gets perspectives from the best AI for each subtask. Context carries forward seamlessly across tools and models, enabling complex workflows like: code reviews with multiple models → automated planning → implementation → pre-commit validation.

> **You're in control.** Your CLI of choice orchestrates the AI team, but you decide the workflow. Craft powerful prompts that bring in Gemini Pro, GPT 5, Flash, or local offline models exactly when needed.

<details>
<summary><b>Reasons to Use Unison MCP</b></summary>

A typical workflow with Claude Code as an example:

1. **Multi-Model Orchestration** - Claude coordinates with Gemini Pro, O3, GPT-5, and 50+ other models to get the best analysis for each task

2. **Context Revival Magic** - Even after Claude's context resets, continue conversations seamlessly by having other models "remind" Claude of the discussion

3. **Guided Workflows** - Enforces systematic investigation phases that prevent rushed analysis and ensure thorough code examination

4. **Extended Context Windows** - Break Claude's limits by delegating to Gemini (1M tokens) or O3 (200K tokens) for massive codebases

5. **True Conversation Continuity** - Full context flows across tools and models - Gemini remembers what O3 said 10 steps ago

6. **Model-Specific Strengths** - Extended thinking with Gemini Pro, blazing speed with Flash, strong reasoning with O3, privacy with local Ollama

7. **Professional Code Reviews** - Multi-pass analysis with severity levels, actionable feedback, and consensus from multiple AI experts

8. **Smart Debugging Assistant** - Systematic root cause analysis with hypothesis tracking and confidence levels

9. **Automatic Model Selection** - Claude intelligently picks the right model for each subtask (or you can specify)

10. **Vision Capabilities** - Analyze screenshots, diagrams, and visual content with vision-enabled models

11. **Local Model Support** - Run Llama, Mistral, or other models locally for complete privacy and zero API costs

12. **Bypass MCP Token Limits** - Automatically works around MCP's 25K limit for large prompts and responses

**The Killer Feature:** When Claude's context resets, just ask to "continue with O3" - the other model's response magically revives Claude's understanding without re-ingesting documents!

#### Example: Multi-Model Code Review Workflow

1. `Perform a codereview using gemini pro and o3 and use planner to generate a detailed plan, implement the fixes and do a final precommit check by continuing from the previous codereview`
2. This triggers a [`codereview`](docs/tools/codereview.md) workflow where Claude walks the code, looking for all kinds of issues
3. After multiple passes, collects relevant code and makes note of issues along the way
4. Maintains a `confidence` level between `exploring`, `low`, `medium`, `high` and `certain` to track how confidently it's been able to find and identify issues
5. Generates a detailed list of critical -> low issues
6. Shares the relevant files, findings, etc with **Gemini Pro** to perform a deep dive for a second [`codereview`](docs/tools/codereview.md)
7. Comes back with a response and next does the same with o3, adding to the prompt if a new discovery comes to light
8. When done, Claude takes in all the feedback and combines a single list of all critical -> low issues, including good patterns in your code. The final list includes new findings or revisions in case Claude misunderstood or missed something crucial and one of the other models pointed this out
9. It then uses the [`planner`](docs/tools/planner.md) workflow to break the work down into simpler steps if a major refactor is required
10. Claude then performs the actual work of fixing highlighted issues
11. When done, Claude returns to Gemini Pro for a [`precommit`](docs/tools/precommit.md) review

All within a single conversation thread! Gemini Pro in step 11 _knows_ what was recommended by O3 in step 7! Taking that context
and review into consideration to aid with its final pre-commit review.

**Think of it as Claude Code _for_ Claude Code.** This MCP isn't magic. It's just **super-glue**.

> **Remember:** Claude stays in full control — but **YOU** call the shots.
> Unison is designed to have Claude engage other models only when needed — and to follow through with meaningful back-and-forth.
> **You're** the one who crafts the powerful prompt that makes Claude bring in Gemini, Flash, O3 — or fly solo.
> You're the guide. The prompter. The puppeteer.
> #### You are the AI - **Actually Intelligent**.
</details>

#### Recommended AI Stack

<details>
<summary>For Claude Code Users</summary>

For best results when using [Claude Code](https://claude.ai/code):  

- **Sonnet 4.5** - All agentic work and orchestration
- **Gemini 3.0 Pro** OR **GPT-5.2 / Pro** - Deep thinking, additional code reviews, debugging and validations, pre-commit analysis
</details>

<details>
<summary>For Codex Users</summary>

For best results when using [Codex CLI](https://developers.openai.com/codex/cli):  

- **GPT-5.2 Codex Medium** - All agentic work and orchestration
- **Gemini 3.0 Pro** OR **GPT-5.2-Pro** - Deep thinking, additional code reviews, debugging and validations, pre-commit analysis
</details>

## Differences from PAL MCP

Unison is forked from [BeehiveInnovations/pal-mcp-server](https://github.com/BeehiveInnovations/pal-mcp-server). It inherits the full PAL feature set and adds the following:

| Area | PAL MCP | Unison MCP |
|------|---------|------------|
| **Model Discovery** | Static JSON files only — manual updates required when providers release new models | Automatic model discovery via [LiteLLM](https://github.com/BerriAI/litellm) at startup; new models appear without JSON changes |
| **Model Catalog** | Limited to manually curated entries per provider | 2000+ models auto-discovered across all providers, with curated overrides for tuned metadata |
| **Discovered vs Curated** | All models treated equally | `listmodels` distinguishes curated models (with hand-tuned intelligence scores, aliases) from auto-discovered ones |
| **Model Selection** | Hardcoded preference lists per provider — go stale when models change | Data-driven selection using `intelligence_score` and capability flags; auto-mode always picks the best available model |
| **Model Freshness** | Manual JSON updates only | Weekly CI workflow fetches the latest LiteLLM catalog and opens a PR with new/updated models for human review |
| **Conversation Storage** | In-memory only — lost on restart | Optional persistent SQLite backend (`STORAGE_BACKEND=sqlite`) — survives restarts with zero config |
| **Changelog** | Git-log style | [Keep a Changelog](https://keepachangelog.com/) format with Unreleased section |
| **Branding** | PAL (Provider Abstraction Layer) | Unison — Providers Together |

> All core tools, providers, workflows, and conversation continuity features from PAL are fully preserved. See [docs/name-change.md](docs/name-change.md) for migration notes.
>
> **How model routing works:** User-facing aliases (e.g., `"gemini"`, `"pro"`, `"flash"`) are curated in `conf/*.json` and always point to a specific model. When auto-mode selects a model (no alias specified), it uses `intelligence_score` and capability flags dynamically — so newly discovered models with high scores are preferred automatically. The weekly LiteLLM refresh PR flags new models that need alias curation.

## Quick Start (5 minutes)

**Prerequisites:** Python 3.10+, Git, [uv installed](https://docs.astral.sh/uv/getting-started/installation/)

**1. Get API Keys** (choose one or more):
- **[OpenRouter](https://openrouter.ai/)** - Access multiple models with one API
- **[Gemini](https://makersuite.google.com/app/apikey)** - Google's latest models
- **[OpenAI](https://platform.openai.com/api-keys)** - O3, GPT-5 series
- **[Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/)** - Enterprise deployments of GPT-4o, GPT-4.1, GPT-5 family
- **[X.AI](https://console.x.ai/)** - Grok models
- **[DIAL](https://dialx.ai/)** - Vendor-agnostic model access
- **[Ollama](https://ollama.ai/)** - Local models (free)

**2. Install** (choose one):

**Option A: Clone and Automatic Setup** (recommended)
```bash
git clone https://github.com/izzoa/unison-mcp-server.git
cd unison-mcp-server

# Handles everything: setup, config, API keys from system environment. 
# Auto-configures Claude Desktop, Claude Code, Gemini CLI, Codex CLI, Qwen CLI
# Enable / disable additional settings in .env
./run-server.sh  
```

**Option B: Instant Setup with [uvx](https://docs.astral.sh/uv/getting-started/installation/)**
```json
// Add to ~/.claude/settings.json or .mcp.json
// Don't forget to add your API keys under env
{
  "mcpServers": {
    "unison": {
      "command": "bash",
      "args": ["-c", "for p in $(which uvx 2>/dev/null) $HOME/.local/bin/uvx /opt/homebrew/bin/uvx /usr/local/bin/uvx uvx; do [ -x \"$p\" ] && exec \"$p\" --from git+https://github.com/izzoa/unison-mcp-server.git unison-mcp-server; done; echo 'uvx not found' >&2; exit 1"],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:~/.local/bin",
        "GEMINI_API_KEY": "your-key-here",
        "DISABLED_TOOLS": "analyze,refactor,testgen,secaudit,docgen,tracer",
        "DEFAULT_MODEL": "auto"
      }
    }
  }
}
```

**3. Start Using!**
```
"Use unison to analyze this code for security issues with gemini pro"
"Debug this error with o3 and then get flash to suggest optimizations"
"Plan the migration strategy with unison, get consensus from multiple models"
"clink with cli_name=\"gemini\" role=\"planner\" to draft a phased rollout plan"
```

👉 **[Complete Setup Guide](docs/getting-started.md)** with detailed installation, configuration for Gemini / Codex / Qwen, and troubleshooting
👉 **[Cursor & VS Code Setup](docs/getting-started.md#ide-clients)** for IDE integration instructions
📺 **[Watch tools in action](#-watch-tools-in-action)** to see real-world examples

## Provider Configuration

Unison activates any provider that has credentials in your `.env`. See `.env.example` for deeper customization.

**Circuit Breaker** — Each provider has a built-in circuit breaker that detects sustained failures (outages, revoked keys, quota exhaustion) and fails fast instead of waiting through the full retry cycle. When a provider fails `CIRCUIT_FAILURE_THRESHOLD` consecutive times (default 5), its circuit opens and requests return immediately for `CIRCUIT_RESET_TIMEOUT_SECONDS` (default 60s) before probing recovery. The consensus tool automatically skips unavailable providers and synthesizes from the rest. See `.env.example` for configuration options.

## Core Tools

> **Note:** Each tool comes with its own multi-step workflow, parameters, and descriptions that consume valuable context window space even when not in use. To optimize performance, some tools are disabled by default. See [Tool Configuration](#tool-configuration) below to enable them.

**Collaboration & Planning** *(Enabled by default)*
- **[`clink`](docs/tools/clink.md)** - Bridge requests to external AI CLIs (Gemini planner, codereviewer, etc.). Supports `read_only` mode for safe analysis without file modifications
- **[`chat`](docs/tools/chat.md)** - Brainstorm ideas, get second opinions, validate approaches. With capable models (GPT-5.2 Pro, Gemini 3.0 Pro), generates complete code / implementation
- **[`thinkdeep`](docs/tools/thinkdeep.md)** - Extended reasoning, edge case analysis, alternative perspectives
- **[`planner`](docs/tools/planner.md)** - Break down complex projects into structured, actionable plans
- **[`consensus`](docs/tools/consensus.md)** - Get expert opinions from multiple AI models with stance steering

**Code Analysis & Quality**
- **[`debug`](docs/tools/debug.md)** - Systematic investigation and root cause analysis
- **[`precommit`](docs/tools/precommit.md)** - Validate changes before committing, prevent regressions
- **[`codereview`](docs/tools/codereview.md)** - Professional reviews with severity levels and actionable feedback
- **[`analyze`](docs/tools/analyze.md)** *(disabled by default - [enable](#tool-configuration))* - Understand architecture, patterns, dependencies across entire codebases

**Development Tools** *(Disabled by default - [enable](#tool-configuration))*
- **[`refactor`](docs/tools/refactor.md)** - Intelligent code refactoring with decomposition focus
- **[`testgen`](docs/tools/testgen.md)** - Comprehensive test generation with edge cases
- **[`secaudit`](docs/tools/secaudit.md)** - Security audits with OWASP Top 10 analysis
- **[`docgen`](docs/tools/docgen.md)** - Generate documentation with complexity analysis

**Utilities**
- **[`apilookup`](docs/tools/apilookup.md)** - Forces current-year API/SDK documentation lookups in a sub-process (saves tokens within the current context window), prevents outdated training data responses
- **[`challenge`](docs/tools/challenge.md)** - Prevent "You're absolutely right!" responses with critical analysis
- **[`tracer`](docs/tools/tracer.md)** *(disabled by default - [enable](#tool-configuration))* - Static analysis prompts for call-flow mapping

<details>
<summary><b id="tool-configuration">👉 Tool Configuration</b></summary>

### Default Configuration

To optimize context window usage, only essential tools are enabled by default:

**Enabled by default:**
- `chat`, `thinkdeep`, `planner`, `consensus` - Core collaboration tools
- `codereview`, `precommit`, `debug` - Essential code quality tools
- `apilookup` - Rapid API/SDK information lookup
- `challenge` - Critical thinking utility

**Disabled by default:**
- `analyze`, `refactor`, `testgen`, `secaudit`, `docgen`, `tracer`

### Enabling Additional Tools

To enable additional tools, remove them from the `DISABLED_TOOLS` list:

**Option 1: Edit your .env file**
```bash
# Default configuration (from .env.example)
DISABLED_TOOLS=analyze,refactor,testgen,secaudit,docgen,tracer

# To enable specific tools, remove them from the list
# Example: Enable analyze tool
DISABLED_TOOLS=refactor,testgen,secaudit,docgen,tracer

# To enable ALL tools
DISABLED_TOOLS=
```

**Option 2: Configure in MCP settings**
```json
// In ~/.claude/settings.json or .mcp.json
{
  "mcpServers": {
    "unison": {
      "env": {
        // Tool configuration
        "DISABLED_TOOLS": "refactor,testgen,secaudit,docgen,tracer",
        "DEFAULT_MODEL": "pro",
        "DEFAULT_THINKING_MODE_THINKDEEP": "high",
        
        // API configuration
        "GEMINI_API_KEY": "your-gemini-key",
        "OPENAI_API_KEY": "your-openai-key",
        "OPENROUTER_API_KEY": "your-openrouter-key",
        
        // Logging and performance
        "LOG_LEVEL": "INFO",
        "CONVERSATION_TIMEOUT_HOURS": "6",
        "MAX_CONVERSATION_TURNS": "50"
      }
    }
  }
}
```

**Option 3: Enable all tools**
```json
// Remove or empty the DISABLED_TOOLS to enable everything
{
  "mcpServers": {
    "unison": {
      "env": {
        "DISABLED_TOOLS": ""
      }
    }
  }
}
```

**Note:**
- Essential tools (`version`, `listmodels`) cannot be disabled
- After changing tool configuration, restart your Claude session for changes to take effect
- Each tool adds to context window usage, so only enable what you need

</details>

## 📺 Watch Tools In Action

<details>
<summary><b>Chat Tool</b> - Collaborative decision making and multi-turn conversations</summary>

**Picking Redis vs Memcached:**

[Chat Redis or Memcached_web.webm](https://github.com/user-attachments/assets/41076cfe-dd49-4dfc-82f5-d7461b34705d)

**Multi-turn conversation with continuation:**

[Chat With Gemini_web.webm](https://github.com/user-attachments/assets/37bd57ca-e8a6-42f7-b5fb-11de271e95db)

</details>

<details>
<summary><b>Consensus Tool</b> - Multi-model debate and decision making</summary>

**Multi-model consensus debate:**

[Unison Consensus Debate](https://github.com/user-attachments/assets/76a23dd5-887a-4382-9cf0-642f5cf6219e)

</details>

<details>
<summary><b>PreCommit Tool</b> - Comprehensive change validation</summary>

**Pre-commit validation workflow:**

<div align="center">
  <img src="https://github.com/user-attachments/assets/584adfa6-d252-49b4-b5b0-0cd6e97fb2c6" width="950">
</div>

</details>

<details>
<summary><b>API Lookup Tool</b> - Current vs outdated API documentation</summary>

**Without Unison - outdated APIs:**

[API without Unison](https://github.com/user-attachments/assets/01a79dc9-ad16-4264-9ce1-76a56c3580ee)

**With Unison - current APIs:**

[API with Unison](https://github.com/user-attachments/assets/5c847326-4b66-41f7-8f30-f380453dce22)

</details>

<details>
<summary><b>Challenge Tool</b> - Critical thinking vs reflexive agreement</summary>

**Without Unison:**

![without_pal@2x](https://github.com/user-attachments/assets/64f3c9fb-7ca9-4876-b687-25e847edfd87)

**With Unison:**

![with_pal@2x](https://github.com/user-attachments/assets/9d72f444-ba53-4ab1-83e5-250062c6ee70)

</details>

## Key Features

**AI Orchestration**
- **Auto model selection** - Claude picks the right AI for each task
- **Multi-model workflows** - Chain different models in single conversations
- **Conversation continuity** - Context preserved across tools and models
- **[Context revival](docs/context-revival.md)** - Continue conversations even after context resets

**Model Support**
- **Multiple providers** - Gemini, OpenAI, Azure, X.AI, OpenRouter, DIAL, Ollama
- **Latest models** - GPT-5, Gemini 3.0 Pro, O3, Grok-4, local Llama
- **Automatic model discovery** - New models appear at startup via [LiteLLM](https://github.com/BerriAI/litellm) integration, no manual config needed
- **[Thinking modes](docs/advanced-usage.md#thinking-modes)** - Control reasoning depth vs cost
- **Vision support** - Analyze images, diagrams, screenshots

**Developer Experience**
- **Guided workflows** - Systematic investigation prevents rushed analysis
- **Smart file handling** - Auto-expand directories, manage token limits
- **Web search integration** - Access current documentation and best practices
- **[Large prompt support](docs/advanced-usage.md#working-with-large-prompts)** - Bypass MCP's 25K token limit

## Example Workflows

**Multi-model Code Review:**
```
"Perform a codereview using gemini pro and o3, then use planner to create a fix strategy"
```
→ Claude reviews code systematically → Consults Gemini Pro → Gets O3's perspective → Creates unified action plan

**Collaborative Debugging:**
```
"Debug this race condition with max thinking mode, then validate the fix with precommit"
```
→ Deep investigation → Expert analysis → Solution implementation → Pre-commit validation

**Architecture Planning:**
```
"Plan our microservices migration, get consensus from pro and o3 on the approach"
```
→ Structured planning → Multiple expert opinions → Consensus building → Implementation roadmap

👉 **[Advanced Usage Guide](docs/advanced-usage.md)** for complex workflows, model configuration, and power-user features

## Quick Links

**📖 Documentation**
- [Docs Overview](docs/index.md) - High-level map of major guides
- [Getting Started](docs/getting-started.md) - Complete setup guide
- [Tools Reference](docs/tools/) - All tools with examples
- [Advanced Usage](docs/advanced-usage.md) - Power user features
- [Configuration](docs/configuration.md) - Environment variables, restrictions
- [Adding Providers](docs/adding_providers.md) - Provider-specific setup (OpenAI, Azure, custom gateways)
- [Model Ranking Guide](docs/model_ranking.md) - How intelligence scores drive auto-mode suggestions

**🔧 Setup & Support**
- [WSL Setup](docs/wsl-setup.md) - Windows users
- [Troubleshooting](docs/troubleshooting.md) - Common issues
- [Contributing](docs/contributions.md) - Code standards, PR process

## License

AGPL-3.0 License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is based on [BeehiveInnovations/pal-mcp-server](https://github.com/BeehiveInnovations/pal-mcp-server). Full credit to the original authors for the foundation this work builds upon.

Built with the power of **Multi-Model AI** collaboration 🤝
- **A**ctual **I**ntelligence by real Humans
- [MCP (Model Context Protocol)](https://modelcontextprotocol.com)
- [Codex CLI](https://developers.openai.com/codex/cli)
- [Claude Code](https://claude.ai/code)
- [Gemini](https://ai.google.dev/)
- [OpenAI](https://openai.com/)
- [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/)

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=izzoa/unison-mcp-server&type=Date)](https://www.star-history.com/#izzoa/unison-mcp-server&Date)
