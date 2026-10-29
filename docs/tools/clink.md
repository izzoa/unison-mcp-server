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
- `working_dir`: Optional **absolute path** the spawned CLI runs in (set as both its cwd and its PWD, so CLIs that root their session at `$PWD` such as opencode honor it too). **It may be omitted**: clink then derives the directory from the conversation, the files you pass, the host's MCP roots, the CLI manifest, `CLINK_DEFAULT_WORKING_DIR` and the server's own working directory, in that order, and refuses with a precise error instead of guessing — see [Working directory](#working-directory). Pass your project or worktree root when you know it — some CLIs (Copilot) root their file tools at their cwd and refuse paths outside it. Broad directories (`/`, `/etc`, `/Users`, `/tmp` itself, …) are refused and your home directory is accepted with a warning. The directory used is reported back as `metadata.working_dir`, with `metadata.working_dir_source` and `metadata.working_dir_warnings` saying how it was chosen
- `read_only`: Restrict the CLI to read-only operations (sandbox flags + prompt instruction + post-execution snapshot verification)
- `absolute_file_paths`: Optional absolute file paths for context (references only, CLI opens files itself). When `working_dir` is omitted they also identify the project the CLI runs in
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
- **Let clink find the project**: Pass `absolute_file_paths` from your project (or an explicit `working_dir`) so the CLI runs where you mean, and check `metadata.working_dir_source` when in doubt — see [Working directory](#working-directory)

## Working directory

clink starts CLIs with relaxed permission flags (see the caution above) and roots their file and shell tools at the call's working directory, so a wrong directory means reads, writes and shell commands in the wrong project. `working_dir` is therefore optional but never guessed silently: clink derives it from evidence, **refuses** when the evidence is ambiguous or missing, and says how it decided in every response.

### Precedence

The first tier that yields a valid directory wins:

| # | Tier | `working_dir_source` | What it means |
|---|---|---|---|
| 1 | Explicit `working_dir` | `explicit` | The value you pass. Validated, then used as its resolved path. Never fenced by roots; no roots request is made. |
| 2 | Continuation | `continuation` | The call continues a clink conversation (`continuation_id`) and passes no `working_dir`: the directory the conversation resolved earlier is reused if it is still valid. |
| 3 | Files | `files` | The project root of the `absolute_file_paths` passed on **this** call. |
| 4 | MCP roots | `roots` | Roots declared by the host. They fence every derived directory, and are the directory itself only when no files were passed (see [MCP roots](#mcp-roots)). |
| 5 | Manifest | `manifest` | `working_dir` in the CLI's manifest (`conf/cli_clients/<cli>.json`; none of the shipped presets sets one). |
| 6 | Environment default | `env` | `CLINK_DEFAULT_WORKING_DIR`. |
| 7 | Server cwd | `server_cwd` | The Unison server's own current directory: guarded, and warned about unless you silence it (see [Fallback policy](#fallback-policy-and-operator-defaults)). |

**Explicit.** An absolute path to an existing, accessible directory that is not broad; anything else fails with `invalid_explicit` before any CLI starts. Your home directory is accepted with a warning that the CLI can read and write everything in it. A path containing `..`, or reached through a symlink, is resolved, and the resolved path is what is spawned in, exported as `PWD`, reported as `metadata.working_dir` and stored for the conversation. Files you pass that lie outside it are listed in `working_dir_warnings`.

**Declared directories** (continuation, manifest, environment default, server cwd) are skipped with a warning, never an error, when they fail the [broadness guard](#broadness-guard), do not exist or are not directories, lie outside the host's roots, do not contain every file used for selection, or are a different checkout than the files (a git worktree nested inside the checkout the candidate points at). A call whose files include one with no project root can use a declared directory only if that directory contains the file.

**Final outcomes.** `ambiguous`, `roots_conflict`, `fallback_refused` and a timed-out search are final: clink never falls through to a lower tier after them, because that would run a permission-relaxed CLI somewhere unrelated to the evidence.

### Project root from files

For each file passed on this call, clink resolves symlinks (a link cannot smuggle in a file from another tree) and walks up at most 12 ancestors to the nearest:

- **VCS root**: a `.git` directory, a `.git` pointer file (git worktrees and submodules), `.hg`, `.svn` or `.jj`; else
- **package marker**: `pyproject.toml`, `setup.py`, `package.json`, `Cargo.toml`, `go.mod`, `pom.xml`, `build.gradle`, `build.gradle.kts`, `Gemfile`, `composer.json`, `mix.exs`, `deno.json` or `CMakeLists.txt`.

Rules that follow:

- **A VCS root outranks package markers.** A file in a `package.json` package inside a git repository resolves to the repository root.
- **Git worktrees resolve to the worktree.** The top of a worktree has a `.git` pointer file; clink never follows it to the main checkout, so a worktree nested inside another checkout resolves to the worktree.
- **One root must contain every file.** Across files, the outermost per-file root that contains all of them wins. Files in different projects are `ambiguous`: the error lists the candidate roots plus an enclosing marker root as a suggestion, never auto-picked, because a `.git` above two repositories would widen what the CLI can touch.
- **No marker, no derivation.** Files with no project root within the 12-ancestor bound yield nothing from this tier and derivation continues. The walk stops at a broad directory without evaluating it, so a dotfile directly in your home directory does not select your home directory.
- **Scratch files do not select.** A file under a temp directory with no project root is scratch: it is ignored for selection and containment while at least one non-scratch file exists, and named in the warnings. Only scratch files means the files tier yields nothing. A repository under `/tmp` is a project, not scratch.
- **Bad paths are ignored, not fatal.** A non-existent absolute path is ignored and named in the warnings; relative paths are rejected earlier by file-path validation.
- **A derived root can be larger than what you would have passed**, for example the VCS root of a monorepo. The read-only snapshot covers it (within `CLINK_SNAPSHOT_BUDGET_SECONDS`); pass `working_dir` to run narrower.

### Refusals

A refusal fails the call before any CLI is spawned. The error carries a code, a message naming what to pass, and the details in its metadata (see [Response metadata](#response-metadata)).

| `error_code` | When | What to do |
|---|---|---|
| `ambiguous` | The files belong to different projects, so no single project root contains them all; or roots are available, no files were passed, and several roots are usable. | Pass `working_dir` (the candidates are listed), or pass files from one project. |
| `roots_conflict` | The host declared roots and any file passed lies outside every one of them, or no single root contains them all while the project is wider than every root. clink never widens to the project root. | Pass `working_dir`, which overrides roots, or pass files inside a declared root. |
| `fallback_refused` | `CLINK_WORKING_DIR_FALLBACK=refuse` and the only directory left was the server cwd (one that passed every guard). | Pass `working_dir` or set `CLINK_DEFAULT_WORKING_DIR`, or use a less strict fallback policy. |
| `underivable` | No tier yielded a valid directory: no usable files, nothing configured, and the server cwd is broad, substituted, outside the roots or does not contain the files. Also returned when the [2 s search budget](#search-budget) runs out, in which case the message says the project-root search timed out. | Pass `working_dir`, or set `CLINK_DEFAULT_WORKING_DIR`. The `tried` list says which tiers were visited and why each was skipped. |
| `invalid_explicit` | `working_dir` is relative, does not exist, is a file, is inaccessible, or is a broad directory. | Pass an absolute path to a project directory. |

### Broadness guard

One guard applies to every tier: explicit, continuation, derived, manifest, environment default and server cwd. It refuses:

- filesystem and drive roots, and ancestors of your home directory
- your home directory itself, except as an explicit `working_dir` (or a stored explicit one), where it is allowed with a warning
- system directories **and everything beneath them**: `/etc /usr /bin /sbin /lib /lib64 /boot /var /root /proc /sys /dev /run`, and on Windows `C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)` and `C:\ProgramData`
- shared containers, **the exact directory only**: `/Users /home /Volumes /mnt /media /opt /srv /Library /Applications /System /System/Volumes/Data /private /tmp`
- `/var/tmp` and the platform's own temp directory, exactly

A project inside a shared container is fine: `/opt/myproject` and `/tmp/myproject` are accepted, while `/var/tmp/x` and the macOS per-user temp directory (`/private/var/folders/…`) stay refused as beneath `/var`. A directory strictly beneath your home directory is your own tree and never a "system directory", so `/root/project` is valid when your home is `/root`.

Directories are compared by filesystem identity, not by path text, so `/users` on a case-insensitive volume and `/System/Volumes/Data/Users` are the same directory as `/Users`.

There is **no switch to turn the guard off**. It is a deliberate tightening: an explicit `/`, `/etc` or `/tmp` that used to be accepted now fails with a reason.

### Fallback policy and operator defaults

**`CLINK_DEFAULT_WORKING_DIR`** is the operator-level answer for hosts with no project concept (Claude Desktop). It must be an absolute path to an existing directory that is not broad, and it is checked on every call. It ranks after the manifest and before the server cwd (source `env`). An invalid value, like an invalid manifest `working_dir`, is skipped with a warning and never fails the call; a directory created after the server started is honoured.

**`CLINK_WORKING_DIR_FALLBACK=allow|warn|refuse`** decides what happens when derivation reaches the server's current directory (the value is trimmed and case-insensitive; empty counts as unset; default `warn`):

- `warn`: run there with source `server_cwd` and the warning `working_dir was not given and could not be derived; ran in the server's current directory <dir>`, followed by `(it has no project marker)` when neither it nor any ancestor within the search bound has one, or by `(it is not itself a project root; it lies inside the project at <root>)` when the launch directory is a subdirectory of a project.
- `allow`: same directory, same source, no warning. **This only silences the warning.** Every guard still applies: a broad, substituted or fenced-out server cwd is still skipped, and the reasons stay in the error details. It is not a rollback of the old behavior, which accepted `/`.
- `refuse`: fail with `fallback_refused`. Applies only to a server cwd that passed every guard; a broad, substituted or fenced-out one is `underivable`.

An unrecognised value behaves as `warn` and adds a warning naming it to every call that has no explicit `working_dir`.

**Server-cwd substitution.** Claude Desktop launches MCP servers from `/` (on Windows, `C:\Windows\System32`). At startup the server switches into its own install directory for such launches and now records that it did. clink never treats that substituted directory as a project: a call with no files and no other defaults fails with `underivable`, asking for `working_dir` or `CLINK_DEFAULT_WORKING_DIR`, instead of running the CLI in the Unison install directory. A deliberate launch from a project directory is not a substitution.

### Continuations

clink stores the resolved directory and its source with the conversation thread, and a continuation without `working_dir` reuses it with source `continuation`, provided it still exists, is not broad, lies inside the host's current roots when roots are available, contains the files passed on this call, and is the same checkout as those files. Otherwise clink derives again, names the stored directory and the reason in `working_dir_warnings`, and stores the new result for later turns.

- An explicit `working_dir` on a later call replaces the stored one. An explicit directory from the first call is reported as `continuation`, not `explicit`, on later turns.
- Only the files passed on **this** call take part in derivation; files replayed from earlier turns do not.
- A stored home directory (the first call named it explicitly) may be reused, with the home-directory warning.
- A conversation started in another tool and continued in clink has no stored directory and derives afresh, as does one persisted before this feature existed.

### Response metadata

Every response (success, a CLI failure after a directory was chosen, and background-job results and errors) carries:

| Field | Meaning |
|---|---|
| `working_dir` | The one directory used: spawned in, exported as `PWD`, covered by the read-only snapshot, and reported. Always the resolved path. |
| `working_dir_source` | `explicit`, `continuation`, `files`, `roots`, `manifest`, `env` or `server_cwd` |
| `working_dir_warnings` | What was skipped or ignored and why (skipped declared directories, scratch or missing files, files outside an explicit `working_dir`, roots that were dropped, the server-cwd fallback). Also logged at WARNING level. |

```json
{
  "working_dir": "/Users/you/code/app",
  "working_dir_source": "files",
  "working_dir_warnings": []
}
```

A refusal carries `error_code`, `candidates` (directories you could pass as `working_dir`: the candidate roots, the declared roots, or the server cwd), `tried` (the tiers visited) and `warnings` in the error's metadata:

```json
{
  "error_code": "ambiguous",
  "candidates": ["/Users/you/code/app-a", "/Users/you/code/app-b"],
  "tried": ["explicit: not given", "continuation: no stored directory"],
  "warnings": []
}
```

Derivation needs no live MCP session and finishes before the CLI is spawned, so blocking, hybrid and in-process use derive the same directory.

### Search budget

The whole derivation is limited to about 2 seconds of filesystem work: it stats a handful of marker names per ancestor, at most 12 ancestors per file, and never scans a directory. When the budget runs out, the call fails with `underivable` (the message says the project-root search timed out and asks for `working_dir`) and does **not** fall through to the manifest, `CLINK_DEFAULT_WORKING_DIR` or the server cwd. An outer 2.5 s backstop covers a probe stuck inside the kernel and raises the same error.

### Per-host behavior

What clink can see depends on the host that launched it:

| Host | What clink sees | With no `working_dir` and no files |
|---|---|---|
| **Claude Code** | The server's cwd, its `PWD` and `CLAUDE_PROJECT_DIR` all equal the directory `claude` was launched in (verified on 5 of 5 live servers). That is a launch directory, not necessarily the git root, and it is frozen for the session. Roots are answered only through the modern-era exchange, which clink uses by default (see [MCP roots](#mcp-roots)); that single root is the launch directory, so launching from a repository subdirectory fences calls to that subdirectory (a file elsewhere in the same repository is `roots_conflict`; pass `working_dir` to override). | By default, the launch directory as `roots`, with no warning. The launch directory as `server_cwd` with the warning when `CLINK_MCP_ROOTS=legacy` or `off`, or when the host does not answer. A launch directory that is broad (your home directory, say) is refused as `underivable`. |
| **Gemini CLI**, **opencode** (v1 and v2) | Advertise roots and answer with the project directory. | The project directory as `roots`, when the roots request succeeds. |
| **Codex** | Advertises no roots and passes no project variable. | Nothing identifies the project beyond the files you pass. The server's own cwd applies as `server_cwd` (with the warning) if it is a narrow directory, otherwise the call is `underivable`; the cwd Codex gives the server is unverified. Pass files or `working_dir`, or set `CLINK_DEFAULT_WORKING_DIR`. |
| **Claude Desktop** | The process cwd is `/` and no roots are inferred. | `underivable`: the substituted install directory is never used. Pass files or `working_dir`, or set `CLINK_DEFAULT_WORKING_DIR`. |
| **VS Code**, **Cursor** | Documentation only: documented to provide roots, not verified. | Not verified. |

**Observed with Claude Code 2.1.312** (real server from a working tree, a fake CLI that records its cwd and `PWD`, launch directory without project markers). A file in a git repository, no `working_dir`: source `files`, the repository root, no warnings; the CLI's real cwd and `PWD` were that root. A file in a git worktree while the server was launched from the main checkout: source `files`, the worktree root. No files with `CLINK_MCP_ROOTS=auto` (what applies by default): source `roots`, the launch directory, no warnings (3 of 3 runs; from a repository subdirectory the root is that subdirectory, not the git root), and a file outside the single root is `roots_conflict`. No files with `CLINK_MCP_ROOTS=legacy` (observed with the variable unset, before the default became `auto`): source `server_cwd`, the launch directory, with the warning (reproduced 3 times, including a launch from a repository subdirectory); `off` was not run separately. Blocking and hybrid execution returned identical provenance. In a later interactive Claude Code session launched from the repository root (the default `auto`), a call with no files and no `working_dir` reported source `roots`, the repository root and no warnings, and a call passing a file inside the repository reported source `files` and the same root.

**Worktree subagents (known limit).** Claude Code freezes the launch directory when the session starts, so a subagent working in a git worktree still has the main checkout as the server's cwd, and clink cannot know that. When such a call passes files, the files tier selects the worktree (its nearest `.git` pointer file) and the main-checkout server cwd is skipped. When it passes no files, clink uses the stale launch directory: with source `roots` and no warning under the default `auto` (Claude Code's single root is the launch directory), or with source `server_cwd` and only the generic server-cwd warning under `CLINK_MCP_ROOTS=legacy` or `off`. Pass files or `working_dir` from worktree subagents; `CLINK_WORKING_DIR_FALLBACK=refuse` turns the `server_cwd` fallback into an error but does not affect the `roots` case.

### MCP roots

MCP hosts can declare **roots**, the folders they have open. clink uses them opportunistically as a fence and a fallback. A failure to *obtain* roots never fails a call (the one failure clink cannot see is a host that mishandles the input-request exchange; see the Claude Code caveats below), but roots can still make a call refuse (`roots_conflict`, `ambiguous`), and an explicit `working_dir` overrides them.

**Files select, roots fence.** The files you pass name the project you mean; roots bound it.

- A project root inside a declared root stands.
- A declared root inside the project narrows it to the innermost root that contains all the files (source `roots`, with a warning naming the project root). clink never widens beyond a root.
- Any file outside every root fails with `roots_conflict` (one stray file is enough, even when the others are inside a root), as do files that no single root contains when the project is wider than every root. A worktree nested inside a root is selected by its files and stands.
- A marker-less root holding several repositories, with files in one of them, yields that repository's root (source `files`).
- Manifest, environment-default, stored and server-cwd candidates outside every root are skipped with a warning naming the roots.
- With **no files** and no valid continuation directory, a single usable root is the directory (source `roots`); several usable roots are `ambiguous`; broad roots (such as your home directory) are dropped with a warning.
- An explicit `working_dir` is never fenced: it triggers no roots request and no roots warning.

**Root URIs** are accepted only as `file://` with an empty or `localhost` host, percent-decoded as UTF-8. Relative results, NUL bytes and encoded path separators (`%2F`) are rejected, and Windows drive-letter and UNC forms map to native paths. A rejected URI is reported with its reason; a reply containing any non-`file` URI is treated as unavailable as a whole.

**`CLINK_MCP_ROOTS`** (default `auto`) selects how clink asks:

- `auto`: ask with a server request (`roots/list`) where the connection can carry one, gated on the roots capability the host declared, waiting at most 3 seconds. On a modern-era connection that cannot carry server requests, ask through the input-request exchange instead: clink returns a roots request instead of running the call, and the host re-executes the call with its answer. The era comes from the negotiated protocol version, never from whether the connection can carry server requests, and the exchange is never used on a legacy-era connection. A retry that arrives without a roots reply counts as unavailable, with a warning: clink never asks twice.
- `legacy`: the server request only, never the exchange. A modern-era connection that cannot carry server requests (Claude Code's) is never asked, so no roots arrive from it.
- `off`: never ask.

An unrecognised value behaves as `legacy` (the conservative mode, not the default) and adds a warning naming it. The value is trimmed and case-insensitive; empty counts as unset. To opt out of the exchange, set `CLINK_MCP_ROOTS=legacy`; to stop asking for roots at all, set `off`.

**Never fails a call.** A failure after a request or reply (timeout, protocol error, a non-`file` or invalid URI, a retry with no reply) is logged at debug level, adds a warning, and derivation continues as if roots were unavailable. A host that did not declare roots, a connection with no back-channel, and an empty list are silent.

**Resolved in the foreground.** Roots are requested during the originating request, before the call becomes a background job and before any conversation state is written, because the exchange re-executes the call (the conversation records the turn once). They are not cached and list-change notifications are not subscribed to, so changed roots are honoured on the next call. A background job uses the roots its originating request resolved and never asks itself.

**Claude Code (verified against 2.1.312).** Claude Code negotiates the 2026-07-28 protocol era, where a server cannot send `roots/list`: the request raises `NoBackChannelError` immediately even though roots are declared (observed in every variant, including a capability-gated one). Claude Code does answer the input-request exchange: it declares `roots` (`listChanged`), replies within milliseconds with one root, `file://<launch directory>` (percent-encoded, no name), and retries the call with the same arguments and the echoed request state. Only the exchange can reach Claude Code's roots, which is why the exchange is used by default: under `legacy` or `off`, roots are never used on Claude Code. Caveats:

- The reply is the **launch directory, not the git root**. For Claude Code, roots add provenance (source `roots`) and the fence rather than new information, and the fence is the launch directory: a launch from a repository subdirectory fences calls to that subdirectory.
- The evidence covers **Claude Code 2.1.312 only**: a default-model session, reproduced in 5 sessions (launch directories with and without a space, one inside a git subdirectory, two calls in one session). `auto` applies to every modern-era host that declares roots, and no other modern-era host has been verified. A host that declares roots but mishandles the exchange (does not retry the call with the roots reply, say) would fail the clink call, and clink cannot detect that from its side. Set `CLINK_MCP_ROOTS=legacy` or `off` for such a host.

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
