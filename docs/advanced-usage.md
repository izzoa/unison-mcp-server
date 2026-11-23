# Advanced Usage Guide

This guide covers advanced features, configuration options, and workflows for power users of the Unison MCP server.

## Table of Contents

- [Model Configuration](#model-configuration)
- [Model Usage Restrictions](#model-usage-restrictions)
- [Thinking Modes](#thinking-modes)
- [Tool Parameters](#tool-parameters)
- [Context Revival: AI Memory Beyond Context Limits](#context-revival-ai-memory-beyond-context-limits)
- [Collaborative Workflows](#collaborative-workflows)
- [Working with Large Prompts](#working-with-large-prompts)
- [Vision Support](#vision-support)
- [Web Search Integration](#web-search-integration)
- [System Prompts](#system-prompts)

## Model Configuration

**For basic configuration**, see the [Configuration Guide](configuration.md) which covers API keys, model selection, and environment variables.

This section focuses on **advanced model usage patterns** for power users:

**Per-Request Model Override:**
Regardless of your default configuration, you can specify models per request:
- "Use **pro** for deep security analysis of auth.py"
- "Use **flash** to quickly format this code"
- "Use **o3** to debug this logic error"
- "Review with **o4-mini** for balanced analysis"
- "Use **gpt4.1** for comprehensive codebase analysis"

**Claude's Auto Mode Decision Matrix:**

| Model | Provider | Context | Strengths | Auto Mode Usage |
|-------|----------|---------|-----------|------------------|
| **`pro`** (Gemini 3.1 Pro Preview) | Google | 1M tokens | Extended thinking (up to 32K tokens), deep analysis | Complex architecture, security reviews, deep debugging |
| **`flash`** (Gemini 2.5 Flash) | Google | 1M tokens | Ultra-fast responses with thinking | Quick checks, formatting, simple analysis |
| **`flash3`** (Gemini 3 Flash Preview) | Google | 1M tokens | Fast Gemini 3 model with thinking | Quick analysis with stronger reasoning than Flash 2.5 |
| **`flashlite`** (Gemini 2.5 Flash Lite) | Google | 1M tokens | Lightweight, cost-efficient, high throughput, vision supported | Fast, low-cost bulk processing |
| **`o3`** | OpenAI | 200K tokens | Strong logical reasoning | Debugging logic errors, systematic analysis |
| **`o3-mini`** | OpenAI | 200K tokens | Balanced speed/quality | Moderate complexity tasks |
| **`o4-mini`** | OpenAI | 200K tokens | Latest reasoning model | Optimized for shorter contexts |
| **`gpt4.1`** | OpenAI | 1M tokens | Latest GPT-4 with extended context | Large codebase analysis, comprehensive reviews |
| **`gpt5.2`** (GPT-5.2) | OpenAI | 400K tokens | Flagship reasoning model with configurable thinking effort | Complex problems, balanced agent/coding flows |
| **`gpt5.1-codex`** (GPT-5.1 Codex) | OpenAI | 400K tokens | Agentic coding specialization (Responses API) | Advanced coding tasks, structured code generation |
| **`gpt5.1-codex-mini`** (GPT-5.1 Codex mini) | OpenAI | 400K tokens | Cost-efficient Codex variant with streaming | Balanced coding tasks, cost-conscious development |
| **`gpt5`** (GPT-5) | OpenAI | 400K tokens | Advanced model with reasoning support | Complex problems requiring advanced reasoning |
| **`gpt5-mini`** (GPT-5 Mini) | OpenAI | 400K tokens | Efficient variant with reasoning | Balanced performance and capability |
| **`gpt5-nano`** (GPT-5 Nano) | OpenAI | 400K tokens | Fastest, cheapest GPT-5 variant | Summarization and classification tasks |
| **`fable`** (Claude Fable 5.1) | Anthropic | 1M tokens | Anthropic's most capable model, adaptive thinking | Hardest reasoning, deep architectural analysis |
| **`opus`** (Claude Opus 5.5) | Anthropic | 1M tokens | Flagship Claude with adaptive thinking | Complex analysis, critical code reviews |
| **`sonnet`** (Claude Sonnet 5.5) | Anthropic | 1M tokens | Balanced Claude, adaptive thinking + vision | Everyday deep work, agentic flows |
| **`haiku`** (Claude Haiku 5.5) | Anthropic | 1M tokens | Fast, cost-efficient Claude | Quick checks, summaries |
| **`grok-4`** | X.AI | 256K tokens | Latest flagship Grok model with reasoning, vision | Complex analysis, reasoning tasks |
| **`grok-4.1-fast-reasoning`** | X.AI | 2M tokens | High-performance Grok 4.1 Fast Reasoning with vision | Fast responses and light reasoning |
| **`llama`** (Llama 3.2) | Custom/Local | 128K tokens | Local inference, privacy | On-device analysis, cost-free processing |
| **Any model** | OpenRouter | Varies | Access to GPT-4, Claude, Llama, etc. | User-specified or based on task requirements |

**Mix & Match Providers:** Use multiple providers simultaneously! Set both `OPENROUTER_API_KEY` and `CUSTOM_API_URL` to access 
cloud models (expensive/powerful) AND local models (free/private) in the same conversation.

**Model Capabilities:**
- **Gemini Models**: Support thinking modes (minimal to max), web search, 1M context
  - **Pro 3.1 (preview)**: Deep analysis with max 32K thinking tokens
  - **Flash 2.5**: Ultra-fast with thinking support (24K thinking tokens)
  - **Flash 3 (preview)**: Fast Gemini 3 model (24K thinking tokens)
  - **Flash Lite 2.5**: Lightweight, cost-efficient, high throughput; supports thinking and vision
- **Claude Models (native)**: Thinking modes mapped onto adaptive-thinking effort levels (extended-thinking budgets on Sonnet 4.6 and Haiku 4.5), vision support
  - **Fable 5.1 / Opus 5.5 / Sonnet 5.5 / Haiku 5.5**: current models behind the bare aliases; 1M context, up to 128K output
  - **Fable 5 / Opus 5 / Sonnet 5**: superseded point releases, pinned via `fable5` / `opus5` / `sonnet5`
  - **Opus 4.8 / Sonnet 4.6**: previous generation, still first-class
  - **Haiku 4.5**: 200K context, budget-based extended thinking, pinned via `haiku4`
- **O3/O4 Models**: Excellent reasoning, systematic analysis, 200K context
- **GPT-4.1**: Extended context window (1M tokens), general capabilities
- **GPT-5.2 Series**: Latest flagship reasoning models, 400K context
  - **GPT-5.2**: Flagship model with configurable thinking effort and vision
  - **GPT-5.1 Codex**: Agentic coding specialization (Responses API, non-streaming)
  - **GPT-5.1 Codex mini**: Cost-efficient Codex variant with streaming support
- **GPT-5 Series**: Advanced reasoning models, 400K context
  - **GPT-5**: Full-featured with reasoning support and vision
  - **GPT-5 Mini**: Balanced efficiency and capability
  - **GPT-5 Nano**: Optimized for fast, low-cost tasks
- **Grok-4 / Grok-4.1-fast-reasoning**: Extended thinking support, vision capabilities (256K / 2M context)

## Model Usage Restrictions

**For complete restriction configuration**, see the [Configuration Guide](configuration.md#model-usage-restrictions).

**Advanced Restriction Strategies:**

**Cost Control Examples:**
```env
# Development: Allow experimentation
GOOGLE_ALLOWED_MODELS=flash,pro
OPENAI_ALLOWED_MODELS=o4-mini,o3-mini

# Production: Cost-optimized  
GOOGLE_ALLOWED_MODELS=flash
OPENAI_ALLOWED_MODELS=o4-mini

# High-performance: Quality over cost
GOOGLE_ALLOWED_MODELS=pro
OPENAI_ALLOWED_MODELS=o3,o4-mini
```

**Important Notes:**
- Restrictions apply to all usage including auto mode
- `OPENROUTER_ALLOWED_MODELS` only affects models defined in `conf/openrouter_models.json`
- Custom local models (from `conf/custom_models.json`) are not affected by OpenRouter restrictions

## Thinking Modes

**Claude automatically manages thinking modes based on task complexity**, but you can also manually control Gemini's reasoning depth to balance between response quality and token consumption. Each thinking mode uses a different amount of tokens, directly affecting API costs and response time.

### Thinking Modes & Token Budgets

These only apply to Gemini and native Anthropic models that support extended thinking; other providers ignore `thinking_mode`. Each mode is a share of the model's maximum thinking budget, not a fixed token count:

| Mode | Share of max thinking budget | Use Case |
|------|------------------------------|----------|
| `minimal` | about 0.5% | Simple, straightforward tasks |
| `low` | 8% | Basic reasoning tasks |
| `medium` | 33% | **Default for simple tools such as `chat`** - Most development tasks |
| `high` | 67% | Complex problems requiring thorough analysis (default for `thinkdeep` and for the expert-analysis call of the workflow tools) |
| `max` | 100% | Exhaustive reasoning |

For example, Gemini 2.5 Pro (32,768 max thinking tokens) gives about 160, 2,600, 10,800, 22,000 and 32,800 tokens. `gemini-3-pro-preview` upgrades `medium` to `high`. Adaptive Claude models map the mode to an effort level instead of a token budget, and other Claude models enforce a 1,024-token minimum. Higher modes use more tokens, which raises cost and response time.

### How to Use Thinking Modes

**Claude automatically selects appropriate thinking modes**, but you can override this by explicitly requesting a specific mode in your prompts. Remember: higher thinking modes = more tokens = higher cost but better quality:

#### Optimizing Token Usage & Costs

**In most cases, let Claude automatically manage thinking modes** for optimal balance of cost and quality. Override manually when you have specific requirements:

**Use lower modes (`minimal`, `low`) to save tokens when:**
- Doing simple formatting or style checks
- Getting quick explanations of basic concepts
- Working with straightforward code
- You need faster responses
- Working within tight token budgets

**Use higher modes (`high`, `max`) when quality justifies the cost:**
- Debugging complex issues (worth the extra tokens to find root causes)
- Reviewing security-critical code (cost of tokens < cost of vulnerabilities)
- Analyzing system architecture (comprehensive analysis saves development time)
- Finding subtle bugs or edge cases
- Working on performance optimizations

**Token Cost Examples:**
- `minimal` (about 0.5% of the budget) vs `max` (100%) = roughly a 200x difference in thinking tokens
- For a simple formatting check, using `minimal` instead of the default `medium` saves about a third of the model's thinking budget
- For critical security reviews, the extra tokens in `high` or `max` mode are a worthwhile investment

**Examples by scenario:**
```
# Quick style check with o3
"Use flash to review formatting in utils.py"

# Security audit with o3
"Get o3 to do a security review of auth/ with thinking mode high"

# Complex debugging, letting claude pick the best model
"Use unison to debug this race condition with max thinking mode"

# Architecture analysis with Gemini Pro
"Analyze the entire src/ directory architecture with high thinking using pro"
```

## Tool Parameters

All tools that work with files support **both individual files and entire directories**. The server automatically expands directories, filters for relevant code files, and manages token limits.

### File-Processing Tools

These are multi-step workflow tools. Every call carries `step`, `step_number`, `total_steps`, `next_step_required` and `findings`; files go in `relevant_files` (and `files_checked`), not in a `files` parameter, and `relevant_files` is required in step 1 for most of them. They also share `relevant_context`, `images`, `confidence` and `use_assistant_model` (where the tool uses them), `model`, `temperature`, `thinking_mode` and `continuation_id`. See `docs/tools/<tool>.md` for each tool's full parameter list.

**`analyze`** - Analyze files or directories
- `step` / `findings`: What to analyze and what you found (step 1 carries the request text)
- `relevant_files`: Files or directories to analyze (required in step 1)
- `model`: auto|pro|flash|flash3|flashlite|o3|o3-mini|o4-mini|gpt4.1|gpt5.2|gpt5.1-codex|gpt5.1-codex-mini|gpt5|gpt5-mini|gpt5-nano|opus|sonnet|haiku|fable (default: server default)
- `analysis_type`: architecture|performance|security|quality|general
- `output_format`: summary|detailed|actionable
- `thinking_mode`: minimal|low|medium|high|max (default: high; honored by Gemini and Anthropic models, ignored by providers without a thinking control)

```
"Analyze the src/ directory for architectural patterns" (auto mode picks best model)
"Use flash to quickly analyze main.py and tests/ to understand test coverage" 
"Use o3 for logical analysis of the algorithm in backend/core.py"
"Use pro for deep analysis of the entire backend/ directory structure"
```

**`codereview`** - Review code files or directories
- `relevant_files`: Files or directories to review (required in step 1)
- `model`: auto|pro|flash|flash3|flashlite|o3|o3-mini|o4-mini|gpt4.1|gpt5.2|gpt5.1-codex|gpt5.1-codex-mini|gpt5|gpt5-mini|gpt5-nano|opus|sonnet|haiku|fable (default: server default)
- `review_type`: full|security|performance|quick
- `focus_on`: Specific aspects to focus on
- `standards`: Coding standards to enforce
- `severity_filter`: critical|high|medium|low|all
- `review_validation_type`: external|internal (default: external)
- `thinking_mode`: minimal|low|medium|high|max (default: high; honored by Gemini and Anthropic models, ignored by providers without a thinking control)

```
"Review the entire api/ directory for security issues" (auto mode picks best model)
"Use pro to review auth/ for deep security analysis"
"Use o3 to review logic in algorithms/ for correctness"
"Use flash to quickly review src/ with focus on performance, only show critical issues"
```

**`debug`** - Debug with file context
- `step` / `findings`: Describe the issue (error messages, stack traces and logs go in `step`/`findings`), then the evidence you gather
- `relevant_files`: Files or directories related to the issue
- `hypothesis` and `confidence`: Your current best theory and how sure you are
- `model`: auto|pro|flash|flash3|flashlite|o3|o3-mini|o4-mini|gpt4.1|gpt5.2|gpt5.1-codex|gpt5.1-codex-mini|gpt5|gpt5-mini|gpt5-nano|opus|sonnet|haiku|fable (default: server default)
- `thinking_mode`: minimal|low|medium|high|max (default: high; honored by Gemini and Anthropic models, ignored by providers without a thinking control)

```
"Debug this logic error with context from backend/" (auto mode picks best model)
"Use o3 to debug this algorithm correctness issue"
"Use pro to debug this complex architecture problem"
```

**`thinkdeep`** - Extended analysis with file context
- `step`: Your current thinking to extend and validate (step 1)
- `relevant_files`: Files or directories for context
- `model`: auto|pro|flash|flash3|flashlite|o3|o3-mini|o4-mini|gpt4.1|gpt5.2|gpt5.1-codex|gpt5.1-codex-mini|gpt5|gpt5-mini|gpt5-nano|opus|sonnet|haiku|fable (default: server default)
- `problem_context`: Additional context
- `focus_areas`: Specific aspects to focus on
- `thinking_mode`: minimal|low|medium|high|max (default: high; honored by Gemini and Anthropic models, ignored by providers without a thinking control)

```
"Think deeper about my design with reference to src/models/" (auto mode picks best model)
"Use pro to think deeper about this architecture with extended thinking"
"Use o3 to think deeper about the logical flow in this algorithm"
```

**`testgen`** - Comprehensive test generation with edge case coverage
- `step`: What to test, the testing objectives and the scope (step 1)
- `relevant_files`: Code files or directories to generate tests for (required in step 1); add existing test files here and say in `step`/`findings` that they are the style reference
- `model`: auto|pro|flash|flash3|flashlite|o3|o3-mini|o4-mini|gpt4.1|gpt5.2|gpt5.1-codex|gpt5.1-codex-mini|gpt5|gpt5-mini|gpt5-nano|opus|sonnet|haiku|fable (default: server default)
- `thinking_mode`: minimal|low|medium|high|max (default: high; honored by Gemini and Anthropic models, ignored by providers without a thinking control)

```
"Generate tests for User.login() method with edge cases" (auto mode picks best model)
"Use pro to generate comprehensive tests for src/payment.py with max thinking mode"
"Use o3 to generate tests for algorithm correctness in sort_functions.py"
"Generate tests following patterns from tests/unit/ for new auth module"
```

**`refactor`** - Intelligent code refactoring with decomposition focus
- `step`: Refactoring goals, context and areas of focus (step 1)
- `relevant_files`: Code files or directories to analyze for refactoring opportunities (required in step 1)
- `refactor_type`: codesmells|decompose|modernize|organization (default: codesmells)
- `model`: auto|pro|flash|flash3|flashlite|o3|o3-mini|o4-mini|gpt4.1|gpt5.2|gpt5.1-codex|gpt5.1-codex-mini|gpt5|gpt5-mini|gpt5-nano|opus|sonnet|haiku|fable (default: server default)
- `focus_areas`: Specific areas to focus on (e.g., 'performance', 'readability', 'maintainability', 'security')
- `style_guide_examples`: Optional existing code files to use as style/pattern reference
- `thinking_mode`: minimal|low|medium|high|max (default: high; honored by Gemini and Anthropic models, ignored by providers without a thinking control)
- `continuation_id`: Thread continuation ID for multi-turn conversations

```
"Analyze legacy codebase for decomposition opportunities" (auto mode picks best model)
"Use pro to identify code smells in the authentication module with max thinking mode"
"Use pro to modernize this JavaScript code following examples/modern-patterns.js"
"Refactor src/ for better organization, focus on maintainability and readability"
```

## Context Revival: AI Memory Beyond Context Limits

**The Unison MCP Server's most revolutionary feature** is its ability to maintain conversation context even after Claude's memory resets. This enables truly persistent AI collaboration across multiple sessions and context boundaries.

### **The Breakthrough**

Even when Claude's context resets or compacts, conversations can continue seamlessly because other models (O3, Gemini) have access to the complete conversation history stored in memory and can "remind" Claude of everything that was discussed.

### Key Benefits

- **Persistent conversations** across Claude's context resets
- **Cross-tool continuation** with full context preservation
- **Multi-session workflows** that maintain complete history
- **True AI orchestration** where models can build on each other's work
- **Seamless handoffs** between different tools and models

### Quick Example

```
Session 1: "Design a RAG system with gemini pro"
[Claude's context resets]
Session 2: "Continue our RAG discussion with o3"
→ O3 receives the full history and reminds Claude of everything discussed
```

**📖 [Read the complete Context Revival guide](context-revival.md)** for detailed examples, technical architecture, configuration options, and best practices.

**See also:** [AI-to-AI Collaboration Guide](ai-collaboration.md) for multi-model coordination and conversation threading.

## Collaborative Workflows

### Design → Review → Implement
```
Think hard about designing and developing a fun calculator app in swift. Review your design plans with o3, taking in
their suggestions but keep the feature-set realistic and doable without adding bloat. Begin implementing and in between
implementation, get a codereview done by Gemini Pro and chat with Flash if you need to for creative directions.   
```

### Code → Review → Fix
```
Implement a new screen where the locations taken from the database display on a map, with pins falling from
the top and landing with animation. Once done, codereview with gemini pro and o3 both and ask them to critique your
work. Fix medium to critical bugs / concerns / issues and show me the final product
```

### Debug → Analyze → Solution → Precommit Check → Publish
```
Take a look at these log files saved under subfolder/diagnostics.log there's a bug where the user says the app
crashes at launch. Think hard and go over each line, tallying it with corresponding code within the project. After
you've performed initial investigation, ask gemini pro to analyze the log files and the related code where you 
suspect lies the bug and then formulate and implement a bare minimal fix. Must not regress. Perform a precommit
with unison in the end using gemini pro to confirm we're okay to publish the fix 
```

### Refactor → Review → Implement → Test
```
Use unison to analyze this legacy authentication module for decomposition opportunities. The code is getting hard to 
maintain and we need to break it down. Use gemini pro with high thinking mode to identify code smells and suggest 
a modernization strategy. After reviewing the refactoring plan, implement the changes step by step and then 
generate comprehensive tests with unison to ensure nothing breaks.
```

### Tool Selection Guidance
To help choose the right tool for your needs:

**Decision Flow:**
1. **Have a specific error/exception?** → Use `debug`
2. **Want to find bugs/issues in code?** → Use `codereview`
3. **Want to understand how code works?** → Use `analyze`
4. **Need comprehensive test coverage?** → Use `testgen`
5. **Want to refactor/modernize code?** → Use `refactor`
6. **Have analysis that needs extension/validation?** → Use `thinkdeep`
7. **Want to brainstorm or discuss?** → Use `chat`

**Key Distinctions:**
- `analyze` vs `codereview`: analyze explains, codereview prescribes fixes
- `chat` vs `thinkdeep`: chat is open-ended, thinkdeep extends specific analysis
- `debug` vs `codereview`: debug diagnoses runtime errors, review finds static issues
- `testgen` vs `debug`: testgen creates test suites, debug just finds issues and recommends solutions
- `refactor` vs `codereview`: refactor suggests structural improvements, codereview finds bugs/issues
- `refactor` vs `analyze`: refactor provides actionable refactoring steps, analyze provides understanding

## Vision Support

The Unison MCP server supports vision-capable models for analyzing images, diagrams, screenshots, and visual content. Vision support works seamlessly with all tools and conversation threading.

**Supported Models:**
- **Gemini 3.0 Pro & Flash**: Excellent for diagrams, architecture analysis, UI mockups (up to 20MB total)
- **OpenAI O3/O4 series**: Strong for visual debugging, error screenshots (up to 20MB total)
- **Claude models (native)**: All ten curated models (Fable, Opus, Sonnet, Haiku) are vision-capable via `ANTHROPIC_API_KEY`
- **Claude models via OpenRouter**: Good for code screenshots, visual analysis (up to 5MB total)
- **Custom models**: Support varies by model, with 40MB maximum enforced for abuse prevention

**Usage Examples:**
```bash
# Debug with error screenshots
"Use unison to debug this error with the stack trace screenshot and error.py"

# Architecture analysis with diagrams  
"Analyze this system architecture diagram with gemini pro for bottlenecks"

# UI review with mockups
"Chat with flash about this UI mockup - is the layout intuitive?"

# Code review with visual context
"Review this authentication code along with the error dialog screenshot"
```

**Image Formats Supported:**
- **Images**: JPG, PNG, GIF, WebP, BMP, SVG, TIFF
- **Documents**: PDF (where supported by model)
- **Data URLs**: Base64-encoded images from Claude

**Key Features:**
- **Automatic validation**: File type, magic bytes, and size validation
- **Conversation context**: Images persist across tool switches and continuation
- **Budget management**: Automatic dropping of old images when limits exceeded
- **Model capability-aware**: Only sends images to vision-capable models

**Best Practices:**
- Describe images when including them: "screenshot of login error", "system architecture diagram"
- Use appropriate models: Gemini for complex diagrams, O3 for debugging visuals
- Consider image sizes: Larger images consume more of the model's capacity

## Working with Large Prompts

The MCP protocol has a combined request+response limit of approximately 25K tokens. This server intelligently works around this limitation by automatically handling large prompts as files:

**How it works:**
1. When you send a prompt larger than the configured limit (default: 50K characters ~10-12K tokens), the server detects this
2. It responds with a special status asking Claude to save the prompt to a file named `prompt.txt`
3. Claude saves the prompt and resends the request with the file path instead
4. The server reads the file content directly into Gemini's 1M token context
5. The full MCP token capacity is preserved for the response

**Example scenario:**
```
# You have a massive code review request with detailed context
User: "Use gemini to review this code: [50,000+ character detailed analysis]"

# Server detects the large prompt and responds:
Unison MCP: "The prompt is too large for MCP's token limits (>50,000 characters). 
Please save the prompt text to a temporary file named 'prompt.txt' and resend 
the request with an empty prompt string and the absolute file path included 
in the absolute_file_paths parameter, along with any other files you wish to share as context."

# Claude automatically handles this:
- Saves your prompt to /tmp/prompt.txt
- Resends: "Use gemini to review this code" with absolute_file_paths=["/tmp/prompt.txt", "/path/to/code.py"]

# Server processes the large prompt through Gemini's 1M context
# Returns comprehensive analysis within MCP's response limits
```

This feature ensures you can send arbitrarily large prompts to Gemini without hitting MCP's protocol limitations, while maximizing the available space for detailed responses.

## Web Search Integration

**Web search recommendations in `chat`**

Unison itself never performs web searches. The `chat` tool appends web-search guidance to the prompt it sends the consulted model: when the model judges that current documentation, API references or community solutions would strengthen its answer, it tells the calling agent which searches to run and to continue with the `continuation_id`. The workflow tools (`analyze`, `codereview`, `consensus`, `debug`, `docgen`, `planner`, `precommit`, `refactor`, `secaudit`, `testgen`, `thinkdeep`, `tracer`) add no such guidance. For documentation lookups use [`apilookup`](tools/apilookup.md), which tells the calling agent to search with its own web tool.

**How it works in `chat`:**
1. The model analyzes the request and identifies areas where current information would be valuable
2. It answers from its training data
3. If searches would help, it asks the calling agent to run specific ones, marked as valuable or required
4. The agent runs them and continues the thread with the same `continuation_id`

**Example:**
```
User: "Use gemini to chat about this FastAPI async error"

Model response:
[... analysis ...]
Please perform a web search on "FastAPI BackgroundTasks memory leak" and then continue this analysis
using the continuation_id from this response if you find relevant information.
```

The calling agent keeps control over the actual searches; if you want the model to rely on its training data only, say so in the prompt.

## System Prompts

The server uses carefully crafted system prompts to give each tool specialized expertise:

### Prompt Architecture
- **Centralized Prompts**: Each tool's system prompt lives in `systemprompts/` (for example, `systemprompts/chat_prompt.py`)
- **Tool Integration**: Each tool inherits from `BaseTool` and implements `get_system_prompt()`
- **Prompt Flow**: `User Request → Tool Selection → System Prompt + Context → Model Response`

### Specialized Expertise
Each tool has a unique system prompt that defines its role and approach:
- **`thinkdeep`**: Acts as a senior development partner, challenging assumptions and finding edge cases
- **`codereview`**: Expert code reviewer with security/performance focus, uses severity levels
- **`debug`**: Systematic debugger providing root cause analysis and prevention strategies
- **`analyze`**: Code analyst focusing on architecture, patterns, and actionable insights

### Customization
To modify tool behavior, you can:
1. Edit the prompt file in `systemprompts/` (and export it via `systemprompts/__init__.py`) for global changes
2. Override `get_system_prompt()` in a tool class for tool-specific changes
3. Use the `temperature` parameter to adjust response style (not available on `consensus`). The default is 1.0 for every tool, accepted from 0 to 2, and each model's own rule then applies: many models accept only 1.0 and ignore or override other values
