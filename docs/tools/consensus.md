# Consensus Tool - Multi-Model Perspective Gathering

**Get diverse expert opinions from multiple AI models on technical proposals and decisions**

The `consensus` tool orchestrates multiple AI models to provide diverse perspectives on your proposals, enabling structured decision-making through for/against analysis and multi-model expert opinions.

## Thinking Mode

Thinking mode is fixed at `medium` for every consulted model; it is not a parameter.

## Model Recommendation

The models consulted are exactly the roster you pass in `models`; the tool picks none for you. Choose reasoning-strong models for complex decisions that benefit from multiple perspectives and deep analysis.

## How It Works

The consensus tool orchestrates multiple AI models to provide diverse perspectives on your proposals:

1. **Assign stances**: Each model can take a specific viewpoint (`for`, `against`, or `neutral`)
2. **Gather opinions in one call**: A single call (`step_number` 1, `total_steps` 1, `next_step_required` false) consults every model concurrently, each seeing only your proposal and files with its assigned perspective and built-in common-sense guardrails, and returns all verdicts together in `accumulated_responses`, in the order you listed the models
3. **Synthesize results**: Claude (the calling agent) combines all perspectives into a balanced recommendation; the tool does not synthesize for you
4. **Natural language**: Use simple descriptions like "supportive", "critical", or "against" - the calling agent maps them onto the `for`/`against`/`neutral` stance values

## Watch In Action

The following is a hypothetical example designed to demonstrate how one consensus can be built upon another (via [continuation](../context-revival.md)). In this scenario, we start with a _blinded_ consensus, where one model is tasked with taking a **for** stance and another with an **against** stance. This approach allows us to see how each model evaluates a particular option relative to the alternative. We then conduct a second consensus — all initiated by a single prompt and orchestrated by Claude Code in this video — to gather each model’s final conclusions.

<div style="center">
  
  [Unison Consensus Debate](https://github.com/user-attachments/assets/76a23dd5-887a-4382-9cf0-642f5cf6219e)
  
</div>

## Example Prompts

**For/Against Analysis:**
```
Use unison consensus with flash taking a supportive stance and pro being critical to evaluate whether 
we should migrate from REST to GraphQL for our API
```

**Multi-Model Technical Decision:**
```
Get consensus from o3, flash, and pro on our new authentication architecture. Have o3 focus on 
security implications, flash on implementation speed, and pro stay neutral for overall assessment
```

**Natural Language Stance Assignment:**
```
Use consensus tool with gemini being "for" the proposal and grok being "against" to debate 
whether we should adopt microservices architecture
```

```
I want to work on module X and Y, unsure which is going to be more popular with users of my app. 
Get a consensus from gemini supporting the idea for implementing X, grok opposing it, and flash staying neutral
```

## Key Features

- **Stance steering**: Assign specific perspectives (for/against/neutral) to each model
- **Custom stance prompts**: Provide specific instructions for how each model should approach the analysis
- **Ethical guardrails**: Models will refuse to support truly bad ideas regardless of assigned stance
- **Unknown stance handling**: An unrecognised stance runs with the neutral prompt (no warning is emitted)
- **Natural language support**: Say "supportive", "critical", "oppose" or "favor"; the calling agent maps these onto `for`/`against`/`neutral`
- **Concurrent consultation**: Every model is consulted in parallel inside one call (per-model 120s timeout; one model's failure or timeout does not cancel the others), so wall-clock time is roughly the slowest model, not the sum. Batch mode runs one provider batch per distinct model plus the synchronous entries concurrently
- **File context support**: Include relevant files for informed decision-making
- **Image support**: Analyze architectural diagrams, UI mockups, or design documents
- **Conversation continuation**: Build on previous consensus analysis with additional rounds

## Tool Parameters

- `step`: The proposal or question every model will see, written as "Evaluate..." (required). Sent verbatim, plus `relevant_files`, to every model and echoed back as `agent_analysis.initial_analysis`
- `step_number`: Send 1 (required)
- `total_steps`: Send 1 (required). The whole consultation is one step; the server treats step 1 as `total_steps` 1 whatever you send
- `next_step_required`: Send false (required). The server ignores it on step 1, which always finishes the consensus
- `findings`: Your own analysis, kept for your synthesis (required). Never sent to the models; echoed back as `agent_analysis.findings`
- `models`: List of model configurations with optional stance, custom instructions and, per entry, `execution_mode` (required on step 1)
- `relevant_files`: Context files for informed analysis (absolute paths)
- `images`: Visual references like diagrams or mockups (absolute paths)
- `continuation_id`: Continue previous consensus discussions in the same thread. A new round is a new `step_number` 1 call; the models never see earlier rounds or each other's answers, and every round consults and bills all models again
- `execution_mode`: `"batch"` consults the models through OpenRouter's asynchronous Batch API at about half price, one batch per model; also accepted on each `models[]` entry (omitted or `null` inherits the request value). Step 1 only. See the [batch guide](batch.md)
- `batch_handle`: Resume (or, with `cancel`, cancel) a batched run that returned `status: "running"`. Pass only this field
- `wait_seconds`: How long this call may wait for the batches (0-50, default 45)
- `cancel`: With `batch_handle`, cancel locally; OpenRouter has no cancel endpoint, so provider batches continue and are still billed

Temperature is not a parameter: the tool uses its analytical default (1.0), clamped to each consulted model's published range.

A missing, `null` or empty `stance` is the neutral stance, so it counts as a duplicate of an explicit `neutral` for the same model.

`current_model_index` and `model_responses` (the bookkeeping of the old one-model-per-step flow) are no longer published or read. They are not part of the request model or the published input schema (which sets `additionalProperties: false`), so an MCP client that still sends them gets an `Input validation error` from the server's schema check before the tool runs; only a direct, in-process construction of the request model ignores them silently.

## Response

Every consensus response (the step-1 verdicts, the later-step replies and a batch run rebuilt on resume) carries `consensus_workflow_status` with the same value as `status`, so the two never disagree. The batch status payloads (`running`, and the `cancelled` one a cancel returns) are the exception: they are not consensus responses and have no `consensus_workflow_status`:

| `status` (and `consensus_workflow_status`, except for the batch status payloads) | Returned by | Meaning |
|---|---|---|
| `consensus_workflow_complete` | step 1 | At least one model answered; `accumulated_responses` holds every verdict (failed models appear with an error entry) |
| `all_providers_unavailable` | step 1 | Every consultation failed; there is nothing to synthesize |
| `running` | step 1, batch mode only | The verdicts are not ready; call again with only the `batch_handle` (a batch status payload: no `consensus_workflow_status`) |
| `ready_for_synthesis` | step 2 or later with `next_step_required` true | No model is consulted; synthesize what you already have, or start a new step 1 |
| `consensus_workflow_complete` | step 2 or later with `next_step_required` false | No model is consulted; the response only closes the workflow (`consensus_complete`, `complete_consensus`) and tells you to synthesize |

`metadata` always has the same shape: `tool_name`, `workflow_type` (`multi_model_consensus`), `models_consulted` (the `model:stance` list from the request), `consensus_complete` (true) and `total_models`. There is no in-progress shape: the consultation is one step, so `consultation_step`, `total_consultation_steps` and `models_to_consult` are never returned.

Later steps are accepted only for compatibility and consult no model. A `step_number` of 2 or more with `next_step_required` true answers `status: "ready_for_synthesis"` and says no further models will be consulted; with `next_step_required` false it answers `status: "consensus_workflow_complete"`; `consensus_workflow_status` carries the same value in both cases. Neither consults a model. `accumulated_responses` and the `metadata` roster come from the server process that ran step 1: that process repeats the verdicts it holds, while a process that did not run step 1 returns no `accumulated_responses` and reports an empty `models_consulted` and `total_models` 0. To run another round, send a new step 1 call. A call carrying only a `batch_handle` rebuilds the step-1 response from the stored record.

## MCP annotations

Consensus is published with `readOnlyHint: false`. A plain run only calls provider APIs, but `execution_mode: "batch"` submits a billed provider batch and persists its state to the batch store (`.unison/batches.db`), so the tool is not read-only and a client may ask for confirmation before calling it.

## Model Configuration Examples

**Basic For/Against:**
```json
[
    {"model": "flash", "stance": "for"},
    {"model": "pro", "stance": "against"}
]
```

**Custom Stance Instructions:**
```json
[
    {"model": "o3", "stance": "for", "stance_prompt": "Focus on implementation benefits and user value"},
    {"model": "flash", "stance": "against", "stance_prompt": "Identify potential risks and technical challenges"}
]
```

**Neutral Analysis:**
```json
[
    {"model": "pro", "stance": "neutral"},
    {"model": "o3", "stance": "neutral"}
]
```

**Batched models (half price, slower):**
```json
[
    {"model": "anthropic/claude-opus-4.5", "stance": "for", "execution_mode": "batch"},
    {"model": "openai/gpt-5", "stance": "against", "execution_mode": "batch"}
]
```
Each distinct batched model becomes one provider batch, joined under one `batch_handle`. If they do not finish within the bounded wait the call returns `status: "running"`; resume with only the handle, and the final result is rebuilt from the stored record even after a server restart. Only models OpenRouter lists as `:batch` qualify: a family alias (`gemini`, `opus`, ...) resolves under batch to the latest batch-listed model of its family, and each entry reports `model_requested` / `model_resolved` when it did. See the [batch guide](batch.md).

## Usage Examples

**Architecture Decision:**
```
"Get consensus from pro and o3 on whether to use microservices vs monolith for our e-commerce platform"
```

**Technology Migration:**
```
"Use consensus with flash supporting and pro opposing to evaluate migrating from MySQL to PostgreSQL"
```

**Feature Priority:**
```
"Get consensus from multiple models on whether to prioritize mobile app vs web dashboard development first"
```

**With Visual Context:**
```
"Use consensus to evaluate this new UI design mockup - have flash support it and pro be critical"
```

## Best Practices

- **Provide detailed context**: Include project constraints, requirements, and background
- **Use balanced stances**: Mix supportive and critical perspectives for thorough analysis
- **State what matters in the proposal**: Name the aspects to weigh (security, performance, etc.) in `step`, or steer one model with a custom `stance_prompt`
- **Include relevant files**: Provide code, documentation, or specifications for context
- **Build on discussions**: Use continuation for follow-up analysis and refinement
- **Leverage visual context**: Include diagrams, mockups, or design documents when relevant

## Ethical Guardrails

The consensus tool includes built-in ethical safeguards:
- Models won't support genuinely harmful proposals regardless of assigned stance
- An unrecognised stance runs with the neutral prompt (no warning is emitted)
- Focus on constructive technical decision-making

## When to Use Consensus vs Other Tools

- **Use `consensus`** for: Multi-perspective analysis, for/against evaluation, major technical decisions
- **Use `chat`** for: Open-ended discussions and brainstorming
- **Use `thinkdeep`** for: Extending specific analysis with deeper reasoning
- **Use `analyze`** for: Understanding existing systems without debate
