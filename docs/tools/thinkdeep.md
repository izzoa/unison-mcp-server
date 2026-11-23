# ThinkDeep Tool - Extended Reasoning Partner

**Get a second opinion to augment Claude's own extended thinking**

The `thinkdeep` tool provides extended reasoning capabilities, offering a second perspective to augment Claude's analysis. It's designed to challenge assumptions, find edge cases, and provide alternative approaches to complex problems.

## Thinking Mode

**Default is `high` (about two thirds of the model's thinking budget) for deep analysis.** Claude will automatically choose the best mode based on complexity - use `low` for quick validations, `medium` for standard problems, `high` for complex issues (default), or `max` for extremely complex challenges requiring deepest analysis.

## Example Prompt

```
Think deeper about my authentication design with pro using max thinking mode and brainstorm to come up 
with the best architecture for my project
```

## Key Features

- **Works with any configured model**, using extended thinking on native Gemini and Anthropic models (see `thinking_mode`)
- **Provides a second opinion** on Claude's analysis
- **Challenges assumptions** and identifies edge cases Claude might miss
- **Offers alternative perspectives** and approaches
- **Validates architectural decisions** and design patterns
- **File reference support** (via `relevant_files` / `files_checked`): `"Use gemini to think deeper about my API design with reference to api/routes.py"`
- **Image support**: Analyze architectural diagrams, flowcharts, design mockups: `"Think deeper about this system architecture diagram with gemini pro using max thinking mode"`
- **Enhanced Critical Evaluation (v2.10.0)**: After the consulted model's analysis, Claude is prompted to critically evaluate the suggestions, consider context and constraints, identify risks, and synthesize a final recommendation - ensuring a balanced, well-considered solution

## Tool Parameters

**Workflow Investigation Parameters (used during step-by-step process):**
- `step`: Current thinking step (required for each step). In step 1 it carries your current thinking/analysis to extend and validate
- `step_number`: Current step number (required)
- `total_steps`: Estimated total steps (required, adjustable)
- `next_step_required`: Whether another step is needed (required)
- `findings`: Discoveries and insights collected in this step (required)
- `files_checked`: All files examined during the investigation
- `relevant_files`: Files directly relevant to the problem (absolute paths)
- `relevant_context`: Methods/functions/classes central to the findings
- `issues_found`: Issues or concerns identified, with severity levels
- `hypothesis`: Current working hypothesis
- `confidence`: exploring/low/medium/high/very_high/almost_certain/certain (`certain` with `next_step_required` false skips expert analysis)
- `images`: Optional images for visual analysis (absolute paths)
- `use_assistant_model`: Whether to run the expert analysis phase (default: true; set to false to rely on your own investigation only and skip the expert model call)

**Configuration:**
- `model`: auto|pro|flash|flash3|flashlite|o3|o3-mini|o4-mini|gpt4.1|gpt5.2|gpt5.1-codex|gpt5.1-codex-mini|gpt5|gpt5-mini|gpt5-nano|opus|sonnet|haiku|fable (default: server default)
- `problem_context`: Additional context about the problem or goal
- `focus_areas`: Specific aspects to focus on (architecture, performance, security, etc.)
- `temperature`: Temperature for creative thinking (default 1.0). Accepted from 0 to 2; the selected model's own rule then applies (Gemini, xAI and some OpenAI models 0-2, older Claude 0-1; models that take only temperature 1 or none ignore or override it).
- `thinking_mode`: minimal|low|medium|high|max (default: high; honored by Gemini and Anthropic models, ignored by providers without a thinking control)
- `continuation_id`: Continue previous conversations

## Usage Examples

**Architecture Design:**
```
"Think deeper about my microservices authentication strategy with pro using max thinking mode"
```

**With File Context:**
```
"Use gemini to think deeper about my API design with reference to api/routes.py and models/user.py"
```

**Visual Analysis:**
```
"Think deeper about this system architecture diagram with gemini pro - identify potential bottlenecks"
```

**Problem Solving:**
```
"I'm considering using GraphQL vs REST for my API. Think deeper about the trade-offs with o3 using high thinking mode"
```

**Code Review Enhancement:**
```
"Think deeper about the security implications of this authentication code with pro"
```

## Best Practices

- **Provide detailed context**: Share your current thinking, constraints, and objectives
- **Be specific about focus areas**: Mention what aspects need deeper analysis
- **Include relevant files**: Reference code, documentation, or configuration files
- **Use appropriate thinking modes**: Higher modes for complex problems, lower for quick validations
- **Leverage visual context**: Include diagrams or mockups for architectural discussions
- **Build on discussions**: Use continuation to extend previous analyses

## Enhanced Critical Evaluation Process

The `thinkdeep` tool includes a unique two-stage process:

1. **Model analysis**: Extended reasoning from the model you chose
2. **Claude's Critical Evaluation**: Claude reviews the model's suggestions, considers:
   - Context and constraints of your specific situation
   - Potential risks and implementation challenges
   - Trade-offs and alternatives
   - Final synthesized recommendation

This ensures you get both deep reasoning and practical, context-aware advice.

## When to Use ThinkDeep vs Other Tools

- **Use `thinkdeep`** for: Extending specific analysis, challenging assumptions, architectural decisions
- **Use `chat`** for: Open-ended brainstorming and general discussions
- **Use `analyze`** for: Understanding existing code without extending analysis
- **Use `codereview`** for: Finding specific bugs and security issues
