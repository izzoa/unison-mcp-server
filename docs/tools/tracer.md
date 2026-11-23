# Tracer Tool - Step-by-Step Code Tracing

**Guides the calling agent through tracing call flows and mapping dependencies**

The `tracer` tool is a multi-step workflow that guides Claude through investigating and tracing code: either the execution flow of a method or function (`precision`) or the structural relationships of a class, module or protocol (`dependencies`). It returns required actions for each step and, on the final step, formatting and rendering instructions for the call-flow or dependency output. The tracing runs inside the calling agent: no external model is called, so it is fast and uses no provider tokens.

## Two Analysis Modes

**`precision` Mode**: For methods/functions
- Traces execution flow, call chains, and usage patterns
- Detailed branching analysis and side effects
- Shows when and how functions are called throughout the system

**`dependencies` Mode**: For classes/modules/protocols  
- Maps bidirectional dependencies and structural relationships
- Identifies coupling and architectural dependencies
- Shows how components interact and depend on each other

## Key Features

- **Multi-step workflow** that tells the agent what to investigate at each step and when the trace is complete
- **No model call**: the workflow runs inside the calling agent, with no external model involved
- **Returns rendering instructions** for the final call-flow or dependency output, so the format is consistent
- **Mode choice up front**: `trace_mode` is required on every call; with `ask`, step 1 stops and tells the agent to ask the user to choose `precision` or `dependencies` before continuing
- **Supports any programming language** with automatic convention detection
- **Output can be used as input** into another tool, such as `chat` along with related code files
- **Image support**: Analyze visual call flow diagrams, sequence diagrams: `"Generate tracer analysis for this payment flow using the sequence diagram"`

## Tool Parameters

**Workflow Investigation Parameters (used during step-by-step process):**
- `step`: Current tracing step description (required for each step; in step 1 it carries what you are about to trace)
- `step_number`: Current step number (required)
- `total_steps`: Estimated total tracing steps (required, adjustable)
- `next_step_required`: Whether another tracing step is needed (required)
- `findings`: Discoveries collected in this step (required)
- `target_description`: What to trace and WHY (required in the published schema)
- `trace_mode`: precision|dependencies|ask (required on every call; `ask` makes step 1 pause so the agent can ask the user to choose a mode)
- `files_checked`: All files examined during the trace
- `relevant_files`: Files directly relevant to the tracing target
- `relevant_context`: Methods/functions central to the trace, as `ClassName.methodName`
- `confidence`: Confidence in the trace (exploring/low/medium/high/very_high/almost_certain/certain); it only shapes the next-step guidance, the tracer never calls an external model
- `images`: Optional images of system architecture diagrams, flow charts, or visual references (absolute paths)
- `continuation_id`: Continue a previous tracing session

## Usage Examples

**Method Execution Tracing:**
```
"Use unison tracer to analyze how UserAuthManager.authenticate is used and why"
```
→ Uses `precision` mode to trace the method's execution flow

**Class Dependency Mapping:**
```
"Use unison to generate a dependency trace for the PaymentProcessor class to understand its relationships"
```
→ Uses `dependencies` mode to map structural relationships

**With Visual Context:**
```
"Generate tracer analysis for the authentication flow using this sequence diagram"
```

**Complex System Analysis:**
```
"Use tracer to understand how the OrderProcessor.processPayment method flows through the entire system"
```

## Precision Mode Output

When using `precision` mode for methods/functions, the final step instructs Claude to render:

**Call Chain Analysis:**
- Where the method is defined
- All locations where it's called
- Direct and indirect callers
- Call hierarchy and depth

**Execution Flow Mapping:**
- Step-by-step execution path
- Branching conditions and logic
- Side effects and state changes
- Return value usage

**Usage Pattern Analysis:**
- Frequency and context of calls
- Parameter passing patterns
- Error handling approaches
- Performance implications

## Dependencies Mode Output

When using `dependencies` mode for classes/modules, the final step instructs Claude to render:

**Structural Relationships:**
- Inheritance hierarchies
- Composition and aggregation
- Interface implementations
- Module imports and exports

**Bidirectional Dependencies:**
- What the component depends on
- What depends on the component
- Circular dependencies
- Coupling strength analysis

**Architectural Impact:**
- Layer violations
- Dependency inversion opportunities
- Refactoring impact assessment
- Testability implications

## Example Step 1 Call

```json
{
  "step": "Trace how UserAuthManager.authenticate is used and why",
  "step_number": 1,
  "total_steps": 3,
  "next_step_required": true,
  "findings": "Starting the trace: locating the definition and call sites of authenticate.",
  "target_description": "UserAuthManager.authenticate: all callers, execution flow and side effects",
  "trace_mode": "precision"
}
```

Later steps carry the files examined, the relevant files and the call chain discovered so far. When the trace is complete, the final response tells the agent how to format the call-flow (precision) or dependency graph (dependencies) output.

## Best Practices

- **Be specific about goals**: Clearly state what you need to understand and why
- **Describe context**: Mention if you're debugging, refactoring, or learning the codebase
- **Choose appropriate mode**: Use `precision` for method flows, `dependencies` for architecture
- **Include visual context**: Reference diagrams or documentation when available
- **Follow up with analysis**: Pass the finished trace to `chat` or `analyze` along with the relevant files

## Integration with Other Tools

The `tracer` tool works best when combined with other analysis tools:

**Tracer + Chat:**
```
1. Run the tracer workflow to map the call flow or dependencies
2. Give the result to the chat tool along with relevant code files
3. Get detailed call-flow or dependency discussion
```

**Tracer + Analyze:**
```
1. Use tracer to map the structure
2. Use analyze for systematic code exploration of the same files
3. Get architectural insights and dependency mapping
```

## When to Use Tracer vs Other Tools

- **Use `tracer`** for: Step-by-step call-flow tracing and dependency mapping with a consistent output format
- **Use `analyze`** for: General code analysis with expert model validation
- **Use `debug`** for: Specific runtime error investigation
- **Use `chat`** for: Open-ended code discussions and exploration