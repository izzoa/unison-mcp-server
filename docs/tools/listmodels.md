# ListModels Tool - List Available Models

**Display all available AI models organized by provider**

The `listmodels` tool shows which providers are configured, available models, their aliases, context windows, and capabilities. This is useful for understanding what models can be used and their characteristics.

## Usage

```
"Use unison to list available models"
```

## Key Features

- **Provider organization**: Shows all configured providers and their status
- **Model details**: Context window and description for each model
- **Alias mapping**: Shows shorthand names and their full model mappings
- **Configuration status**: Indicates which providers are available based on API keys
- **Context window information**: Helps you choose models based on your content size needs
- **Batch tag**: Models that can run through OpenRouter's asynchronous Batch API at reduced pricing (`execution_mode="batch"`, see the [batch guide](batch.md)) are tagged `batch`; an entry whose aliases float under batch also shows what they resolve to (`batch: opus, claude-opus → anthropic/claude-opus-5.5`), read from the batch catalog already loaded, without a network call

## Output Information

The tool returns markdown. For each provider it shows:

**Provider Status:**
- `Configured and available`, or `Not configured (set <ENV_KEY>)`; API keys are never revealed
- Providers appear in a fixed order; there is no priority ranking in the output

**Models:**
- Each model with its context window size and description (native providers add a "Supports structured code generation" line where applicable)
- Aliases (`alias → model`)
- Models auto-discovered from LiteLLM, listed separately
- OpenRouter entries as `(score N, <context> context, thinking, batch)`, plus what a family alias resolves to under batch
- Custom/local endpoint models

**Summary:**
- Number of configured providers and total available models, followed by usage tips

There is no capability summary (largest context windows, fastest models) and no thinking or vision tags for native providers' models.

## Example Output

```
# Available AI Models

## Google Gemini ✅
**Status**: Configured and available

**Models**:
- `gemini-3.1-pro-preview` - 1M context
  - Latest flagship (1M context) - Deep reasoning, thinking mode, complex problems, architecture
- `gemini-2.5-flash` - 1M context
  - Stable fast model (1M context) - Quick analysis, simple queries, rapid iterations

**Aliases**:
- `pro` → `gemini-3.1-pro-preview`
- `flash` → `gemini-2.5-flash`

## OpenRouter ❌
**Status**: Not configured (set OPENROUTER_API_KEY)

## Summary
**Configured Providers**: 1
**Total Available Models**: 2
```

## When to Use ListModels

- **Model selection**: When you're unsure which models are available
- **Capability checking**: To verify what features each model supports
- **Configuration validation**: To confirm your API keys are working
- **Context planning**: To choose models based on content size requirements
- **Performance optimization**: To select the right model for speed vs quality trade-offs

## Configuration Dependencies

The available models depend on your configuration:

**API Keys Required:**
- `GEMINI_API_KEY` - Enables Gemini Pro and Flash models
- `OPENAI_API_KEY` - Enables OpenAI O3, O4-mini, and GPT models
- `ANTHROPIC_API_KEY` - Enables native Claude models (Fable, Opus, Sonnet, Haiku)
- `OPENROUTER_API_KEY` - Enables access to multiple providers through OpenRouter
- `CUSTOM_API_URL` - Enables local/custom models (Ollama, vLLM, etc.)

**Model Restrictions:**
If you've set model usage restrictions via environment variables, a restricted provider lists only its allowed models under "Models (policy restricted)" (or says that none are allowed), and OpenRouter adds an allow-list note. The output does not list restricted models or explain how to change a policy; see the configuration guide for the `*_ALLOWED_MODELS` variables.

## Tool Parameters

This tool requires no parameters - it simply queries the server configuration and displays all available information.

## Best Practices

- **Check before planning**: Use this tool to understand your options before starting complex tasks
- **Verify configuration**: Confirm your API keys are working as expected
- **Choose appropriate models**: Match model capabilities to your specific needs
- **Understand limits**: Be aware of context windows when working with large files

## When to Use ListModels vs Other Tools

- **Use `listmodels`** for: Understanding available options and model capabilities
- **Use `chat`** for: General discussions about which model to use for specific tasks
- **Use `version`** for: Server configuration and version information
- **Use other tools** for: Actual analysis, debugging, or development work