# Version Tool - Server Information

**Get the server version, update status, model-selection mode, configured providers and available-model count**

The `version` tool reports which Unison MCP Server version is running, whether a newer one exists on GitHub, how models are selected and which providers are configured. It is useful for debugging and for verifying your installation. It does not list tools or models: use `listmodels` for models, and your MCP client's tool list (MCP `tools/list`) for the tool inventory.

## Usage

```
"Get unison to show its version"
```

## Key Features

- **Server version information**: Current version, last-updated date and author
- **Update status**: Compares your version with the latest published one (this makes a network call to GitHub with a 10 second timeout; offline it reports "Could not check for updates")
- **Configuration overview**: Model-selection mode and default model, connected client and installation path
- **Provider status**: One line per provider saying whether it is configured, plus the available-model count
- **Debug information**: Helpful details for troubleshooting

## Output Information

The tool returns markdown with these sections:

- **Server Information**: version, last updated, author, model selection (auto mode or the default model), connected client, installation path
- **Quick Summary** and **Agent Reporting Guidance**: version and model-selection bullets the agent is told to relay as is
- **Update Status**: latest GitHub version, and whether you are up to date, behind (with the `git pull` commands) or ahead (a development build)
- **Configuration**: each provider as ✅ Configured or ❌ Not configured, and the **Available Models** count

The response metadata also carries `python_version` and `platform`.

## Example Output

```
# Unison MCP Server Version

## Server Information
**Current Version**: 14.7.0
**Last Updated**: 2026-08-16
**Author**: Fahad Gilani
**Model Selection**: Auto model selection mode (call `listmodels` to inspect options)
**Installation Path**: `/path/to/unison-mcp-server`

## Quick Summary — relay everything below
- Version `14.7.0` (updated 2026-08-16)
- Auto model selection (use `listmodels` for options)
- Run `listmodels` for the complete model catalog and capabilities

## Update Status
**Latest Version (GitHub)**: 14.7.0
✅ **UP TO DATE**
You are running the latest version.

## Configuration
**Providers**:
- **Google**: ✅ Configured
- **OpenAI**: ✅ Configured
- **OpenRouter**: ❌ Not configured

**Available Models**: 24
```

## When to Use Version Tool

- **Troubleshooting**: When experiencing issues with the server or tools
- **Configuration verification**: To confirm which providers are configured and how models are selected
- **Support requests**: To provide system information when asking for help
- **Update checking**: To verify you're running the latest version

## Debug Information

The version tool can help diagnose common issues:

**Connection Problems:**
- Confirms the server is running and responsive

**Configuration Issues:**
- Shows which providers have API keys configured
- Shows the model-selection mode and default model

## Tool Parameters

This tool requires no parameters.

## Best Practices

- **Include in bug reports**: Always include version output when reporting issues
- **Check after updates**: Verify version information after server updates
- **Validate configuration**: Confirm providers and model selection match your expectations

## When to Use Version vs Other Tools

- **Use `version`** for: Server diagnostics, configuration verification, update checks
- **Use `listmodels`** for: Model availability and capability information
- **Use other tools** for: Actual development and analysis tasks
