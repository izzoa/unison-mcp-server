# Gemini CLI Setup

> **Note**: While Unison MCP Server connects successfully to Gemini CLI, tool invocation is not working
> correctly yet. We'll update this guide once the integration is fully functional.

This guide explains how to configure Unison MCP Server to work with [Gemini CLI](https://github.com/google-gemini/gemini-cli).

## Prerequisites

- Unison MCP Server installed and configured
- Gemini CLI installed
- At least one API key configured in your `.env` file

## Configuration

1. Edit `~/.gemini/settings.json` and add:

```json
{
  "mcpServers": {
    "unison": {
      "command": "/path/to/unison-mcp-server/unison-mcp-server"
    }
  }
}
```

2. Replace `/path/to/unison-mcp-server` with your actual Unison MCP installation path (the folder name may still be `unison-mcp-server`).

3. If the `unison-mcp-server` wrapper script doesn't exist, create it:

```bash
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
exec .unison_venv/bin/python server.py "$@"
```

Then make it executable: `chmod +x unison-mcp-server`

4. Restart Gemini CLI.

All 15 Unison tools are now available in your Gemini CLI session.
