# Tool Plugin Development Guide

Unison can load third-party tools from pip-installed packages — no fork, no
registry edit. This guide covers the tool interface, packaging, and the
validation your tool must pass.

## The tool interface

A plugin tool is a concrete subclass of `BaseTool` implementing the
**method-based** contract (there is no class-attribute contract):

- `get_name() -> str` — must return the same name your entry point declares
- `get_description() -> str` — non-empty
- `get_input_schema() -> dict` — the MCP input schema
- your own `execute()` override (or the framework's workflow execution path).
  Inheriting `BaseTool.execute` unchanged fails validation — the default only
  raises `NotImplementedError`.

## Declaring the entry point

In your plugin package's `pyproject.toml`:

```toml
[project.entry-points."unison.tools"]
my_tool = "my_package.tools:MyTool"
```

**The entry-point key (`my_tool`) becomes the tool name.** After
`pip install my-package` into the server's environment, the tool is loaded at
server startup — the module is imported then, but your class is only
instantiated on first use.

## Validation and quarantine

Registration-time (class-level) checks: non-empty unique name (first
registration wins; built-ins always win), concrete `BaseTool` subclass, a
usable execution method. Failures are logged at WARNING and the tool is
skipped — the server still starts.

First-instantiation checks: `get_name()` must agree with the registered name,
`get_description()` must be non-empty, `get_input_schema()` must return a
dict. A failure **quarantines** the tool: it disappears from availability and
direct calls receive a controlled error, never a raw traceback.

`DISABLED_TOOLS` applies to plugin tools exactly as to built-ins.

## Alternatives to entry points

- `ToolRegistry.register("name", MyTool)` — direct registration before startup
  freeze.
- `@register_tool("name")` — decorator; registers when its module is imported
  (the decorator itself imports nothing).
- `UNISON_TOOL_AUTODISCOVERY=true` — opt-in, import-only scan of the `tools/`
  package for local development; registration still happens via the decorator.

After startup the registry is frozen; late imports log and are ignored.
