# clink CLI manifests

Each `*.json` file in this directory configures one CLI that the `clink` tool can bridge to. The loader
reads only `*.json`, so this file is documentation. See `docs/tools/clink.md` for the full reference.

## Model strings are forwarded verbatim

The `model` argument of a clink call is handed to the external CLI exactly as supplied (for opencode,
`-m <model>`). Unison does not parse it, split it on a colon, or strip a suffix. A provider-qualified
string with a suffix, such as `openrouter/openai/gpt-6-astra-pro:batch`, reaches the CLI unchanged, and
whether it means anything (a variant, a batch-priced listing, or an error) is decided by the CLI and its
provider. Unison makes no claim about it.

Server-side batch execution (`execution_mode`) does not apply to clink: Unison runs a subprocess and
never sees the request the CLI sends, so it cannot submit a provider batch on its behalf. A clink call
that sets `execution_mode` is rejected.

## The `supported_models` allowlist matches exactly

A manifest may declare `"supported_models": [...]`. When that list is non-empty, a call that names a
`model` is checked against it by **exact string equality** before the CLI is started.

**The trap:** a value that differs from a listed entry only by a colon-delimited suffix does *not*
match that entry. With

```json
"supported_models": ["openrouter/openai/gpt-6-astra-pro"]
```

the call `model: "openrouter/openai/gpt-6-astra-pro:batch"` is rejected before any subprocess is
launched (the error names the rejected model and lists the allowed values). The reverse also holds:
listing only the suffixed form does not admit the unsuffixed one.

To permit a suffixed variant, list the suffixed form explicitly:

```json
"supported_models": [
  "openrouter/openai/gpt-6-astra-pro",
  "openrouter/openai/gpt-6-astra-pro:batch"
]
```

Leave `supported_models` out (or empty) to skip validation entirely: every value is then forwarded and
any problem surfaces as a CLI-level error in the response metadata.
