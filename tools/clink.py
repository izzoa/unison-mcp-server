"""clink tool - bridge Unison MCP requests to external AI CLIs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from mcp.types import TextContent
from pydantic import BaseModel, Field

from clink import get_registry
from clink.agents import AgentOutput, CLIAgentError, create_agent
from clink.constants import CLINK_DEPTH_ENV_VAR, CLINK_MAX_DEPTH_ENV_VAR, DEFAULT_CLINK_MAX_DEPTH
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from config import TEMPERATURE_BALANCED
from tools.models import ToolModelCategory, ToolOutput
from tools.shared.base_models import COMMON_FIELD_DESCRIPTIONS
from tools.shared.exceptions import ToolExecutionError
from tools.simple.base import SchemaBuilder, SimpleTool
from utils.env import get_env
from utils.fs_snapshot import SnapshotStats, capture_snapshot, classify_changes, diff_snapshots
from utils.mcp_context import get_current_request_context

# Wall-clock ceiling for each read-only verification snapshot. Observed live:
# an unbounded walk of a large OneDrive-synced repo took 60-90s per snapshot
# (twice per call), starving the actual CLI run of the MCP host's tool-timeout
# budget. Override with CLINK_SNAPSHOT_BUDGET_SECONDS.
_DEFAULT_SNAPSHOT_BUDGET_SECONDS = 30.0

logger = logging.getLogger(__name__)

MAX_RESPONSE_CHARS = 20_000
SUMMARY_PATTERN = re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.IGNORECASE | re.DOTALL)


def _check_recursion_guard() -> None:
    """Raise ``ToolExecutionError`` when the clink recursion depth is exceeded.

    Reads ``UNISON_CLINK_DEPTH`` (default 0) and compares against
    ``CLINK_MAX_RECURSION_DEPTH`` (default :data:`DEFAULT_CLINK_MAX_DEPTH`).
    The depth is incremented by :meth:`BaseCLIAgent._build_environment` for
    every CLI we spawn, so a clink-spawned CLI that itself invokes Unison
    via MCP sees a higher depth at this entry point.

    With the default max of 1, depth 0 and 1 succeed (so the user's primary
    CLI → Unison → a clink-spawned CLI works normally), but depth 2+ fails
    (the spawned CLI re-invoking Unison creates the loop).
    """
    raw_depth = os.environ.get(CLINK_DEPTH_ENV_VAR, "")
    try:
        current_depth = int(raw_depth) if raw_depth else 0
    except ValueError:
        current_depth = 0

    raw_max = os.environ.get(CLINK_MAX_DEPTH_ENV_VAR, "")
    try:
        max_depth = int(raw_max) if raw_max else DEFAULT_CLINK_MAX_DEPTH
    except ValueError:
        max_depth = DEFAULT_CLINK_MAX_DEPTH

    if current_depth > max_depth:
        raise ToolExecutionError(
            f"clink recursion limit exceeded "
            f"({CLINK_DEPTH_ENV_VAR}={current_depth}, max={max_depth}). "
            f"This usually means the calling CLI has Unison wired as an MCP "
            f"server while also invoking the clink tool, creating a loop. "
            f"Remove Unison from the calling CLI's MCP server config, or "
            f"raise {CLINK_MAX_DEPTH_ENV_VAR} in your environment if the "
            f"depth is intentional."
        )


class CLinkRequest(BaseModel):
    """Request model for clink tool."""

    prompt: str = Field(..., description="Prompt forwarded to the target CLI.")
    cli_name: str | None = Field(
        default=None,
        description=(
            "Configured CLI client name to invoke. Required when more than one CLI is "
            "configured; may be omitted only when a single CLI is configured, in which case "
            "that one is used."
        ),
    )
    role: str | None = Field(
        default=None,
        description="Optional role preset defined in the CLI configuration (defaults to 'default').",
    )
    absolute_file_paths: list[str] = Field(
        default_factory=list,
        description=COMMON_FIELD_DESCRIPTIONS["absolute_file_paths"],
    )
    images: list[str] = Field(
        default_factory=list,
        description=COMMON_FIELD_DESCRIPTIONS["images"],
    )
    continuation_id: str | None = Field(
        default=None,
        description=COMMON_FIELD_DESCRIPTIONS["continuation_id"],
    )
    read_only: bool = Field(
        default=False,
        description=(
            "When true, restricts the external CLI to read-only operations via three enforcement layers: "
            "(1) CLI-specific sandbox flags, (2) prompt-level instruction, (3) post-execution filesystem verification. "
            "Violations are reported in metadata but do not block the response."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional model identifier forwarded to the selected CLI. Opencode uses the "
            "'provider/model' convention (e.g. 'anthropic/claude-sonnet-4-5', 'openai/gpt-5', "
            "'ollama/llama3.2'); other CLIs accept their own forms (e.g. 'sonnet' for claude, "
            "model aliases for gemini/codex). When omitted, the CLI uses whatever default is "
            "set by its manifest. Validation is best-effort: if the selected CLI defines a "
            "'supported_models' allowlist in its manifest, the value is checked against it; "
            "otherwise invalid model strings surface as CLI-level errors in response metadata."
        ),
    )
    working_dir: str | None = Field(
        default=None,
        description=(
            "Absolute path to the directory the spawned CLI should run in. Pass your project or "
            "worktree root so the CLI can see your files — some CLIs (e.g. Copilot) root their "
            "file tools at their working directory and refuse paths outside it. When omitted, "
            "falls back to the CLI manifest's working_dir, then the MCP server's own working "
            "directory."
        ),
    )


class CLinkTool(SimpleTool):
    """Bridge MCP requests to configured CLI agents.

    Schema metadata is cached at construction time and execution relies on the shared
    SimpleTool hooks for conversation memory. Prompt preparation is customised so we
    pass instructions and file references suitable for another CLI agent.
    """

    def __init__(self) -> None:
        # Cache registry metadata so the schema surfaces concrete enum values.
        self._registry = get_registry()
        self._cli_names = self._registry.list_clients()
        self._role_map: dict[str, list[str]] = {name: self._registry.list_roles(name) for name in self._cli_names}
        self._all_roles: list[str] = sorted({role for roles in self._role_map.values() for role in roles})
        # No vendor preference: a single configured client is the only implicit
        # default. With more than one, `cli_name` is required — see
        # _resolve_client() and the matching rule in get_input_schema().
        self._default_cli_name = self._cli_names[0] if len(self._cli_names) == 1 else None
        self._active_system_prompt: str = ""
        # Background jobs (tools/clink_jobs.py) run execute() on a private
        # instance after the originating MCP request has already returned;
        # sending progress against that completed request would violate the
        # spec, so job runners disable the heartbeat per instance.
        self._progress_heartbeat_enabled = True
        super().__init__()

    def disable_progress_heartbeat(self) -> None:
        """Suppress MCP progress notifications for this instance (job mode)."""
        self._progress_heartbeat_enabled = False

    def _resolve_client(self, cli_name: str | None) -> ResolvedCLIClient:
        """Resolve the effective CLI client for a request.

        The single resolution path shared by every entry point that assembles a
        prompt or dispatches to a CLI, so no caller passes an unresolved value
        into the registry (where ``None`` would surface as an ``AttributeError``
        rather than an actionable message).

        ``cli_name`` is required whenever more than one client is configured —
        matching what :meth:`get_input_schema` already advertises. Selecting an
        arbitrary client instead would silently send the request somewhere the
        caller did not ask for.
        """
        if not self._cli_names:
            self._raise_tool_error("No CLI clients are configured for clink.")

        selected = (cli_name or "").strip() or self._default_cli_name
        if not selected:
            available = ", ".join(self._cli_names)
            self._raise_tool_error(
                f"'cli_name' is required when multiple CLI clients are configured. " f"Available clients: {available}."
            )

        try:
            return self._registry.get_client(selected)
        except KeyError as exc:
            self._raise_tool_error(str(exc))

    def get_name(self) -> str:
        return "clink"

    def get_description(self) -> str:
        # Deliberately names no CLI: the generated `cli_name` enum is the
        # authoritative list of configured targets, and a hardcoded example
        # list goes stale (this previously advertised Qwen, never a target).
        return (
            "Link a request to an external AI CLI through Unison MCP to reuse their capabilities "
            "inside existing workflows. See the 'cli_name' enum for the configured targets."
        )

    def get_annotations(self) -> dict[str, Any]:
        return {"readOnlyHint": True}

    def requires_model(self) -> bool:
        return False

    def get_model_category(self) -> ToolModelCategory:
        return ToolModelCategory.BALANCED

    def get_default_temperature(self) -> float:
        return TEMPERATURE_BALANCED

    def get_system_prompt(self) -> str:
        return self._active_system_prompt or ""

    def get_request_model(self):
        return CLinkRequest

    def get_input_schema(self) -> dict[str, Any]:
        # Surface configured CLI names and roles directly in the schema so MCP clients
        # (and downstream agents) can discover available options without consulting
        # a separate registry call.
        role_descriptions = []
        for name in self._cli_names:
            roles = ", ".join(sorted(self._role_map.get(name, ["default"]))) or "default"
            role_descriptions.append(f"{name}: {roles}")

        if role_descriptions:
            cli_available = ", ".join(self._cli_names) if self._cli_names else "(none configured)"
            default_text = (
                f" Default: {self._default_cli_name}." if self._default_cli_name and len(self._cli_names) <= 1 else ""
            )
            cli_description = (
                "Configured CLI client name (from conf/cli_clients). Available: " + cli_available + default_text
            )
            role_description = (
                "Optional role preset defined for the selected CLI (defaults to 'default'). Roles per CLI: "
                + "; ".join(role_descriptions)
            )
        else:
            cli_description = "Configured CLI client name (from conf/cli_clients)."
            role_description = "Optional role preset defined for the selected CLI (defaults to 'default')."

        properties = {
            "prompt": {
                "type": "string",
                "description": "User request forwarded to the CLI (conversation context is pre-applied).",
            },
            "cli_name": {
                "type": "string",
                "enum": self._cli_names,
                "description": cli_description,
            },
            "role": {
                "type": "string",
                "enum": self._all_roles or ["default"],
                "description": role_description,
            },
            "absolute_file_paths": SchemaBuilder.SIMPLE_FIELD_SCHEMAS["absolute_file_paths"],
            "images": SchemaBuilder.COMMON_FIELD_SCHEMAS["images"],
            "continuation_id": SchemaBuilder.COMMON_FIELD_SCHEMAS["continuation_id"],
            "read_only": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Restrict the external CLI to read-only operations. "
                    "Enforced via CLI sandbox flags, prompt instruction, and post-execution filesystem verification."
                ),
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional model forwarded to the selected CLI. Opencode uses 'provider/model' "
                    "(e.g. 'anthropic/claude-sonnet-4-5', 'openai/gpt-5', 'ollama/llama3.2'); other CLIs "
                    "accept their own forms (e.g. 'sonnet' for claude). Omit to use the CLI's manifest "
                    "default. Validation is best-effort — invalid models surface as CLI-level errors in "
                    "response metadata unless the manifest declares a 'supported_models' allowlist."
                ),
            },
            "working_dir": {
                "type": "string",
                "description": (
                    "Absolute path to the directory the spawned CLI should run in. Pass your "
                    "project or worktree root so the CLI can see your files (some CLIs root "
                    "their file tools at their cwd). Omit to use the CLI manifest's working_dir, "
                    "then the server's own working directory."
                ),
            },
        }

        schema = {
            "type": "object",
            "properties": properties,
            "required": ["prompt"],
            "additionalProperties": False,
        }

        if len(self._cli_names) > 1:
            schema["required"].append("cli_name")

        return schema

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        """Unused by clink because we override the schema end-to-end."""
        return {}

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        # Recursion guard: if we're already running inside a clink-spawned CLI
        # that itself wired Unison as an MCP server, refuse to spawn another
        # CLI rather than enter a context-window-exploding loop. See
        # clink-multi-cli-infrastructure spec for the env-var propagation
        # contract.
        _check_recursion_guard()

        self._current_arguments = arguments
        request = self.get_request_model()(**arguments)

        path_error = self._validate_file_paths(request)
        if path_error:
            self._raise_tool_error(path_error)

        client_config = self._resolve_client(request.cli_name)

        try:
            role_config = client_config.get_role(request.role)
        except KeyError as exc:
            self._raise_tool_error(str(exc))

        requested_model = (request.model or "").strip() or None
        if requested_model and requested_model.startswith("-"):
            # A model value beginning with "-" can be interpreted as a CLI flag by
            # the spawned CLI (or collide with an adjacent flag during read-only
            # flag stripping), corrupting the command. Real model identifiers never
            # start with "-".
            self._raise_tool_error(f"Invalid model '{requested_model}': model identifiers cannot start with '-'.")
        if requested_model and client_config.supported_models:
            if requested_model not in client_config.supported_models:
                allowed = ", ".join(client_config.supported_models)
                self._raise_tool_error(
                    f"Model '{requested_model}' is not in the supported_models allowlist for CLI "
                    f"'{client_config.name}'. Allowed values: {allowed}."
                )

        working_dir_override: Path | None = None
        raw_working_dir = (request.working_dir or "").strip()
        if raw_working_dir:
            candidate = Path(raw_working_dir)
            if not candidate.is_absolute():
                self._raise_tool_error(f"working_dir must be an absolute path, got: {raw_working_dir}")
            if not candidate.is_dir():
                self._raise_tool_error(f"working_dir does not exist or is not a directory: {raw_working_dir}")
            working_dir_override = candidate

        absolute_file_paths = self.get_request_files(request)
        images = self.get_request_images(request)
        continuation_id = self.get_request_continuation_id(request)

        from utils.tool_execution_context import ToolExecutionContext

        _exec_ctx = ToolExecutionContext.from_arguments(arguments)
        self._model_context = _exec_ctx.model_context if _exec_ctx else None

        system_prompt_text = role_config.prompt_path.read_text(encoding="utf-8")
        include_system_prompt = not self._use_external_system_prompt(client_config)

        try:
            prompt_text = await self._prepare_prompt_for_role(
                request,
                role_config,
                client=client_config,
                system_prompt=system_prompt_text,
                include_system_prompt=include_system_prompt,
            )
        except Exception as exc:
            logger.exception("Failed to prepare clink prompt")
            self._raise_tool_error(f"Failed to prepare prompt: {exc}")

        # Capture pre-execution filesystem snapshot for read-only verification.
        # Rooted at the same directory the CLI will actually run in.
        pre_snapshot = None
        read_only = getattr(request, "read_only", False)
        effective_working_dir = working_dir_override or client_config.working_dir
        snapshot_dir = str(effective_working_dir) if effective_working_dir else "."
        snapshot_budget = self._snapshot_budget_seconds()
        pre_stats = SnapshotStats()
        if read_only:
            # Full-depth, include gitignored/transient so a deep or gitignored
            # write (e.g. to .env) cannot silently evade read-only verification.
            # Bulk dirs (.git, node_modules, ...) are pruned and the walk is
            # time-budgeted so verification can never starve the CLI call
            # itself of the host's tool-timeout budget.
            # Off the event loop: a budgeted walk can still take tens of
            # seconds, and background jobs must not stall poll responses.
            pre_snapshot = await asyncio.to_thread(
                capture_snapshot,
                snapshot_dir,
                include_ignored=True,
                time_budget_seconds=snapshot_budget,
                stats=pre_stats,
            )

        agent = create_agent(client_config)
        heartbeat = asyncio.create_task(self._progress_heartbeat(client_config.name))
        try:
            result = await agent.run(
                role=role_config,
                prompt=prompt_text,
                system_prompt=system_prompt_text if system_prompt_text.strip() else None,
                files=absolute_file_paths,
                images=images,
                read_only=read_only,
                model=requested_model,
                working_dir=working_dir_override,
            )
        except CLIAgentError as exc:
            metadata = self._build_error_metadata(client_config, exc)
            self._raise_tool_error(
                f"CLI '{client_config.name}' execution failed: {exc}",
                metadata=metadata,
            )
        finally:
            # Stop the heartbeat before any response (success or error) is
            # built: the spec forbids progress after request completion.
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

        metadata = self._build_success_metadata(client_config, role_config, result, requested_model=requested_model)
        metadata = self._prune_metadata(metadata, client_config, reason="normal")

        # Report where the CLI actually ran so callers never have to ask it.
        metadata["working_dir"] = str(effective_working_dir) if effective_working_dir else os.getcwd()

        # Post-execution read-only verification
        if read_only and pre_snapshot is not None:
            post_stats = SnapshotStats()
            post_snapshot = await asyncio.to_thread(
                capture_snapshot,
                snapshot_dir,
                include_ignored=True,
                time_budget_seconds=snapshot_budget,
                stats=post_stats,
            )
            diff = diff_snapshots(pre_snapshot, post_snapshot)
            sandbox_flags = agent.get_read_only_args()
            # Report enforcement honestly. `read_only_enforced` reflects whether a
            # layer-1 CLI/OS sandbox flag was actually applied. When it is empty,
            # the only enforcement is the prompt instruction plus this best-effort,
            # detective (not preventive) snapshot diff, which covers the
            # working-directory subtree only and cannot see command execution or
            # out-of-tree writes. Do not let an empty diff read as proof of no
            # side effects.
            metadata["read_only_sandbox_flags"] = sandbox_flags
            metadata["read_only_enforced"] = bool(sandbox_flags)
            metadata["read_only_enforcement"] = {
                "sandbox_flags": bool(sandbox_flags),
                "prompt_instruction": True,
                "post_execution_verification": True,
            }
            verification_partial = pre_stats.truncated or post_stats.truncated
            metadata["read_only_verification_coverage"] = (
                "working_dir_subtree (partial)" if verification_partial else "working_dir_subtree"
            )
            metadata["read_only_verification_stats"] = {
                "pre_entries": pre_stats.entry_count,
                "post_entries": post_stats.entry_count,
                "pre_elapsed_seconds": round(pre_stats.elapsed_seconds, 2),
                "post_elapsed_seconds": round(post_stats.elapsed_seconds, 2),
                "truncated": verification_partial,
            }
            by_model, by_bookkeeping = classify_changes(diff, agent.fs_violation_ignore_patterns)
            metadata["read_only_violations"] = {
                "by_model": by_model.to_dict(),
                "by_cli_bookkeeping": by_bookkeeping.to_dict(),
            }
            if by_model.has_changes:
                logger.warning(
                    "Read-only violation detected for CLI '%s': %s",
                    client_config.name,
                    by_model.to_dict(),
                )

        content, metadata = self._apply_output_limit(
            client_config,
            result.parsed.content,
            metadata,
        )

        model_info = {
            "provider": client_config.name,
            "model_name": result.parsed.metadata.get("model_used"),
        }

        if continuation_id:
            try:
                self._record_assistant_turn(continuation_id, content, request, model_info)
            except Exception:
                logger.debug("Failed to record assistant turn for continuation %s", continuation_id, exc_info=True)

        continuation_offer = self._create_continuation_offer(request, model_info)
        if continuation_offer:
            tool_output = self._create_continuation_offer_response(
                content,
                continuation_offer,
                request,
                model_info,
            )
            tool_output.metadata = self._merge_metadata(tool_output.metadata, metadata)
        else:
            tool_output = ToolOutput(
                status="success",
                content=content,
                content_type="text",
                metadata=metadata,
            )

        return [TextContent(type="text", text=tool_output.model_dump_json())]

    async def prepare_prompt(self, request) -> str:
        client_config = self._resolve_client(getattr(request, "cli_name", None))
        role_config = client_config.get_role(request.role)
        system_prompt_text = role_config.prompt_path.read_text(encoding="utf-8")
        include_system_prompt = not self._use_external_system_prompt(client_config)
        return await self._prepare_prompt_for_role(
            request,
            role_config,
            client=client_config,
            system_prompt=system_prompt_text,
            include_system_prompt=include_system_prompt,
        )

    async def _prepare_prompt_for_role(
        self,
        request: CLinkRequest,
        role: ResolvedCLIRole,
        *,
        client: ResolvedCLIClient,
        system_prompt: str,
        include_system_prompt: bool,
    ) -> str:
        """Load the role prompt and assemble the final user message."""
        self._active_system_prompt = system_prompt
        try:
            user_content = self.handle_prompt_file_with_fallback(request).strip()
            guidance = self._agent_capabilities_guidance(client)
            file_section = self._format_file_references(self.get_request_files(request))

            sections: list[str] = []
            active_prompt = self.get_system_prompt().strip()
            if include_system_prompt and active_prompt:
                sections.append(active_prompt)

            # Read-only enforcement: prompt-level instruction
            if getattr(request, "read_only", False):
                # Stated behaviorally rather than by tool name. Enumerated names go
                # stale — the five previously listed here match no current target,
                # including Gemini — and naming a handful implicitly narrows a
                # prohibition meant to cover every write path, including shell
                # redirection and tools that did not exist when this was written.
                sections.append(
                    "=== READ-ONLY MODE ===\n"
                    "CRITICAL CONSTRAINT: You are operating in READ-ONLY mode. "
                    "You MUST NOT create, modify, delete, or rename any file. "
                    "Do not use any tool that writes to the filesystem, whatever it is called, "
                    "and do not achieve the same effect by other means such as shell commands "
                    "or output redirection. "
                    "Only read files and provide analysis. Any file modification is a violation."
                )

            sections.append(guidance)
            sections.append("=== USER REQUEST ===\n" + user_content)
            if file_section:
                sections.append("=== FILE REFERENCES ===\n" + file_section)
            sections.append("Provide your response below using your own CLI tools as needed:")
            return "\n\n".join(sections)
        finally:
            self._active_system_prompt = ""

    def _use_external_system_prompt(self, client: ResolvedCLIClient) -> bool:
        runner_name = (client.runner or client.name).lower()
        return runner_name == "claude"

    def _build_success_metadata(
        self,
        client: ResolvedCLIClient,
        role: ResolvedCLIRole,
        result: AgentOutput,
        *,
        requested_model: str | None = None,
    ) -> dict[str, Any]:
        """Capture execution metadata for successful CLI calls."""
        metadata: dict[str, Any] = {
            "cli_name": client.name,
            "role": role.name,
            "command": result.sanitized_command,
            "duration_seconds": round(result.duration_seconds, 3),
            "parser": result.parser_name,
            "return_code": result.returncode,
        }
        metadata.update(result.parsed.metadata)
        if requested_model:
            metadata["model_requested"] = requested_model

        if result.stderr.strip():
            metadata.setdefault("stderr", result.stderr.strip())
        if result.output_file_content and "raw" not in metadata:
            metadata["raw_output_file"] = result.output_file_content
        return metadata

    def _merge_metadata(self, base: dict[str, Any] | None, extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base or {})
        merged.update(extra)
        return merged

    def _apply_output_limit(
        self,
        client: ResolvedCLIClient,
        content: str,
        metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if len(content) <= MAX_RESPONSE_CHARS:
            return content, metadata

        summary = self._extract_summary(content)
        if summary:
            summary_text = summary
            if len(summary_text) > MAX_RESPONSE_CHARS:
                logger.debug(
                    "Clink summary from %s exceeded %d chars; truncating summary to fit.",
                    client.name,
                    MAX_RESPONSE_CHARS,
                )
                summary_text = summary_text[:MAX_RESPONSE_CHARS]
            summary_metadata = self._prune_metadata(metadata, client, reason="summary")
            summary_metadata.update(
                {
                    "output_summarized": True,
                    "output_original_length": len(content),
                    "output_summary_length": len(summary_text),
                    "output_limit": MAX_RESPONSE_CHARS,
                }
            )
            logger.info(
                "Clink compressed %s output via <SUMMARY>: original=%d chars, summary=%d chars",
                client.name,
                len(content),
                len(summary_text),
            )
            return summary_text, summary_metadata

        truncated_metadata = self._prune_metadata(metadata, client, reason="truncated")
        truncated_metadata.update(
            {
                "output_truncated": True,
                "output_original_length": len(content),
                "output_limit": MAX_RESPONSE_CHARS,
            }
        )

        excerpt_limit = min(4000, MAX_RESPONSE_CHARS // 2)
        excerpt = content[:excerpt_limit]
        truncated_metadata["output_excerpt_length"] = len(excerpt)

        logger.warning(
            "Clink truncated %s output: original=%d chars exceeds limit=%d; excerpt_length=%d",
            client.name,
            len(content),
            MAX_RESPONSE_CHARS,
            len(excerpt),
        )

        message = (
            f"CLI '{client.name}' produced {len(content)} characters, exceeding the configured clink limit "
            f"({MAX_RESPONSE_CHARS} characters). The full output was suppressed to stay within MCP response caps. "
            "Please narrow the request (review fewer files, summarize results) or run the CLI directly for the full log.\n\n"
            f"--- Begin excerpt ({len(excerpt)} of {len(content)} chars) ---\n{excerpt}\n--- End excerpt ---"
        )

        return message, truncated_metadata

    def _extract_summary(self, content: str) -> str | None:
        match = SUMMARY_PATTERN.search(content)
        if not match:
            return None
        summary = match.group(1).strip()
        return summary or None

    def _prune_metadata(
        self,
        metadata: dict[str, Any],
        client: ResolvedCLIClient,
        *,
        reason: str,
    ) -> dict[str, Any]:
        cleaned = dict(metadata)
        events = cleaned.pop("events", None)
        if events is not None:
            cleaned[f"events_removed_for_{reason}"] = True
            logger.debug(
                "Clink dropped %s events metadata for %s response (%s)",
                client.name,
                reason,
                type(events).__name__,
            )
        return cleaned

    #: Seconds between MCP progress heartbeats while a CLI runs. Class-level so
    #: tests can shrink it.
    _PROGRESS_HEARTBEAT_INTERVAL_SECONDS = 10.0

    async def _progress_heartbeat(self, cli_name: str) -> None:
        """Emit MCP progress notifications while the spawned CLI runs.

        ``ServerSession.report_progress`` is a no-op when the caller sent no
        ``progressToken``, so this costs nothing for hosts that don't ask.
        Hosts that reset their tool timeout on progress keep long CLI runs
        alive; the rest at least surface liveness to the user. A heartbeat
        failure must never break the CLI call itself.
        """
        if not self._progress_heartbeat_enabled:
            return
        ctx = get_current_request_context()
        session = getattr(ctx, "session", None)
        report = getattr(session, "report_progress", None)
        if report is None:
            return
        started = time.monotonic()
        sequence = 0
        while True:
            await asyncio.sleep(self._PROGRESS_HEARTBEAT_INTERVAL_SECONDS)
            sequence += 1
            elapsed = int(time.monotonic() - started)
            try:
                await report(float(sequence), None, f"clink: {cli_name} still running ({elapsed}s elapsed)")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("clink progress heartbeat failed; stopping", exc_info=True)
                return

    def _snapshot_budget_seconds(self) -> float:
        """Wall-clock budget per read-only verification snapshot."""
        raw = (get_env("CLINK_SNAPSHOT_BUDGET_SECONDS") or "").strip()
        if raw:
            try:
                value = float(raw)
            except ValueError:
                logger.warning("Ignoring invalid CLINK_SNAPSHOT_BUDGET_SECONDS=%r (not a number)", raw)
            else:
                if value > 0:
                    return value
                logger.warning("Ignoring invalid CLINK_SNAPSHOT_BUDGET_SECONDS=%r (must be positive)", raw)
        return _DEFAULT_SNAPSHOT_BUDGET_SECONDS

    def _build_error_metadata(self, client: ResolvedCLIClient, exc: CLIAgentError) -> dict[str, Any]:
        """Assemble metadata for failed CLI calls.

        Read-only verification (``read_only_enforced``, ``read_only_sandbox_flags``,
        ``read_only_violations``) is intentionally NOT included in error metadata
        because verification only runs after a successful ``agent.run()``. If a
        future change needs to surface partial verification on error paths,
        wire it explicitly here rather than relying on shape compatibility.
        """
        metadata: dict[str, Any] = {
            "cli_name": client.name,
            "return_code": exc.returncode,
        }
        if exc.stdout:
            metadata["stdout"] = exc.stdout.strip()
        if exc.stderr:
            metadata["stderr"] = exc.stderr.strip()
        return metadata

    def _raise_tool_error(self, message: str, metadata: dict[str, Any] | None = None) -> NoReturn:
        error_output = ToolOutput(status="error", content=message, content_type="text", metadata=metadata)
        raise ToolExecutionError(error_output.model_dump_json())

    def _agent_capabilities_guidance(self, client: ResolvedCLIClient) -> str:
        """Build the capabilities guidance for the CLI actually being invoked.

        Interpolates the resolved client's name rather than consulting a lookup
        table, so a newly registered target is named correctly with no change
        here. Deliberately asserts no specific capability: ``ResolvedCLIClient``
        models none, targets differ in what they can do (aider has no web
        search, for instance), and promising a facility the target lacks
        misinforms the model about the tools available to it.
        """
        return (
            f"You are operating through the {client.name} CLI agent. Use whatever tools are "
            "available to you to gather the information you need. Deliver the final answer "
            "yourself without asking the Unison MCP host to perform searches or file reads "
            "on your behalf."
        )

    def _format_file_references(self, files: list[str]) -> str:
        if not files:
            return ""

        references: list[str] = []
        for file_path in files:
            try:
                path = Path(file_path)
                stat = path.stat()
                modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                size = stat.st_size
                references.append(f"- {file_path} (last modified {modified}, {size} bytes)")
            except OSError:
                references.append(f"- {file_path} (unavailable)")
        return "\n".join(references)
