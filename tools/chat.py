"""
Chat tool - General development chat and collaborative thinking

This tool provides a conversational interface for general development assistance,
brainstorming, problem-solving, and collaborative thinking. It supports file context,
images, and conversation continuation for seamless multi-turn interactions.
"""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from mcp.types import TextContent
from pydantic import Field, model_validator

if TYPE_CHECKING:
    from providers.shared import ModelCapabilities
    from tools.models import ToolModelCategory

from config import TEMPERATURE_BALANCED
from systemprompts import CHAT_PROMPT, GENERATE_CODE_PROMPT
from tools.models import ToolOutput
from tools.shared.base_models import COMMON_FIELD_DESCRIPTIONS, MAX_TEMPERATURE, ToolRequest
from tools.shared.batch_support import (
    BatchControlFields,
    BatchResumableToolMixin,
    admitted_resolution,
    batch_schema_properties,
    require_inputs_unless_resuming,
)
from tools.shared.exceptions import ToolExecutionError
from utils.batch_coordinator import ChildPlan

from .simple.base import SimpleTool

# Field descriptions matching the original Chat tool exactly
CHAT_FIELD_DESCRIPTIONS = {
    "prompt": (
        "Your question or idea for collaborative thinking to be sent to the external model. Provide detailed context, "
        "including your goal, what you've tried, and any specific challenges. "
        "WARNING: Large inline code must NOT be shared in prompt. Provide full-path to files on disk as separate parameter."
    ),
    "absolute_file_paths": ("Full, absolute file paths to relevant code in order to share with external model"),
    "images": "Image paths (absolute) or base64 strings for optional visual context.",
    "working_directory_absolute_path": (
        "Absolute path to an existing directory where generated code artifacts can be saved."
    ),
}


class ChatRequest(BatchControlFields, ToolRequest):
    """Request model for Chat tool.

    ``prompt`` and ``working_directory_absolute_path`` are schema-optional so a call that
    carries only a ``batch_handle`` validates; they are enforced at runtime (see
    :meth:`_require_start_inputs`) whenever the call starts new work.
    """

    prompt: str | None = Field(default=None, description=CHAT_FIELD_DESCRIPTIONS["prompt"])
    absolute_file_paths: list[str] | None = Field(
        default_factory=list,
        description=CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
    )
    images: list[str] | None = Field(default_factory=list, description=CHAT_FIELD_DESCRIPTIONS["images"])
    working_directory_absolute_path: str | None = Field(
        default=None,
        description=CHAT_FIELD_DESCRIPTIONS["working_directory_absolute_path"],
    )

    @model_validator(mode="after")
    def _require_start_inputs(self) -> "ChatRequest":
        """Starting new work still needs a prompt and a working directory; resuming needs neither."""

        require_inputs_unless_resuming(
            self,
            "chat",
            {
                "prompt": self.prompt,
                "working_directory_absolute_path": self.working_directory_absolute_path,
            },
        )
        return self


class ChatTool(BatchResumableToolMixin, SimpleTool):
    """
    General development chat and collaborative thinking tool using SimpleTool architecture.

    This tool provides identical functionality to the original Chat tool but uses the new
    SimpleTool architecture for cleaner code organization and better maintainability.

    Migration note: This tool is designed to be a drop-in replacement for the original
    Chat tool with 100% behavioral compatibility.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_recordable_response: str | None = None

    def get_name(self) -> str:
        return "chat"

    def get_description(self) -> str:
        return (
            "General chat and collaborative thinking partner for brainstorming, development discussion, "
            "getting second opinions, and exploring ideas. Use for ideas, validations, questions, and thoughtful explanations."
        )

    def get_annotations(self) -> dict[str, Any] | None:
        """Chat writes generated artifacts when code-generation is enabled."""

        return {"readOnlyHint": False}

    def get_system_prompt(self) -> str:
        return CHAT_PROMPT

    def get_capability_system_prompts(self, capabilities: Optional["ModelCapabilities"]) -> list[str]:
        prompts = list(super().get_capability_system_prompts(capabilities))
        if capabilities and capabilities.allow_code_generation:
            prompts.append(GENERATE_CODE_PROMPT)
        return prompts

    def get_default_temperature(self) -> float:
        return TEMPERATURE_BALANCED

    def get_model_category(self) -> "ToolModelCategory":
        """Chat prioritizes fast responses and cost efficiency"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.FAST_RESPONSE

    def get_request_model(self):
        """Return the Chat-specific request model"""
        return ChatRequest

    # === Schema Generation Utilities ===

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema matching the original Chat tool expectations."""

        # No schema-level required inputs, not even ``model`` in auto mode: the MCP boundary
        # validates every call against this schema before dispatch, and a call carrying only
        # batch_handle (resume/cancel) must validate. prompt and working_directory_absolute_path
        # are enforced at runtime when starting work (ChatRequest._require_start_inputs). ``model``
        # in auto mode is the exception: a tool cannot demand it at runtime, because the MCP
        # boundary (handlers/tool_handlers.py) resolves a missing model to the category's
        # preferred one before ``execute`` runs. Reaching ``execute`` directly with no model in
        # auto mode still raises the "Model parameter is required in auto mode" error.
        required_fields: list[str] = []

        schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": CHAT_FIELD_DESCRIPTIONS["prompt"],
                },
                "absolute_file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": CHAT_FIELD_DESCRIPTIONS["images"],
                },
                "working_directory_absolute_path": {
                    "type": "string",
                    "description": CHAT_FIELD_DESCRIPTIONS["working_directory_absolute_path"],
                },
                "model": self.get_model_field_schema(),
                "temperature": {
                    "type": "number",
                    "description": COMMON_FIELD_DESCRIPTIONS["temperature"],
                    "minimum": 0,
                    "maximum": MAX_TEMPERATURE,
                },
                "thinking_mode": {
                    "type": "string",
                    "enum": ["minimal", "low", "medium", "high", "max"],
                    "description": COMMON_FIELD_DESCRIPTIONS["thinking_mode"],
                },
                "continuation_id": {
                    "type": "string",
                    "description": COMMON_FIELD_DESCRIPTIONS["continuation_id"],
                },
                **batch_schema_properties(),
            },
            "required": required_fields,
            "additionalProperties": False,
        }

        return schema

    # === Batch execution (execution_mode="batch") ===

    async def execute_batch_start(self, arguments: dict[str, Any], providers: Any) -> list[Any]:
        """Submit this chat call as a one-request provider batch and wait a bounded time for it.

        Admission has already checked that nothing non-text is attached and that ``providers`` holds a
        provider that can batch the model. The prompt, system prompt and temperature are assembled the
        way ``SimpleTool.execute`` assembles them for a synchronous call (that method is deliberately
        untouched), then the call returns the completed reply when the batch finishes inside the wait
        and a resumable ``status='running'`` handle otherwise. Reasoning effort (``thinking_mode``) is a
        synchronous-path feature and is not carried into the batch request.
        """

        from providers.shared.batch import BatchRequest
        from utils.model_context import ModelContext
        from utils.model_resolution import trim_model_identifier

        request = ChatRequest(**arguments)
        path_error = self._validate_file_paths(request)
        if path_error:
            self._raise_batch_error(path_error, code="invalid_path")

        model = trim_model_identifier(str(arguments.get("model") or ""))
        provider = providers[model]
        self._current_arguments = arguments
        self._current_model_name = model
        # The limits are those of the model that will run, resolved once at admission (an alias such as opus is
        # not described by its older curated entry); a provider that resolves nothing keeps its own capabilities.
        resolution = admitted_resolution(providers, model)
        self._model_context = ModelContext(
            model_name=model,
            provider=provider,
            capabilities=resolution.capabilities if resolution is not None else None,
        )

        try:
            prompt = await self._assemble_batch_prompt(request, arguments)
            temperature, warnings = self.get_validated_temperature(request, self._model_context)
            self.log_batch_temperature_warnings(warnings)
            capabilities = self._model_context.capabilities
            system_prompt = self.get_language_instruction() + self._augment_system_prompt_with_capabilities(
                self.get_system_prompt(), capabilities
            )
            self.check_batch_input_fits(
                self._model_context,
                resolution.slug if resolution is not None else model,
                [prompt, system_prompt],
                resolution=resolution,
            )
        except ToolExecutionError:
            raise
        except Exception as exc:
            logger.exception("Could not prepare the batched chat request")
            self._raise_batch_error(
                f"Could not prepare the batched chat request ({exc}). Nothing was submitted and nothing was run.",
                code="batch_preparation_failed",
            )
            return []  # unreachable

        plan = ChildPlan(
            provider=provider,
            model=model,
            requests=[
                BatchRequest(
                    custom_id="chat",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    extra={"model": model},
                )
            ],
            resolution=resolution,
        )
        result = await self.run_batch_submission(
            [plan],
            continuation_id=request.continuation_id,
            payload={"working_directory_absolute_path": request.working_directory_absolute_path, "model": model},
            wait_seconds=request.wait_seconds,
        )
        return self.render_batch_result(result)

    async def _assemble_batch_prompt(self, request: ChatRequest, arguments: dict[str, Any]) -> str:
        """The full prompt a synchronous chat call would send, history and follow-up instructions included.

        Mirrors the prompt assembly in ``SimpleTool.execute`` (a continuation either arrives with its
        history already embedded by the server boundary or is reconstructed here, and a new
        conversation gets the follow-up instructions).
        """

        continuation_id = self.get_request_continuation_id(request)
        if not continuation_id:
            from utils.request_helpers import get_follow_up_instructions

            prompt = await self.prepare_prompt(request)
            return f"{prompt}\n\n{get_follow_up_instructions(0)}"

        from utils.context_reconstructor import CONVERSATION_HISTORY_HEADER_PREFIX
        from utils.conversation_memory import add_turn, build_conversation_history, get_thread
        from utils.tool_execution_context import ToolExecutionContext

        exec_ctx = ToolExecutionContext.from_arguments(arguments)
        field_value = self.get_request_prompt(request)
        if (exec_ctx is not None and bool(exec_ctx.original_user_prompt)) or (
            CONVERSATION_HISTORY_HEADER_PREFIX in field_value
        ):
            return field_value  # the boundary already embedded the history and recorded the user turn

        thread_context = get_thread(continuation_id)
        if not thread_context:
            logger.warning("Thread %s not found, preparing prompt normally", continuation_id)
            return await self.prepare_prompt(request)
        if field_value:
            add_turn(continuation_id, "user", field_value, files=self.get_request_files(request))
            thread_context = get_thread(continuation_id) or thread_context
        conversation_history, _ = build_conversation_history(thread_context, self._model_context)
        base_prompt = await self.prepare_prompt(request)
        if conversation_history:
            return f"{conversation_history}\n\n=== NEW USER INPUT ===\n{base_prompt}"
        return base_prompt

    def render_completed_batch(self, result: Any) -> list[TextContent]:
        """Render a completed chat batch through the same formatting a synchronous reply gets.

        The model named is the one recorded at submission. Request-dependent formatting uses
        the fields the submitting call stored in the record's payload (currently
        ``working_directory_absolute_path``); code-block extraction needs a live model
        context, which a resumption does not have, so generated code blocks are left in the
        text rather than written to disk.
        """

        outcomes = [o for item in result.children for o in item.outcomes]
        success = next((o for o in outcomes if o.success and o.response is not None and o.response.content), None)
        if success is None or success.response is None:
            failure = next((o for o in outcomes if not o.success and o.error is not None), None)
            reason = (
                failure.error.message
                if failure is not None and failure.error is not None
                else next((i.error for i in result.children if i.error), "the batch produced no usable response")
            )
            error = ToolOutput(
                status="error",
                content=f"The batched chat request failed: {reason}",
                content_type="text",
                metadata={"tool_name": self.get_name(), "batch_handle": result.handle},
            )
            raise ToolExecutionError(error.model_dump_json())

        payload = result.record.payload if result.record is not None and isinstance(result.record.payload, dict) else {}
        request = ChatRequest.model_construct(
            working_directory_absolute_path=payload.get("working_directory_absolute_path"),
            continuation_id=result.record.continuation_id if result.record is not None else None,
        )
        response = success.response
        # format_response consults ``_model_context`` (left over from an earlier synchronous call on this
        # shared instance) to decide on code extraction; a resumption has no model context, so mask it.
        # This runs without yielding to the event loop, so no other call can observe the masked state.
        stale_context = getattr(self, "_model_context", None)
        self._model_context = None
        try:
            body = self.format_response(response.content, request, {"model_name": response.model_name})
        finally:
            self._model_context = stale_context
        metadata: dict[str, Any] = {
            "tool_name": self.get_name(),
            "batch_handle": result.handle,
            "batch_execution": True,
            "model_used": response.model_name,
        }
        provider = getattr(response.provider, "value", None)
        if provider:
            metadata["provider_used"] = provider
        # A family alias (opus, flash ...) resolves at submission to the latest batch-listed model; say which,
        # from the record, so the caller sees what the batch actually ran on.
        from utils.model_resolution import resolution_metadata

        answered = next((item.child for item in result.children if item.outcomes), None)
        requested = answered.requested_model if answered is not None else None
        metadata.update(resolution_metadata(requested, response.model_name))
        # Newer batch models the alias did not pick, as recorded at submission (a resume reads the record, never
        # the catalog); the keys are absent when there is nothing to report.
        if answered is not None:
            from providers.shared.batch_resolution import note_from_record

            metadata.update(note_from_record(answered.resolution))

        continuation_id = request.continuation_id
        offer = None
        if continuation_id:
            # A batched call on an existing conversation recorded its user turn at submission, as a
            # synchronous one does before the model call; its reply is recorded here, once, so the thread
            # is never left with an unanswered user turn.
            self._record_batch_reply(continuation_id, result.handle, response)
            offer = self._create_continuation_offer(request)
        if offer:
            from tools.models import ContinuationOffer

            metadata["conversation_ready"] = True
            output = ToolOutput(
                status="continuation_available",
                content=body,
                content_type="text",
                continuation_offer=ContinuationOffer(
                    continuation_id=offer["continuation_id"],
                    note=offer["note"],
                    remaining_turns=offer["remaining_turns"],
                ),
                metadata=metadata,
            )
        else:
            output = ToolOutput(status="success", content=body, content_type="text", metadata=metadata)
        return [TextContent(type="text", text=output.model_dump_json())]

    def _record_batch_reply(self, continuation_id: str, handle: str, response: Any) -> None:
        """Record a completed batch's reply as the assistant turn of its conversation, at most once per handle.

        Resuming a finished handle renders it again, so the turn is tagged with the handle in its metadata
        and an existing tagged turn means it is already there. Best effort: a thread that is gone (expired)
        has nothing to record into, and a failure here must not lose the reply the caller is waiting for.
        """

        try:
            from utils.conversation_memory import add_turn, get_thread

            thread = get_thread(continuation_id)
            if thread is None:
                return
            for turn in thread.turns:
                if turn.role == "assistant" and (turn.model_metadata or {}).get("batch_handle") == handle:
                    return
            provider = getattr(response.provider, "value", None)
            add_turn(
                continuation_id,
                "assistant",
                response.content,
                tool_name=self.get_name(),
                model_provider=provider,
                model_name=response.model_name,
                model_metadata={"usage": response.usage, "batch_handle": handle},
            )
        except Exception:
            logger.warning(
                "Could not record the batched reply for %s in its conversation", continuation_id, exc_info=True
            )

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        """Tool-specific field definitions used by SimpleTool scaffolding."""

        return {
            "prompt": {
                "type": "string",
                "description": CHAT_FIELD_DESCRIPTIONS["prompt"],
            },
            "absolute_file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": CHAT_FIELD_DESCRIPTIONS["images"],
            },
            "working_directory_absolute_path": {
                "type": "string",
                "description": CHAT_FIELD_DESCRIPTIONS["working_directory_absolute_path"],
            },
        }

    def get_required_fields(self) -> list[str]:
        """Required fields for ChatSimple tool"""
        return ["prompt", "working_directory_absolute_path"]

    # === Hook Method Implementations ===

    async def prepare_prompt(self, request: ChatRequest) -> str:
        """
        Prepare the chat prompt with optional context files.

        This implementation matches the original Chat tool exactly while using
        SimpleTool convenience methods for cleaner code.
        """
        # Use SimpleTool's Chat-style prompt preparation
        return self.prepare_chat_style_prompt(request)

    def _validate_file_paths(self, request) -> str | None:
        """Extend validation to cover the working directory path."""

        files = self.get_request_files(request)
        if files:
            expanded_files: list[str] = []
            for file_path in files:
                expanded = os.path.expanduser(file_path)
                if not os.path.isabs(expanded):
                    return (
                        "Error: All file paths must be FULL absolute paths to real files / folders - DO NOT SHORTEN. "
                        f"Received: {file_path}"
                    )
                expanded_files.append(expanded)
            self.set_request_files(request, expanded_files)

        error = super()._validate_file_paths(request)
        if error:
            return error

        working_directory = request.working_directory_absolute_path
        if working_directory:
            expanded = os.path.expanduser(working_directory)
            if not os.path.isabs(expanded):
                return (
                    "Error: 'working_directory_absolute_path' must be an absolute path (you may use '~' which will be expanded). "
                    f"Received: {working_directory}"
                )
            if not os.path.isdir(expanded):
                return (
                    "Error: 'working_directory_absolute_path' must reference an existing directory. "
                    f"Received: {working_directory}"
                )
            # Sandbox the WRITE destination with the same policy applied to reads.
            # Writing pal_generated.code is more dangerous than reading, so it must
            # not be held to a weaker standard: refuse dangerous system dirs,
            # pseudo-filesystems, and the home-directory root.
            from utils.file_utils import is_home_directory_root
            from utils.security_config import is_dangerous_path

            resolved_wd = Path(expanded).resolve()
            if is_dangerous_path(resolved_wd) or is_home_directory_root(resolved_wd):
                return (
                    "Error: 'working_directory_absolute_path' resolves to a protected location "
                    f"({resolved_wd}). Choose a project subdirectory instead."
                )
        return None

    def format_response(self, response: str, request: ChatRequest, model_info: dict | None = None) -> str:
        """
        Format the chat response to match the original Chat tool exactly.
        """
        self._last_recordable_response = None
        body = response
        recordable_override: str | None = None

        if self._model_supports_code_generation():
            block, remainder, _ = self._extract_generated_code_block(response)
            if block:
                sanitized_text = remainder.strip()
                target_directory = request.working_directory_absolute_path
                try:
                    artifact_path = self._persist_generated_code_block(block, target_directory)
                except Exception as exc:  # pragma: no cover - rare filesystem failures
                    logger.error("Failed to persist generated code block: %s", exc, exc_info=True)
                    warning = (
                        f"WARNING: Unable to write pal_generated.code inside '{target_directory}'. "
                        "Check the path permissions and re-run. The generated code block is included below for manual handling."
                    )

                    history_copy_base = sanitized_text
                    history_copy = self._join_sections(history_copy_base, warning) if history_copy_base else warning
                    recordable_override = history_copy

                    sanitized_warning = history_copy.strip()
                    body = f"{sanitized_warning}\n\n{block.strip()}".strip()
                else:
                    if not sanitized_text:
                        base_message = (
                            "Generated code saved to pal_generated.code.\n"
                            "\n"
                            "CRITICAL: Contains mixed instructions + partial snippets - NOT complete code to copy as-is!\n"
                            "\n"
                            "You MUST:\n"
                            "  1. Read as a proposal from partial context - you may need to read the file in sections\n"
                            "  2. Implement ideas using YOUR complete codebase context and understanding\n"
                            "  3. Never paste wholesale - snippets may be partial with missing lines, pasting will corrupt your code!\n"
                            "  4. Adapt to fit your actual structure and style\n"
                            "  5. Build/lint/test after implementation to verify correctness\n"
                            "\n"
                            "Treat as guidance to implement thoughtfully, not ready-to-paste code."
                        )
                        sanitized_text = base_message

                    instruction = self._build_agent_instruction(artifact_path)
                    body = self._join_sections(sanitized_text, instruction)

        final_output = (
            f"{body}\n\n---\n\nAGENT'S TURN: Evaluate this perspective alongside your analysis to "
            "form a comprehensive solution and continue with the user's request and task at hand."
        )

        if recordable_override is not None:
            self._last_recordable_response = (
                f"{recordable_override}\n\n---\n\nAGENT'S TURN: Evaluate this perspective alongside your analysis to "
                "form a comprehensive solution and continue with the user's request and task at hand."
            )
        else:
            self._last_recordable_response = final_output

        return final_output

    def _record_assistant_turn(
        self, continuation_id: str, response_text: str, request, model_info: dict | None
    ) -> None:
        recordable = self._last_recordable_response if self._last_recordable_response is not None else response_text
        try:
            super()._record_assistant_turn(continuation_id, recordable, request, model_info)
        finally:
            self._last_recordable_response = None

    def _model_supports_code_generation(self) -> bool:
        context = getattr(self, "_model_context", None)
        if not context:
            return False

        try:
            capabilities = context.capabilities
        except Exception:  # pragma: no cover - defensive fallback
            return False

        return bool(capabilities.allow_code_generation)

    def _extract_generated_code_block(self, text: str) -> tuple[str | None, str, int]:
        matches = list(re.finditer(r"<GENERATED-CODE>.*?</GENERATED-CODE>", text, flags=re.DOTALL | re.IGNORECASE))
        if not matches:
            return None, text, 0

        last_match = matches[-1]
        block = last_match.group(0).strip()

        # Merge the text before and after the final block while trimming excess whitespace
        before = text[: last_match.start()]
        after = text[last_match.end() :]
        remainder = self._join_sections(before, after)

        return block, remainder, len(matches)

    def _persist_generated_code_block(self, block: str, working_directory: str) -> Path:
        expanded = os.path.expanduser(working_directory)
        target_dir = Path(expanded).resolve()
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Absolute working directory path '{working_directory}' does not exist")

        target_file = target_dir / "pal_generated.code"
        content = block if block.endswith("\n") else f"{block}\n"

        # Write without following symlinks. O_NOFOLLOW makes open() fail if the
        # final path component is a symlink, so a planted `pal_generated.code`
        # symlink (CWE-59) cannot redirect the write to an arbitrary file, and
        # there is no unlink->write TOCTOU window. O_TRUNC replaces the content
        # of an existing regular file in place; we never fall through to a write
        # after an ambiguous/failed open.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(target_file, flags, 0o600)
        except OSError as exc:
            raise OSError(f"Refused to write pal_generated.code at '{target_file}': {exc}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        logger.info("Generated code artifact written to %s", target_file)
        return target_file

    @staticmethod
    def _build_agent_instruction(artifact_path: Path) -> str:
        return (
            f"CONTINUING FROM PREVIOUS DISCUSSION: Implementation plan saved to `{artifact_path}`.\n"
            "\n"
            f"CRITICAL WARNING: `{artifact_path}` may contain partial code snippets from another AI with limited context. "
            "Wholesale copy-pasting MAY CORRUPT your codebase with incomplete logic and missing lines.\n"
            "\n"
            "Required workflow:\n"
            "1. For <UPDATED_EXISTING_FILE:...> blocks: Partial excerpts only. Understand the intent and implement using YOUR full context. "
            "DO NOT copy wholesale - adapt ideas to fit actual structure.\n"
            "2. For <NEWFILE:...> blocks: Understand proposal and create properly. Verify completeness (imports, syntax, logic).\n"
            "3. Validation: After ALL changes, verify correctness using available tools (build/compile, linters, tests, type checks, etc.).\n"
            f"4. Cleanup: After you're done reading and applying changes, delete `{artifact_path}` once verified to prevent stale instructions.\n"
            "\n"
            "Treat this as a patch-set requiring manual integration, not ready-to-paste code. You have full codebase context - use it."
        )

    @staticmethod
    def _join_sections(*sections: str) -> str:
        chunks: list[str] = []
        for section in sections:
            if section:
                trimmed = section.strip()
                if trimmed:
                    chunks.append(trimmed)
        return "\n\n".join(chunks)

    def get_websearch_guidance(self) -> str:
        """
        Return Chat tool-style web search guidance.
        """
        return self.get_chat_style_websearch_guidance()


logger = logging.getLogger(__name__)
