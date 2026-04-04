"""
MCP prompt handler implementations for the Unison MCP Server.

Contains the list_prompts and get_prompt handler logic, wired onto the
MCP server instance via the register() function.
"""

import logging
from typing import Any

from mcp.types import GetPromptResult, Prompt, PromptMessage

from conf.prompt_templates import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


def register(server, tool_registry) -> None:
    """
    Register list_prompts and get_prompt handlers on the MCP server.

    Args:
        server: The MCP Server instance.
        tool_registry: A ToolRegistry for checking tool existence.
    """

    @server.list_prompts()
    async def handle_list_prompts() -> list[Prompt]:
        """
        List all available prompts for CLI Code shortcuts.

        Automatically generates prompts from all tools (1:1 mapping) plus
        a few marketing aliases with richer templates for commonly used tools.
        """
        logger.debug("MCP client requested prompt list")
        prompts = []

        available = tool_registry.get_available_tools()

        for tool_name, tool in available.items():
            if tool_name in PROMPT_TEMPLATES:
                template_info = PROMPT_TEMPLATES[tool_name]
                prompts.append(
                    Prompt(
                        name=template_info["name"],
                        description=template_info["description"],
                        arguments=[],
                    )
                )
            else:
                prompts.append(
                    Prompt(
                        name=tool_name,
                        description=f"Use {tool.name} tool",
                        arguments=[],
                    )
                )

        # Add special "continue" prompt
        prompts.append(
            Prompt(
                name="continue",
                description="Continue the previous conversation using the chat tool",
                arguments=[],
            )
        )

        logger.debug("Returning %d prompts to MCP client", len(prompts))
        return prompts

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, Any] = None) -> GetPromptResult:
        """
        Get prompt details and generate the actual prompt text.

        Supports structured prompt names like "chat:gpt5" where "chat" is
        the tool name and "gpt5" is the model to use.
        """
        logger.debug("MCP client requested prompt: %s with args: %s", name, arguments)

        available = tool_registry.get_available_tools()

        # Handle special "continue" case
        if name.lower() == "continue":
            tool_name = "chat"
            template_info = {
                "name": "continue",
                "description": "Continue the previous conversation",
                "template": "Continue the conversation",
            }
            logger.debug("Using /unison:continue - defaulting to chat tool")
        else:
            tool_name = None
            template_info = None

            # Check if it's a known prompt name
            for t_name, t_info in PROMPT_TEMPLATES.items():
                if t_info["name"] == name:
                    tool_name = t_name
                    template_info = t_info
                    break

            # If not found, check if it's a direct tool name
            if not tool_name and name in available:
                tool_name = name
                template_info = {
                    "name": name,
                    "description": f"Use {name} tool",
                    "template": f"Use {name}",
                }

            if not tool_name:
                logger.error("Unknown prompt requested: %s", name)
                raise ValueError(f"Unknown prompt: {name}")

        # Get the template
        template = template_info.get("template", f"Use {tool_name}")

        # Safe template expansion with defaults
        final_model = arguments.get("model", "auto") if arguments else "auto"

        prompt_args = {
            "model": final_model,
            "thinking_mode": arguments.get("thinking_mode", "medium") if arguments else "medium",
        }

        logger.debug("Using model '%s' for prompt '%s'", final_model, name)

        # Safely format the template
        try:
            prompt_text = template.format(**prompt_args)
        except KeyError as e:
            logger.warning("Missing template argument %s for prompt %s, using raw template", e, name)
            prompt_text = template

        # Generate tool call instruction
        if name.lower() == "continue":
            tool_instruction = (
                f"Continue the previous conversation using the {tool_name} tool. "
                "CRITICAL: You MUST provide the continuation_id from the previous response "
                "to maintain conversation context. "
                "Additionally, you should reuse the same model that was used in the previous "
                "exchange for consistency, unless the user specifically asks for a different "
                "model name to be used."
            )
        else:
            tool_instruction = prompt_text

        return GetPromptResult(
            prompt=Prompt(
                name=name,
                description=template_info["description"],
                arguments=[],
            ),
            messages=[
                PromptMessage(
                    role="user",
                    content={"type": "text", "text": tool_instruction},
                )
            ],
        )
