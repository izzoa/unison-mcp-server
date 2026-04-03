"""
Prompt template definitions for all MCP tools.

Each template provides a name, description, and template string used by
the MCP prompt handlers (handle_list_prompts, handle_get_prompt).
"""

PROMPT_TEMPLATES = {
    "chat": {
        "name": "chat",
        "description": "Chat and brainstorm ideas",
        "template": "Chat with {model} about this",
    },
    "clink": {
        "name": "clink",
        "description": "Forward a request to a configured AI CLI (e.g., Gemini)",
        "template": "Use clink with cli_name=<cli> to run this prompt",
    },
    "thinkdeep": {
        "name": "thinkdeeper",
        "description": "Step-by-step deep thinking workflow with expert analysis",
        "template": "Start comprehensive deep thinking workflow with {model} using {thinking_mode} thinking mode",
    },
    "planner": {
        "name": "planner",
        "description": "Break down complex ideas, problems, or projects into multiple manageable steps",
        "template": "Create a detailed plan with {model}",
    },
    "consensus": {
        "name": "consensus",
        "description": "Step-by-step consensus workflow with multi-model analysis",
        "template": "Start comprehensive consensus workflow with {model}",
    },
    "codereview": {
        "name": "review",
        "description": "Perform a comprehensive code review",
        "template": "Perform a comprehensive code review with {model}",
    },
    "precommit": {
        "name": "precommit",
        "description": "Step-by-step pre-commit validation workflow",
        "template": "Start comprehensive pre-commit validation workflow with {model}",
    },
    "debug": {
        "name": "debug",
        "description": "Debug an issue or error",
        "template": "Help debug this issue with {model}",
    },
    "secaudit": {
        "name": "secaudit",
        "description": "Comprehensive security audit with OWASP Top 10 coverage",
        "template": "Perform comprehensive security audit with {model}",
    },
    "docgen": {
        "name": "docgen",
        "description": "Generate comprehensive code documentation with complexity analysis",
        "template": "Generate comprehensive documentation with {model}",
    },
    "analyze": {
        "name": "analyze",
        "description": "Analyze files and code structure",
        "template": "Analyze these files with {model}",
    },
    "refactor": {
        "name": "refactor",
        "description": "Refactor and improve code structure",
        "template": "Refactor this code with {model}",
    },
    "tracer": {
        "name": "tracer",
        "description": "Trace code execution paths",
        "template": "Generate tracer analysis with {model}",
    },
    "testgen": {
        "name": "testgen",
        "description": "Generate comprehensive tests",
        "template": "Generate comprehensive tests with {model}",
    },
    "challenge": {
        "name": "challenge",
        "description": "Challenge a statement critically without automatic agreement",
        "template": "Challenge this statement critically",
    },
    "apilookup": {
        "name": "apilookup",
        "description": "Look up the latest API or SDK information",
        "template": "Lookup latest API docs for {model}",
    },
    "listmodels": {
        "name": "listmodels",
        "description": "List available AI models",
        "template": "List all available models",
    },
    "version": {
        "name": "version",
        "description": "Show server version and system information",
        "template": "Show Unison MCP Server version",
    },
}
