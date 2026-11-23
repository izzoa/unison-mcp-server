"""
Simple tools for Unison MCP.

Simple tools follow a basic request → AI model → response pattern.
They inherit from SimpleTool which provides streamlined functionality
for tools that don't need multi-step workflows.

Available simple tools:
- chat: General chat and collaborative thinking
- challenge: Critical-thinking wrapper for pushback on a statement
- apilookup: Current API and SDK documentation lookup
- clink: Bridge requests to configured external AI CLIs
"""

from .base import SimpleTool

__all__ = ["SimpleTool"]
