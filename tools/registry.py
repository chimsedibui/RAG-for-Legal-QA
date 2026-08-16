"""Minimal tool registry: a name -> Tool dict plus a helper to build the
OpenAI function-calling schema list. Deliberately not a plugin-loading
system — adding a tool means creating one module implementing the Tool
protocol (core/interfaces.py) and calling .register() once at the
composition root (api/app.py); no edits to services/rag_pipeline.py needed.
"""
from typing import Dict, List, Optional

from core.interfaces import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def schemas(self) -> List[dict]:
        """Returns the OpenAI-format `tools` list for chat completions."""
        return [{"type": "function", "function": t.schema} for t in self._tools.values()]
