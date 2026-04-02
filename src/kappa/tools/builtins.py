"""Built-in system tools for VFS memory operations.

These are the first tools registered in the registry, allowing the
agent to persist and retrieve knowledge through the self-healing loop.
"""

from __future__ import annotations

from kappa.memory.vfs import VFSManager
from kappa.tools.registry import ToolResult


class ReadMemoryTool:
    """Read a file from the agent's long-term memory (VFS)."""

    def __init__(self, vfs: VFSManager) -> None:
        self._vfs = vfs

    @property
    def name(self) -> str:
        return "read_memory"

    @property
    def description(self) -> str:
        return "Read a file from long-term memory. Args: path (str)."

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(success=False, output="", error="Missing required arg: path")

        try:
            content = self._vfs.read(path)
        except ValueError as e:
            return ToolResult(success=False, output="", error=str(e))

        if content is None:
            return ToolResult(success=True, output="", error=f"File not found: {path}")

        return ToolResult(success=True, output=content)


class WriteMemoryTool:
    """Write content to a file in the agent's long-term memory (VFS)."""

    def __init__(self, vfs: VFSManager) -> None:
        self._vfs = vfs

    @property
    def name(self) -> str:
        return "write_memory"

    @property
    def description(self) -> str:
        return "Write content to long-term memory. Args: path (str), content (str)."

    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        content = kwargs.get("content")

        if not path:
            return ToolResult(success=False, output="", error="Missing required arg: path")
        if content is None:
            return ToolResult(success=False, output="", error="Missing required arg: content")

        try:
            self._vfs.write(path, content)
        except ValueError as e:
            return ToolResult(success=False, output="", error=str(e))

        return ToolResult(success=True, output=f"Written to {path}")
