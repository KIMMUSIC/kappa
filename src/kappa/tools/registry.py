"""Tool registry with protocol-based tool abstraction.

Provides a plug-and-play registry where tools conforming to the
``BaseTool`` protocol can be registered by name and dispatched at
runtime.  All tool executions are tracked through the budget system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from kappa.budget.tracker import BudgetTracker
from kappa.exceptions import ToolExecutionError


@dataclass(frozen=True)
class ToolResult:
    """Structured output from a tool execution."""

    success: bool
    output: str
    error: str | None = None


@runtime_checkable
class BaseTool(Protocol):
    """Interface that every tool must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def execute(self, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
    """Manages registration, lookup, and execution of tools.

    Each ``execute()`` call records a fixed token cost against the
    ``BudgetTracker`` so that runaway tool loops are bounded by the
    same budget ceiling as LLM calls.

    Args:
        tracker: Optional BudgetTracker for cost accounting.
            If provided, each tool execution records a fixed token cost.
        cost_per_tool_call: Virtual token cost charged per tool invocation.
    """

    def __init__(
        self,
        tracker: BudgetTracker | None = None,
        cost_per_tool_call: int = 50,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._tracker = tracker
        self._cost_per_tool_call = cost_per_tool_call

    def register(self, tool: BaseTool) -> None:
        """Register a tool.  Overwrites if name already exists."""
        if not isinstance(tool, BaseTool):
            raise TypeError(
                f"Tool must satisfy BaseTool protocol, got {type(tool).__name__}"
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Look up a tool by name.

        Raises:
            ToolExecutionError: If the tool is not registered.
        """
        if name not in self._tools:
            raise ToolExecutionError(
                f"Unknown tool: {name!r}. "
                f"Available: {sorted(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> list[dict[str, str]]:
        """Return metadata for all registered tools (sorted by name)."""
        return sorted(
            [{"name": t.name, "description": t.description} for t in self._tools.values()],
            key=lambda d: d["name"],
        )

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Look up and execute a tool, recording cost against the budget.

        Raises:
            ToolExecutionError: If the tool is not registered.
            BudgetExceededException: If the budget is exhausted.
        """
        tool = self.get(name)

        # Record virtual cost before execution
        if self._tracker:
            self._tracker.record_usage(
                prompt_tokens=self._cost_per_tool_call,
                completion_tokens=0,
            )

        return tool.execute(**kwargs)
