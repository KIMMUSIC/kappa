"""Tests for the MCP tool registry and built-in tools (Task 2).

Covers:
- BaseTool protocol conformance
- ToolRegistry: register / get / list / execute
- Budget tracking on tool execution
- Built-in tools (read_memory, write_memory) integration with VFS
- Error handling (unknown tool, missing args, path traversal)
"""

from __future__ import annotations

import pytest

from kappa.budget.tracker import BudgetTracker
from kappa.config import BudgetConfig, MemoryConfig
from kappa.exceptions import BudgetExceededException, ToolExecutionError
from kappa.memory.vfs import VFSManager
from kappa.tools.builtins import ReadMemoryTool, WriteMemoryTool
from kappa.tools.registry import BaseTool, ToolRegistry, ToolResult


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture()
def vfs(tmp_path):
    config = MemoryConfig(workspace_root="workspace")
    return VFSManager(config=config, base_dir=tmp_path)


@pytest.fixture()
def registry():
    return ToolRegistry()


@pytest.fixture()
def tracked_registry():
    """Registry with a budget tracker (1000 token limit)."""
    tracker = BudgetTracker(BudgetConfig(max_total_tokens=1000, max_cost_usd=100.0))
    return ToolRegistry(tracker=tracker, cost_per_tool_call=50)


# ── Helpers ─────────────────────────────────────────────────────


class EchoTool:
    """Minimal tool for testing — echoes its input."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes the message back."

    def execute(self, **kwargs) -> ToolResult:
        msg = kwargs.get("message", "")
        return ToolResult(success=True, output=msg)


class FailTool:
    """Always-failing tool for error path testing."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "Always fails."

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=False, output="", error="intentional failure")


# ── BaseTool Protocol ──────────────────────────────────────────


class TestBaseToolProtocol:

    def test_echo_satisfies_protocol(self):
        assert isinstance(EchoTool(), BaseTool)

    def test_read_memory_satisfies_protocol(self, vfs):
        assert isinstance(ReadMemoryTool(vfs), BaseTool)

    def test_write_memory_satisfies_protocol(self, vfs):
        assert isinstance(WriteMemoryTool(vfs), BaseTool)


# ── ToolResult ─────────────────────────────────────────────────


class TestToolResult:

    def test_result_is_immutable(self):
        r = ToolResult(success=True, output="ok")
        with pytest.raises(AttributeError):
            r.success = False

    def test_result_error_default_none(self):
        r = ToolResult(success=True, output="ok")
        assert r.error is None

    def test_result_with_error(self):
        r = ToolResult(success=False, output="", error="boom")
        assert r.error == "boom"
        assert r.success is False


# ── ToolRegistry ───────────────────────────────────────────────


class TestToolRegistry:

    def test_register_and_get(self, registry):
        registry.register(EchoTool())
        tool = registry.get("echo")
        assert tool.name == "echo"

    def test_get_unknown_raises(self, registry):
        with pytest.raises(ToolExecutionError, match="Unknown tool"):
            registry.get("nonexistent")

    def test_list_tools_empty(self, registry):
        assert registry.list_tools() == []

    def test_list_tools_sorted(self, registry):
        registry.register(EchoTool())
        registry.register(FailTool())
        tools = registry.list_tools()
        assert [t["name"] for t in tools] == ["echo", "fail"]

    def test_list_tools_has_description(self, registry):
        registry.register(EchoTool())
        tools = registry.list_tools()
        assert tools[0]["description"] == "Echoes the message back."

    def test_execute_dispatches_correctly(self, registry):
        registry.register(EchoTool())
        result = registry.execute("echo", message="hello")
        assert result.success is True
        assert result.output == "hello"

    def test_execute_unknown_tool_raises(self, registry):
        with pytest.raises(ToolExecutionError, match="Unknown tool"):
            registry.execute("ghost")

    def test_register_overwrites_existing(self, registry):
        registry.register(EchoTool())
        registry.register(EchoTool())  # no error
        assert len(registry.list_tools()) == 1

    def test_register_rejects_non_protocol(self, registry):
        with pytest.raises(TypeError, match="BaseTool protocol"):
            registry.register("not a tool")  # type: ignore


# ── Budget Tracking ────────────────────────────────────────────


class TestBudgetTracking:

    def test_tool_execution_records_cost(self, tracked_registry):
        tracked_registry.register(EchoTool())
        tracked_registry.execute("echo", message="hi")
        tracker = tracked_registry._tracker
        assert tracker.total_tokens == 50  # cost_per_tool_call

    def test_multiple_calls_accumulate_cost(self, tracked_registry):
        tracked_registry.register(EchoTool())
        for _ in range(5):
            tracked_registry.execute("echo", message="hi")
        tracker = tracked_registry._tracker
        assert tracker.total_tokens == 250

    def test_budget_exceeded_blocks_tool(self, tracked_registry):
        tracked_registry.register(EchoTool())
        # 1000 token limit, 50 per call → 20th call hits the limit exactly
        for _ in range(19):
            tracked_registry.execute("echo", message="hi")
        with pytest.raises(BudgetExceededException):
            tracked_registry.execute("echo", message="hits limit")

    def test_no_tracker_skips_cost(self, registry):
        registry.register(EchoTool())
        # Should not raise — no tracker attached
        result = registry.execute("echo", message="hi")
        assert result.success is True


# ── Built-in Tools ─────────────────────────────────────────────


class TestReadMemoryTool:

    def test_read_existing_file(self, vfs):
        vfs.write("notes.md", "important stuff")
        tool = ReadMemoryTool(vfs)
        result = tool.execute(path="notes.md")
        assert result.success is True
        assert result.output == "important stuff"

    def test_read_nonexistent_file(self, vfs):
        tool = ReadMemoryTool(vfs)
        result = tool.execute(path="ghost.md")
        assert result.success is True  # not a failure, just empty
        assert result.output == ""
        assert "not found" in result.error.lower()

    def test_read_missing_path_arg(self, vfs):
        tool = ReadMemoryTool(vfs)
        result = tool.execute()
        assert result.success is False
        assert "path" in result.error.lower()

    def test_read_path_traversal_blocked(self, vfs):
        tool = ReadMemoryTool(vfs)
        result = tool.execute(path="../secret.txt")
        assert result.success is False
        assert "traversal" in result.error.lower()


class TestWriteMemoryTool:

    def test_write_and_verify(self, vfs):
        tool = WriteMemoryTool(vfs)
        result = tool.execute(path="LEARNINGS.md", content="# Lesson 1\nDon't repeat mistakes.")
        assert result.success is True
        assert vfs.read("LEARNINGS.md") == "# Lesson 1\nDon't repeat mistakes."

    def test_write_missing_path(self, vfs):
        tool = WriteMemoryTool(vfs)
        result = tool.execute(content="data")
        assert result.success is False
        assert "path" in result.error.lower()

    def test_write_missing_content(self, vfs):
        tool = WriteMemoryTool(vfs)
        result = tool.execute(path="file.md")
        assert result.success is False
        assert "content" in result.error.lower()

    def test_write_path_traversal_blocked(self, vfs):
        tool = WriteMemoryTool(vfs)
        result = tool.execute(path="../../escape.txt", content="pwned")
        assert result.success is False
        assert "traversal" in result.error.lower()

    def test_write_creates_nested_dirs(self, vfs):
        tool = WriteMemoryTool(vfs)
        result = tool.execute(path="deep/nested/file.md", content="deep")
        assert result.success is True
        assert vfs.read("deep/nested/file.md") == "deep"


# ── Integration: Registry + Builtins + VFS ──────────────────────


class TestRegistryWithBuiltins:

    def test_full_round_trip(self, vfs):
        registry = ToolRegistry()
        registry.register(ReadMemoryTool(vfs))
        registry.register(WriteMemoryTool(vfs))

        # Write via registry
        w = registry.execute("write_memory", path="rules.md", content="no loops")
        assert w.success is True

        # Read back via registry
        r = registry.execute("read_memory", path="rules.md")
        assert r.success is True
        assert r.output == "no loops"

    def test_list_shows_both_builtins(self, vfs):
        registry = ToolRegistry()
        registry.register(ReadMemoryTool(vfs))
        registry.register(WriteMemoryTool(vfs))
        names = [t["name"] for t in registry.list_tools()]
        assert names == ["read_memory", "write_memory"]
