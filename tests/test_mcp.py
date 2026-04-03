"""Tests for MCP Bridge and Tool Adapter (Phase 4, Task 1).

Validates:
  - MCPTransport protocol compliance
  - MCPBridge connection, discovery, and invocation
  - MCPToolAdapter normalization to ToolResult
  - ToolRegistry integration (register_all)
  - BudgetTracker cost accounting (no budget bypass)
  - Error handling and edge cases
"""

from __future__ import annotations

import pytest

from kappa.budget.tracker import BudgetTracker
from kappa.config import BudgetConfig, MCPConfig
from kappa.exceptions import ToolExecutionError
from kappa.tools.mcp import MCPBridge, MCPToolAdapter, MCPTransport
from kappa.tools.registry import ToolRegistry, ToolResult


# ── Fake MCP Transport ─────────────────────────────────────────


class FakeMCPTransport:
    """In-process fake MCP server for deterministic testing."""

    def __init__(
        self,
        tools: list[dict] | None = None,
        call_results: dict[str, dict] | None = None,
        fail_on_connect: bool = False,
        fail_on_method: str | None = None,
    ) -> None:
        self._tools = tools or []
        self._call_results = call_results or {}
        self._fail_on_connect = fail_on_connect
        self._fail_on_method = fail_on_method
        self._connected = False
        self._requests: list[dict] = []

    def connect(self) -> None:
        if self._fail_on_connect:
            raise ConnectionError("Fake connection failure")
        self._connected = True

    def send(self, request: dict) -> dict:
        self._requests.append(request)
        method = request.get("method", "")
        req_id = request.get("id", 0)

        if method == self._fail_on_method:
            return {"jsonrpc": "2.0", "id": req_id, "error": f"{method} failed"}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-server", "version": "1.0"},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self._tools},
            }

        if method == "tools/call":
            tool_name = request["params"]["name"]
            if tool_name in self._call_results:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": self._call_results[tool_name],
                }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": f"Unknown tool: {tool_name}",
            }

        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    def close(self) -> None:
        self._connected = False


# ── Fixtures ───────────────────────────────────────────────────


SAMPLE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the filesystem",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    },
]

SAMPLE_CALL_RESULTS = {
    "read_file": {
        "content": [{"type": "text", "text": "Hello, World!"}],
        "isError": False,
    },
    "write_file": {
        "content": [{"type": "text", "text": "File written successfully."}],
        "isError": False,
    },
    "search_web": {
        "content": [
            {"type": "text", "text": "Result 1: Kappa project"},
            {"type": "text", "text": "Result 2: LangGraph docs"},
        ],
        "isError": False,
    },
}


@pytest.fixture
def fake_transport():
    return FakeMCPTransport(tools=SAMPLE_TOOLS, call_results=SAMPLE_CALL_RESULTS)


@pytest.fixture
def bridge(fake_transport):
    b = MCPBridge("test-server", fake_transport)
    b.connect()
    return b


@pytest.fixture
def budget_tracker():
    config = BudgetConfig(max_total_tokens=10_000, max_cost_usd=1.0)
    return BudgetTracker(config)


@pytest.fixture
def registry(budget_tracker):
    return ToolRegistry(tracker=budget_tracker, cost_per_tool_call=50)


# ── Transport Protocol Tests ──────────────────────────────────


class TestMCPTransportProtocol:
    """Verify FakeMCPTransport satisfies MCPTransport protocol."""

    def test_fake_transport_is_mcp_transport(self, fake_transport):
        assert isinstance(fake_transport, MCPTransport)

    def test_connect_and_close(self, fake_transport):
        fake_transport.connect()
        assert fake_transport._connected is True
        fake_transport.close()
        assert fake_transport._connected is False


# ── MCPBridge Tests ────────────────────────────────────────────


class TestMCPBridge:
    """MCPBridge lifecycle, discovery, and invocation."""

    def test_connect_initializes_protocol(self, fake_transport):
        bridge = MCPBridge("srv", fake_transport)
        assert bridge.connected is False

        bridge.connect()
        assert bridge.connected is True
        assert bridge.server_name == "srv"

        # Verify initialize request was sent
        assert fake_transport._requests[0]["method"] == "initialize"

    def test_connect_failure_raises(self):
        transport = FakeMCPTransport(fail_on_method="initialize")
        bridge = MCPBridge("fail-srv", transport)
        with pytest.raises(ToolExecutionError, match="MCP initialize failed"):
            bridge.connect()

    def test_connect_transport_error(self):
        transport = FakeMCPTransport(fail_on_connect=True)
        bridge = MCPBridge("err-srv", transport)
        with pytest.raises(ConnectionError):
            bridge.connect()

    def test_close(self, bridge, fake_transport):
        assert bridge.connected is True
        bridge.close()
        assert bridge.connected is False

    def test_discover_tools(self, bridge):
        tools = bridge.discover_tools()
        assert len(tools) == 3
        names = [t["name"] for t in tools]
        assert "read_file" in names
        assert "write_file" in names
        assert "search_web" in names

    def test_discover_tools_with_schema(self, bridge):
        tools = bridge.discover_tools()
        read_tool = next(t for t in tools if t["name"] == "read_file")
        assert "inputSchema" in read_tool
        assert read_tool["inputSchema"]["required"] == ["path"]

    def test_discover_tools_not_connected_raises(self, fake_transport):
        bridge = MCPBridge("no-conn", fake_transport)
        with pytest.raises(ToolExecutionError, match="not connected"):
            bridge.discover_tools()

    def test_discover_tools_failure(self):
        transport = FakeMCPTransport(fail_on_method="tools/list")
        bridge = MCPBridge("srv", transport)
        bridge.connect()
        with pytest.raises(ToolExecutionError, match="tools/list failed"):
            bridge.discover_tools()

    def test_call_tool_success(self, bridge):
        result = bridge.call_tool("read_file", {"path": "/tmp/test.txt"})
        assert result["content"][0]["text"] == "Hello, World!"

    def test_call_tool_unknown_tool(self, bridge):
        with pytest.raises(ToolExecutionError, match="tools/call failed"):
            bridge.call_tool("nonexistent", {})

    def test_call_tool_not_connected_raises(self, fake_transport):
        bridge = MCPBridge("no-conn", fake_transport)
        with pytest.raises(ToolExecutionError, match="not connected"):
            bridge.call_tool("read_file", {})

    def test_discover_empty_catalog(self):
        transport = FakeMCPTransport(tools=[])
        bridge = MCPBridge("empty", transport)
        bridge.connect()
        tools = bridge.discover_tools()
        assert tools == []

    def test_request_ids_increment(self, bridge):
        bridge.discover_tools()
        bridge.discover_tools()
        # initialize=1, discover=2, discover=3
        ids = [r["id"] for r in bridge._transport._requests]
        assert ids == [1, 2, 3]


# ── MCPToolAdapter Tests ──────────────────────────────────────


class TestMCPToolAdapter:
    """MCPToolAdapter normalization and BaseTool compliance."""

    def test_name_is_namespaced(self, bridge):
        adapter = MCPToolAdapter(
            server_name="github",
            tool_name="list_repos",
            description="List repos",
            input_schema={},
            bridge=bridge,
        )
        assert adapter.name == "mcp:github:list_repos"

    def test_custom_prefix(self, bridge):
        adapter = MCPToolAdapter(
            server_name="db",
            tool_name="query",
            description="Run SQL",
            input_schema={},
            bridge=bridge,
            name_prefix="ext",
        )
        assert adapter.name == "ext:db:query"

    def test_description_passthrough(self, bridge):
        adapter = MCPToolAdapter(
            server_name="s",
            tool_name="t",
            description="A detailed description",
            input_schema={},
            bridge=bridge,
        )
        assert adapter.description == "A detailed description"

    def test_input_schema_passthrough(self, bridge):
        schema = {"type": "object", "properties": {"x": {"type": "int"}}}
        adapter = MCPToolAdapter(
            server_name="s",
            tool_name="t",
            description="",
            input_schema=schema,
            bridge=bridge,
        )
        assert adapter.input_schema == schema

    def test_execute_success_single_content(self, bridge):
        adapter = MCPToolAdapter(
            server_name="test-server",
            tool_name="read_file",
            description="Read file",
            input_schema={},
            bridge=bridge,
        )
        result = adapter.execute(path="/tmp/test.txt")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "Hello, World!"
        assert result.error is None

    def test_execute_success_multi_content(self, bridge):
        adapter = MCPToolAdapter(
            server_name="test-server",
            tool_name="search_web",
            description="Search",
            input_schema={},
            bridge=bridge,
        )
        result = adapter.execute(query="kappa")
        assert result.success is True
        assert "Result 1" in result.output
        assert "Result 2" in result.output

    def test_execute_mcp_error(self, bridge):
        """Tool call to unknown tool returns failed ToolResult (no exception)."""
        adapter = MCPToolAdapter(
            server_name="test-server",
            tool_name="nonexistent",
            description="",
            input_schema={},
            bridge=bridge,
        )
        result = adapter.execute()
        assert result.success is False
        assert result.error is not None
        assert "tools/call failed" in result.error

    def test_execute_is_error_flag(self):
        """MCP result with isError=True produces failed ToolResult."""
        transport = FakeMCPTransport(
            tools=[{"name": "bad_tool", "description": "", "inputSchema": {}}],
            call_results={
                "bad_tool": {
                    "content": [{"type": "text", "text": "Permission denied"}],
                    "isError": True,
                }
            },
        )
        bridge = MCPBridge("srv", transport)
        bridge.connect()

        adapter = MCPToolAdapter(
            server_name="srv",
            tool_name="bad_tool",
            description="",
            input_schema={},
            bridge=bridge,
        )
        result = adapter.execute()
        assert result.success is False
        assert "Permission denied" in result.error

    def test_normalize_flat_result(self, bridge):
        """Normalize a result with no content array."""
        raw = {"data": "flat value"}
        result = MCPToolAdapter._normalize(raw)
        assert result.success is True
        # Falls back to json.dumps
        assert "flat value" in result.output

    def test_normalize_string_content(self, bridge):
        """Content list with raw strings instead of dicts."""
        raw = {"content": ["plain text", "more text"], "isError": False}
        result = MCPToolAdapter._normalize(raw)
        assert result.success is True
        assert "plain text" in result.output
        assert "more text" in result.output


# ── ToolRegistry Integration ──────────────────────────────────


class TestMCPRegistryIntegration:
    """End-to-end: discover → register → execute through ToolRegistry."""

    def test_register_all(self, bridge, registry):
        registered = bridge.register_all(registry)
        assert len(registered) == 3
        assert "mcp:test-server:read_file" in registered
        assert "mcp:test-server:write_file" in registered
        assert "mcp:test-server:search_web" in registered

    def test_registered_tools_appear_in_list(self, bridge, registry):
        bridge.register_all(registry)
        tool_names = [t["name"] for t in registry.list_tools()]
        assert "mcp:test-server:read_file" in tool_names

    def test_execute_through_registry(self, bridge, registry):
        bridge.register_all(registry)
        result = registry.execute("mcp:test-server:read_file", path="/tmp/test.txt")
        assert result.success is True
        assert result.output == "Hello, World!"

    def test_register_all_empty_catalog(self, registry):
        transport = FakeMCPTransport(tools=[])
        bridge = MCPBridge("empty", transport)
        bridge.connect()
        registered = bridge.register_all(registry)
        assert registered == []

    def test_register_all_custom_prefix(self, registry, fake_transport):
        config = MCPConfig(tool_name_prefix="ext")
        bridge = MCPBridge("srv", fake_transport, config=config)
        bridge.connect()
        registered = bridge.register_all(registry)
        assert all(name.startswith("ext:") for name in registered)


# ── Budget Tracking Tests ─────────────────────────────────────


class TestMCPBudgetTracking:
    """Verify MCP tools are properly tracked by BudgetTracker."""

    def test_mcp_tool_execution_costs_budget(self, bridge, registry, budget_tracker):
        bridge.register_all(registry)

        initial_tokens = budget_tracker.total_tokens
        registry.execute("mcp:test-server:read_file", path="/tmp/x")
        assert budget_tracker.total_tokens == initial_tokens + 50

    def test_multiple_mcp_calls_accumulate_cost(self, bridge, registry, budget_tracker):
        bridge.register_all(registry)

        registry.execute("mcp:test-server:read_file", path="/a")
        registry.execute("mcp:test-server:write_file", path="/b", content="c")
        registry.execute("mcp:test-server:search_web", query="test")

        assert budget_tracker.total_tokens == 150  # 3 × 50

    def test_budget_exceeded_blocks_mcp_tool(self, fake_transport):
        """MCP tools cannot bypass budget — BudgetExceededException is raised."""
        config = BudgetConfig(max_total_tokens=60, max_cost_usd=10.0)
        tracker = BudgetTracker(config)
        reg = ToolRegistry(tracker=tracker, cost_per_tool_call=50)

        bridge = MCPBridge("srv", fake_transport)
        bridge.connect()
        bridge.register_all(reg)

        # First call: 50 tokens — OK
        result = reg.execute("mcp:srv:read_file", path="/a")
        assert result.success is True

        # Second call: 100 tokens total — exceeds budget of 60
        from kappa.exceptions import BudgetExceededException

        with pytest.raises(BudgetExceededException):
            reg.execute("mcp:srv:read_file", path="/b")


# ── MCPConfig Tests ───────────────────────────────────────────


class TestMCPConfig:
    """MCPConfig dataclass."""

    def test_defaults(self):
        config = MCPConfig()
        assert config.request_timeout == 30.0
        assert config.tool_name_prefix == "mcp"
        assert config.max_retries == 2

    def test_custom_values(self):
        config = MCPConfig(request_timeout=10.0, tool_name_prefix="ext", max_retries=5)
        assert config.request_timeout == 10.0
        assert config.tool_name_prefix == "ext"
        assert config.max_retries == 5

    def test_frozen(self):
        config = MCPConfig()
        with pytest.raises(AttributeError):
            config.request_timeout = 999
