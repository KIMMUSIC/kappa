"""MCP (Model Context Protocol) client bridge and tool adapter.

Connects to external MCP servers, discovers their tool catalogs,
and wraps each discovered tool as a ``BaseTool`` for seamless
``ToolRegistry`` integration.  All MCP tool calls flow through the
existing ``BudgetTracker`` pipeline — no budget bypass is possible.

Architecture::

    MCPBridge ─── MCPTransport (Protocol) ─── External MCP Server
        │
        ├── discover_tools()  →  [tool_spec, ...]
        ├── call_tool(name, args)  →  raw result
        └── register_all(registry)
                │
                └── MCPToolAdapter (BaseTool) × N  →  ToolRegistry

"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from kappa.config import MCPConfig
from kappa.exceptions import ToolExecutionError
from kappa.tools.registry import ToolResult


# ── Transport Protocol ─────────────────────────────────────────


@runtime_checkable
class MCPTransport(Protocol):
    """Injectable transport layer for MCP server communication.

    Implementations may use stdio (subprocess), SSE (HTTP), or any
    other transport that the MCP specification supports.
    """

    def connect(self) -> None:
        """Establish connection to the MCP server."""
        ...

    def send(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and return the parsed response."""
        ...

    def close(self) -> None:
        """Close the connection gracefully."""
        ...


# ── MCP Bridge ─────────────────────────────────────────────────


class MCPBridge:
    """Manages connection, discovery, and invocation for one MCP server.

    Args:
        server_name: Logical name for this server (used as namespace).
        transport: Concrete ``MCPTransport`` implementation.
        config: Optional ``MCPConfig`` overrides.
    """

    def __init__(
        self,
        server_name: str,
        transport: MCPTransport,
        config: MCPConfig | None = None,
    ) -> None:
        self._server_name = server_name
        self._transport = transport
        self._config = config or MCPConfig()
        self._connected = False
        self._request_id = 0

    # ── Lifecycle ──────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to the MCP server and perform protocol initialization."""
        self._transport.connect()
        init_resp = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "kappa", "version": "0.4.0"},
        })
        if "error" in init_resp:
            raise ToolExecutionError(
                f"MCP initialize failed for {self._server_name!r}: "
                f"{init_resp['error']}"
            )
        self._connected = True

    def close(self) -> None:
        """Close the transport connection."""
        self._transport.close()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def server_name(self) -> str:
        return self._server_name

    # ── Discovery ──────────────────────────────────────────────

    def discover_tools(self) -> list[dict[str, Any]]:
        """Fetch the tool catalog from the MCP server.

        Returns a list of tool specs, each containing at minimum
        ``name``, ``description``, and ``inputSchema``.

        Raises:
            ToolExecutionError: If discovery fails.
        """
        self._ensure_connected()
        resp = self._send_request("tools/list", {})
        if "error" in resp:
            raise ToolExecutionError(
                f"MCP tools/list failed for {self._server_name!r}: "
                f"{resp['error']}"
            )
        return resp.get("result", {}).get("tools", [])

    # ── Invocation ─────────────────────────────────────────────

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a single tool on the MCP server.

        Args:
            name: The tool name as reported by the server.
            arguments: Keyword arguments matching the tool's inputSchema.

        Returns:
            Raw result dict from the MCP server.

        Raises:
            ToolExecutionError: If the call fails or returns an error.
        """
        self._ensure_connected()
        resp = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        if "error" in resp:
            raise ToolExecutionError(
                f"MCP tools/call failed for {self._server_name!r}/{name}: "
                f"{resp['error']}"
            )
        return resp.get("result", {})

    # ── Registry integration ───────────────────────────────────

    def register_all(self, registry: "ToolRegistry") -> list[str]:
        """Discover tools and register each as a ``MCPToolAdapter``.

        Args:
            registry: The ``ToolRegistry`` to populate.

        Returns:
            List of registered tool names (namespaced).
        """
        from kappa.tools.registry import ToolRegistry as _TR  # noqa: F811

        tools = self.discover_tools()
        registered: list[str] = []
        prefix = self._config.tool_name_prefix

        for spec in tools:
            adapter = MCPToolAdapter(
                server_name=self._server_name,
                tool_name=spec["name"],
                description=spec.get("description", ""),
                input_schema=spec.get("inputSchema", {}),
                bridge=self,
                name_prefix=prefix,
            )
            registry.register(adapter)
            registered.append(adapter.name)

        return registered

    # ── Internal helpers ───────────────────────────────────────

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise ToolExecutionError(
                f"MCPBridge {self._server_name!r} is not connected. "
                f"Call connect() first."
            )

    def _send_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        return self._transport.send(request)


# ── MCP Tool Adapter ───────────────────────────────────────────


class MCPToolAdapter:
    """Wraps a single MCP tool as a ``BaseTool`` for ``ToolRegistry``.

    The adapter normalizes every MCP result into a ``ToolResult``,
    ensuring uniform error handling and budget tracking.

    Args:
        server_name: Logical MCP server name.
        tool_name: Tool name as reported by the MCP server.
        description: Human-readable description from the server.
        input_schema: JSON Schema for the tool's input parameters.
        bridge: The ``MCPBridge`` to invoke the tool through.
        name_prefix: Namespace prefix (default ``"mcp"``).
    """

    def __init__(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        bridge: MCPBridge,
        name_prefix: str = "mcp",
    ) -> None:
        self._server_name = server_name
        self._tool_name = tool_name
        self._description = description
        self._input_schema = input_schema
        self._bridge = bridge
        self._name_prefix = name_prefix

    @property
    def name(self) -> str:
        """Namespaced tool name: ``mcp:<server>:<tool>``."""
        return f"{self._name_prefix}:{self._server_name}:{self._tool_name}"

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for this tool's input parameters."""
        return self._input_schema

    def execute(self, **kwargs: Any) -> ToolResult:
        """Invoke the MCP tool and normalize the result to ``ToolResult``.

        Any exception from the bridge is caught and converted into a
        failed ``ToolResult`` — the adapter never raises on tool errors.
        """
        try:
            raw = self._bridge.call_tool(self._tool_name, kwargs)
            return self._normalize(raw)
        except ToolExecutionError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output="",
                error=f"Unexpected MCP error: {exc}",
            )

    # ── Normalization ──────────────────────────────────────────

    @staticmethod
    def _normalize(raw: dict[str, Any]) -> ToolResult:
        """Convert raw MCP result to ``ToolResult``.

        MCP tools/call results follow the shape::

            {"content": [{"type": "text", "text": "..."}], "isError": false}

        This method handles both structured and flat response formats.
        """
        is_error = raw.get("isError", False)

        # Extract text from content array
        content_parts = raw.get("content", [])
        text_parts: list[str] = []
        for part in content_parts:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)

        text = "\n".join(text_parts) if text_parts else json.dumps(raw)

        if is_error:
            return ToolResult(success=False, output="", error=text)

        return ToolResult(success=True, output=text)
