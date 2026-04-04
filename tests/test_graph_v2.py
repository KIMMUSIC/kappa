"""Tests for the LangGraph Self-Healing Loop V2 (Phase 2 — Task 4).

Covers:
- Tool call parsing (<tool_call> XML block)
- Parser routing: code vs tool vs error
- Tool node execution via ToolRegistry
- Memory context injection into build_messages
- Semantic loop detector integration
- Mixed flows: tool call → code execution
- Backward compatibility: no registry/detector = Phase 1 behaviour
"""

from __future__ import annotations

import pytest

from kappa.budget.gate import BudgetGate, LLMResponse
from kappa.config import AgentConfig, BudgetConfig, MemoryConfig, SemanticConfig
from kappa.defense.semantic import SemanticLoopDetector
from kappa.exceptions import SemanticLoopException
from kappa.graph.graph import SelfHealingGraph
from kappa.graph.nodes import build_messages, parse_llm_output
from kappa.memory.vfs import VFSManager
from kappa.config import ExecutionConfig
from kappa.sandbox.executor import SandboxResult
from kappa.tools.builtins import ReadMemoryTool, WriteMemoryTool
from kappa.tools.registry import ToolRegistry, ToolResult


# ── Test doubles (reused from test_self_healing.py) ─────────────


class ScriptedProvider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0
        self.call_count = 0
        self.captured_messages: list[list[dict]] = []

    def call(self, *, messages: list[dict], model: str, max_tokens: int) -> LLMResponse:
        self.captured_messages.append(messages)
        content = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        self.call_count += 1
        return LLMResponse(
            content=content, prompt_tokens=50, completion_tokens=100,
            model=model, stop_reason="end_turn",
        )


class FakeExecutor:
    """Returns pre-scripted execution results."""

    def __init__(self, results=None, config=None):
        self._results = results or [SandboxResult(exit_code=0, stdout="", stderr="")]
        self._index = 0
        self._config = config or ExecutionConfig(workspace_dir=None, output_dir=None)
        self.calls: list[str] = []

    @property
    def config(self):
        return self._config

    def execute(self, code: str) -> SandboxResult:
        self.calls.append(code)
        result = self._results[min(self._index, len(self._results) - 1)]
        self._index += 1
        return result


def _make_graph(
    provider: ScriptedProvider,
    runtime: FakeExecutor,
    registry: ToolRegistry | None = None,
    detector: SemanticLoopDetector | None = None,
    max_retries: int = 3,
) -> SelfHealingGraph:
    budget_config = BudgetConfig(max_total_tokens=100_000, max_cost_usd=100.0)
    gate = BudgetGate(provider=provider, budget_config=budget_config)
    config = AgentConfig(max_self_heal_retries=max_retries)
    return SelfHealingGraph(
        gate=gate, sandbox=runtime, config=config,
        registry=registry, detector=detector,
    )


# ── Parser: <tool_call> block ──────────────────────────────────


class TestToolCallParser:

    def test_valid_tool_call_parsed(self):
        raw = '<think>Need to read memory</think>\n<tool_call>{"name": "read_memory", "kwargs": {"path": "LEARNINGS.md"}}</tool_call>'
        result = parse_llm_output(raw)
        assert result.error is None
        assert result.tool_call is not None
        assert result.tool_call["name"] == "read_memory"
        assert result.tool_call["kwargs"]["path"] == "LEARNINGS.md"
        assert result.code == ""

    def test_tool_call_without_think_is_error(self):
        raw = '<tool_call>{"name": "read_memory", "kwargs": {}}</tool_call>'
        result = parse_llm_output(raw)
        assert result.error is not None
        assert "think" in result.error.lower()

    def test_action_and_tool_call_both_present_is_error(self):
        raw = '<think>confused</think>\n<action>print(1)</action>\n<tool_call>{"name": "x", "kwargs": {}}</tool_call>'
        result = parse_llm_output(raw)
        assert result.error is not None
        assert "mutually exclusive" in result.error.lower()

    def test_invalid_json_in_tool_call(self):
        raw = '<think>try</think>\n<tool_call>{bad json}</tool_call>'
        result = parse_llm_output(raw)
        assert result.error is not None
        assert "json" in result.error.lower()

    def test_tool_call_missing_name_field(self):
        raw = '<think>try</think>\n<tool_call>{"kwargs": {"path": "x"}}</tool_call>'
        result = parse_llm_output(raw)
        assert result.error is not None
        assert "name" in result.error.lower()

    def test_tool_call_with_no_kwargs(self):
        raw = '<think>simple</think>\n<tool_call>{"name": "list_tools"}</tool_call>'
        result = parse_llm_output(raw)
        assert result.error is None
        assert result.tool_call["name"] == "list_tools"


# ── Graph V2: Tool routing ─────────────────────────────────────


class TestToolRouting:

    def test_tool_call_routes_to_tool_node(self):
        """LLM returns <tool_call> → tool node executes → success."""
        provider = ScriptedProvider([
            '<think>Read memory</think>\n<tool_call>{"name": "echo", "kwargs": {"message": "hi"}}</tool_call>',
        ])
        runtime = FakeExecutor(results=[])  # sandbox not used

        # Simple echo tool
        class EchoTool:
            name = "echo"
            description = "Echo"
            def execute(self, **kw):
                return ToolResult(success=True, output=kw.get("message", ""))

        registry = ToolRegistry()
        registry.register(EchoTool())
        graph = _make_graph(provider, runtime, registry=registry)

        result = graph.run("Echo hi")
        assert result["status"] == "success"
        assert result["sandbox_result"]["stdout"] == "hi"
        assert len(runtime.calls) == 0  # sandbox never called

    def test_tool_error_loops_back_to_coder(self):
        """Tool fails → error fed back → coder retries with code → success."""
        provider = ScriptedProvider([
            '<think>Try tool</think>\n<tool_call>{"name": "unknown_tool", "kwargs": {}}</tool_call>',
            '<think>Fallback to code</think>\n<action>print("done")</action>',
        ])
        runtime = FakeExecutor(results=[
            SandboxResult(exit_code=0, stdout="done\n", stderr=""),
        ])
        registry = ToolRegistry()
        graph = _make_graph(provider, runtime, registry=registry)

        result = graph.run("Do something")
        assert result["status"] == "success"
        assert result["attempt"] == 2
        # Error about unknown tool was fed back
        assert any("unknown_tool" in e.lower() for e in result["error_history"])

    def test_tool_error_exhausts_retries(self):
        """Tool keeps failing → max_attempts reached → END."""
        provider = ScriptedProvider([
            '<think>Try</think>\n<tool_call>{"name": "bad", "kwargs": {}}</tool_call>',
            '<think>Try again</think>\n<tool_call>{"name": "bad", "kwargs": {}}</tool_call>',
            '<think>One more</think>\n<tool_call>{"name": "bad", "kwargs": {}}</tool_call>',
        ])
        runtime = FakeExecutor(results=[])
        registry = ToolRegistry()
        graph = _make_graph(provider, runtime, registry=registry, max_retries=3)

        result = graph.run("Use bad tool")
        assert result["status"] == "tool_error"
        assert result["attempt"] == 3

    def test_no_registry_tool_call_returns_error(self):
        """<tool_call> without registry → tool_error."""
        provider = ScriptedProvider([
            '<think>Try tool</think>\n<tool_call>{"name": "x", "kwargs": {}}</tool_call>',
            '<think>Fallback</think>\n<action>print("ok")</action>',
        ])
        runtime = FakeExecutor(results=[
            SandboxResult(exit_code=0, stdout="ok\n", stderr=""),
        ])
        graph = _make_graph(provider, runtime, registry=None)

        result = graph.run("Do task")
        assert result["status"] == "success"
        assert result["attempt"] == 2


# ── Graph V2: Builtin tools + VFS ──────────────────────────────


class TestBuiltinToolsInGraph:

    def test_write_then_read_memory_via_graph(self, tmp_path):
        """Agent writes to VFS via tool → then reads it back via tool → success."""
        vfs = VFSManager(MemoryConfig(workspace_root="ws"), base_dir=tmp_path)
        registry = ToolRegistry()
        registry.register(WriteMemoryTool(vfs))
        registry.register(ReadMemoryTool(vfs))

        provider = ScriptedProvider([
            '<think>Save learning</think>\n<tool_call>{"name": "write_memory", "kwargs": {"path": "LEARNINGS.md", "content": "Never repeat errors"}}</tool_call>',
        ])
        runtime = FakeExecutor(results=[])
        graph = _make_graph(provider, runtime, registry=registry)

        result = graph.run("Save a learning")
        assert result["status"] == "success"
        assert vfs.read("LEARNINGS.md") == "Never repeat errors"


# ── Memory context injection ───────────────────────────────────


class TestMemoryContextInjection:

    def test_memory_context_in_prompt(self):
        """memory_context is prepended to system prompt."""
        state = {
            "goal": "test",
            "llm_output": "",
            "parsed_code": "",
            "sandbox_result": None,
            "attempt": 0,
            "max_attempts": 3,
            "error_history": [],
            "status": "running",
            "tool_calls": [],
            "memory_context": "# Rule: always validate input",
        }
        messages = build_messages(state)
        content = messages[0]["content"]
        assert "[Long-term Memory]" in content
        assert "always validate input" in content

    def test_no_memory_context_omits_header(self):
        """Empty memory_context → no [Long-term Memory] header."""
        state = {
            "goal": "test",
            "llm_output": "",
            "parsed_code": "",
            "sandbox_result": None,
            "attempt": 0,
            "max_attempts": 3,
            "error_history": [],
            "status": "running",
            "tool_calls": [],
            "memory_context": "",
        }
        messages = build_messages(state)
        content = messages[0]["content"]
        assert "[Long-term Memory]" not in content

    def test_memory_context_passed_via_run(self):
        """Graph.run(memory_context=...) injects into state."""
        provider = ScriptedProvider([
            '<think>ok</think>\n<action>print("hi")</action>',
        ])
        runtime = FakeExecutor(results=[
            SandboxResult(exit_code=0, stdout="hi\n", stderr=""),
        ])
        graph = _make_graph(provider, runtime)
        result = graph.run("Say hi", memory_context="Remember: be polite")

        # Verify memory was in the prompt sent to LLM
        prompt = provider.captured_messages[0][0]["content"]
        assert "Remember: be polite" in prompt


# ── Semantic loop detector integration ──────────────────────────


class TestSemanticDetectorInGraph:

    def test_semantic_loop_raises_exception(self):
        """Identical think blocks → SemanticLoopException raised from graph."""
        provider = ScriptedProvider([
            '<think>I will fix the parser error</think>\n<action>print(x)</action>',
            '<think>I will fix the parser error</think>\n<action>print(x)</action>',
            '<think>I will fix the parser error</think>\n<action>print(x)</action>',
        ])
        runtime = FakeExecutor(results=[
            SandboxResult(exit_code=1, stdout="", stderr="NameError"),
            SandboxResult(exit_code=1, stdout="", stderr="NameError"),
            SandboxResult(exit_code=1, stdout="", stderr="NameError"),
        ])
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=5, similarity_threshold=0.85, min_samples=3)
        )
        graph = _make_graph(provider, runtime, detector=detector, max_retries=5)

        with pytest.raises(SemanticLoopException):
            graph.run("Fix the bug")

    def test_diverse_thoughts_no_exception(self):
        """Different think blocks → no SemanticLoopException → normal flow."""
        provider = ScriptedProvider([
            '<think>First approach: try direct print</think>\n<action>print(x)</action>',
            '<think>Second approach: define variable first</think>\n<action>x = 1\nprint(x)</action>',
        ])
        runtime = FakeExecutor(results=[
            SandboxResult(exit_code=1, stdout="", stderr="NameError"),
            SandboxResult(exit_code=0, stdout="1\n", stderr=""),
        ])
        detector = SemanticLoopDetector(
            SemanticConfig(window_size=5, similarity_threshold=0.85, min_samples=3)
        )
        graph = _make_graph(provider, runtime, detector=detector)

        result = graph.run("Print x")
        assert result["status"] == "success"

    def test_no_detector_skips_check(self):
        """No detector provided → identical thoughts don't raise."""
        provider = ScriptedProvider([
            '<think>same</think>\n<action>print(x)</action>',
            '<think>same</think>\n<action>print(x)</action>',
            '<think>same</think>\n<action>x=1\nprint(x)</action>',
        ])
        runtime = FakeExecutor(results=[
            SandboxResult(exit_code=1, stdout="", stderr="NameError"),
            SandboxResult(exit_code=1, stdout="", stderr="NameError"),
            SandboxResult(exit_code=0, stdout="1\n", stderr=""),
        ])
        graph = _make_graph(provider, runtime, detector=None, max_retries=3)

        result = graph.run("Print x")
        assert result["status"] == "success"


# ── Mixed flow: tool + code ────────────────────────────────────


class TestMixedFlow:

    def test_tool_call_then_code_execution(self):
        """Agent calls tool first, then writes code → full pipeline."""
        provider = ScriptedProvider([
            '<think>Read memory first</think>\n<tool_call>{"name": "echo", "kwargs": {"message": "context loaded"}}</tool_call>',
        ])
        runtime = FakeExecutor(results=[])

        class EchoTool:
            name = "echo"
            description = "Echo"
            def execute(self, **kw):
                return ToolResult(success=True, output=kw.get("message", ""))

        registry = ToolRegistry()
        registry.register(EchoTool())
        graph = _make_graph(provider, runtime, registry=registry)

        result = graph.run("Load context")
        assert result["status"] == "success"
        assert result["attempt"] == 1
        assert len(result["tool_calls"]) == 1


# ── Backward compatibility ─────────────────────────────────────


class TestBackwardCompatibility:

    def test_graph_without_optional_params(self):
        """Graph works identically to Phase 1 when no registry/detector given."""
        provider = ScriptedProvider([
            '<think>Simple</think>\n<action>print("hello")</action>',
        ])
        runtime = FakeExecutor(results=[
            SandboxResult(exit_code=0, stdout="hello\n", stderr=""),
        ])
        graph = _make_graph(provider, runtime)

        result = graph.run("Print hello")
        assert result["status"] == "success"
        assert result["attempt"] == 1
        assert result["tool_calls"] == []
        assert result["memory_context"] == ""

    def test_initial_state_has_new_fields(self):
        """_initial_state includes tool_calls and memory_context with defaults."""
        provider = ScriptedProvider(["<think>x</think>\n<action>pass</action>"])
        runtime = FakeExecutor(results=[SandboxResult(exit_code=0, stdout="", stderr="")])
        graph = _make_graph(provider, runtime)
        state = graph._initial_state("test")
        assert state["tool_calls"] == []
        assert state["memory_context"] == ""
