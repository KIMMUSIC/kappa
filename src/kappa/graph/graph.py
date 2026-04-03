"""LangGraph-based single self-healing agent loop (V2).

Assembles the coder → parser → linter → sandbox pipeline with
conditional edges that loop back on failure (up to a hard retry limit).

V2 additions:
  - Tool branching: parser routes <tool_call> to a dedicated tool node.
  - Semantic loop detection: coder checks for repetitive behaviour.
  - Memory context: VFS content injected into the system prompt.

Verification layers covered:
  Layer 1 — Atomic XML Matching (parser node)
  Layer 2 — Syntactic Broom (linter node)
  Layer 4 — Context-Aware Self-Healing (conditional edges)
  Layer 5 — Runtime Smoke Test (sandbox node)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from kappa.budget.gate import BudgetGate
from kappa.config import AgentConfig
from kappa.defense.semantic import SemanticLoopDetector
from kappa.graph.nodes import build_messages, lint_code, parse_llm_output
from kappa.graph.state import AgentState
from kappa.sandbox.executor import SandboxExecutor
from kappa.tools.registry import ToolRegistry


class SelfHealingGraph:
    """Single-agent self-healing loop assembled as a LangGraph state machine.

    Flow (V2)::

        coder → parser ─┬─ <action>    → linter ─┬─ ok  → sandbox ─┬─ exit 0 → END
                         │                        │                  │
                         ├─ <tool_call> → tool ───┤                  │
                         │                        │                  │
                         └─ parse_error ──────────┤──────────────────┤→ coder*
                                                                     │
                                                              * attempt < max

    Raises:
        BudgetExceededException: propagated from BudgetGate if budget runs out.
        SandboxExecutionError:   propagated if sandbox infrastructure fails.
        SemanticLoopException:   if semantic repetition is detected.
    """

    def __init__(
        self,
        gate: BudgetGate,
        sandbox: SandboxExecutor,
        config: AgentConfig | None = None,
        registry: ToolRegistry | None = None,
        detector: SemanticLoopDetector | None = None,
    ) -> None:
        self._gate = gate
        self._sandbox = sandbox
        self._config = config or AgentConfig()
        self._registry = registry
        self._detector = detector
        self._app = self._build()

    # ── Node implementations ────────────────────────────────────

    def _coder_node(self, state: AgentState) -> dict:
        """Invoke LLM via BudgetGate to generate code."""
        messages = build_messages(state)
        response = self._gate.call(
            messages=messages,
            model=self._config.model,
        )
        return {
            "attempt": state["attempt"] + 1,
            "llm_output": response.content,
            "status": "running",
        }

    def _parser_node(self, state: AgentState) -> dict:
        """Extract <think>/<action>/<tool_call> blocks — reject malformed output."""
        result = parse_llm_output(state["llm_output"])

        # Feed think content to semantic detector
        if self._detector and result.think:
            self._detector.record(result.think)
            self._detector.check()  # may raise SemanticLoopException

        if result.error:
            return {
                "parsed_code": "",
                "tool_calls": [],
                "error_history": state["error_history"] + [
                    f"Parse error: {result.error}"
                ],
                "status": "parse_error",
            }

        # Tool call path
        if result.tool_call is not None:
            return {
                "parsed_code": "",
                "tool_calls": state.get("tool_calls", []) + [result.tool_call],
                "status": "tool_call",
            }

        # Code path (original)
        return {
            "parsed_code": result.code,
            "status": "running",
        }

    def _linter_node(self, state: AgentState) -> dict:
        """AST syntax check — catch fatal errors before sandbox."""
        error = lint_code(state["parsed_code"])
        if error:
            return {
                "error_history": state["error_history"] + [error],
                "status": "lint_error",
            }
        return {"status": "running"}

    def _sandbox_node(self, state: AgentState) -> dict:
        """Execute code in isolated container, capture result."""
        result = self._sandbox.execute(state["parsed_code"])
        sandbox_dict = {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }
        if result.exit_code == 0:
            return {
                "sandbox_result": sandbox_dict,
                "status": "success",
            }
        return {
            "sandbox_result": sandbox_dict,
            "error_history": state["error_history"] + [
                f"Runtime error (exit_code={result.exit_code}):\n{result.stderr}"
            ],
            "status": "runtime_error",
        }

    def _tool_node(self, state: AgentState) -> dict:
        """Execute a tool via the ToolRegistry."""
        tool_calls = state.get("tool_calls", [])
        if not tool_calls or self._registry is None:
            return {
                "error_history": state["error_history"] + [
                    "Tool execution failed: no registry or no tool calls."
                ],
                "status": "tool_error",
            }

        call = tool_calls[-1]  # most recent tool call
        name = call.get("name", "")
        kwargs = call.get("kwargs", {})

        try:
            result = self._registry.execute(name, **kwargs)
        except Exception as e:
            return {
                "error_history": state["error_history"] + [
                    f"Tool error ({name}): {e}"
                ],
                "status": "tool_error",
            }

        if result.success:
            return {
                "sandbox_result": {
                    "exit_code": 0,
                    "stdout": result.output,
                    "stderr": "",
                    "timed_out": False,
                },
                "status": "success",
            }
        return {
            "error_history": state["error_history"] + [
                f"Tool error ({name}): {result.error}"
            ],
            "status": "tool_error",
        }

    # ── Conditional routing ─────────────────────────────────────

    def _route_after_parse(self, state: AgentState) -> str:
        if state["status"] == "parse_error":
            if state["attempt"] >= state["max_attempts"]:
                return END
            return "coder"
        if state["status"] == "tool_call":
            return "tool"
        return "linter"

    def _route_after_lint(self, state: AgentState) -> str:
        if state["status"] == "lint_error":
            if state["attempt"] >= state["max_attempts"]:
                return END
            return "coder"
        return "sandbox"

    def _route_after_sandbox(self, state: AgentState) -> str:
        if state["status"] == "success":
            return END
        if state["attempt"] >= state["max_attempts"]:
            return END
        return "coder"

    def _route_after_tool(self, state: AgentState) -> str:
        if state["status"] == "success":
            return END
        if state["attempt"] >= state["max_attempts"]:
            return END
        return "coder"

    # ── Graph assembly ──────────────────────────────────────────

    def _build(self):
        graph = StateGraph(AgentState)

        graph.add_node("coder", self._coder_node)
        graph.add_node("parser", self._parser_node)
        graph.add_node("linter", self._linter_node)
        graph.add_node("sandbox", self._sandbox_node)
        graph.add_node("tool", self._tool_node)

        graph.set_entry_point("coder")
        graph.add_edge("coder", "parser")
        graph.add_conditional_edges("parser", self._route_after_parse)
        graph.add_conditional_edges("linter", self._route_after_lint)
        graph.add_conditional_edges("sandbox", self._route_after_sandbox)
        graph.add_conditional_edges("tool", self._route_after_tool)

        return graph.compile()

    # ── Public API ──────────────────────────────────────────────

    def _initial_state(self, goal: str) -> AgentState:
        return {
            "goal": goal,
            "llm_output": "",
            "parsed_code": "",
            "sandbox_result": None,
            "attempt": 0,
            "max_attempts": self._config.max_self_heal_retries,
            "error_history": [],
            "status": "running",
            "tool_calls": [],
            "memory_context": "",
        }

    def run(self, goal: str, memory_context: str = "") -> AgentState:
        """Execute the self-healing loop for the given goal.

        Args:
            goal: Task description for the agent.
            memory_context: Optional VFS content to inject into the prompt.

        Returns the final AgentState after success or retry exhaustion.

        Raises:
            BudgetExceededException: If budget is exceeded mid-loop.
            SandboxExecutionError: If sandbox infrastructure fails.
            SemanticLoopException: If semantic repetition is detected.
        """
        state = self._initial_state(goal)
        if memory_context:
            state["memory_context"] = memory_context
        return self._app.invoke(state)

    def stream(self, goal: str, memory_context: str = ""):
        """Yield (node_name, state_update) for each step.

        Use this to observe the graph executing node by node.
        """
        state = self._initial_state(goal)
        if memory_context:
            state["memory_context"] = memory_context
        yield from self._app.stream(state)
