"""LangGraph-based single self-healing agent loop.

Assembles the coder → parser → linter → sandbox pipeline with
conditional edges that loop back on failure (up to a hard retry limit).

GEODE layers covered:
  Layer 1 — Atomic XML Matching (parser node)
  Layer 2 — Syntactic Broom (linter node)
  Layer 4 — Context-Aware Self-Healing (conditional edges)
  Layer 5 — Runtime Smoke Test (sandbox node)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from kappa.budget.gate import BudgetGate
from kappa.config import AgentConfig
from kappa.graph.nodes import build_messages, lint_code, parse_llm_output
from kappa.graph.state import AgentState
from kappa.sandbox.executor import SandboxExecutor


class SelfHealingGraph:
    """Single-agent self-healing loop assembled as a LangGraph state machine.

    Flow::

        coder → parser ─┬─ ok ──→ linter ─┬─ ok ──→ sandbox ─┬─ exit 0 → END (success)
                         │                  │                   │
                         └─ err ─→ coder*   └─ err ─→ coder*   └─ err ──→ coder*

        * loops back only if attempt < max_attempts, otherwise → END

    Raises:
        BudgetExceededException: propagated from BudgetGate if budget runs out.
        SandboxExecutionError:   propagated if sandbox infrastructure fails.
    """

    def __init__(
        self,
        gate: BudgetGate,
        sandbox: SandboxExecutor,
        config: AgentConfig | None = None,
    ) -> None:
        self._gate = gate
        self._sandbox = sandbox
        self._config = config or AgentConfig()
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
        """Extract <think>/<action> blocks — reject malformed output."""
        result = parse_llm_output(state["llm_output"])
        if result.error:
            return {
                "parsed_code": "",
                "error_history": state["error_history"] + [
                    f"Parse error: {result.error}"
                ],
                "status": "parse_error",
            }
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

    # ── Conditional routing ─────────────────────────────────────

    def _route_after_parse(self, state: AgentState) -> str:
        if state["status"] == "parse_error":
            if state["attempt"] >= state["max_attempts"]:
                return END
            return "coder"
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

    # ── Graph assembly ──────────────────────────────────────────

    def _build(self):
        graph = StateGraph(AgentState)

        graph.add_node("coder", self._coder_node)
        graph.add_node("parser", self._parser_node)
        graph.add_node("linter", self._linter_node)
        graph.add_node("sandbox", self._sandbox_node)

        graph.set_entry_point("coder")
        graph.add_edge("coder", "parser")
        graph.add_conditional_edges("parser", self._route_after_parse)
        graph.add_conditional_edges("linter", self._route_after_lint)
        graph.add_conditional_edges("sandbox", self._route_after_sandbox)

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
        }

    def run(self, goal: str) -> AgentState:
        """Execute the self-healing loop for the given goal.

        Returns the final AgentState after success or retry exhaustion.

        Raises:
            BudgetExceededException: If budget is exceeded mid-loop.
            SandboxExecutionError: If sandbox infrastructure fails.
        """
        return self._app.invoke(self._initial_state(goal))

    def stream(self, goal: str):
        """Yield (node_name, state_update) for each step.

        Use this to observe the graph executing node by node.
        """
        yield from self._app.stream(self._initial_state(goal))
