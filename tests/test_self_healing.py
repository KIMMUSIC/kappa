"""Tests for the LangGraph Self-Healing Loop (Phase 1 — Task 3).

Verification scenarios:
1. Parser: valid XML → extracts think/code; missing blocks → error.
2. Linter: valid syntax → None; broken syntax → error string.
3. Happy path: correct code on first try → success in 1 attempt.
4. Self-heal from runtime error: sandbox fails → stderr fed back → LLM fixes.
5. Self-heal from syntax error: linter catches → LLM fixes (sandbox skipped).
6. Parse error recovery: missing <action> → retry with format instructions.
7. Multi-stage healing: syntax error → fix → runtime error → fix → success.
8. Max retries exhausted: keeps failing → stops at hard limit.
9. Budget exceeded: BudgetExceededException propagates out of the graph.
"""

from __future__ import annotations

import pytest

from kappa.budget.gate import BudgetGate, LLMResponse
from kappa.config import AgentConfig, BudgetConfig
from kappa.exceptions import BudgetExceededException
from kappa.graph.graph import SelfHealingGraph
from kappa.graph.nodes import lint_code, parse_llm_output
from kappa.sandbox.executor import SandboxExecutor, SandboxResult


# ── Test doubles ────────────────────────────────────────────────


class ScriptedProvider:
    """Returns pre-scripted LLM responses in sequence."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0
        self.call_count = 0
        self.captured_messages: list[list[dict]] = []

    def call(
        self, *, messages: list[dict], model: str, max_tokens: int
    ) -> LLMResponse:
        self.captured_messages.append(messages)
        content = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        self.call_count += 1
        return LLMResponse(
            content=content,
            prompt_tokens=50,
            completion_tokens=100,
            model=model,
            stop_reason="end_turn",
        )


class ScriptedRuntime:
    """Returns pre-scripted sandbox results in sequence."""

    def __init__(self, results: list[SandboxResult]) -> None:
        self._results = results
        self._index = 0
        self.calls: list[str] = []

    def run(
        self,
        *,
        image: str,
        command: list[str],
        mem_limit: str,
        network_disabled: bool,
        timeout: int,
        volumes: dict | None = None,
    ) -> SandboxResult:
        self.calls.append(command[-1] if command else "")
        result = self._results[min(self._index, len(self._results) - 1)]
        self._index += 1
        return result


def _make_components(
    provider: ScriptedProvider,
    runtime: ScriptedRuntime,
    max_tokens: int = 100_000,
) -> tuple[BudgetGate, SandboxExecutor]:
    budget_config = BudgetConfig(max_total_tokens=max_tokens, max_cost_usd=100.0)
    gate = BudgetGate(provider=provider, budget_config=budget_config)
    sandbox = SandboxExecutor(runtime=runtime)
    return gate, sandbox


# ── Parser unit tests (Layer 1) ────────────────────────────────


class TestParser:

    def test_valid_output_extracts_both_blocks(self):
        result = parse_llm_output(
            "<think>reasoning here</think>\n<action>print('hi')</action>"
        )
        assert result.think == "reasoning here"
        assert result.code == "print('hi')"
        assert result.error is None

    def test_missing_action_block_returns_error(self):
        result = parse_llm_output("<think>reasoning</think>\nprint('hi')")
        assert result.error is not None
        assert "action" in result.error.lower()

    def test_missing_think_block_returns_error(self):
        result = parse_llm_output("<action>print('hi')</action>")
        assert result.error is not None
        assert "think" in result.error.lower()

    def test_both_missing_returns_error(self):
        result = parse_llm_output("Just some plain text")
        assert result.error is not None

    def test_multiline_code_extracted(self):
        raw = (
            "<think>multi-step</think>\n"
            "<action>\nx = 1\ny = 2\nprint(x + y)\n</action>"
        )
        result = parse_llm_output(raw)
        assert "x = 1" in result.code
        assert "print(x + y)" in result.code
        assert result.error is None


# ── Linter unit tests (Layer 2) ────────────────────────────────


class TestLinter:

    def test_valid_code_returns_none(self):
        assert lint_code("print('hello')") is None

    def test_multiline_valid_code(self):
        assert lint_code("def f():\n  return 1\nprint(f())") is None

    def test_syntax_error_returns_message(self):
        error = lint_code("def f(\n  return")
        assert error is not None
        assert "SyntaxError" in error

    def test_empty_string_is_valid(self):
        assert lint_code("") is None


# ── SelfHealingGraph integration tests ──────────────────────────


class TestSelfHealingGraph:

    def test_success_on_first_attempt(self):
        """Correct code on first try → success, no retries."""
        provider = ScriptedProvider([
            "<think>Simple print</think>\n<action>print('hello')</action>",
        ])
        runtime = ScriptedRuntime([
            SandboxResult(exit_code=0, stdout="hello\n", stderr=""),
        ])
        gate, sandbox = _make_components(provider, runtime)
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox)

        result = graph.run("Print hello")

        assert result["status"] == "success"
        assert result["attempt"] == 1
        assert result["sandbox_result"]["exit_code"] == 0
        assert "hello" in result["sandbox_result"]["stdout"]
        assert provider.call_count == 1
        assert len(runtime.calls) == 1

    def test_self_heal_from_runtime_error(self):
        """Runtime error → stderr fed back to LLM → LLM fixes → success."""
        provider = ScriptedProvider([
            "<think>Print x</think>\n<action>print(x)</action>",
            "<think>Define x first</think>\n<action>x = 42\nprint(x)</action>",
        ])
        runtime = ScriptedRuntime([
            SandboxResult(exit_code=1, stdout="", stderr="NameError: name 'x' is not defined"),
            SandboxResult(exit_code=0, stdout="42\n", stderr=""),
        ])
        gate, sandbox = _make_components(provider, runtime)
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox)

        result = graph.run("Print the value of x")

        assert result["status"] == "success"
        assert result["attempt"] == 2
        assert provider.call_count == 2
        assert len(runtime.calls) == 2
        # Verify error was fed back to LLM on second call
        second_prompt = provider.captured_messages[1][0]["content"]
        assert "NameError" in second_prompt

    def test_self_heal_from_syntax_error(self):
        """Linter catches syntax error → sandbox skipped → LLM fixes → success."""
        provider = ScriptedProvider([
            "<think>Define function</think>\n<action>def f(\n  return 1</action>",
            "<think>Fix syntax</think>\n<action>def f():\n  return 1\nprint(f())</action>",
        ])
        runtime = ScriptedRuntime([
            # Only one sandbox call — first attempt never reaches sandbox
            SandboxResult(exit_code=0, stdout="1\n", stderr=""),
        ])
        gate, sandbox = _make_components(provider, runtime)
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox)

        result = graph.run("Create a function that returns 1")

        assert result["status"] == "success"
        assert result["attempt"] == 2
        # Sandbox was called only once (syntax error skipped it)
        assert len(runtime.calls) == 1
        # Error feedback contains SyntaxError
        second_prompt = provider.captured_messages[1][0]["content"]
        assert "SyntaxError" in second_prompt

    def test_parse_error_recovery(self):
        """LLM output missing XML tags → parse error → retry → success."""
        provider = ScriptedProvider([
            "Here's the code: print('hello')",  # No XML anchors
            "<think>Use proper format</think>\n<action>print('hello')</action>",
        ])
        runtime = ScriptedRuntime([
            SandboxResult(exit_code=0, stdout="hello\n", stderr=""),
        ])
        gate, sandbox = _make_components(provider, runtime)
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox)

        result = graph.run("Print hello")

        assert result["status"] == "success"
        assert result["attempt"] == 2
        # Parse error was fed back
        second_prompt = provider.captured_messages[1][0]["content"]
        assert "Parse error" in second_prompt

    def test_combined_syntax_then_runtime_healing(self):
        """Syntax error → fix → runtime error → fix → success (multi-stage)."""
        provider = ScriptedProvider([
            "<think>Try</think>\n<action>def f(\n  return</action>",
            "<think>Fix syntax</think>\n<action>print(undefined)</action>",
            "<think>Fix everything</think>\n<action>print('done')</action>",
        ])
        runtime = ScriptedRuntime([
            # Call 1: attempt 2 code (attempt 1 caught by linter)
            SandboxResult(exit_code=1, stdout="", stderr="NameError: name 'undefined'"),
            # Call 2: attempt 3 code
            SandboxResult(exit_code=0, stdout="done\n", stderr=""),
        ])
        config = AgentConfig(max_self_heal_retries=3)
        gate, sandbox = _make_components(provider, runtime)
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox, config=config)

        result = graph.run("Print done")

        assert result["status"] == "success"
        assert result["attempt"] == 3
        assert len(runtime.calls) == 2  # linter skipped sandbox on attempt 1
        assert len(result["error_history"]) == 2  # syntax + runtime errors

    def test_max_retries_exhausted(self):
        """LLM keeps generating bad code → stops at max_attempts."""
        provider = ScriptedProvider([
            "<think>Try</think>\n<action>print(x)</action>",
            "<think>Try again</think>\n<action>print(y)</action>",
            "<think>Try once more</think>\n<action>print(z)</action>",
        ])
        runtime = ScriptedRuntime([
            SandboxResult(exit_code=1, stdout="", stderr="NameError: name 'x'"),
            SandboxResult(exit_code=1, stdout="", stderr="NameError: name 'y'"),
            SandboxResult(exit_code=1, stdout="", stderr="NameError: name 'z'"),
        ])
        config = AgentConfig(max_self_heal_retries=3)
        gate, sandbox = _make_components(provider, runtime)
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox, config=config)

        result = graph.run("Print something")

        assert result["status"] == "runtime_error"
        assert result["attempt"] == 3
        assert len(result["error_history"]) == 3

    def test_budget_exceeded_propagates(self):
        """BudgetExceededException raised during coder → propagates out."""
        provider = ScriptedProvider([
            "<think>ok</think>\n<action>print('hi')</action>",
        ])
        runtime = ScriptedRuntime([
            SandboxResult(exit_code=0, stdout="hi\n", stderr=""),
        ])
        # Tiny budget — first LLM call (150 tokens) exceeds limit of 10
        budget_config = BudgetConfig(max_total_tokens=10, max_cost_usd=100.0)
        gate = BudgetGate(provider=provider, budget_config=budget_config)
        sandbox = SandboxExecutor(runtime=runtime)
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox)

        with pytest.raises(BudgetExceededException):
            graph.run("Print hi")

    def test_error_history_accumulates_across_retries(self):
        """Each failure appends to error_history — full trace preserved."""
        provider = ScriptedProvider([
            "plain text, no XML",  # parse error
            "<think>fix</think>\n<action>def f(:</action>",  # syntax error
            "<think>fix</think>\n<action>print(z)</action>",  # runtime error
        ])
        runtime = ScriptedRuntime([
            SandboxResult(exit_code=1, stdout="", stderr="NameError: z"),
        ])
        config = AgentConfig(max_self_heal_retries=3)
        gate, sandbox = _make_components(provider, runtime)
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox, config=config)

        result = graph.run("Do something")

        assert result["attempt"] == 3
        assert len(result["error_history"]) == 3
        assert "Parse error" in result["error_history"][0]
        assert "SyntaxError" in result["error_history"][1]
        assert "NameError" in result["error_history"][2]
