"""State definition for the self-healing agent loop."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """Typed state flowing through the LangGraph self-healing state machine.

    Fields:
        goal:           Original task description from the user.
        llm_output:     Raw LLM response text (may contain XML anchors).
        parsed_code:    Code extracted from the <action> block.
        sandbox_result: Structured output from sandbox execution, or None.
        attempt:        Current attempt number (starts at 0, incremented by coder).
        max_attempts:   Hard limit — graph stops after this many coder invocations.
        error_history:  Accumulated error messages across all retries.
        status:         Current loop status — drives conditional routing.
    """

    goal: str
    llm_output: str
    parsed_code: str
    sandbox_result: dict | None  # {exit_code, stdout, stderr, timed_out}
    attempt: int
    max_attempts: int
    error_history: list[str]
    status: str  # running | success | parse_error | lint_error | runtime_error
