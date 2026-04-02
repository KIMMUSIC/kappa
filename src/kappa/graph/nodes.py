"""Node utilities for the self-healing graph.

Pure functions for parsing LLM output, linting code, and building
prompt messages.  These are consumed by the graph assembly in graph.py.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from kappa.graph.state import AgentState

# ── XML anchor patterns (GEODE Layer 1) ─────────────────────────

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)

# ── System prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a code-generation agent. You MUST respond in EXACTLY this format:

<think>
[Your step-by-step reasoning about the task]
</think>
<action>
[Python code to execute]
</action>

Rules:
- You MUST include both <think> and <action> blocks.
- The <action> block MUST contain ONLY valid, executable Python code.
- Do NOT include any text outside these blocks.
- Do NOT use markdown code fences inside <action>."""


# ── Parser (GEODE Layer 1: Atomic XML Matching) ─────────────────


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing LLM output into structured blocks."""

    think: str
    code: str
    error: str | None = None


def parse_llm_output(raw: str) -> ParseResult:
    """Extract <think> and <action> blocks from raw LLM output.

    Returns a ParseResult with an error message if either block is missing.
    """
    think_match = THINK_PATTERN.search(raw)
    action_match = ACTION_PATTERN.search(raw)

    if not action_match:
        return ParseResult(
            think=think_match.group(1).strip() if think_match else "",
            code="",
            error="Missing required <action> block in LLM output.",
        )
    if not think_match:
        return ParseResult(
            think="",
            code=action_match.group(1).strip(),
            error="Missing required <think> block in LLM output.",
        )

    return ParseResult(
        think=think_match.group(1).strip(),
        code=action_match.group(1).strip(),
    )


# ── Linter (GEODE Layer 2: Syntactic Broom) ─────────────────────


def lint_code(code: str) -> str | None:
    """Return None if code is syntactically valid, else an error string."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"


# ── Message builder ──────────────────────────────────────────────


def build_messages(state: AgentState) -> list[dict]:
    """Build the LLM message list from current graph state.

    First attempt gets a clean prompt.  Subsequent attempts include the
    most recent error so the LLM can self-correct.
    """
    goal = state["goal"]
    error_history = state["error_history"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]

    if not error_history:
        content = f"{SYSTEM_PROMPT}\n\nGoal: {goal}"
    else:
        last_error = error_history[-1]
        content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Your previous attempt failed with this error:\n"
            f"```\n{last_error}\n```\n\n"
            f"Fix the code and try again. "
            f"Attempt {attempt + 1}/{max_attempts}."
        )

    return [{"role": "user", "content": content}]
