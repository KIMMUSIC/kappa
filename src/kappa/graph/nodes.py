"""Node utilities for the self-healing graph.

Pure functions for parsing LLM output, linting code, and building
prompt messages.  These are consumed by the graph assembly in graph.py.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from kappa.graph.state import AgentState

# ── XML anchor patterns (Layer 1) ───────────────────────────────

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# ── System prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a code-generation agent. You MUST respond in EXACTLY this format:

<think>
[Your step-by-step reasoning about the task]
</think>
<action>
[Python code to execute]
</action>

OR, to call a tool instead of executing code:

<think>
[Your step-by-step reasoning about which tool to use]
</think>
<tool_call>
{"name": "tool_name", "kwargs": {"key": "value"}}
</tool_call>

Rules:
- You MUST include a <think> block.
- Include EITHER <action> OR <tool_call>, NOT both.
- The <action> block MUST contain ONLY valid, executable Python code.
- If the task requires creating files (HTML, CSS, JS, etc.), write Python code \
that creates those files on disk (e.g., using open() and write()). \
Do NOT just describe what the file should contain — produce the actual file.
- The <tool_call> block MUST contain valid JSON with "name" and "kwargs".
- Do NOT include any text outside these blocks.
- Do NOT use markdown code fences inside <action>.
- Before creating or modifying a file, check if it already exists using os.path.exists().
- If modifying an existing file, read its current contents first with open(..., 'r'), \
then make targeted changes — do NOT overwrite with completely new unrelated content.
- Preserve existing functionality when updating files."""


# ── Parser (Layer 1: Atomic XML Matching) ───────────────────────


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing LLM output into structured blocks."""

    think: str
    code: str
    tool_call: dict | None = None
    error: str | None = None


def parse_llm_output(raw: str) -> ParseResult:
    """Extract <think> and <action>/<tool_call> blocks from raw LLM output.

    Routes to either the code path (<action>) or the tool path (<tool_call>).
    The two are mutually exclusive — both present is a parse error.

    Returns a ParseResult with an error message if blocks are missing or
    malformed.
    """
    think_match = THINK_PATTERN.search(raw)
    action_match = ACTION_PATTERN.search(raw)
    tool_match = TOOL_CALL_PATTERN.search(raw)

    think_text = think_match.group(1).strip() if think_match else ""

    # Both <action> and <tool_call> present → ambiguous, reject
    if action_match and tool_match:
        return ParseResult(
            think=think_text,
            code="",
            error="<action> and <tool_call> are mutually exclusive — include only one.",
        )

    # <tool_call> path
    if tool_match:
        if not think_match:
            return ParseResult(
                think="",
                code="",
                error="Missing required <think> block in LLM output.",
            )
        raw_json = tool_match.group(1).strip()
        try:
            tool_data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            return ParseResult(
                think=think_text,
                code="",
                error=f"Invalid JSON in <tool_call>: {e}",
            )
        if "name" not in tool_data:
            return ParseResult(
                think=think_text,
                code="",
                error='<tool_call> JSON must contain a "name" field.',
            )
        return ParseResult(
            think=think_text,
            code="",
            tool_call=tool_data,
        )

    # <action> path (original logic, unchanged)
    if not action_match:
        return ParseResult(
            think=think_text,
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
        think=think_text,
        code=action_match.group(1).strip(),
    )


# ── Linter (Layer 2: Syntactic Broom) ───────────────────────────


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

    If ``memory_context`` is present in the state, it is prepended to
    the system prompt so the agent can recall long-term knowledge.

    If ``workspace_path`` is set, workspace file-writing instructions
    are injected into the system prompt.
    """
    goal = state["goal"]
    error_history = state["error_history"]
    attempt = state["attempt"]
    max_attempts = state["max_attempts"]
    memory_context = state.get("memory_context", "")
    workspace_path = state.get("workspace_path", "")

    prompt = SYSTEM_PROMPT
    if workspace_path:
        prompt += (
            f"\n- All output files MUST be written under the {workspace_path} directory."
            f"\n- Example: open('{workspace_path}/output.html', 'w') instead of open('output.html', 'w')."
            f"\n- The {workspace_path} directory is persisted — files written there survive after execution."
            f"\n- Files written outside {workspace_path} will be lost when the container is destroyed."
        )
    if memory_context:
        prompt = (
            f"[Long-term Memory]\n{memory_context}\n\n"
            f"{prompt}"
        )

    if not error_history:
        content = f"{prompt}\n\nGoal: {goal}"
    else:
        last_error = error_history[-1]
        content = (
            f"{prompt}\n\n"
            f"Goal: {goal}\n\n"
            f"Your previous attempt failed with this error:\n"
            f"```\n{last_error}\n```\n\n"
            f"Fix the code and try again. "
            f"Attempt {attempt + 1}/{max_attempts}."
        )

    return [{"role": "user", "content": content}]
