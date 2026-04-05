"""Meta-Prompter: Gap analysis, prompt structurization, and strategy selection.

Analyzes user input to identify missing information, restructure the prompt
into a clear engineering specification, and select the optimal prompting
strategy (CoT, ReAct, Direct) for downstream workers.

Output is a JSON structure consumed by the orchestrator to decide whether
to activate the Interview node (high ambiguity) or proceed directly to
the Planner (low ambiguity).
"""

from __future__ import annotations

import json
import re
from typing import TypedDict


# ── Prompt ─────────────────────────────────────────────────────────

META_PROMPTER_PROMPT = """\
You are a meta-prompt engineer. Analyze the user's goal and produce an \
enhanced, structured version optimized for a code-generation agent.

Perform three tasks:

1. RESTRUCTURE: Reorganize the goal into clear sections:
   - Goal: What the user wants to achieve (one sentence)
   - Constraints: Any limitations, technologies, or requirements mentioned
   - Role: What role the agent should assume (e.g., Python developer)
   - Output Format: Expected deliverables (files, scripts, apps)

2. STRATEGIZE: Select the optimal prompting strategy:
   - "CoT" (Chain of Thought): For complex reasoning, multi-step logic, \
algorithm design
   - "ReAct" (Reasoning + Acting): For tasks requiring tool use, file I/O, \
API calls, or iterative exploration
   - "direct": For simple, well-defined tasks with clear output

3. GAP ANALYSIS: Identify missing information that would cause the agent \
to guess or hallucinate. Score ambiguity from 0.0 (crystal clear, all \
details specified) to 1.0 (completely vague, almost nothing specified). \
List each gap as a question that could be asked to the user.

User's goal: {goal}
{workspace_context}
Output ONLY valid JSON (no markdown, no commentary):
{{"enhanced_goal": "Restructured goal with all sections...", \
"ambiguity_score": 0.7, \
"gaps": ["What data source will be used?", "What UI framework?"], \
"strategy": "CoT"}}"""


# ── Result Type ────────────────────────────────────────────────────


class MetaPromptResult(TypedDict):
    """Parsed output from the Meta-Prompter LLM call."""

    enhanced_goal: str
    ambiguity_score: float
    gaps: list[str]
    strategy: str


# ── Parser ─────────────────────────────────────────────────────────


def parse_meta_prompt_response(raw: str) -> MetaPromptResult | None:
    """Extract MetaPromptResult JSON from LLM response.

    Tolerates markdown code fences and partial JSON.
    Returns None if parsing fails entirely.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    if "enhanced_goal" not in data:
        return None

    score = float(data.get("ambiguity_score", 1.0))
    score = max(0.0, min(1.0, score))

    return MetaPromptResult(
        enhanced_goal=str(data["enhanced_goal"]),
        ambiguity_score=score,
        gaps=[str(g) for g in data.get("gaps", [])],
        strategy=str(data.get("strategy", "direct")),
    )
