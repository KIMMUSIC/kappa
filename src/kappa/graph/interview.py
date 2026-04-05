"""Interactive Interview Engine for user intent clarification.

When the Meta-Prompter detects high ambiguity, this module conducts a
Rich CLI interview to gather missing information from the user. The
answers are synthesized with the original goal into a "Golden Goal" —
a precise engineering specification with zero ambiguity.

The interview node itself is a sentinel-based non-blocking state setter.
Actual I/O (Rich Panel + Prompt.ask) happens in the CLI layer via
``run_interview()``, which is called by ``run_dashboard`` when it
detects the ``"awaiting_interview"`` sentinel status.
"""

from __future__ import annotations

import json
import re
from typing import TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from kappa.budget.gate import BudgetGate


# ── Prompt ─────────────────────────────────────────────────────────

INTERVIEW_SYNTHESIZER_PROMPT = """\
You are synthesizing a user's original goal with their interview answers \
into a precise, unambiguous engineering specification.

Original goal: {goal}

Interview Q&A:
{qa_pairs}

Create a single, comprehensive goal statement that:
1. Eliminates ALL ambiguity from the original goal
2. Incorporates specific technologies, data formats, UI choices, and \
validation criteria from the answers
3. Is detailed enough for a code-generation agent to implement without \
any guessing
4. Includes concrete acceptance criteria derived from the answers

Output ONLY the enhanced goal text (no JSON, no markdown, no commentary). \
Write it as a clear engineering specification paragraph."""


# ── Result Type ────────────────────────────────────────────────────


class InterviewResult(TypedDict):
    """Output from a completed interview session."""

    golden_goal: str
    qa_pairs: list[dict[str, str]]
    original_goal: str


# ── Interview Runner ───────────────────────────────────────────────


def run_interview(
    console: Console,
    goal: str,
    gaps: list[str],
    max_questions: int,
    gate: BudgetGate,
    model: str,
) -> InterviewResult:
    """Conduct a Rich CLI interview and synthesize a Golden Goal.

    This function is called from the CLI layer (``run_dashboard``)
    when the graph emits the ``"awaiting_interview"`` sentinel.
    It blocks on user input via ``Prompt.ask()``, so it must only
    be called when Rich ``Live`` rendering is paused.

    Args:
        console: Rich Console for rendering.
        goal: The user's original goal string.
        gaps: List of gap questions from Meta-Prompter analysis.
        max_questions: Maximum number of questions to ask.
        gate: BudgetGate for LLM calls (Golden Goal synthesis).
        model: LLM model identifier for synthesis.

    Returns:
        InterviewResult with the synthesized golden_goal.
    """
    questions = gaps[:max_questions]
    if not questions:
        return InterviewResult(
            golden_goal=goal,
            qa_pairs=[],
            original_goal=goal,
        )

    # ── Render interview header ──────────────────────────────────
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]목표:[/] \"{goal}\"\n"
                f"완벽한 도구를 설계하기 위해 [bold]{len(questions)}[/]가지 "
                f"질문을 드리겠습니다."
            ),
            title="[bold cyan]🤖 KAPPA PRODUCT MANAGER: 요구사항 구체화 인터뷰[/]",
            border_style="cyan",
        )
    )
    console.print()

    # ── Q&A loop ─────────────────────────────────────────────────
    qa_pairs: list[dict[str, str]] = []
    for i, question in enumerate(questions, 1):
        console.print(
            f"[bold cyan]\\[Q{i}/{len(questions)}][/] {question}"
        )
        answer = Prompt.ask("[bold]❯[/]")
        qa_pairs.append({"question": question, "answer": answer.strip()})
        console.print()

    # ── Synthesize Golden Goal via LLM ───────────────────────────
    qa_text = "\n".join(
        f"Q: {qa['question']}\nA: {qa['answer']}"
        for qa in qa_pairs
    )
    prompt = INTERVIEW_SYNTHESIZER_PROMPT.format(
        goal=goal,
        qa_pairs=qa_text,
    )

    messages = [{"role": "user", "content": prompt}]
    response = gate.call(messages=messages, model=model)
    golden_goal = response.content.strip()

    # ── Display synthesized goal ─────────────────────────────────
    console.print(
        Panel(
            golden_goal,
            title="[bold green]📝 확정 목표 (Golden Goal)[/]",
            border_style="green",
        )
    )
    console.print()

    return InterviewResult(
        golden_goal=golden_goal,
        qa_pairs=qa_pairs,
        original_goal=goal,
    )
