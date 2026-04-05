"""Tests for Phase 6: Interview Engine module.

Covers:
  - INTERVIEW_SYNTHESIZER_PROMPT formatting
  - InterviewResult structure
  - run_interview with mocked console and gate
  - Empty gaps → passthrough (no questions asked)
  - Question limiting via max_questions
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kappa.budget.gate import LLMResponse
from kappa.graph.interview import (
    INTERVIEW_SYNTHESIZER_PROMPT,
    InterviewResult,
    run_interview,
)


class TestInterviewSynthesizerPrompt:
    """INTERVIEW_SYNTHESIZER_PROMPT formatting."""

    def test_format_with_goal_and_qa(self):
        result = INTERVIEW_SYNTHESIZER_PROMPT.format(
            goal="Build a dashboard",
            qa_pairs="Q: What framework?\nA: Streamlit",
        )
        assert "Build a dashboard" in result
        assert "Streamlit" in result


class TestRunInterview:
    """run_interview behavior with mocked I/O."""

    def _make_gate(self, response_text: str) -> MagicMock:
        gate = MagicMock()
        gate.call.return_value = LLMResponse(
            content=response_text,
            prompt_tokens=10,
            completion_tokens=20,
            model="test",
            stop_reason="end_turn",
        )
        return gate

    @patch("kappa.graph.interview.Prompt.ask")
    def test_empty_gaps_generates_questions_via_llm(self, mock_ask):
        """No pre-generated gaps → LLM generates questions → interview proceeds."""
        mock_ask.return_value = "Use PostgreSQL"
        console = MagicMock()
        gate = MagicMock()
        # First call: generate questions, second call: synthesize golden goal
        gate.call.side_effect = [
            LLMResponse(
                content='["What database?", "What framework?"]',
                prompt_tokens=10, completion_tokens=20,
                model="test", stop_reason="end_turn",
            ),
            LLMResponse(
                content="Build a dashboard with PostgreSQL",
                prompt_tokens=10, completion_tokens=20,
                model="test", stop_reason="end_turn",
            ),
        ]

        result = run_interview(
            console=console,
            goal="Build a dashboard",
            gaps=[],
            max_questions=5,
            gate=gate,
            model="test-model",
        )

        assert result["golden_goal"] == "Build a dashboard with PostgreSQL"
        assert len(result["qa_pairs"]) == 2
        assert gate.call.call_count == 2  # generate questions + synthesize

    @patch("kappa.graph.interview.Prompt.ask")
    def test_single_gap_single_question(self, mock_ask):
        """One gap → one question → synthesis."""
        mock_ask.return_value = "Use PostgreSQL"
        console = MagicMock()
        gate = self._make_gate("Build a dashboard with PostgreSQL backend")

        result = run_interview(
            console=console,
            goal="Build a dashboard",
            gaps=["What database?"],
            max_questions=5,
            gate=gate,
            model="test-model",
        )

        assert result["golden_goal"] == "Build a dashboard with PostgreSQL backend"
        assert len(result["qa_pairs"]) == 1
        assert result["qa_pairs"][0]["question"] == "What database?"
        assert result["qa_pairs"][0]["answer"] == "Use PostgreSQL"
        assert result["original_goal"] == "Build a dashboard"
        gate.call.assert_called_once()

    @patch("kappa.graph.interview.Prompt.ask")
    def test_max_questions_limits_gaps(self, mock_ask):
        """Only ask up to max_questions, even if more gaps exist."""
        mock_ask.return_value = "answer"
        console = MagicMock()
        gate = self._make_gate("synthesized goal")

        result = run_interview(
            console=console,
            goal="Vague goal",
            gaps=["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            max_questions=2,
            gate=gate,
            model="test-model",
        )

        assert len(result["qa_pairs"]) == 2
        assert mock_ask.call_count == 2

    @patch("kappa.graph.interview.Prompt.ask")
    def test_multiple_answers_synthesized(self, mock_ask):
        """Multiple Q&A pairs are all passed to synthesis prompt."""
        mock_ask.side_effect = ["CSV files", "Streamlit", "Heatmap chart"]
        console = MagicMock()
        gate = self._make_gate("Complete spec with CSV, Streamlit, and heatmap")

        result = run_interview(
            console=console,
            goal="Data analysis tool",
            gaps=["Data source?", "UI framework?", "Chart type?"],
            max_questions=5,
            gate=gate,
            model="test-model",
        )

        assert len(result["qa_pairs"]) == 3
        # Verify the synthesis prompt contains all answers
        call_args = gate.call.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        prompt_text = messages[0]["content"]
        assert "CSV files" in prompt_text
        assert "Streamlit" in prompt_text
        assert "Heatmap chart" in prompt_text
