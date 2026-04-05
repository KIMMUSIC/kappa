"""Tests for Phase 6: Meta-Prompter module.

Covers:
  - META_PROMPTER_PROMPT formatting
  - parse_meta_prompt_response: valid JSON, code fences, partial, invalid
  - Ambiguity score clamping (0.0–1.0)
  - Gaps extraction
  - Strategy extraction
"""

from __future__ import annotations

import json

import pytest

from kappa.graph.meta_prompter import (
    META_PROMPTER_PROMPT,
    MetaPromptResult,
    parse_meta_prompt_response,
)


class TestMetaPromptPrompt:
    """META_PROMPTER_PROMPT formatting."""

    def test_format_with_goal_and_context(self):
        result = META_PROMPTER_PROMPT.format(
            goal="Build a web scraper",
            workspace_context="Files: main.py, utils.py",
        )
        assert "Build a web scraper" in result
        assert "Files: main.py, utils.py" in result

    def test_format_empty_context(self):
        result = META_PROMPTER_PROMPT.format(goal="Hello", workspace_context="")
        assert "Hello" in result


class TestParseMetaPromptResponse:
    """parse_meta_prompt_response edge cases."""

    def test_valid_json(self):
        raw = json.dumps({
            "enhanced_goal": "Build a REST API with Flask",
            "ambiguity_score": 0.3,
            "gaps": ["What database?"],
            "strategy": "CoT",
        })
        result = parse_meta_prompt_response(raw)
        assert result is not None
        assert result["enhanced_goal"] == "Build a REST API with Flask"
        assert result["ambiguity_score"] == 0.3
        assert result["gaps"] == ["What database?"]
        assert result["strategy"] == "CoT"

    def test_code_fence_json(self):
        raw = '```json\n{"enhanced_goal": "Test", "ambiguity_score": 0.5, "gaps": [], "strategy": "direct"}\n```'
        result = parse_meta_prompt_response(raw)
        assert result is not None
        assert result["enhanced_goal"] == "Test"

    def test_json_embedded_in_text(self):
        raw = 'Here is the result:\n{"enhanced_goal": "Test", "ambiguity_score": 0.2}\nDone.'
        result = parse_meta_prompt_response(raw)
        assert result is not None
        assert result["enhanced_goal"] == "Test"

    def test_missing_enhanced_goal_returns_none(self):
        raw = json.dumps({"ambiguity_score": 0.5})
        result = parse_meta_prompt_response(raw)
        assert result is None

    def test_completely_invalid_returns_none(self):
        result = parse_meta_prompt_response("This is not JSON at all")
        assert result is None

    def test_empty_string_returns_none(self):
        result = parse_meta_prompt_response("")
        assert result is None

    def test_ambiguity_score_clamped_high(self):
        raw = json.dumps({
            "enhanced_goal": "Test",
            "ambiguity_score": 1.5,
            "gaps": [],
            "strategy": "direct",
        })
        result = parse_meta_prompt_response(raw)
        assert result is not None
        assert result["ambiguity_score"] == 1.0

    def test_ambiguity_score_clamped_low(self):
        raw = json.dumps({
            "enhanced_goal": "Test",
            "ambiguity_score": -0.3,
        })
        result = parse_meta_prompt_response(raw)
        assert result is not None
        assert result["ambiguity_score"] == 0.0

    def test_default_gaps_and_strategy(self):
        raw = json.dumps({"enhanced_goal": "Test"})
        result = parse_meta_prompt_response(raw)
        assert result is not None
        assert result["gaps"] == []
        assert result["strategy"] == "direct"

    def test_gaps_coerced_to_strings(self):
        raw = json.dumps({
            "enhanced_goal": "Test",
            "gaps": [123, True, "valid"],
        })
        result = parse_meta_prompt_response(raw)
        assert result is not None
        assert result["gaps"] == ["123", "True", "valid"]

    def test_ambiguity_score_default_when_missing(self):
        raw = json.dumps({"enhanced_goal": "Test"})
        result = parse_meta_prompt_response(raw)
        assert result is not None
        assert result["ambiguity_score"] == 1.0
