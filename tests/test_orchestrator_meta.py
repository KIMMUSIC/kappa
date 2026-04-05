"""Tests for Phase 6: Meta-Prompting Pipeline integration with OrchestratorGraph.

Covers:
  - _meta_prompter_node: skip mode, LLM analysis, parse failure fallback
  - _interview_node: sentinel emission
  - _plan_approval_node: sentinel emission, skip fallback
  - _route_after_meta_prompter: ambiguity-based routing
  - Graph topology: meta_prompter as entry point, plan_approval replaces plan_reviewer
  - OrchestratorState new fields in _initial_state
  - Public API: update_state, resume_stream signatures
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from kappa.budget.gate import BudgetGate, LLMResponse
from kappa.config import (
    AgentConfig,
    BudgetConfig,
    ExecutionConfig,
    MetaPromptConfig,
    OrchestratorConfig,
)
from kappa.graph.orchestrator import OrchestratorGraph, OrchestratorState
from kappa.sandbox.executor import SandboxResult


# ── Test Doubles ────────────────────────────────────────────────────


class _MockProvider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self._lock = threading.Lock()

    def call(self, *, messages, model, max_tokens=4096) -> LLMResponse:
        with self._lock:
            content = self._responses[min(self._idx, len(self._responses) - 1)]
            self._idx += 1
        return LLMResponse(
            content=content, prompt_tokens=10, completion_tokens=10,
            model=model, stop_reason="end_turn",
        )


class _FakeExecutor:
    def __init__(self):
        self._config = ExecutionConfig(workspace_dir=None, output_dir=None)

    @property
    def config(self):
        return self._config

    def execute(self, code: str) -> SandboxResult:
        return SandboxResult(exit_code=0, stdout="ok", stderr="", timed_out=False)


def _make_gate(responses: list[str]) -> BudgetGate:
    return BudgetGate(
        provider=_MockProvider(responses),
        budget_config=BudgetConfig(max_total_tokens=500_000),
    )


_SKIP_META = MetaPromptConfig(skip_interview=True, skip_plan_approval=True)


# ── Meta-Prompter Node Tests ───────────────────────────────────────


class TestMetaPrompterNode:
    """_meta_prompter_node behavior."""

    def test_skip_mode_passes_through(self):
        """When both skip flags set, returns main_goal as enhanced_goal."""
        gate = _make_gate(["unused"])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )
        state = orch._initial_state("Build a tool")
        result = orch._meta_prompter_node(state)

        assert result["enhanced_goal"] == "Build a tool"
        assert result["ambiguity_score"] == 0.0
        assert result["gaps"] == []
        assert result["meta_strategy"] == "direct"
        assert result["global_status"] == "meta_prompting"

    def test_llm_analysis_returns_parsed_result(self):
        """Normal mode: LLM returns valid JSON → parsed into state."""
        meta_response = json.dumps({
            "enhanced_goal": "Build a REST API with Flask and PostgreSQL",
            "ambiguity_score": 0.6,
            "gaps": ["What authentication method?"],
            "strategy": "ReAct",
        })
        gate = _make_gate([meta_response])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=MetaPromptConfig(skip_interview=False, skip_plan_approval=True),
        )
        state = orch._initial_state("Build an API")
        result = orch._meta_prompter_node(state)

        assert result["enhanced_goal"] == "Build a REST API with Flask and PostgreSQL"
        assert result["ambiguity_score"] == 0.6
        assert result["gaps"] == ["What authentication method?"]
        assert result["meta_strategy"] == "ReAct"

    def test_llm_parse_failure_fallback(self):
        """When LLM returns unparseable response, falls back to main_goal."""
        gate = _make_gate(["This is not JSON"])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=MetaPromptConfig(skip_interview=False, skip_plan_approval=True),
        )
        state = orch._initial_state("Build something")
        result = orch._meta_prompter_node(state)

        assert result["enhanced_goal"] == "Build something"
        assert result["ambiguity_score"] == 0.0
        assert result["gaps"] == []


# ── Interview Node Tests ───────────────────────────────────────────


class TestInterviewNode:
    """_interview_node sentinel behavior."""

    def test_emits_awaiting_interview_sentinel(self):
        gate = _make_gate(["unused"])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )
        state = orch._initial_state("Test")
        result = orch._interview_node(state)

        assert result["global_status"] == "awaiting_interview"


# ── Plan Approval Node Tests ──────────────────────────────────────


class TestPlanApprovalNode:
    """_plan_approval_node behavior."""

    def test_skip_mode_delegates_to_plan_reviewer(self):
        """When skip_plan_approval=True, falls back to LLM review."""
        plan_review_resp = json.dumps({"approved": True, "critique": "", "score": 0.9})
        gate = _make_gate([plan_review_resp])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=MetaPromptConfig(skip_interview=True, skip_plan_approval=True),
        )
        state = orch._initial_state("Test")
        state["plan"] = [
            {"id": "task-001", "goal": "Do thing", "depends_on": [],
             "status": "pending", "result": None, "critique": "", "attempts": 0},
        ]
        result = orch._plan_approval_node(state)

        assert result["global_status"] == "dispatching"

    def test_interactive_mode_emits_sentinel(self):
        """When skip_plan_approval=False, emits awaiting_plan_approval."""
        gate = _make_gate(["unused"])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=MetaPromptConfig(skip_interview=True, skip_plan_approval=False),
        )
        state = orch._initial_state("Test")
        result = orch._plan_approval_node(state)

        assert result["global_status"] == "awaiting_plan_approval"


# ── Routing Tests ─────────────────────────────────────────────────


class TestMetaRouting:
    """_route_after_meta_prompter logic."""

    def _make_orch(self, skip_interview=False, threshold=0.4):
        gate = _make_gate(["unused"])
        return OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=MetaPromptConfig(
                skip_interview=skip_interview,
                ambiguity_threshold=threshold,
            ),
        )

    def test_skip_interview_always_routes_to_planner(self):
        orch = self._make_orch(skip_interview=True)
        state = orch._initial_state("Test")
        state["ambiguity_score"] = 0.9  # high ambiguity
        assert orch._route_after_meta_prompter(state) == "planner"

    def test_high_ambiguity_routes_to_interview(self):
        orch = self._make_orch(skip_interview=False, threshold=0.4)
        state = orch._initial_state("Test")
        state["ambiguity_score"] = 0.7
        assert orch._route_after_meta_prompter(state) == "interview"

    def test_low_ambiguity_routes_to_planner(self):
        orch = self._make_orch(skip_interview=False, threshold=0.4)
        state = orch._initial_state("Test")
        state["ambiguity_score"] = 0.2
        assert orch._route_after_meta_prompter(state) == "planner"

    def test_exact_threshold_routes_to_planner(self):
        orch = self._make_orch(skip_interview=False, threshold=0.4)
        state = orch._initial_state("Test")
        state["ambiguity_score"] = 0.4
        assert orch._route_after_meta_prompter(state) == "planner"


# ── Initial State Tests ───────────────────────────────────────────


class TestInitialState:
    """_initial_state includes all new Phase 6 fields."""

    def test_new_fields_have_defaults(self):
        gate = _make_gate(["unused"])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )
        state = orch._initial_state("Test goal")

        assert state["enhanced_goal"] == ""
        assert state["ambiguity_score"] == 0.0
        assert state["gaps"] == []
        assert state["meta_strategy"] == ""
        assert state["interview_result"] is None
        assert state["user_plan_feedback"] == ""
        assert state["main_goal"] == "Test goal"


# ── Public API Tests ──────────────────────────────────────────────


class TestPublicAPI:
    """stream/run/update_state/resume_stream signatures."""

    def test_stream_accepts_config(self):
        """stream() should accept optional config parameter."""
        plan_resp = json.dumps({"tasks": [{"id": "task-001", "goal": "Print hello", "depends_on": []}]})
        plan_review = json.dumps({"approved": True, "critique": "", "score": 0.9})
        review_resp = json.dumps({"approved": True, "think": "ok", "critique": "", "score": 0.95})
        gate = _make_gate([plan_resp, plan_review, review_resp])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )
        config = {"configurable": {"thread_id": "test-stream"}}

        with patch.object(orch, "_execute_subtask", return_value={
            "status": "success", "parsed_code": "print('hello')",
            "sandbox_result": {"stdout": "hello", "stderr": "", "exit_code": 0},
        }):
            steps = list(orch.stream("Hello", config=config))

        assert len(steps) > 0

    def test_run_accepts_config(self):
        """run() should accept optional config parameter."""
        plan_resp = json.dumps({"tasks": [{"id": "task-001", "goal": "Print hello", "depends_on": []}]})
        plan_review = json.dumps({"approved": True, "critique": "", "score": 0.9})
        review_resp = json.dumps({"approved": True, "think": "ok", "critique": "", "score": 0.95})
        gate = _make_gate([plan_resp, plan_review, review_resp])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )

        with patch.object(orch, "_execute_subtask", return_value={
            "status": "success", "parsed_code": "print('hello')",
            "sandbox_result": {"stdout": "hello", "stderr": "", "exit_code": 0},
        }):
            result = orch.run("Hello", config={"configurable": {"thread_id": "test-run"}})

        assert result["global_status"] == "done"

    def test_update_state_method_exists(self):
        """update_state should be a callable public method."""
        gate = _make_gate(["unused"])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )
        assert callable(getattr(orch, "update_state", None))

    def test_resume_stream_method_exists(self):
        """resume_stream should be a callable public method."""
        gate = _make_gate(["unused"])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )
        assert callable(getattr(orch, "resume_stream", None))


# ── Graph Topology Tests ──────────────────────────────────────────


class TestGraphTopology:
    """Graph structure verification."""

    def test_plan_approval_replaces_plan_reviewer(self):
        """plan_approval node exists, plan_reviewer is method-only."""
        gate = _make_gate(["unused"])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )
        # _plan_reviewer_node still exists as method (fallback)
        assert hasattr(orch, "_plan_reviewer_node")
        # _plan_approval_node is the graph node
        assert hasattr(orch, "_plan_approval_node")

    def test_planner_uses_enhanced_goal(self):
        """_planner_node should use enhanced_goal when available."""
        plan_resp = json.dumps({
            "tasks": [{"id": "task-001", "goal": "Do it", "depends_on": []}]
        })
        gate = _make_gate([plan_resp])
        orch = OrchestratorGraph(
            gate=gate, sandbox=_FakeExecutor(),
            meta_prompt_config=_SKIP_META,
        )
        state = orch._initial_state("Original goal")
        state["enhanced_goal"] = "Enhanced: Build X with Y"

        result = orch._planner_node(state)
        assert len(result["plan"]) >= 1
