"""Tests for Phase 3 Task 2: Centralized Orchestrator (Quality Gate).

Covers:
  - JSON parsing helper (code fences, raw JSON, invalid input)
  - Planner node: task decomposition, truncation, failure on bad JSON
  - Dispatcher node: parallel execution, SessionLane integration,
    worker exception handling, dependency gating
  - Reviewer node: approve, reject with critique, unparseable review
  - Routing decisions: finalizer / dispatcher / failed
  - Full orchestrator happy path (2 tasks → done)
  - Rejection → retry → success flow
  - Max-rejection → failed flow
  - Planner failure → failed flow
  - Telemetry record accumulation
  - Backward compatibility: existing SelfHealingGraph untouched
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from kappa.budget.gate import BudgetGate, LLMResponse
from kappa.config import AgentConfig, BudgetConfig, OrchestratorConfig
from kappa.graph.orchestrator import (
    OrchestratorGraph,
    OrchestratorState,
    SubTask,
)
from kappa.infra.session_lane import SyncSessionLane
from kappa.config import ExecutionConfig
from kappa.sandbox.executor import SandboxResult


# ═════════════════════════════════════════════════════════════════════
# Test doubles
# ═════════════════════════════════════════════════════════════════════


class _ThreadSafeProvider:
    """Mock LLM provider that returns responses in insertion order.

    Thread-safe so that concurrent workers can call it safely.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self._lock = threading.Lock()
        self.call_count = 0

    def call(
        self, *, messages: list[dict], model: str, max_tokens: int = 4096
    ) -> LLMResponse:
        with self._lock:
            content = self._responses[min(self._idx, len(self._responses) - 1)]
            self._idx += 1
            self.call_count += 1
        return LLMResponse(
            content=content,
            prompt_tokens=10,
            completion_tokens=10,
            model=model,
            stop_reason="end_turn",
        )


class _FakeExecutor:
    """Minimal sandbox executor that always succeeds."""

    def __init__(self):
        self._config = ExecutionConfig(workspace_dir=None, output_dir=None)

    @property
    def config(self):
        return self._config

    def execute(self, code: str) -> SandboxResult:
        return SandboxResult(
            exit_code=0,
            stdout=f"ok:{code[:30]}",
            stderr="",
            timed_out=False,
        )


def _make_gate(responses: list[str]) -> tuple[BudgetGate, _ThreadSafeProvider]:
    provider = _ThreadSafeProvider(responses)
    gate = BudgetGate(
        provider=provider,
        budget_config=BudgetConfig(max_total_tokens=500_000, max_cost_usd=50.0),
    )
    return gate, provider


def _make_sandbox() -> _FakeExecutor:
    return _FakeExecutor()


def _successful_worker_result(goal: str = "test") -> dict:
    """Return a minimal AgentState dict representing a successful worker run."""
    return {
        "goal": goal,
        "llm_output": "<think>ok</think>\n<action>print('hi')</action>",
        "parsed_code": "print('hi')",
        "sandbox_result": {
            "exit_code": 0,
            "stdout": f"result_of_{goal[:20]}",
            "stderr": "",
            "timed_out": False,
        },
        "attempt": 1,
        "max_attempts": 3,
        "error_history": [],
        "status": "success",
        "tool_calls": [],
        "memory_context": "",
        "workspace_path": "",
    }


def _plan_json(*tasks: tuple[str, str, list[str]]) -> str:
    """Build a planner JSON response from (id, goal, deps) tuples."""
    items = [
        {"id": tid, "goal": goal, "depends_on": deps}
        for tid, goal, deps in tasks
    ]
    return json.dumps({"tasks": items})


def _review_json(approved: bool, score: float = 0.9, critique: str = "") -> str:
    return json.dumps(
        {
            "approved": approved,
            "think": "Reviewed.",
            "critique": critique,
            "score": score,
        }
    )


def _plan_review_approve() -> str:
    """Plan reviewer response that approves the plan."""
    return json.dumps({"approved": True, "critique": "", "score": 0.9})


# ═════════════════════════════════════════════════════════════════════
# _parse_json helper
# ═════════════════════════════════════════════════════════════════════


class TestParseJson:
    def test_raw_json(self):
        assert OrchestratorGraph._parse_json('{"a": 1}') == {"a": 1}

    def test_code_fenced_json(self):
        text = '```json\n{"a": 1}\n```'
        assert OrchestratorGraph._parse_json(text) == {"a": 1}

    def test_json_embedded_in_text(self):
        text = 'Here is the plan: {"tasks": []} and more text'
        assert OrchestratorGraph._parse_json(text) == {"tasks": []}

    def test_invalid_json_returns_none(self):
        assert OrchestratorGraph._parse_json("not json at all") is None

    def test_empty_string(self):
        assert OrchestratorGraph._parse_json("") is None

    def test_nested_json(self):
        text = json.dumps({"tasks": [{"id": "t1", "goal": "x", "depends_on": []}]})
        result = OrchestratorGraph._parse_json(text)
        assert result is not None
        assert len(result["tasks"]) == 1


# ═════════════════════════════════════════════════════════════════════
# Planner node
# ═════════════════════════════════════════════════════════════════════


class TestPlannerNode:
    def test_decomposes_into_subtasks(self):
        plan_resp = _plan_json(
            ("task-001", "Do A", []),
            ("task-002", "Do B", ["task-001"]),
        )
        gate, _ = _make_gate([plan_resp])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        state = orch._initial_state("Build something")
        result = orch._planner_node(state)

        assert result["global_status"] == "plan_review"
        assert len(result["plan"]) == 2
        assert result["plan"][0]["id"] == "task-001"
        assert result["plan"][0]["status"] == "pending"
        assert result["plan"][1]["depends_on"] == ["task-001"]

    def test_truncates_to_max_subtasks(self):
        tasks = [(f"task-{i:03d}", f"Goal {i}", []) for i in range(20)]
        plan_resp = _plan_json(*tasks)
        gate, _ = _make_gate([plan_resp])
        orch = OrchestratorGraph(
            gate=gate,
            sandbox=_make_sandbox(),
            orchestrator_config=OrchestratorConfig(max_subtasks=3),
        )

        result = orch._planner_node(orch._initial_state("Big goal"))
        assert len(result["plan"]) == 3

    def test_bad_json_sets_failed(self):
        gate, _ = _make_gate(["This is not JSON"])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        result = orch._planner_node(orch._initial_state("Goal"))
        assert result["global_status"] == "failed"
        assert result["plan"] == []

    def test_missing_tasks_key_sets_failed(self):
        gate, _ = _make_gate(['{"items": []}'])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        result = orch._planner_node(orch._initial_state("Goal"))
        assert result["global_status"] == "failed"


# ═════════════════════════════════════════════════════════════════════
# Dispatcher node
# ═════════════════════════════════════════════════════════════════════


class TestDispatcherNode:
    def _make_state(self, plan: list[SubTask], completed=None) -> OrchestratorState:
        return {
            "main_goal": "test",
            "plan": plan,
            "completed": completed or [],
            "rejected_count": 0,
            "max_retries_per_task": 3,
            "global_status": "dispatching",
            "final_output": "",
            "telemetry_records": [],
        }

    def test_dispatches_ready_tasks(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "pending",
             "result": None, "critique": "", "attempts": 0},
        ]
        gate, _ = _make_gate([])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        worker_result = _successful_worker_result("A")

        with patch.object(orch, "_execute_subtask", return_value=worker_result):
            result = orch._dispatcher_node(self._make_state(plan))

        assert result["plan"][0]["status"] == "awaiting_review"
        assert result["plan"][0]["result"] is not None
        assert result["global_status"] == "reviewing"

    def test_skips_tasks_with_unmet_deps(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "pending",
             "result": None, "critique": "", "attempts": 0},
            {"id": "t2", "goal": "B", "depends_on": ["t1"], "status": "pending",
             "result": None, "critique": "", "attempts": 0},
        ]
        gate, _ = _make_gate([])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        worker_result = _successful_worker_result("A")

        with patch.object(orch, "_execute_subtask", return_value=worker_result):
            result = orch._dispatcher_node(self._make_state(plan))

        # t1 dispatched, t2 still pending
        assert result["plan"][0]["status"] == "awaiting_review"
        assert result["plan"][1]["status"] == "pending"

    def test_no_ready_tasks_fails(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": ["t0"], "status": "pending",
             "result": None, "critique": "", "attempts": 0},
        ]
        gate, _ = _make_gate([])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        result = orch._dispatcher_node(self._make_state(plan))
        assert result["global_status"] == "failed"

    def test_worker_exception_captured(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "pending",
             "result": None, "critique": "", "attempts": 0},
        ]
        gate, _ = _make_gate([])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        with patch.object(
            orch, "_execute_subtask", side_effect=RuntimeError("boom")
        ):
            result = orch._dispatcher_node(self._make_state(plan))

        assert result["plan"][0]["status"] == "awaiting_review"
        assert result["plan"][0]["result"]["status"] == "runtime_error"

    def test_dispatches_rejected_tasks(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "rejected",
             "result": None, "critique": "Fix it", "attempts": 1},
        ]
        gate, _ = _make_gate([])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        worker_result = _successful_worker_result("A")

        with patch.object(orch, "_execute_subtask", return_value=worker_result) as mock_exec:
            result = orch._dispatcher_node(self._make_state(plan))

        assert result["plan"][0]["status"] == "awaiting_review"
        # Verify critique was passed to worker
        called_subtask = mock_exec.call_args[0][0]
        assert called_subtask["critique"] == "Fix it"


# ═════════════════════════════════════════════════════════════════════
# Reviewer node
# ═════════════════════════════════════════════════════════════════════


class TestReviewerNode:
    def _make_state(
        self, plan: list[SubTask], completed=None, rejected_count=0
    ) -> OrchestratorState:
        return {
            "main_goal": "test",
            "plan": plan,
            "completed": completed or [],
            "rejected_count": rejected_count,
            "max_retries_per_task": 5,
            "global_status": "reviewing",
            "final_output": "",
            "telemetry_records": [],
        }

    def test_approves_successful_task(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "awaiting_review",
             "result": _successful_worker_result("A"), "critique": "", "attempts": 0},
        ]
        review_resp = _review_json(approved=True, score=0.95)
        gate, _ = _make_gate([review_resp])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        result = orch._reviewer_node(self._make_state(plan))

        assert result["plan"][0]["status"] == "completed"
        assert "t1" in result["completed"]
        assert result["rejected_count"] == 0

    def test_rejects_with_critique(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "awaiting_review",
             "result": _successful_worker_result("A"), "critique": "", "attempts": 0},
        ]
        review_resp = _review_json(approved=False, score=0.3, critique="Output wrong")
        gate, _ = _make_gate([review_resp])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        result = orch._reviewer_node(self._make_state(plan))

        assert result["plan"][0]["status"] == "rejected"
        assert result["plan"][0]["critique"] == "Output wrong"
        assert result["plan"][0]["attempts"] == 1
        assert result["rejected_count"] == 1
        assert "t1" not in result["completed"]

    def test_unparseable_review_rejects(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "awaiting_review",
             "result": _successful_worker_result("A"), "critique": "", "attempts": 0},
        ]
        gate, _ = _make_gate(["NOT JSON AT ALL"])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        result = orch._reviewer_node(self._make_state(plan))

        assert result["plan"][0]["status"] == "rejected"
        assert "could not be parsed" in result["plan"][0]["critique"]

    def test_skips_non_awaiting_tasks(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "completed",
             "result": _successful_worker_result("A"), "critique": "", "attempts": 0},
            {"id": "t2", "goal": "B", "depends_on": [], "status": "pending",
             "result": None, "critique": "", "attempts": 0},
        ]
        gate, _ = _make_gate([])  # no LLM calls expected
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        result = orch._reviewer_node(self._make_state(plan, completed=["t1"]))

        # Both tasks unchanged
        assert result["plan"][0]["status"] == "completed"
        assert result["plan"][1]["status"] == "pending"

    def test_telemetry_recorded(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "A", "depends_on": [], "status": "awaiting_review",
             "result": _successful_worker_result("A"), "critique": "", "attempts": 0},
        ]
        review_resp = _review_json(approved=True, score=0.88)
        gate, _ = _make_gate([review_resp])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        result = orch._reviewer_node(self._make_state(plan))

        assert len(result["telemetry_records"]) == 1
        record = result["telemetry_records"][0]
        assert record["task_id"] == "t1"
        assert record["score"] == 0.88
        assert record["outcome"] == "success"


# ═════════════════════════════════════════════════════════════════════
# Routing decisions
# ═════════════════════════════════════════════════════════════════════


class TestRouting:
    def _make_orch(self) -> OrchestratorGraph:
        gate, _ = _make_gate([])
        return OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

    def test_route_after_plan_success(self):
        orch = self._make_orch()
        state: OrchestratorState = {
            "main_goal": "", "plan": [{"id": "t1"}],  # type: ignore[typeddict-item]
            "completed": [], "rejected_count": 0, "max_retries_per_task": 3,
            "global_status": "dispatching", "final_output": "", "telemetry_records": [],
        }
        assert orch._route_after_plan(state) == "plan_reviewer"

    def test_route_after_plan_failure(self):
        orch = self._make_orch()
        state: OrchestratorState = {
            "main_goal": "", "plan": [],
            "completed": [], "rejected_count": 0, "max_retries_per_task": 3,
            "global_status": "failed", "final_output": "", "telemetry_records": [],
        }
        assert orch._route_after_plan(state) == "failed"

    def test_route_after_review_all_done(self):
        orch = self._make_orch()
        state: OrchestratorState = {
            "main_goal": "", "plan": [
                {"id": "t1", "goal": "", "depends_on": [], "status": "completed",
                 "result": None, "critique": "", "attempts": 0},
            ],
            "completed": ["t1"], "rejected_count": 0, "max_retries_per_task": 3,
            "global_status": "reviewing", "final_output": "", "telemetry_records": [],
        }
        assert orch._route_after_review(state) == "finalizer"

    def test_route_after_review_has_rejected(self):
        orch = self._make_orch()
        state: OrchestratorState = {
            "main_goal": "", "plan": [
                {"id": "t1", "goal": "", "depends_on": [], "status": "rejected",
                 "result": None, "critique": "bad", "attempts": 1},
            ],
            "completed": [], "rejected_count": 1, "max_retries_per_task": 3,
            "global_status": "reviewing", "final_output": "", "telemetry_records": [],
        }
        assert orch._route_after_review(state) == "dispatcher"

    def test_route_after_review_task_failed(self):
        orch = self._make_orch()
        state: OrchestratorState = {
            "main_goal": "", "plan": [
                {"id": "t1", "goal": "", "depends_on": [], "status": "failed",
                 "result": None, "critique": "bad", "attempts": 3},
            ],
            "completed": [], "rejected_count": 3, "max_retries_per_task": 3,
            "global_status": "reviewing", "final_output": "", "telemetry_records": [],
        }
        assert orch._route_after_review(state) == "failed"


# ═════════════════════════════════════════════════════════════════════
# Finalizer node
# ═════════════════════════════════════════════════════════════════════


class TestFinalizerNode:
    def test_merges_outputs(self):
        plan: list[SubTask] = [
            {"id": "t1", "goal": "Task A", "depends_on": [], "status": "completed",
             "result": {"sandbox_result": {"stdout": "output_A", "stderr": "", "exit_code": 0, "timed_out": False}},
             "critique": "", "attempts": 0},
            {"id": "t2", "goal": "Task B", "depends_on": [], "status": "completed",
             "result": {"sandbox_result": {"stdout": "output_B", "stderr": "", "exit_code": 0, "timed_out": False}},
             "critique": "", "attempts": 0},
        ]
        state: OrchestratorState = {
            "main_goal": "Goal", "plan": plan,
            "completed": ["t1", "t2"], "rejected_count": 0, "max_retries_per_task": 3,
            "global_status": "reviewing", "final_output": "", "telemetry_records": [],
        }
        gate, _ = _make_gate([])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        result = orch._finalizer_node(state)

        assert result["global_status"] == "done"
        assert "output_A" in result["final_output"]
        assert "output_B" in result["final_output"]
        assert result["final_output"].index("[t1]") < result["final_output"].index("[t2]")


# ═════════════════════════════════════════════════════════════════════
# Full orchestrator integration — happy path
# ═════════════════════════════════════════════════════════════════════


class TestOrchestratorHappyPath:
    def test_two_independent_tasks_succeed(self):
        """Planner → Dispatcher → Reviewer → Finalizer with 2 tasks."""
        plan_resp = _plan_json(
            ("task-001", "Write hello", []),
            ("task-002", "Write goodbye", []),
        )
        review_1 = _review_json(approved=True, score=0.9)
        review_2 = _review_json(approved=True, score=0.95)

        gate, provider = _make_gate([plan_resp, review_1, review_2])
        orch = OrchestratorGraph(
            gate=gate,
            sandbox=_make_sandbox(),
            orchestrator_config=OrchestratorConfig(max_parallel_workers=1),
        )

        worker_result = _successful_worker_result("task")
        with patch.object(orch, "_execute_subtask", return_value=worker_result):
            result = orch.run("Build a greeting app")

        assert result["global_status"] == "done"
        assert set(result["completed"]) == {"task-001", "task-002"}
        assert len(result["telemetry_records"]) == 2
        assert result["final_output"] != ""

    def test_sequential_dependency_chain(self):
        """Task B depends on Task A — must execute in order."""
        plan_resp = _plan_json(
            ("task-001", "Step 1", []),
            ("task-002", "Step 2", ["task-001"]),
        )
        review_a = _review_json(approved=True, score=0.9)
        review_b = _review_json(approved=True, score=0.9)

        gate, _ = _make_gate([plan_resp, review_a, review_b])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        execution_order: list[str] = []

        def mock_execute(subtask):
            execution_order.append(subtask["id"])
            return _successful_worker_result(subtask["goal"])

        with patch.object(orch, "_execute_subtask", side_effect=mock_execute):
            result = orch.run("Two-step process")

        assert result["global_status"] == "done"
        assert execution_order == ["task-001", "task-002"]

    def test_stream_yields_steps(self):
        """stream() should yield step dicts with node names."""
        plan_resp = _plan_json(("task-001", "Do it", []))
        review = _review_json(approved=True, score=0.9)

        gate, _ = _make_gate([plan_resp, review])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())

        with patch.object(
            orch, "_execute_subtask",
            return_value=_successful_worker_result("Do it"),
        ):
            steps = list(orch.stream("Simple goal"))

        node_names = [name for step in steps for name in step.keys()]
        assert "planner" in node_names
        assert "dispatcher" in node_names
        assert "reviewer" in node_names
        assert "finalizer" in node_names


# ═════════════════════════════════════════════════════════════════════
# Rejection and retry flow
# ═════════════════════════════════════════════════════════════════════


class TestOrchestratorRejection:
    def test_reject_then_approve(self):
        """Reviewer rejects once, task loops back, succeeds on retry."""
        plan_resp = _plan_json(("task-001", "Do A", []))
        review_reject = _review_json(approved=False, score=0.3, critique="Bad output")
        review_approve = _review_json(approved=True, score=0.9)

        gate, _ = _make_gate([plan_resp, _plan_review_approve(), review_reject, review_approve])
        orch = OrchestratorGraph(
            gate=gate,
            sandbox=_make_sandbox(),
            orchestrator_config=OrchestratorConfig(max_retries_per_task=5),
        )

        call_count = {"n": 0}

        def mock_execute(subtask):
            call_count["n"] += 1
            return _successful_worker_result(subtask["goal"])

        with patch.object(orch, "_execute_subtask", side_effect=mock_execute):
            result = orch.run("Tricky goal")

        assert result["global_status"] == "done"
        assert call_count["n"] == 2  # first attempt + retry
        assert result["rejected_count"] == 1
        assert len(result["telemetry_records"]) == 2

    def test_max_retries_per_task_triggers_failure(self):
        """Repeated rejections beyond max_retries_per_task → failed."""
        plan_resp = _plan_json(("task-001", "Impossible", []))
        reject = _review_json(approved=False, score=0.1, critique="Still wrong")

        # Enough rejects to exceed max_retries_per_task=2
        gate, _ = _make_gate([plan_resp, _plan_review_approve(), reject, reject, reject])
        orch = OrchestratorGraph(
            gate=gate,
            sandbox=_make_sandbox(),
            orchestrator_config=OrchestratorConfig(max_retries_per_task=2),
        )

        with patch.object(
            orch, "_execute_subtask",
            return_value=_successful_worker_result("Impossible"),
        ):
            result = orch.run("Impossible goal")

        assert result["global_status"] == "failed"
        assert result["rejected_count"] >= 2


# ═════════════════════════════════════════════════════════════════════
# Failure modes
# ═════════════════════════════════════════════════════════════════════


class TestOrchestratorFailure:
    def test_planner_bad_json_fails(self):
        gate, _ = _make_gate(["This is garbage, not JSON"])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        result = orch.run("Anything")
        assert result["global_status"] == "failed"

    def test_planner_empty_tasks_fails(self):
        gate, _ = _make_gate(['{"tasks": []}'])
        orch = OrchestratorGraph(gate=gate, sandbox=_make_sandbox())
        result = orch.run("Anything")
        assert result["global_status"] == "failed"


# ═════════════════════════════════════════════════════════════════════
# SessionLane integration
# ═════════════════════════════════════════════════════════════════════


class TestSessionLaneIntegration:
    def test_session_lane_used_during_dispatch(self):
        """Verify SessionLane.lane() is called for each worker execution."""
        lane = SyncSessionLane()
        acquired_keys: list[str] = []
        original_lane = lane.lane

        from contextlib import contextmanager

        @contextmanager
        def tracking_lane(key):
            acquired_keys.append(key)
            with original_lane(key):
                yield

        plan_resp = _plan_json(("task-001", "A", []), ("task-002", "B", []))
        review_1 = _review_json(approved=True, score=0.9)
        review_2 = _review_json(approved=True, score=0.9)

        # Worker responses: each SelfHealingGraph.run() needs coder output
        worker_code = '<think>ok</think>\n<action>print("hi")</action>'

        gate, _ = _make_gate([
            plan_resp,
            _plan_review_approve(),
            worker_code, worker_code,  # workers
            review_1, review_2,
        ])
        orch = OrchestratorGraph(
            gate=gate,
            sandbox=_make_sandbox(),
            session_lane=lane,
            orchestrator_config=OrchestratorConfig(max_parallel_workers=1),
        )

        with patch.object(lane, "lane", side_effect=tracking_lane):
            result = orch.run("Two tasks with lane")

        assert result["global_status"] == "done"
        assert sorted(acquired_keys) == ["task-001", "task-002"]


# ═════════════════════════════════════════════════════════════════════
# Full integration with real SelfHealingGraph workers
# ═════════════════════════════════════════════════════════════════════


class TestOrchestratorWithRealWorkers:
    def test_end_to_end_with_self_healing_graph(self):
        """Full pipeline: planner → real workers → reviewer → done."""
        plan_resp = _plan_json(("task-001", "Print hello", []))
        worker_code = '<think>I will print hello</think>\n<action>print("hello")</action>'
        review_resp = _review_json(approved=True, score=0.95)

        gate, _ = _make_gate([plan_resp, _plan_review_approve(), worker_code, review_resp])
        orch = OrchestratorGraph(
            gate=gate,
            sandbox=_make_sandbox(),
            orchestrator_config=OrchestratorConfig(max_parallel_workers=1),
        )

        result = orch.run("Print hello world")

        assert result["global_status"] == "done"
        assert "task-001" in result["completed"]
        assert len(result["telemetry_records"]) == 1
        assert result["telemetry_records"][0]["score"] == 0.95


# ═════════════════════════════════════════════════════════════════════
# Backward compatibility — SelfHealingGraph untouched
# ═════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    def test_self_healing_graph_unchanged(self):
        """SelfHealingGraph still works exactly as in Phase 2."""
        from kappa.graph.graph import SelfHealingGraph

        code_resp = '<think>Print hello.</think>\n<action>print("hello")</action>'
        provider = _ThreadSafeProvider([code_resp])
        gate = BudgetGate(
            provider=provider,
            budget_config=BudgetConfig(max_total_tokens=100_000),
        )
        sandbox = _make_sandbox()
        graph = SelfHealingGraph(gate=gate, sandbox=sandbox)
        result = graph.run("Print hello")

        assert result["status"] == "success"

    def test_orchestrator_imports_do_not_break_existing(self):
        """Importing OrchestratorGraph shouldn't affect existing exports."""
        from kappa.graph import AgentState, OrchestratorGraph, SelfHealingGraph

        assert SelfHealingGraph is not None
        assert AgentState is not None
        assert OrchestratorGraph is not None


# ═════════════════════════════════════════════════════════════════════
# OrchestratorConfig defaults
# ═════════════════════════════════════════════════════════════════════


class TestOrchestratorConfig:
    def test_defaults(self):
        cfg = OrchestratorConfig()
        assert cfg.max_retries_per_task == 3
        assert cfg.max_subtasks == 10
        assert cfg.max_parallel_workers == 3

    def test_override(self):
        cfg = OrchestratorConfig(max_retries_per_task=5, max_subtasks=20)
        assert cfg.max_retries_per_task == 5
        assert cfg.max_subtasks == 20
