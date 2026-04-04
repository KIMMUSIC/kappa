"""Tests for Phase 3 Task 3: Agent-RRM Telemetry Pipeline.

Covers:
  - TrajectoryRecord: creation, auto-timestamp, serialisation
  - TelemetryManager: JSONL write, read_all, summary, thread safety,
    disabled mode, empty log, unicode content
  - Orchestrator integration: telemetry records persisted to JSONL
    via the Reviewer node when TelemetryManager is injected
  - JSONL format: each line is valid JSON with think/critique/score
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from kappa.budget.gate import BudgetGate, LLMResponse
from kappa.config import BudgetConfig, OrchestratorConfig, TelemetryConfig
from kappa.graph.orchestrator import OrchestratorGraph
from kappa.sandbox.executor import SandboxExecutor, SandboxResult
from kappa.telemetry.manager import TelemetryManager, TrajectoryRecord


# ═════════════════════════════════════════════════════════════════════
# Test doubles (shared with test_orchestrator.py patterns)
# ═════════════════════════════════════════════════════════════════════


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


class _FakeRuntime:
    def run(self, *, image, command, mem_limit, network_disabled, timeout, volumes=None):
        return SandboxResult(exit_code=0, stdout="ok", stderr="", timed_out=False)


def _make_record(**overrides) -> TrajectoryRecord:
    defaults = {
        "task_id": "task-001",
        "worker_goal": "Do something",
        "think": "I will do the thing.",
        "critique": "",
        "score": 0.9,
        "outcome": "success",
    }
    defaults.update(overrides)
    return TrajectoryRecord(**defaults)


# ═════════════════════════════════════════════════════════════════════
# TrajectoryRecord
# ═════════════════════════════════════════════════════════════════════


class TestTrajectoryRecord:
    def test_auto_timestamp(self):
        rec = _make_record()
        assert rec.timestamp != ""
        assert "T" in rec.timestamp  # ISO 8601

    def test_explicit_timestamp_preserved(self):
        rec = TrajectoryRecord(
            task_id="t1", worker_goal="g", think="", critique="",
            score=0.5, outcome="success", timestamp="2026-01-01T00:00:00",
        )
        assert rec.timestamp == "2026-01-01T00:00:00"

    def test_frozen(self):
        rec = _make_record()
        with pytest.raises(AttributeError):
            rec.score = 0.0  # type: ignore[misc]

    def test_asdict_roundtrip(self):
        rec = _make_record(timestamp="2026-04-03T12:00:00+00:00")
        d = asdict(rec)
        rec2 = TrajectoryRecord(**d)
        assert rec == rec2

    def test_mandatory_fields(self):
        rec = _make_record()
        assert rec.task_id == "task-001"
        assert rec.think == "I will do the thing."
        assert rec.critique == ""
        assert rec.score == 0.9
        assert rec.outcome == "success"

    def test_default_token_usage(self):
        rec = _make_record()
        assert rec.token_usage == 0


# ═════════════════════════════════════════════════════════════════════
# TelemetryManager — basic I/O
# ═════════════════════════════════════════════════════════════════════


class TestTelemetryManagerIO:
    def test_record_creates_jsonl_file(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record())
        assert log.exists()

    def test_record_appends_single_line(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record())
        lines = log.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "task-001"

    def test_multiple_records_append(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record(task_id="t1"))
        mgr.record(_make_record(task_id="t2"))
        mgr.record(_make_record(task_id="t3"))
        lines = log.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_read_all_returns_records(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record(task_id="t1", score=0.8))
        mgr.record(_make_record(task_id="t2", score=0.95))

        records = mgr.read_all()
        assert len(records) == 2
        assert records[0].task_id == "t1"
        assert records[0].score == 0.8
        assert records[1].task_id == "t2"

    def test_read_all_empty_file(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        log.write_text("")
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        assert mgr.read_all() == []

    def test_read_all_no_file(self, tmp_path: Path):
        log = tmp_path / "nonexistent.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        assert mgr.read_all() == []

    def test_creates_parent_directories(self, tmp_path: Path):
        log = tmp_path / "deep" / "nested" / "dir" / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record())
        assert log.exists()


# ═════════════════════════════════════════════════════════════════════
# TelemetryManager — JSONL format validation
# ═════════════════════════════════════════════════════════════════════


class TestJSONLFormat:
    def test_each_line_is_valid_json(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        for i in range(5):
            mgr.record(_make_record(task_id=f"t{i}"))

        for line in log.read_text(encoding="utf-8").strip().split("\n"):
            data = json.loads(line)
            assert "task_id" in data
            assert "think" in data
            assert "critique" in data
            assert "score" in data

    def test_three_mandatory_metadata_present(self, tmp_path: Path):
        """Verify the 3 core RRM fields: think, critique, score."""
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record(
            think="I analyzed the problem step by step.",
            critique="Output contains a logic error in line 3.",
            score=0.42,
        ))

        data = json.loads(log.read_text(encoding="utf-8").strip())
        assert data["think"] == "I analyzed the problem step by step."
        assert data["critique"] == "Output contains a logic error in line 3."
        assert data["score"] == 0.42

    def test_unicode_content(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record(
            think="한국어 추론 과정",
            critique="日本語のフィードバック",
        ))

        records = mgr.read_all()
        assert records[0].think == "한국어 추론 과정"
        assert records[0].critique == "日本語のフィードバック"


# ═════════════════════════════════════════════════════════════════════
# TelemetryManager — summary
# ═════════════════════════════════════════════════════════════════════


class TestTelemetrySummary:
    def test_summary_empty(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        s = mgr.summary()
        assert s["total"] == 0
        assert s["avg_score"] == 0.0
        assert s["rejection_rate"] == 0.0

    def test_summary_all_success(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record(score=0.8, outcome="success"))
        mgr.record(_make_record(score=1.0, outcome="success"))

        s = mgr.summary()
        assert s["total"] == 2
        assert s["avg_score"] == 0.9
        assert s["rejection_rate"] == 0.0
        assert s["success_count"] == 2
        assert s["rejected_count"] == 0

    def test_summary_mixed(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        mgr.record(_make_record(score=0.9, outcome="success"))
        mgr.record(_make_record(score=0.3, outcome="rejected"))
        mgr.record(_make_record(score=0.8, outcome="success"))
        mgr.record(_make_record(score=0.2, outcome="rejected"))

        s = mgr.summary()
        assert s["total"] == 4
        assert s["avg_score"] == 0.55
        assert s["rejection_rate"] == 0.5
        assert s["success_count"] == 2
        assert s["rejected_count"] == 2


# ═════════════════════════════════════════════════════════════════════
# TelemetryManager — disabled mode
# ═════════════════════════════════════════════════════════════════════


class TestTelemetryDisabled:
    def test_disabled_no_file_created(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(enabled=False, log_path=str(log)))
        mgr.record(_make_record())
        assert not log.exists()

    def test_disabled_read_all_empty(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(enabled=False, log_path=str(log)))
        assert mgr.read_all() == []

    def test_disabled_summary_empty(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(enabled=False, log_path=str(log)))
        assert mgr.summary()["total"] == 0

    def test_enabled_property(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        enabled = TelemetryManager(TelemetryConfig(enabled=True, log_path=str(log)))
        disabled = TelemetryManager(TelemetryConfig(enabled=False, log_path=str(log)))
        assert enabled.enabled is True
        assert disabled.enabled is False


# ═════════════════════════════════════════════════════════════════════
# TelemetryManager — thread safety
# ═════════════════════════════════════════════════════════════════════


class TestTelemetryThreadSafety:
    def test_concurrent_writes_no_corruption(self, tmp_path: Path):
        """Multiple threads writing simultaneously must not corrupt JSONL."""
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))

        def writer(n: int) -> None:
            for i in range(10):
                mgr.record(_make_record(task_id=f"thread-{n}-item-{i}"))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(writer, n) for n in range(4)]
            for f in futures:
                f.result()

        records = mgr.read_all()
        assert len(records) == 40  # 4 threads × 10 records

        # Every line must be valid JSON
        for line in log.read_text(encoding="utf-8").strip().split("\n"):
            json.loads(line)  # raises if corrupted

    def test_concurrent_writes_all_ids_present(self, tmp_path: Path):
        """All task_ids from all threads must appear in the log."""
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))

        expected_ids: set[str] = set()
        for n in range(3):
            for i in range(5):
                expected_ids.add(f"t-{n}-{i}")

        def writer(n: int) -> None:
            for i in range(5):
                mgr.record(_make_record(task_id=f"t-{n}-{i}"))

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(writer, n) for n in range(3)]
            for f in futures:
                f.result()

        actual_ids = {r.task_id for r in mgr.read_all()}
        assert actual_ids == expected_ids


# ═════════════════════════════════════════════════════════════════════
# Orchestrator ↔ TelemetryManager integration
# ═════════════════════════════════════════════════════════════════════


class TestOrchestratorTelemetryIntegration:
    def _plan_json(self, *tasks):
        items = [
            {"id": tid, "goal": goal, "depends_on": deps}
            for tid, goal, deps in tasks
        ]
        return json.dumps({"tasks": items})

    def _review_json(self, approved, score=0.9, critique=""):
        return json.dumps({
            "approved": approved, "think": "Reviewed.",
            "critique": critique, "score": score,
        })

    def _worker_result(self, goal="test"):
        return {
            "goal": goal, "llm_output": "", "parsed_code": "print('ok')",
            "sandbox_result": {"exit_code": 0, "stdout": "ok", "stderr": "", "timed_out": False},
            "attempt": 1, "max_attempts": 3, "error_history": [],
            "status": "success", "tool_calls": [], "memory_context": "",
        }

    def test_records_persisted_on_approval(self, tmp_path: Path):
        """Approved tasks should produce a JSONL record with score."""
        log = tmp_path / "tel.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))

        plan_resp = self._plan_json(("task-001", "Print hello", []))
        review_resp = self._review_json(approved=True, score=0.95)

        provider = _MockProvider([plan_resp, review_resp])
        gate = BudgetGate(provider=provider, budget_config=BudgetConfig(max_total_tokens=500_000))
        sandbox = SandboxExecutor(runtime=_FakeRuntime())
        orch = OrchestratorGraph(gate=gate, sandbox=sandbox, telemetry=mgr)

        with patch.object(orch, "_execute_subtask", return_value=self._worker_result()):
            result = orch.run("Print hello")

        assert result["global_status"] == "done"

        records = mgr.read_all()
        assert len(records) == 1
        assert records[0].task_id == "task-001"
        assert records[0].worker_goal == "Print hello"
        assert records[0].score == 0.95
        assert records[0].outcome == "success"
        assert records[0].think == "Reviewed."
        assert records[0].critique == ""

    def test_records_persisted_on_rejection(self, tmp_path: Path):
        """Rejected tasks should produce a JSONL record with critique."""
        log = tmp_path / "tel.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))

        plan_resp = self._plan_json(("task-001", "Complex task", []))
        review_reject = self._review_json(approved=False, score=0.2, critique="Logic error")
        review_approve = self._review_json(approved=True, score=0.85)

        plan_review_ok = json.dumps({"approved": True, "critique": "", "score": 0.9})
        provider = _MockProvider([plan_resp, plan_review_ok, review_reject, review_approve])
        gate = BudgetGate(provider=provider, budget_config=BudgetConfig(max_total_tokens=500_000))
        sandbox = SandboxExecutor(runtime=_FakeRuntime())
        orch = OrchestratorGraph(
            gate=gate, sandbox=sandbox, telemetry=mgr,
            orchestrator_config=OrchestratorConfig(max_retries_per_task=5),
        )

        with patch.object(orch, "_execute_subtask", return_value=self._worker_result()):
            result = orch.run("Complex task")

        assert result["global_status"] == "done"

        records = mgr.read_all()
        assert len(records) == 2

        # First record: rejection
        assert records[0].outcome == "rejected"
        assert records[0].critique == "Logic error"
        assert records[0].score == 0.2

        # Second record: success after retry
        assert records[1].outcome == "success"
        assert records[1].score == 0.85

    def test_multiple_tasks_all_recorded(self, tmp_path: Path):
        """All subtask reviews should be persisted."""
        log = tmp_path / "tel.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))

        plan_resp = self._plan_json(
            ("task-001", "Task A", []),
            ("task-002", "Task B", []),
            ("task-003", "Task C", []),
        )
        reviews = [
            self._review_json(approved=True, score=0.9),
            self._review_json(approved=True, score=0.85),
            self._review_json(approved=True, score=0.95),
        ]

        provider = _MockProvider([plan_resp] + reviews)
        gate = BudgetGate(provider=provider, budget_config=BudgetConfig(max_total_tokens=500_000))
        sandbox = SandboxExecutor(runtime=_FakeRuntime())
        orch = OrchestratorGraph(
            gate=gate, sandbox=sandbox, telemetry=mgr,
            orchestrator_config=OrchestratorConfig(max_parallel_workers=1),
        )

        with patch.object(orch, "_execute_subtask", return_value=self._worker_result()):
            result = orch.run("Three tasks")

        assert result["global_status"] == "done"
        records = mgr.read_all()
        assert len(records) == 3

        s = mgr.summary()
        assert s["total"] == 3
        assert s["success_count"] == 3
        assert s["rejected_count"] == 0

    def test_no_telemetry_manager_still_works(self):
        """Orchestrator without TelemetryManager should work as before."""
        plan_resp = self._plan_json(("task-001", "Simple", []))
        review_resp = self._review_json(approved=True, score=0.9)

        provider = _MockProvider([plan_resp, review_resp])
        gate = BudgetGate(provider=provider, budget_config=BudgetConfig(max_total_tokens=500_000))
        sandbox = SandboxExecutor(runtime=_FakeRuntime())
        orch = OrchestratorGraph(gate=gate, sandbox=sandbox)  # no telemetry

        with patch.object(orch, "_execute_subtask", return_value=self._worker_result()):
            result = orch.run("Simple goal")

        assert result["global_status"] == "done"
        assert len(result["telemetry_records"]) == 1  # still in state


# ═════════════════════════════════════════════════════════════════════
# TelemetryConfig defaults
# ═════════════════════════════════════════════════════════════════════


class TestTelemetryConfig:
    def test_defaults(self):
        cfg = TelemetryConfig()
        assert cfg.enabled is True
        assert "trajectories.jsonl" in cfg.log_path

    def test_override(self):
        cfg = TelemetryConfig(enabled=False, log_path="/tmp/custom.jsonl")
        assert cfg.enabled is False
        assert cfg.log_path == "/tmp/custom.jsonl"

    def test_log_path_property(self, tmp_path: Path):
        log = tmp_path / "test.jsonl"
        mgr = TelemetryManager(TelemetryConfig(log_path=str(log)))
        assert mgr.log_path == log
