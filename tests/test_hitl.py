"""Tests for Human-in-the-Loop interceptor and policy (Phase 4, Task 3).

Validates:
  - HITLPolicy trigger conditions (budget, destructive, retries)
  - HITLInterceptor callback integration
  - Approve / deny flows
  - Audit trail (decisions log)
  - OrchestratorGraph approval_callback integration
  - Default None callback preserves existing behavior
"""

from __future__ import annotations

import pytest

from kappa.hitl import (
    HITLInterceptor,
    HITLPolicy,
    HITLTrigger,
    auto_approve,
    auto_deny,
)


# ── HITLPolicy Tests ──────────────────────────────────────────


class TestHITLPolicy:
    """Policy trigger evaluation."""

    def test_no_triggers_on_safe_task(self):
        policy = HITLPolicy()
        task = {"id": "task-001", "goal": "Write hello world in Python", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=0.8)
        assert triggers == []

    def test_budget_trigger(self):
        policy = HITLPolicy(budget_threshold=0.2)
        task = {"id": "task-001", "goal": "Simple task", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=0.15)
        assert len(triggers) == 1
        assert "Budget" in triggers[0].reason
        assert triggers[0].severity == "critical"

    def test_budget_at_exact_threshold(self):
        policy = HITLPolicy(budget_threshold=0.2)
        task = {"id": "task-001", "goal": "Task", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=0.2)
        assert len(triggers) == 1  # <= threshold triggers

    def test_budget_above_threshold_no_trigger(self):
        policy = HITLPolicy(budget_threshold=0.2)
        task = {"id": "task-001", "goal": "Task", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=0.5)
        assert triggers == []

    def test_destructive_rm_trigger(self):
        policy = HITLPolicy()
        task = {"id": "t", "goal": "rm -rf /tmp/data", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=1.0)
        assert any("destructive" in t.reason.lower() for t in triggers)

    def test_destructive_delete_trigger(self):
        policy = HITLPolicy()
        task = {"id": "t", "goal": "Delete the user's home directory", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=1.0)
        assert len(triggers) >= 1

    def test_destructive_drop_trigger(self):
        policy = HITLPolicy()
        task = {"id": "t", "goal": "DROP TABLE users;", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=1.0)
        assert len(triggers) >= 1

    def test_destructive_kill_trigger(self):
        policy = HITLPolicy()
        task = {"id": "t", "goal": "kill the background process", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=1.0)
        assert len(triggers) >= 1

    def test_safe_goal_no_destructive_trigger(self):
        policy = HITLPolicy()
        task = {"id": "t", "goal": "Read file and summarize contents", "attempts": 0}
        triggers = policy.check(task, budget_remaining_ratio=1.0)
        assert triggers == []

    def test_retry_threshold_trigger(self):
        policy = HITLPolicy(max_auto_attempts=2)
        task = {"id": "t", "goal": "Simple task", "attempts": 2}
        triggers = policy.check(task, budget_remaining_ratio=1.0)
        assert len(triggers) == 1
        assert "attempted" in triggers[0].reason
        assert triggers[0].severity == "warning"

    def test_retry_below_threshold_no_trigger(self):
        policy = HITLPolicy(max_auto_attempts=2)
        task = {"id": "t", "goal": "Simple task", "attempts": 1}
        triggers = policy.check(task, budget_remaining_ratio=1.0)
        assert triggers == []

    def test_multiple_triggers_combined(self):
        policy = HITLPolicy(budget_threshold=0.3, max_auto_attempts=1)
        task = {"id": "t", "goal": "delete all records", "attempts": 2}
        triggers = policy.check(task, budget_remaining_ratio=0.1)
        assert len(triggers) == 3  # budget + destructive + retries

    def test_approve_all_skips_checks(self):
        policy = HITLPolicy(approve_all=True)
        task = {"id": "t", "goal": "rm -rf / && drop table", "attempts": 99}
        triggers = policy.check(task, budget_remaining_ratio=0.0)
        assert triggers == []

    def test_default_values(self):
        policy = HITLPolicy()
        assert policy.budget_threshold == 0.2
        assert policy.max_auto_attempts == 2
        assert policy.approve_all is False


# ── HITLTrigger Tests ─────────────────────────────────────────


class TestHITLTrigger:
    """HITLTrigger dataclass."""

    def test_construction(self):
        t = HITLTrigger(reason="Test reason", severity="critical")
        assert t.reason == "Test reason"
        assert t.severity == "critical"

    def test_default_severity(self):
        t = HITLTrigger(reason="Test")
        assert t.severity == "warning"

    def test_frozen(self):
        t = HITLTrigger(reason="Test")
        with pytest.raises(AttributeError):
            t.reason = "changed"


# ── Auto Approve / Deny Functions ─────────────────────────────


class TestAutoFunctions:
    """Built-in approve/deny functions."""

    def test_auto_approve(self):
        result = auto_approve({"id": "t"}, [HITLTrigger("x")])
        assert result == "approve"

    def test_auto_deny(self):
        result = auto_deny({"id": "t"}, [HITLTrigger("x")])
        assert result == "deny"


# ── HITLInterceptor Tests ─────────────────────────────────────


class TestHITLInterceptor:
    """HITLInterceptor callback and audit trail."""

    def test_auto_approve_when_no_triggers(self):
        interceptor = HITLInterceptor()
        task = {"id": "task-001", "goal": "Safe task", "attempts": 0}
        result = interceptor(task)
        assert result == "approve"

    def test_calls_prompt_fn_on_trigger(self):
        """When policy triggers, prompt_fn is called."""
        called_with = {}

        def capture_prompt(task, triggers):
            called_with["task"] = task
            called_with["triggers"] = triggers
            return "deny"

        policy = HITLPolicy(budget_threshold=0.5)
        interceptor = HITLInterceptor(
            policy=policy,
            prompt_fn=capture_prompt,
            budget_ratio_fn=lambda: 0.3,
        )
        task = {"id": "task-001", "goal": "Task", "attempts": 0}
        result = interceptor(task)
        assert result == "deny"
        assert called_with["task"]["id"] == "task-001"
        assert len(called_with["triggers"]) >= 1

    def test_deny_flow(self):
        interceptor = HITLInterceptor(
            policy=HITLPolicy(max_auto_attempts=0),
            prompt_fn=auto_deny,
        )
        task = {"id": "t", "goal": "Any task", "attempts": 1}
        result = interceptor(task)
        assert result == "deny"

    def test_approve_flow(self):
        interceptor = HITLInterceptor(
            policy=HITLPolicy(max_auto_attempts=0),
            prompt_fn=auto_approve,
        )
        task = {"id": "t", "goal": "Any task", "attempts": 1}
        result = interceptor(task)
        assert result == "approve"

    def test_decisions_audit_trail(self):
        interceptor = HITLInterceptor(
            policy=HITLPolicy(approve_all=True),
        )
        interceptor({"id": "t1", "goal": "A", "attempts": 0})
        interceptor({"id": "t2", "goal": "B", "attempts": 0})
        assert len(interceptor.decisions) == 2
        assert interceptor.decisions[0]["task_id"] == "t1"
        assert interceptor.decisions[0]["decision"] == "approve"
        assert interceptor.decisions[1]["task_id"] == "t2"

    def test_audit_trail_records_triggers(self):
        interceptor = HITLInterceptor(
            policy=HITLPolicy(max_auto_attempts=0),
            prompt_fn=auto_deny,
        )
        interceptor({"id": "t1", "goal": "Task", "attempts": 1})
        assert len(interceptor.decisions) == 1
        assert "attempted" in interceptor.decisions[0]["triggers"][0]
        assert interceptor.decisions[0]["decision"] == "deny"

    def test_budget_ratio_fn_integration(self):
        budget_calls = []

        def track_budget():
            budget_calls.append(1)
            return 0.1

        interceptor = HITLInterceptor(
            policy=HITLPolicy(budget_threshold=0.2),
            prompt_fn=auto_approve,
            budget_ratio_fn=track_budget,
        )
        interceptor({"id": "t", "goal": "Task", "attempts": 0})
        assert len(budget_calls) == 1

    def test_no_budget_fn_skips_budget_check(self):
        """Without budget_ratio_fn, budget is assumed 1.0 (full)."""
        interceptor = HITLInterceptor(
            policy=HITLPolicy(budget_threshold=0.5),
            prompt_fn=auto_deny,  # would deny if triggered
        )
        # No budget_ratio_fn → ratio defaults to 1.0 → no trigger
        task = {"id": "t", "goal": "Safe task", "attempts": 0}
        result = interceptor(task)
        assert result == "approve"

    def test_policy_property(self):
        policy = HITLPolicy(budget_threshold=0.1)
        interceptor = HITLInterceptor(policy=policy)
        assert interceptor.policy.budget_threshold == 0.1


# ── OrchestratorGraph Integration ─────────────────────────────


class TestOrchestratorHITLIntegration:
    """Verify approval_callback works within OrchestratorGraph."""

    def _make_orchestrator(self, callback=None):
        """Build orchestrator with scripted provider (no real LLM)."""
        import json

        from kappa.budget.gate import BudgetGate
        from kappa.config import (
            AgentConfig,
            BudgetConfig,
            ExecutionConfig,
            OrchestratorConfig,
        )
        from kappa.graph.orchestrator import OrchestratorGraph
        from kappa.sandbox.executor import SandboxResult

        # Scripted LLM responses
        responses = iter([
            # Planner: two independent tasks
            json.dumps({
                "tasks": [
                    {"id": "task-001", "goal": "print hello", "depends_on": []},
                    {"id": "task-002", "goal": "delete database", "depends_on": []},
                ]
            }),
            # Plan reviewer: approve
            json.dumps({"approved": True, "critique": "", "score": 0.9}),
            # Worker 1: coder output (task-001)
            '<think>ok</think>\n<action>\nprint("hello")\n</action>',
            # Worker 2: coder output (task-002) — might not run if denied
            '<think>ok</think>\n<action>\nprint("deleted")\n</action>',
            # Reviewer for task-001
            json.dumps({"approved": True, "think": "good", "critique": "", "score": 0.9}),
            # Reviewer for task-002 (if it ran)
            json.dumps({"approved": True, "think": "good", "critique": "", "score": 0.9}),
        ])

        class ScriptedProvider:
            def call(self, *, messages, model, max_tokens=4096):
                from kappa.budget.gate import LLMResponse
                return LLMResponse(
                    content=next(responses),
                    prompt_tokens=10,
                    completion_tokens=10,
                    model="test",
                )

        budget_config = BudgetConfig(max_total_tokens=100_000, max_cost_usd=10.0)
        gate = BudgetGate(provider=ScriptedProvider(), budget_config=budget_config)

        class FakeExecutor:
            def __init__(self):
                self._config = ExecutionConfig(
                    timeout_seconds=5, workspace_dir=None, output_dir=None
                )

            @property
            def config(self):
                return self._config

            def execute(self, code: str) -> SandboxResult:
                import subprocess
                import sys
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True, text=True, timeout=5,
                )
                return SandboxResult(
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

        from kappa.config import MetaPromptConfig

        return OrchestratorGraph(
            gate=gate,
            sandbox=FakeExecutor(),
            config=AgentConfig(max_self_heal_retries=1),
            orchestrator_config=OrchestratorConfig(max_retries_per_task=3),
            meta_prompt_config=MetaPromptConfig(
                skip_interview=True, skip_plan_approval=True,
            ),
            approval_callback=callback,
        )

    def test_no_callback_all_tasks_run(self):
        """Without callback, all tasks execute (backward compatible)."""
        orch = self._make_orchestrator(callback=None)
        result = orch.run("Do two things")
        # Both tasks should have been executed
        completed = result.get("completed", [])
        assert len(completed) == 2

    def test_deny_all_callback(self):
        """Deny-all callback prevents all task execution."""
        def deny_all(task, context):
            return "deny"

        orch = self._make_orchestrator(callback=deny_all)
        result = orch.run("Do two things")
        # No tasks completed since all denied
        assert result["global_status"] == "failed"

    def test_selective_deny(self):
        """Callback can selectively deny specific tasks."""
        def deny_destructive(task, context):
            if "delete" in task.get("goal", "").lower():
                return "deny"
            return "approve"

        orch = self._make_orchestrator(callback=deny_destructive)
        result = orch.run("Do two things")
        completed = result.get("completed", [])
        # task-001 (print hello) approved, task-002 (delete database) denied
        assert "task-001" in completed

    def test_hitl_interceptor_as_callback(self):
        """Full HITLInterceptor works as approval_callback."""
        interceptor = HITLInterceptor(
            policy=HITLPolicy(),
            prompt_fn=auto_approve,
        )
        orch = self._make_orchestrator(callback=interceptor)
        result = orch.run("Do two things")
        completed = result.get("completed", [])
        assert len(completed) >= 1  # At least safe tasks approved
        assert len(interceptor.decisions) >= 1
