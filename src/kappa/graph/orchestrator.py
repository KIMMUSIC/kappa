"""Centralized Orchestrator Super-Graph for multi-agent task decomposition.

Decomposes a high-level goal into a DAG of subtasks via a Planner,
dispatches them to SelfHealingGraph workers (through SessionLane for
resource safety), and validates results through a Quality Gate
(Reviewer) before producing final output.

The existing ``SelfHealingGraph`` is consumed as a **black-box worker**
— its code is never modified.  The orchestrator wraps it as a
Sub-Graph inside a higher-level Super-Graph.

Flow::

    planner → plan_reviewer ─┬→ dispatcher → [Workers…] → reviewer ─┬→ finalizer → END
       ↑                     │       ↑                               │
       └── plan rejected ────┘       └──── rejected (critique) ──────┘
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from kappa.budget.gate import BudgetGate
from kappa.config import AgentConfig, OrchestratorConfig
from kappa.defense.semantic import SemanticLoopDetector
from kappa.graph.graph import SelfHealingGraph
from kappa.infra.session_lane import SyncSessionLane
from kappa.sandbox.executor import SandboxExecutor
from kappa.telemetry.manager import TelemetryManager, TrajectoryRecord
from kappa.tools.registry import ToolRegistry

# ── Prompts ─────────────────────────────────────────────────────────

PLANNER_PROMPT = """\
You are a task decomposition planner.  Break the given goal into \
concrete, actionable subtasks that a code-generation agent can execute.

Each subtask will be executed by a Python code-generation agent that runs \
Python code in a sandbox. Therefore:
- Frame every goal as a concrete Python programming task.
- If the task involves creating files (HTML, CSS, JS, config, etc.), the goal \
MUST say "Write Python code that creates <filename> with ..." — not just \
"Create the HTML structure".
- The agent can only produce Python code, so each goal must be achievable \
by writing and executing a Python script.
- The sandbox has a /workspace directory mounted from the host. \
All file output MUST go to /workspace/ so it persists after execution.

Rules:
- Each subtask gets a unique id: "task-001", "task-002", etc.
- Use depends_on to express ordering (list of prerequisite task IDs).
- Independent tasks have empty depends_on (they can run in parallel).
- Generate between 2 and {max_subtasks} subtasks.
- Each subtask goal must be self-contained and precise.

Output ONLY valid JSON (no markdown, no commentary):
{{"tasks": [{{"id": "task-001", "goal": "...", "depends_on": []}}]}}

Goal: {goal}"""

REVIEWER_PROMPT = """\
You are a pragmatic quality reviewer.  Evaluate whether the worker \
output fulfils the subtask goal.

Subtask goal: {goal}
Worker status: {status}
Worker output:
{output}

Rules:
- If worker status is "success" and the generated code reasonably \
addresses the goal, APPROVE it.
- Only REJECT for functional problems: wrong logic, missing core \
requirements, or runtime errors.
- Do NOT reject for style, missing docstrings, or minor improvements.
- If workspace file contents are included, verify they match the subtask \
goal — REJECT if a file was overwritten with unrelated content.
- Score from 0.0 (terrible) to 1.0 (perfect).

Output ONLY valid JSON (no markdown, no commentary):
{{"approved": true, "think": "reasoning", "critique": "", "score": 0.95}}"""

PLAN_REVIEWER_PROMPT = """\
You are a plan reviewer.  Evaluate whether the proposed subtask \
decomposition is well-structured and achievable.

Original goal: {goal}

Proposed plan:
{plan}

Rules:
- Each subtask must be a concrete Python programming task achievable \
by a code-generation agent running in a sandboxed environment.
- Dependencies (depends_on) must be logically correct — no circular \
dependencies, no references to non-existent task IDs.
- The subtasks should collectively cover the full scope of the goal.
- Approve if the plan is reasonable and actionable.
- Reject ONLY for structural issues: vague goals, missing steps, \
wrong dependencies, or tasks that cannot be executed as Python code.

Output ONLY valid JSON (no markdown, no commentary):
{{"approved": true, "critique": "", "score": 0.9}}"""


# ── State schemas ───────────────────────────────────────────────────


class SubTask(TypedDict):
    """A single decomposed subtask managed by the orchestrator."""

    id: str
    goal: str
    depends_on: list[str]
    status: str  # pending | awaiting_review | completed | rejected
    result: dict | None
    critique: str
    attempts: int


class OrchestratorState(TypedDict):
    """State flowing through the orchestrator Super-Graph."""

    main_goal: str
    plan: list[SubTask]
    completed: list[str]
    rejected_count: int
    max_retries_per_task: int
    global_status: str  # planning | plan_review | dispatching | reviewing | done | failed
    final_output: str
    telemetry_records: list[dict]
    plan_critique: str
    plan_attempts: int


# ── Orchestrator ────────────────────────────────────────────────────


class OrchestratorGraph:
    """Hierarchical multi-agent orchestrator (Super-Graph).

    The Planner decomposes the goal into a DAG of subtasks.
    The Dispatcher runs ``SelfHealingGraph`` workers in parallel
    (through ``SyncSessionLane`` for resource safety).
    The Reviewer (Quality Gate) validates each result — approved
    results proceed; rejected ones loop back with critique.

    Args:
        gate: BudgetGate for all LLM calls (shared with workers).
        sandbox: SandboxExecutor for worker code execution.
        config: AgentConfig for worker configuration.
        orchestrator_config: Orchestrator-specific thresholds.
        registry: Optional ToolRegistry forwarded to workers.
        detector: Optional SemanticLoopDetector forwarded to workers.
        session_lane: Optional SyncSessionLane for resource serialisation.
        telemetry: Optional TelemetryManager for persistent JSONL logging.
    """

    def __init__(
        self,
        gate: BudgetGate,
        sandbox: SandboxExecutor,
        config: AgentConfig | None = None,
        orchestrator_config: OrchestratorConfig | None = None,
        registry: ToolRegistry | None = None,
        detector: SemanticLoopDetector | None = None,
        session_lane: SyncSessionLane | None = None,
        telemetry: TelemetryManager | None = None,
        *,
        approval_callback: Callable[[dict[str, Any], dict[str, Any] | None], str] | None = None,
    ) -> None:
        self._gate = gate
        self._sandbox = sandbox
        self._config = config or AgentConfig()
        self._orch_config = orchestrator_config or OrchestratorConfig()
        self._registry = registry
        self._detector = detector
        self._session_lane = session_lane
        self._telemetry = telemetry
        self._approval_callback = approval_callback
        self._app = self._build()

    # ── LLM helpers ─────────────────────────────────────────────────

    def _llm_call(self, prompt: str, model: str | None = None) -> str:
        """Send a single prompt to the LLM via BudgetGate."""
        messages = [{"role": "user", "content": prompt}]
        response = self._gate.call(
            messages=messages,
            model=model or self._config.model,
        )
        return response.content

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract JSON from LLM response, tolerating code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
        return None

    # ── Worker execution ────────────────────────────────────────────

    def _execute_subtask(self, subtask: SubTask) -> dict:
        """Run a single subtask through a fresh SelfHealingGraph worker."""
        goal = subtask["goal"]
        if subtask.get("critique"):
            goal = (
                f"{goal}\n\n"
                f"Your previous attempt was rejected:\n"
                f"{subtask['critique']}\n"
                f"Fix the issues and try again."
            )

        worker = SelfHealingGraph(
            gate=self._gate,
            sandbox=self._sandbox,
            config=self._config,
            registry=self._registry,
            detector=self._detector,
        )

        if self._session_lane:
            with self._session_lane.lane(subtask["id"]):
                return dict(worker.run(goal))
        return dict(worker.run(goal))

    # ── Dependency context ──────────────────────────────────────────

    @staticmethod
    def _enrich_with_deps(task: SubTask, plan: list[SubTask]) -> SubTask:
        """Return a copy of *task* whose goal includes predecessor outputs."""
        dep_parts: list[str] = []
        for dep_id in task["depends_on"]:
            dep_task = next(
                (t for t in plan if t["id"] == dep_id and t["status"] == "completed"),
                None,
            )
            if dep_task and dep_task.get("result"):
                sandbox = (dep_task["result"].get("sandbox_result") or {})
                stdout = sandbox.get("stdout", "").strip()
                parsed_code = dep_task["result"].get("parsed_code", "").strip()
                if stdout or parsed_code:
                    part = f"[{dep_id}] {dep_task['goal']}"
                    if parsed_code:
                        part += f"\nCode:\n{parsed_code[:500]}"
                    if stdout:
                        part += f"\nOutput:\n{stdout[:500]}"
                    dep_parts.append(part)
        if not dep_parts:
            return task
        enriched: SubTask = {**task}
        enriched["goal"] = (
            task["goal"]
            + "\n\n[Context from completed prerequisite tasks]\n"
            + "\n\n".join(dep_parts)
        )
        return enriched

    # ── Node implementations ────────────────────────────────────────

    def _planner_node(self, state: OrchestratorState) -> dict:
        """Decompose *main_goal* into a DAG of SubTasks via LLM."""
        prompt = PLANNER_PROMPT.format(
            goal=state["main_goal"],
            max_subtasks=self._orch_config.max_subtasks,
        )

        plan_critique = state.get("plan_critique", "")
        if plan_critique:
            prompt += (
                f"\n\nYour previous plan was rejected:\n"
                f"{plan_critique}\n"
                f"Revise the plan to address the issues."
            )

        raw = self._llm_call(prompt, model=self._orch_config.planner_model)
        parsed = self._parse_json(raw)

        if not parsed or "tasks" not in parsed:
            return {"plan": [], "global_status": "failed"}

        raw_tasks = parsed["tasks"][: self._orch_config.max_subtasks]
        plan: list[SubTask] = []
        for t in raw_tasks:
            plan.append(
                {
                    "id": t.get("id", f"task-{len(plan) + 1:03d}"),
                    "goal": t.get("goal", ""),
                    "depends_on": t.get("depends_on", []),
                    "status": "pending",
                    "result": None,
                    "critique": "",
                    "attempts": 0,
                }
            )

        return {
            "plan": plan,
            "global_status": "plan_review",
            "plan_attempts": state.get("plan_attempts", 0) + 1,
        }

    def _plan_reviewer_node(self, state: OrchestratorState) -> dict:
        """Validate the planner's decomposition before dispatching."""
        plan_text = "\n".join(
            f"  {t['id']}: {t['goal']} (depends_on={t['depends_on']})"
            for t in state["plan"]
        )
        prompt = PLAN_REVIEWER_PROMPT.format(
            goal=state["main_goal"],
            plan=plan_text,
        )
        raw = self._llm_call(prompt, model=self._orch_config.reviewer_model)
        review = self._parse_json(raw)

        if review is None:
            review = {"approved": False, "critique": "Plan review could not be parsed.", "score": 0.0}

        if review.get("approved", False):
            return {"global_status": "dispatching", "plan_critique": ""}

        critique = review.get("critique", "Plan rejected without specific feedback.")
        return {"global_status": "plan_rejected", "plan_critique": critique}

    def _dispatcher_node(self, state: OrchestratorState) -> dict:
        """Find ready subtasks and execute them via workers."""
        plan = state["plan"]
        completed_ids = set(state["completed"])

        ready = [
            t
            for t in plan
            if t["status"] in ("pending", "rejected")
            and all(d in completed_ids for d in t["depends_on"])
        ]

        if not ready:
            return {"global_status": "failed"}

        # ── Phase 4: HITL approval gate ────────────────────────────
        if self._approval_callback:
            approved_ready: list[SubTask] = []
            denied_plan_updates: dict[str, SubTask] = {}
            for task in ready:
                decision = self._approval_callback(task, None)
                if decision == "deny":
                    denied_plan_updates[task["id"]] = {
                        **task,
                        "status": "failed",
                        "critique": "Operator denied execution",
                    }
                else:
                    approved_ready.append(task)
            ready = approved_ready
            if denied_plan_updates:
                plan = state["plan"]
                updated = [
                    denied_plan_updates.get(t["id"], t) for t in plan
                ]
                if not ready:
                    return {
                        "plan": updated,
                        "rejected_count": state["rejected_count"] + len(denied_plan_updates),
                        "global_status": "reviewing",
                    }
                # Merge denied updates into plan for later
                state = {**state, "plan": updated}

        results: dict[str, dict] = {}
        max_w = min(self._orch_config.max_parallel_workers, len(ready))

        # Enrich each task with outputs from completed dependencies
        enriched_ready = [self._enrich_with_deps(t, plan) for t in ready]

        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futures = {
                pool.submit(self._execute_subtask, task): task["id"]
                for task in enriched_ready
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    results[task_id] = future.result()
                except Exception as exc:
                    results[task_id] = {
                        "status": "runtime_error",
                        "error_history": [str(exc)],
                        "sandbox_result": None,
                    }

        updated_plan: list[SubTask] = []
        for task in plan:
            if task["id"] in results:
                updated_plan.append(
                    {
                        "id": task["id"],
                        "goal": task["goal"],
                        "depends_on": task["depends_on"],
                        "status": "awaiting_review",
                        "result": results[task["id"]],
                        "critique": task["critique"],
                        "attempts": task["attempts"],
                    }
                )
            else:
                updated_plan.append(task)

        return {"plan": updated_plan, "global_status": "reviewing"}

    def _reviewer_node(self, state: OrchestratorState) -> dict:
        """Quality-gate: evaluate each awaiting_review subtask via LLM."""
        plan = state["plan"]
        completed = list(state["completed"])
        rejected_count = state["rejected_count"]
        telemetry = list(state["telemetry_records"])

        updated_plan: list[SubTask] = []
        for task in plan:
            if task["status"] != "awaiting_review":
                updated_plan.append(task)
                continue

            result = task.get("result") or {}
            worker_status = result.get("status", "unknown")
            sandbox = result.get("sandbox_result") or {}
            parsed_code = result.get("parsed_code", "")
            stdout = sandbox.get("stdout", "")
            stderr = sandbox.get("stderr", "")

            output_parts: list[str] = []
            if parsed_code:
                output_parts.append(f"[Generated Code]\n{parsed_code}")
            if stdout:
                output_parts.append(f"[Execution stdout]\n{stdout}")
            if stderr:
                output_parts.append(f"[Execution stderr]\n{stderr}")
            output_text = "\n\n".join(output_parts) if output_parts else "(no output)"

            # Include workspace file contents for verification
            if self._sandbox.config.workspace_dir:
                ws_path = Path(self._sandbox.config.workspace_dir).resolve()
                if ws_path.exists():
                    ws_parts: list[str] = []
                    for f in sorted(ws_path.iterdir()):
                        if f.is_file():
                            try:
                                content = f.read_text(errors="replace")[:2000]
                                ws_parts.append(f"--- {f.name} ---\n{content}")
                            except Exception:
                                ws_parts.append(f"--- {f.name} --- (unreadable)")
                    if ws_parts:
                        output_text += (
                            "\n\n[Workspace files on disk]\n"
                            + "\n\n".join(ws_parts)
                        )

            prompt = REVIEWER_PROMPT.format(
                goal=task["goal"],
                status=worker_status,
                output=output_text,
            )
            raw = self._llm_call(prompt, model=self._orch_config.reviewer_model)
            review = self._parse_json(raw)
            if review is None:
                review = {
                    "approved": False,
                    "think": "",
                    "critique": "Reviewer output could not be parsed.",
                    "score": 0.0,
                }

            approved = review.get("approved", False)
            think = review.get("think", "")
            critique = review.get("critique", "")
            score = float(review.get("score", 0.0))

            if approved:
                updated_plan.append(
                    {
                        "id": task["id"],
                        "goal": task["goal"],
                        "depends_on": task["depends_on"],
                        "status": "completed",
                        "result": task["result"],
                        "critique": "",
                        "attempts": task["attempts"],
                    }
                )
                completed.append(task["id"])
            else:
                new_attempts = task["attempts"] + 1
                if new_attempts >= state["max_retries_per_task"]:
                    task_status = "failed"
                else:
                    task_status = "rejected"
                updated_plan.append(
                    {
                        "id": task["id"],
                        "goal": task["goal"],
                        "depends_on": task["depends_on"],
                        "status": task_status,
                        "result": task["result"],
                        "critique": critique
                        or "Review failed: no specific feedback.",
                        "attempts": new_attempts,
                    }
                )
                rejected_count += 1

            record_dict = {
                "task_id": task["id"],
                "think": think,
                "critique": critique,
                "score": score,
                "outcome": "success" if approved else "rejected",
            }
            telemetry.append(record_dict)

            # Persist to JSONL via TelemetryManager (observer pattern)
            if self._telemetry:
                self._telemetry.record(
                    TrajectoryRecord(
                        task_id=task["id"],
                        worker_goal=task["goal"],
                        think=think,
                        critique=critique,
                        score=score,
                        outcome="success" if approved else "rejected",
                    )
                )

        return {
            "plan": updated_plan,
            "completed": completed,
            "rejected_count": rejected_count,
            "telemetry_records": telemetry,
            "global_status": "reviewing",
        }

    def _finalizer_node(self, state: OrchestratorState) -> dict:
        """Merge all completed subtask outputs into a single result."""
        parts: list[str] = []
        for task in sorted(state["plan"], key=lambda t: t["id"]):
            result = task.get("result") or {}
            sandbox = result.get("sandbox_result") or {}
            stdout = sandbox.get("stdout", "")
            parts.append(f"[{task['id']}] {task['goal']}\n{stdout}")

        return {"final_output": "\n\n".join(parts), "global_status": "done"}

    def _failed_node(self, _state: OrchestratorState) -> dict:
        """Terminal node for failed orchestration."""
        return {"global_status": "failed"}

    # ── Routing ─────────────────────────────────────────────────────

    def _route_after_plan(self, state: OrchestratorState) -> str:
        if state["global_status"] == "failed" or not state["plan"]:
            return "failed"
        return "plan_reviewer"

    def _route_after_plan_review(self, state: OrchestratorState) -> str:
        if state["global_status"] == "dispatching":
            return "dispatcher"
        if state.get("plan_attempts", 0) >= self._orch_config.max_plan_retries:
            return "failed"
        return "planner"

    def _route_after_dispatch(self, state: OrchestratorState) -> str:
        if state["global_status"] == "failed":
            return "failed"
        return "reviewer"

    def _route_after_review(self, state: OrchestratorState) -> str:
        if any(t["status"] == "failed" for t in state["plan"]):
            return "failed"
        if all(t["status"] == "completed" for t in state["plan"]):
            return "finalizer"
        return "dispatcher"

    # ── Graph assembly ──────────────────────────────────────────────

    def _build(self):
        graph = StateGraph(OrchestratorState)

        graph.add_node("planner", self._planner_node)
        graph.add_node("plan_reviewer", self._plan_reviewer_node)
        graph.add_node("dispatcher", self._dispatcher_node)
        graph.add_node("reviewer", self._reviewer_node)
        graph.add_node("finalizer", self._finalizer_node)
        graph.add_node("failed", self._failed_node)

        graph.set_entry_point("planner")
        graph.add_conditional_edges("planner", self._route_after_plan)
        graph.add_conditional_edges("plan_reviewer", self._route_after_plan_review)
        graph.add_conditional_edges("dispatcher", self._route_after_dispatch)
        graph.add_conditional_edges("reviewer", self._route_after_review)
        graph.add_edge("finalizer", END)
        graph.add_edge("failed", END)

        return graph.compile()

    # ── Public API ──────────────────────────────────────────────────

    def _initial_state(self, goal: str) -> OrchestratorState:
        return {
            "main_goal": goal,
            "plan": [],
            "completed": [],
            "rejected_count": 0,
            "max_retries_per_task": self._orch_config.max_retries_per_task,
            "global_status": "planning",
            "final_output": "",
            "telemetry_records": [],
            "plan_critique": "",
            "plan_attempts": 0,
        }

    def run(self, goal: str) -> OrchestratorState:
        """Execute the orchestrator for a high-level goal.

        Returns the final ``OrchestratorState`` with results, telemetry
        records, and a merged ``final_output``.

        Raises:
            BudgetExceededException: If the shared budget is exhausted.
        """
        state = self._initial_state(goal)
        return self._app.invoke(state)

    def stream(self, goal: str):
        """Yield ``(node_name, state_update)`` for each orchestrator step."""
        state = self._initial_state(goal)
        yield from self._app.stream(state)
