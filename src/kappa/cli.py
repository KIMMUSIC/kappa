"""Interactive Rich CLI dashboard for the Kappa orchestrator.

Streams ``OrchestratorGraph`` execution in real-time with:
  - Live plan table (subtask status, scores, attempts)
  - Activity log (scrolling event feed)
  - Budget gauge (token / cost progress bar)
  - HITL approval prompts (pauses rendering for [Y/N] input)

Usage::

    python -m kappa.cli "Solve the FizzBuzz problem in three languages"

"""

from __future__ import annotations

import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from typing import Any

from kappa.budget.gate import BudgetGate
from kappa.budget.tracker import BudgetTracker
from kappa.hitl import HITLInterceptor, HITLPolicy, HITLTrigger


# ── Status icons ───────────────────────────────────────────────

_STATUS_ICONS = {
    "pending": "[dim]⏳ pending[/]",
    "running": "[cyan]⚙️  running[/]",
    "awaiting_review": "[yellow]🔍 reviewing[/]",
    "completed": "[green]✅ completed[/]",
    "rejected": "[red]❌ rejected[/]",
    "failed": "[red]💀 failed[/]",
}


# ── Dashboard State ────────────────────────────────────────────


@dataclass
class DashboardState:
    """Mutable state backing the dashboard layout."""

    goal: str = ""
    global_status: str = "idle"
    plan: list[dict[str, Any]] = field(default_factory=list)
    activity: deque[str] = field(default_factory=lambda: deque(maxlen=30))
    budget_used_tokens: int = 0
    budget_max_tokens: int = 100_000
    budget_cost_usd: float = 0.0
    budget_max_cost_usd: float = 5.0
    final_output: str = ""

    def update_from_step(self, node: str, state_update: dict[str, Any]) -> None:
        """Process a single graph step and update dashboard state."""
        now = datetime.now().strftime("%H:%M:%S")

        if "global_status" in state_update:
            self.global_status = state_update["global_status"]

        if "plan" in state_update:
            self.plan = state_update["plan"]

        if "final_output" in state_update:
            self.final_output = state_update["final_output"]

        # Node-specific activity messages
        if node == "planner":
            plan = state_update.get("plan", [])
            status = state_update.get("global_status", "")
            if plan:
                self.activity.append(
                    f"[dim]{now}[/] [cyan]PLANNER[/] Decomposed into "
                    f"[bold]{len(plan)}[/] subtasks"
                )
            elif status == "failed":
                self.activity.append(
                    f"[dim]{now}[/] [red]PLANNER[/] Failed to decompose goal"
                )

        elif node == "dispatcher":
            running = [t for t in self.plan if t.get("status") == "awaiting_review"]
            self.activity.append(
                f"[dim]{now}[/] [blue]DISPATCHER[/] Executed "
                f"[bold]{len(running)}[/] workers"
            )

        elif node == "reviewer":
            for task in self.plan:
                tid = task.get("id", "?")
                if task.get("status") == "completed":
                    self.activity.append(
                        f"[dim]{now}[/] [green]REVIEWER[/] {tid} "
                        f"[green]approved[/]"
                    )
                elif task.get("status") == "rejected":
                    critique = task.get("critique", "")[:60]
                    self.activity.append(
                        f"[dim]{now}[/] [red]REVIEWER[/] {tid} "
                        f"[red]rejected[/]: {critique}"
                    )

        elif node == "finalizer":
            self.activity.append(
                f"[dim]{now}[/] [green]FINALIZER[/] All tasks merged — "
                f"[bold green]done[/]"
            )

        elif node == "meta_prompter":
            score = state_update.get("ambiguity_score", 0)
            strategy = state_update.get("meta_strategy", "")
            self.activity.append(
                f"[dim]{now}[/] [magenta]META[/] Ambiguity: "
                f"{score:.0%} | Strategy: {strategy}"
            )

        elif node == "interview":
            self.activity.append(
                f"[dim]{now}[/] [yellow]INTERVIEW[/] "
                f"Awaiting user input..."
            )

        elif node == "plan_approval":
            self.activity.append(
                f"[dim]{now}[/] [yellow]APPROVAL[/] "
                f"Awaiting plan approval..."
            )

        elif node == "failed":
            self.activity.append(
                f"[dim]{now}[/] [red]FAILED[/] Orchestration terminated"
            )

    def update_budget(self, tracker: BudgetTracker) -> None:
        """Pull latest budget numbers from the tracker."""
        self.budget_used_tokens = tracker.total_tokens
        self.budget_cost_usd = tracker.estimated_cost_usd


# ── Layout Builders ────────────────────────────────────────────


def build_header(state: DashboardState) -> Panel:
    """Top panel: goal + global status."""
    status_color = {
        "idle": "dim",
        "meta_prompting": "magenta",
        "awaiting_interview": "yellow",
        "awaiting_plan_approval": "yellow",
        "planning": "cyan",
        "plan_review": "cyan",
        "dispatching": "blue",
        "reviewing": "yellow",
        "done": "green",
        "failed": "red",
    }.get(state.global_status, "white")

    text = Text()
    text.append("Goal: ", style="bold")
    text.append(state.goal or "(none)", style="white")
    text.append("  |  Status: ", style="dim")
    text.append(state.global_status.upper(), style=f"bold {status_color}")

    return Panel(text, title="[bold]KAPPA Orchestrator[/]", border_style="bright_blue")


def build_plan_table(state: DashboardState) -> Panel:
    """Middle panel: subtask table."""
    table = Table(expand=True, show_lines=False, pad_edge=False)
    table.add_column("ID", style="bold", width=10)
    table.add_column("Goal", ratio=3)
    table.add_column("Status", width=18, justify="center")
    table.add_column("Score", width=8, justify="center")
    table.add_column("Att.", width=5, justify="center")

    for task in state.plan:
        tid = task.get("id", "?")
        goal = task.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        status = task.get("status", "pending")
        status_text = _STATUS_ICONS.get(status, status)
        attempts = str(task.get("attempts", 0))

        # Extract score from telemetry if available
        result = task.get("result") or {}
        score = "—"
        if status == "completed":
            score = "[green]pass[/]"
        elif status == "rejected":
            score = "[red]fail[/]"

        table.add_row(tid, goal, status_text, score, attempts)

    if not state.plan:
        table.add_row("—", "[dim]Waiting for planner...[/]", "—", "—", "—")

    return Panel(table, title="[bold]Plan[/]", border_style="cyan")


def build_activity_log(state: DashboardState) -> Panel:
    """Bottom-left: scrolling event log."""
    lines = list(state.activity) if state.activity else ["[dim]No events yet...[/]"]
    # Show most recent at bottom
    content = "\n".join(lines[-15:])
    return Panel(content, title="[bold]Activity[/]", border_style="yellow")


def build_budget_panel(state: DashboardState) -> Panel:
    """Bottom-right: budget gauge."""
    token_pct = (
        (state.budget_used_tokens / state.budget_max_tokens * 100)
        if state.budget_max_tokens > 0
        else 0
    )
    cost_pct = (
        (state.budget_cost_usd / state.budget_max_cost_usd * 100)
        if state.budget_max_cost_usd > 0
        else 0
    )

    bar_width = 20

    def _bar(pct: float) -> str:
        filled = int(pct / 100 * bar_width)
        filled = min(filled, bar_width)
        empty = bar_width - filled
        color = "green" if pct < 60 else ("yellow" if pct < 80 else "red")
        return f"[{color}]{'█' * filled}[/][dim]{'░' * empty}[/]"

    text = Text.from_markup(
        f"Tokens: {_bar(token_pct)} "
        f"{state.budget_used_tokens:,}/{state.budget_max_tokens:,} "
        f"({token_pct:.0f}%)\n"
        f"  Cost: {_bar(cost_pct)} "
        f"${state.budget_cost_usd:.4f}/${state.budget_max_cost_usd:.2f} "
        f"({cost_pct:.0f}%)"
    )
    return Panel(text, title="[bold]Budget[/]", border_style="green")


def build_layout(state: DashboardState) -> Layout:
    """Compose the full dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="plan", ratio=2),
        Layout(name="bottom", ratio=1),
    )
    layout["bottom"].split_row(
        Layout(name="activity", ratio=3),
        Layout(name="budget", ratio=1),
    )

    layout["header"].update(build_header(state))
    layout["plan"].update(build_plan_table(state))
    layout["activity"].update(build_activity_log(state))
    layout["budget"].update(build_budget_panel(state))

    return layout


# ── HITL Prompt (Rich-based) ───────────────────────────────────


def rich_approval_prompt(
    task: dict[str, Any],
    triggers: list[HITLTrigger],
) -> str:
    """Interactive Rich prompt for HITL approval.

    Displays trigger reasons and asks for [Y/N] confirmation.
    """
    console = Console()
    console.print()
    console.print(
        Panel(
            "\n".join(
                f"  [yellow]⚠[/] {t.reason} [dim]({t.severity})[/]"
                for t in triggers
            ),
            title=f"[bold yellow]Approval Required: {task.get('id', '?')}[/]",
            subtitle=f"[dim]{task.get('goal', '')[:80]}[/]",
            border_style="yellow",
        )
    )

    answer = Prompt.ask(
        "[bold]Approve execution?[/]",
        choices=["y", "n"],
        default="y",
    )
    return "approve" if answer.lower() == "y" else "deny"


# ── Git Safety ────────────────────────────────────────────────────────────


def _git_snapshot() -> str | None:
    """Create a lightweight git snapshot before sandbox execution."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, check=True,
        )
        result = subprocess.run(
            ["git", "stash", "push", "-u", "-m", "kappa-sandbox-snapshot"],
            capture_output=True, text=True,
        )
        if "No local changes" in result.stdout:
            return None
        return "kappa-sandbox-snapshot"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_restore(console: Console) -> None:
    """Pop the kappa snapshot stash to restore pre-execution state."""
    try:
        result = subprocess.run(
            ["git", "stash", "list"], capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if "kappa-sandbox-snapshot" in line:
                ref = line.split(":")[0]
                subprocess.run(
                    ["git", "stash", "pop", ref],
                    capture_output=True, check=True,
                )
                console.print("[yellow]Git snapshot restored.[/]")
                return
    except Exception:
        console.print("[red]Failed to restore git snapshot.[/]")


def _git_show_diff(console: Console) -> None:
    """Show a summary of changes made by the sandbox execution."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True, text=True,
        )
        if result.stdout.strip():
            console.print(
                Panel(
                    result.stdout.strip(),
                    title="[bold cyan]Changes by Sandbox[/]",
                    border_style="cyan",
                )
            )
    except Exception:
        pass

# ── Main CLI Entry Point ───────────────────────────────────────


def create_hitl_interceptor(
    tracker: BudgetTracker,
    policy: HITLPolicy | None = None,
    interactive: bool = True,
) -> HITLInterceptor:
    """Factory for a wired-up HITL interceptor.

    Args:
        tracker: BudgetTracker to monitor remaining budget.
        policy: Override default policy.
        interactive: If True, use Rich prompts; if False, auto-approve.
    """
    from kappa.hitl import auto_approve as _auto_approve

    def _budget_ratio() -> float:
        max_tokens = tracker._config.max_total_tokens
        if max_tokens <= 0:
            return 1.0
        return tracker.remaining_tokens / max_tokens

    return HITLInterceptor(
        policy=policy or HITLPolicy(),
        prompt_fn=rich_approval_prompt if interactive else _auto_approve,
        budget_ratio_fn=_budget_ratio,
    )


def show_plan_approval(
    console: Console,
    plan: list[dict[str, Any]],
    goal: str,
) -> tuple[str, str]:
    """Display plan and collect user approval decision.

    Called from ``run_dashboard`` when the graph emits
    ``"awaiting_plan_approval"`` sentinel. Must only be called
    when Rich ``Live`` rendering is paused.

    Returns:
        (choice, feedback) where choice is "Y", "N", or "M"
        and feedback is the user's modification request (empty for Y/N).
    """
    console.print()
    table = Table(
        title="[bold]PLAN APPROVAL[/] 프로젝트 실행 계획서",
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("ID", style="bold")
    table.add_column("Task Goal", ratio=3)
    table.add_column("Depends On", style="dim")

    for task in plan:
        deps = ", ".join(task.get("depends_on", [])) or "-"
        table.add_row(task.get("id", "?"), task.get("goal", ""), deps)

    console.print(table)
    console.print()

    choice = Prompt.ask(
        "[bold]이 계획을 승인하십니까?[/]",
        choices=["Y", "N", "M"],
        default="Y",
    )

    feedback = ""
    if choice == "M":
        feedback = Prompt.ask("[bold]수정 요청 내용을 입력하세요[/]")

    return choice, feedback


def run_dashboard(
    orchestrator: Any,
    goal: str,
    tracker: BudgetTracker | None = None,
    max_tokens: int = 100_000,
    max_cost_usd: float = 5.0,
    *,
    meta_config: Any | None = None,
    gate: Any | None = None,
    model: str | None = None,
) -> DashboardState:
    """Stream orchestrator execution through the Rich dashboard.

    Supports sentinel-based protocol: when the graph emits
    ``"awaiting_interview"`` or ``"awaiting_plan_approval"``,
    the Live display pauses, interactive I/O is performed, and
    the graph resumes from the checkpointed state.

    Args:
        orchestrator: ``OrchestratorGraph`` instance.
        goal: High-level goal string.
        tracker: Optional BudgetTracker for live budget display.
        max_tokens: Budget ceiling for display.
        max_cost_usd: Cost ceiling for display.
        meta_config: MetaPromptConfig for interview settings.
        gate: BudgetGate for interview LLM calls.
        model: LLM model for interview synthesis.

    Returns:
        Final ``DashboardState`` after orchestration completes.
    """
    console = Console()
    config = {"configurable": {"thread_id": f"kappa-{id(orchestrator)}"}}
    state = DashboardState(
        goal=goal,
        global_status="planning",
        budget_max_tokens=max_tokens,
        budget_max_cost_usd=max_cost_usd,
    )
    accumulated: dict[str, Any] = {}

    with Live(
        build_layout(state),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        stream_iter = orchestrator.stream(goal, config=config)

        while True:
            try:
                step = next(stream_iter)
            except StopIteration:
                break

            for node, update in step.items():
                state.update_from_step(node, update)
                accumulated.update(update)
                if tracker:
                    state.update_budget(tracker)

                gs = update.get("global_status", "")

                if gs == "awaiting_interview" and gate and meta_config:
                    from kappa.graph.interview import run_interview

                    live.stop()
                    try:
                        result = run_interview(
                            console=console,
                            goal=goal,
                            gaps=accumulated.get("gaps", []),
                            max_questions=meta_config.max_interview_questions,
                            gate=gate,
                            model=model or "claude-sonnet-4-20250514",
                        )
                        orchestrator.update_state(config, {
                            "enhanced_goal": result["golden_goal"],
                            "interview_result": dict(result),
                            "global_status": "planning",
                        })
                    except (KeyboardInterrupt, Exception):
                        orchestrator.update_state(config, {
                            "global_status": "failed",
                        })
                    live.start()
                    stream_iter = orchestrator.resume_stream(config=config)

                elif gs == "awaiting_plan_approval":
                    live.stop()
                    choice, feedback = show_plan_approval(
                        console=console,
                        plan=state.plan,
                        goal=accumulated.get("enhanced_goal", goal),
                    )
                    if choice == "Y":
                        patch = {"global_status": "dispatching", "plan_critique": ""}
                    elif choice == "N":
                        patch = {"global_status": "failed"}
                    else:
                        patch = {
                            "global_status": "plan_rejected",
                            "plan_critique": feedback,
                            "user_plan_feedback": feedback,
                        }
                    orchestrator.update_state(config, patch)
                    live.start()
                    stream_iter = orchestrator.resume_stream(config=config)

            live.update(build_layout(state))

    return state


def main() -> None:
    """CLI entry point for the Kappa orchestrator dashboard."""
    from kappa.budget.gate import AnthropicProvider, BudgetGate
    from kappa.config import (
        AgentConfig,
        BudgetConfig,
        ExecutionConfig,
        OrchestratorConfig,
    )
    from kappa.exceptions import BudgetExceededException, ExecutionError
    from kappa.graph.orchestrator import OrchestratorGraph
    from kappa.sandbox.executor import HostExecutor, always_approve

    console = Console()

    console.print(
        Panel(
            "[bold]KAPPA Orchestrator Dashboard[/]\n"
            "[dim]Phase 5: Host Execution + HITL[/]",
            border_style="bright_blue",
        )
    )

    # Setup infrastructure
    try:
        provider = AnthropicProvider()
    except Exception as e:
        console.print(f"[red]API key error:[/] {e}")
        console.print("Set ANTHROPIC_API_KEY in .env or environment.")
        sys.exit(1)

    budget_config = BudgetConfig(max_total_tokens=100_000, max_cost_usd=5.00)
    exec_config = ExecutionConfig(timeout_seconds=15)
    agent_config = AgentConfig(max_self_heal_retries=3)
    orch_config = OrchestratorConfig(max_rejections=3, max_subtasks=5)

    gate = BudgetGate(provider=provider, budget_config=budget_config)
    executor = HostExecutor(config=exec_config, approval_fn=always_approve)

    # Create HITL interceptor
    interceptor = create_hitl_interceptor(gate.tracker, interactive=True)

    orchestrator = OrchestratorGraph(
        gate=gate,
        sandbox=executor,
        config=agent_config,
        orchestrator_config=orch_config,
        approval_callback=interceptor,
    )

    console.print(
        f"[green]Ready.[/]  Budget: {budget_config.max_total_tokens:,} tokens / "
        f"${budget_config.max_cost_usd}"
    )
    console.print('[dim]Type "quit" to exit.[/]\n')

    # REPL
    while True:
        try:
            goal = Prompt.ask("[bold]Goal[/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break

        goal = goal.strip()
        if not goal:
            continue
        if goal.lower() in ("quit", "exit", "q"):
            console.print("[dim]Bye.[/]")
            break

        # Git safety snapshot
        snapshot = _git_snapshot()
        if snapshot:
            console.print("[dim]Git snapshot created.[/]")

        try:
            final = run_dashboard(
                orchestrator,
                goal,
                tracker=gate.tracker,
                max_tokens=budget_config.max_total_tokens,
                max_cost_usd=budget_config.max_cost_usd,
            )

            console.print()
            if final.global_status == "done":
                console.print(
                    Panel(
                        final.final_output or "[dim]No output[/]",
                        title="[bold green]Result: SUCCESS[/]",
                        border_style="green",
                    )
                )
                _git_show_diff(console)
            else:
                console.print(
                    Panel(
                        f"Status: {final.global_status}",
                        title="[bold red]Result: FAILED[/]",
                        border_style="red",
                    )
                )
                if snapshot:
                    answer = Prompt.ask(
                        "[yellow]Restore files to pre-execution state?[/]",
                        choices=["y", "n"],
                        default="y",
                    )
                    if answer.lower() == "y":
                        _git_restore(console)
            console.print()

        except BudgetExceededException as e:
            console.print(f"\n[bold red]BUDGET EXCEEDED:[/] {e}\n")
            break
        except ExecutionError as e:
            console.print(f"\n[bold red]EXECUTION ERROR:[/] {e}\n")
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/]\n")


if __name__ == "__main__":
    main()
