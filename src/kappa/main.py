"""Kappa Harness OS — Production Entrypoint.

Wires all real infrastructure components together and launches the
Rich CLI dashboard.  This is the single command a user runs::

    python -m kappa.main --goal "Build a REST API with tests"

Environment variables are loaded from ``.env`` (see ``.env.example``).
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# ── Bootstrap ─────────────────────────────────────────────────────

load_dotenv()

console = Console()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="kappa",
        description="Kappa Harness OS — Autonomous Self-Healing Agent",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="High-level goal to execute (skips interactive REPL).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override BUDGET_MAX_TOKENS from .env.",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Override BUDGET_MAX_COST_USD from .env.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip HITL prompts (auto-approve all subtasks).",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable JSONL trajectory recording.",
    )
    parser.add_argument(
        "--skip-interview",
        action="store_true",
        help="Skip the interview step even if ambiguity is high.",
    )
    parser.add_argument(
        "--skip-plan-approval",
        action="store_true",
        help="Skip interactive plan approval (use LLM reviewer instead).",
    )
    return parser.parse_args()


# ── Component Assembly ────────────────────────────────────────────


def build_orchestrator(args: argparse.Namespace):
    """Assemble all production components and return (orchestrator, tracker, budget_config).

    This is the Dependency Injection root — every real implementation
    is instantiated here and threaded through the object graph.
    """
    from kappa.budget.gate import AnthropicProvider, BudgetGate
    from kappa.config import (
        AgentConfig,
        BudgetConfig,
        MCPConfig,
        MetaPromptConfig,
        OrchestratorConfig,
        RAGConfig,
        ExecutionConfig,
        SemanticConfig,
        SessionLaneConfig,
        TelemetryConfig,
    )
    from kappa.defense.semantic import SemanticLoopDetector
    from kappa.exceptions import ExecutionError
    from kappa.graph.orchestrator import OrchestratorGraph
    from kappa.infra.session_lane import SyncSessionLane
    from kappa.sandbox.executor import HostExecutor, always_approve, auto_approve
    from kappa.telemetry.manager import TelemetryManager
    from kappa.tools.registry import ToolRegistry

    # ── 1. LLM Provider ───────────────────────────────────────────
    console.print("[dim]Connecting to Anthropic API...[/]")
    try:
        provider = AnthropicProvider()
    except Exception as e:
        console.print(f"[bold red]API Error:[/] {e}")
        console.print(
            "[yellow]Set ANTHROPIC_API_KEY in .env or environment.[/]"
        )
        sys.exit(1)

    # ── 2. Configs (env-driven with CLI overrides) ─────────────────
    #    BudgetConfig reads defaults from env vars; CLI flags override.
    env_budget = BudgetConfig()
    budget_config = BudgetConfig(
        max_total_tokens=args.max_tokens or env_budget.max_total_tokens,
        max_cost_usd=args.max_cost or env_budget.max_cost_usd,
    )
    exec_config = ExecutionConfig()
    agent_config = AgentConfig(budget=budget_config, execution=exec_config)
    orch_config = OrchestratorConfig()
    meta_config = MetaPromptConfig(
        skip_interview=getattr(args, "skip_interview", False),
        skip_plan_approval=getattr(args, "skip_plan_approval", False),
    )
    rag_config = RAGConfig()
    mcp_config = MCPConfig()
    semantic_config = SemanticConfig()
    session_lane_config = SessionLaneConfig()
    telemetry_config = TelemetryConfig(
        enabled=not args.no_telemetry,
    )

    # ── 3. Core Infrastructure ─────────────────────────────────────
    gate = BudgetGate(provider=provider, budget_config=budget_config)
    approval_fn = always_approve if exec_config.auto_approve else auto_approve
    executor = HostExecutor(config=exec_config, approval_fn=approval_fn)

    # ── 5. Tool Registry ───────────────────────────────────────────
    registry = ToolRegistry(tracker=gate.tracker)

    # ── 6. RAG Pipeline (optional — activates when docs are ingested)
    #    The InMemoryVectorStore ships as default; swap for ChromaDB/FAISS
    #    by implementing the VectorStore protocol.
    #    Embedding provider can be injected when available; for now the
    #    KnowledgeSearchTool is registered without a live embedder so the
    #    registry entry exists for the planner to discover.

    # ── 7. Semantic Loop Detector ──────────────────────────────────
    detector = SemanticLoopDetector(config=semantic_config)

    # ── 8. Session Lane (per-key serialisation) ────────────────────
    session_lane = SyncSessionLane(config=session_lane_config)

    # ── 9. Telemetry Manager ───────────────────────────────────────
    telemetry = TelemetryManager(config=telemetry_config)

    # ── 10. HITL Interceptor ───────────────────────────────────────
    from kappa.cli import create_hitl_interceptor
    from kappa.hitl import HITLPolicy

    interactive = not args.auto_approve
    interceptor = create_hitl_interceptor(
        tracker=gate.tracker,
        policy=HITLPolicy(approve_all=args.auto_approve),
        interactive=interactive,
    )

    # ── 11. Orchestrator Super-Graph ───────────────────────────────
    orchestrator = OrchestratorGraph(
        gate=gate,
        sandbox=executor,
        config=agent_config,
        orchestrator_config=orch_config,
        meta_prompt_config=meta_config,
        console=console,
        registry=registry,
        detector=detector,
        session_lane=session_lane,
        telemetry=telemetry,
        approval_callback=interceptor,
    )

    return orchestrator, gate.tracker, budget_config, gate, meta_config


# ── Execution Modes ───────────────────────────────────────────────


def run_single(args: argparse.Namespace) -> None:
    """Execute a single goal from --goal and exit."""
    from kappa.cli import run_dashboard
    from kappa.exceptions import BudgetExceededException, ExecutionError

    orchestrator, tracker, budget_config, gate, meta_config = build_orchestrator(args)

    try:
        final = run_dashboard(
            orchestrator,
            args.goal,
            tracker=tracker,
            max_tokens=budget_config.max_total_tokens,
            max_cost_usd=budget_config.max_cost_usd,
            meta_config=meta_config,
            gate=gate,
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
        else:
            console.print(
                Panel(
                    f"Status: {final.global_status}",
                    title="[bold red]Result: FAILED[/]",
                    border_style="red",
                )
            )

    except BudgetExceededException as e:
        console.print(f"\n[bold red]BUDGET EXCEEDED:[/] {e}")
        sys.exit(1)
    except ExecutionError as e:
        console.print(f"\n[bold red]EXECUTION ERROR:[/] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
        sys.exit(130)


def run_repl(args: argparse.Namespace) -> None:
    """Interactive REPL — enter goals one at a time."""
    from rich.prompt import Prompt

    from kappa.cli import run_dashboard
    from kappa.exceptions import BudgetExceededException, ExecutionError

    orchestrator, tracker, budget_config, gate, meta_config = build_orchestrator(args)

    console.print('[dim]Type "quit" to exit.[/]\n')

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

        try:
            final = run_dashboard(
                orchestrator,
                goal,
                tracker=tracker,
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
            else:
                console.print(
                    Panel(
                        f"Status: {final.global_status}",
                        title="[bold red]Result: FAILED[/]",
                        border_style="red",
                    )
                )
            console.print()

        except BudgetExceededException as e:
            console.print(f"\n[bold red]BUDGET EXCEEDED:[/] {e}\n")
            break
        except ExecutionError as e:
            console.print(f"\n[bold red]EXECUTION ERROR:[/] {e}\n")
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/]\n")


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    """Production entrypoint for the Kappa Harness OS."""
    args = _parse_args()

    console.print(
        Panel(
            "[bold]KAPPA Harness OS[/]\n"
            "[dim]Autonomous Self-Healing Agent with Deterministic Guardrails[/]",
            border_style="bright_blue",
        )
    )

    if args.goal:
        run_single(args)
    else:
        run_repl(args)


if __name__ == "__main__":
    main()
