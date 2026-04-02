"""Interactive self-healing loop CLI.

Run:  python interactive.py

Requirements:
  - ANTHROPIC_API_KEY set in .env or environment
  - Docker Desktop running
"""

from __future__ import annotations

import sys
import textwrap

from kappa.budget.gate import BudgetGate, AnthropicProvider
from kappa.config import AgentConfig, BudgetConfig, SandboxConfig
from kappa.exceptions import BudgetExceededException, SandboxExecutionError
from kappa.graph.graph import SelfHealingGraph
from kappa.sandbox.executor import DockerRuntime, SandboxExecutor

# ── Display helpers ─────────────────────────────────────────────

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner(label: str, color: str, content: str, max_lines: int = 0) -> None:
    print(f"\n{color}{BOLD}[{label}]{RESET}")
    lines = content.rstrip().split("\n")
    if max_lines and len(lines) > max_lines:
        for line in lines[:max_lines]:
            print(f"  {DIM}{line}{RESET}")
        print(f"  {DIM}... ({len(lines) - max_lines} more lines){RESET}")
    else:
        for line in lines:
            print(f"  {DIM}{line}{RESET}")


def separator() -> None:
    print(f"{DIM}{'─' * 60}{RESET}")


# ── Main ────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{BOLD}=== Kappa Self-Healing Loop (Interactive) ==={RESET}")
    print(f"{DIM}Phase 1: coder -> parser -> linter -> sandbox -> heal{RESET}")
    print(f"{DIM}Type 'quit' to exit.{RESET}\n")

    # Setup
    try:
        provider = AnthropicProvider()
    except Exception as e:
        print(f"{RED}API key error: {e}{RESET}")
        print(f"Set ANTHROPIC_API_KEY in .env or environment.")
        sys.exit(1)

    try:
        runtime = DockerRuntime()
    except SandboxExecutionError as e:
        print(f"{RED}Docker error: {e}{RESET}")
        print(f"Start Docker Desktop first.")
        sys.exit(1)

    budget_config = BudgetConfig(max_total_tokens=50_000, max_cost_usd=1.00)
    sandbox_config = SandboxConfig(timeout_seconds=10, memory_limit_mb=128)
    agent_config = AgentConfig(max_self_heal_retries=3)

    gate = BudgetGate(provider=provider, budget_config=budget_config)
    sandbox = SandboxExecutor(runtime=runtime, config=sandbox_config)
    graph = SelfHealingGraph(gate=gate, sandbox=sandbox, config=agent_config)

    print(f"{GREEN}Ready.{RESET}  Budget: {budget_config.max_total_tokens} tokens / ${budget_config.max_cost_usd}")
    print(f"         Sandbox: {sandbox_config.timeout_seconds}s timeout, {sandbox_config.memory_limit_mb}MB mem")
    print(f"         Retries: {agent_config.max_self_heal_retries}\n")

    # REPL
    while True:
        try:
            goal = input(f"{BOLD}Goal > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Bye.{RESET}")
            break

        if not goal:
            continue
        if goal.lower() in ("quit", "exit", "q"):
            print(f"{DIM}Bye.{RESET}")
            break

        separator()
        print(f"{MAGENTA}{BOLD}Goal:{RESET} {goal}")
        separator()

        try:
            final_state = None
            for step in graph.stream(goal):
                for node_name, state_update in step.items():
                    final_state = state_update

                    if node_name == "coder":
                        attempt = state_update.get("attempt", "?")
                        max_att = agent_config.max_self_heal_retries
                        print(f"\n{CYAN}{BOLD}[CODER] Attempt {attempt}/{max_att}{RESET}")
                        llm_out = state_update.get("llm_output", "")
                        banner("LLM Output", CYAN, llm_out, max_lines=20)

                    elif node_name == "parser":
                        status = state_update.get("status", "")
                        code = state_update.get("parsed_code", "")
                        if status == "parse_error":
                            errs = state_update.get("error_history", [])
                            last = errs[-1] if errs else "unknown"
                            banner("PARSER FAIL", RED, last)
                        else:
                            banner("PARSER OK -> code extracted", GREEN, code)

                    elif node_name == "linter":
                        status = state_update.get("status", "")
                        if status == "lint_error":
                            errs = state_update.get("error_history", [])
                            last = errs[-1] if errs else "unknown"
                            banner("LINTER FAIL", YELLOW, last)
                        else:
                            print(f"\n{GREEN}{BOLD}[LINTER OK]{RESET} {DIM}Syntax valid{RESET}")

                    elif node_name == "sandbox":
                        sr = state_update.get("sandbox_result", {})
                        ec = sr.get("exit_code", -1)
                        if ec == 0:
                            banner("SANDBOX OK (exit_code=0)", GREEN, sr.get("stdout", ""))
                        else:
                            timed = " [TIMED OUT]" if sr.get("timed_out") else ""
                            banner(
                                f"SANDBOX FAIL (exit_code={ec}){timed}",
                                RED,
                                sr.get("stderr", ""),
                                max_lines=10,
                            )

            # Final result
            separator()
            if final_state and final_state.get("status") == "success":
                print(f"\n{GREEN}{BOLD}RESULT: SUCCESS{RESET}")
                sr = final_state.get("sandbox_result", {})
                if sr.get("stdout"):
                    print(f"{GREEN}stdout:{RESET} {sr['stdout'].rstrip()}")
            else:
                status = final_state.get("status", "unknown") if final_state else "unknown"
                print(f"\n{RED}{BOLD}RESULT: FAILED ({status}){RESET}")
                errs = final_state.get("error_history", []) if final_state else []
                if errs:
                    print(f"{RED}Last error:{RESET} {errs[-1][:200]}")

            # Budget status
            t = gate.tracker
            print(f"\n{DIM}Tokens used: {t.total_tokens}/{budget_config.max_total_tokens} | "
                  f"Cost: ${t.estimated_cost_usd:.4f}/${budget_config.max_cost_usd}{RESET}")
            separator()
            print()

        except BudgetExceededException as e:
            print(f"\n{RED}{BOLD}BUDGET EXCEEDED: {e}{RESET}\n")
            break
        except SandboxExecutionError as e:
            print(f"\n{RED}{BOLD}SANDBOX ERROR: {e}{RESET}\n")
        except KeyboardInterrupt:
            print(f"\n{DIM}Interrupted.{RESET}\n")


if __name__ == "__main__":
    main()
