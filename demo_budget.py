"""Budget Gate Interactive Demo CLI.

Claude Code 스타일의 대화형 인터페이스로 BudgetGate를 직접 체험합니다.
실제 Anthropic API를 호출하며, 매 응답마다 예산 현황을 실시간 표시합니다.

Usage:
    python demo_budget.py
    python demo_budget.py --max-tokens 3000 --max-cost 0.05
"""

from __future__ import annotations

import argparse
import sys

from kappa.config import BudgetConfig
from kappa.budget.gate import BudgetGate, AnthropicProvider
from kappa.exceptions import BudgetExceededException


# ── ANSI colors ───────────────────────────────────────────────────

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner(config: BudgetConfig) -> None:
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════╗
║         Kappa Budget Gate Demo CLI               ║
╚══════════════════════════════════════════════════╝{RESET}

{DIM}예산 한도: {config.max_total_tokens:,} tokens | ${config.max_cost_usd:.2f}
모델: claude-sonnet-4-20250514
종료: quit / exit / Ctrl+C{RESET}
""")


def print_budget_bar(gate: BudgetGate, config: BudgetConfig) -> None:
    t = gate.tracker
    token_pct = min(100, int(t.total_tokens / config.max_total_tokens * 100))
    cost_pct = min(100, int(t.estimated_cost_usd / config.max_cost_usd * 100))

    # color by usage level
    if token_pct < 50:
        bar_color = GREEN
    elif token_pct < 80:
        bar_color = YELLOW
    else:
        bar_color = RED

    filled = token_pct // 5
    bar = "█" * filled + "░" * (20 - filled)

    print(f"""
{DIM}┌─ Budget Status ──────────────────────────────────┐{RESET}
{DIM}│{RESET} Tokens: {bar_color}{bar} {token_pct:3d}%{RESET}  {t.total_tokens:,} / {config.max_total_tokens:,}
{DIM}│{RESET} Cost:   ${t.estimated_cost_usd:.4f} / ${config.max_cost_usd:.2f}  ({cost_pct}%)
{DIM}│{RESET} Calls:  {t.call_count}
{DIM}└──────────────────────────────────────────────────┘{RESET}
""")


def main() -> None:
    env_defaults = BudgetConfig()  # .env 값 반영

    parser = argparse.ArgumentParser(description="Budget Gate Interactive Demo")
    parser.add_argument("--max-tokens", type=int, default=None, help=f"토큰 한도 (default: .env → {env_defaults.max_total_tokens})")
    parser.add_argument("--max-cost", type=float, default=None, help=f"비용 한도 USD (default: .env → {env_defaults.max_cost_usd})")
    args = parser.parse_args()

    config = BudgetConfig(
        max_total_tokens=args.max_tokens if args.max_tokens is not None else env_defaults.max_total_tokens,
        max_cost_usd=args.max_cost if args.max_cost is not None else env_defaults.max_cost_usd,
    )

    try:
        provider = AnthropicProvider()
    except Exception as e:
        print(f"{RED}[ERROR] Anthropic 클라이언트 초기화 실패: {e}{RESET}")
        print(f"{DIM}.env 파일에 ANTHROPIC_API_KEY가 설정되어 있는지 확인하세요.{RESET}")
        sys.exit(1)

    gate = BudgetGate(provider=provider, budget_config=config)
    messages: list[dict] = []

    print_banner(config)
    print_budget_bar(gate, config)

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}세션 종료.{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{DIM}세션 종료.{RESET}")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            print(f"\n{DIM}[호출 중...]{RESET}")
            resp = gate.call(messages=messages, max_tokens=1024)

            messages.append({"role": "assistant", "content": resp.content})

            print(f"\n{CYAN}{BOLD}AI ▸ {RESET}{resp.content}")
            print(f"{DIM}  ↳ 이번 호출: +{resp.prompt_tokens} prompt / +{resp.completion_tokens} completion{RESET}")
            print_budget_bar(gate, config)

        except BudgetExceededException as e:
            print(f"""
{RED}{BOLD}╔══════════════════════════════════════════════════╗
║  ⚠  BUDGET EXCEEDED — CIRCUIT BREAKER TRIPPED   ║
╚══════════════════════════════════════════════════╝{RESET}

{RED}{e}{RESET}
{DIM}더 이상의 API 호출이 차단되었습니다.
토큰 사용: {e.tokens_used:,} | 비용: ${e.cost_used:.4f}{RESET}
""")
            break

    # final summary
    t = gate.tracker
    print(f"""{DIM}
━━━ Session Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  총 호출:      {t.call_count}회
  총 토큰:      {t.total_tokens:,}
  총 비용:      ${t.estimated_cost_usd:.4f}
  차단기 상태:  {"TRIPPED" if t.is_tripped else "정상"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}""")


if __name__ == "__main__":
    main()
