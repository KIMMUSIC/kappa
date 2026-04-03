"""Phase 3 Interactive Demo — 오케스트레이터 파이프라인을 직접 체험합니다.

사용자가 목표를 입력하면 Planner → Dispatcher → Reviewer → Finalizer
전 과정을 단계별로 시각화합니다.

Usage:
    python demo_phase3.py
"""

import json
import sys
import os
import time
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── 색상 헬퍼 ──────────────────────────────────────────────────────

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(title: str, color: str = CYAN) -> None:
    w = 62
    print(f"\n{color}{'━' * w}")
    print(f"  {BOLD}{title}{RESET}{color}")
    print(f"{'━' * w}{RESET}\n")


def step(label: str, detail: str = "") -> None:
    print(f"  {YELLOW}▶{RESET} {BOLD}{label}{RESET}  {DIM}{detail}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def pause() -> None:
    input(f"\n  {DIM}[Enter를 눌러 다음 단계로 →]{RESET} ")


def type_effect(text: str, delay: float = 0.008) -> None:
    """타이핑 효과로 텍스트 출력."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# ── Mock 구현체들 ──────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass(frozen=True)
class _LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    stop_reason: str | None = None


class InteractiveProvider:
    """사용자가 직접 Planner/Reviewer 역할을 하거나 자동 응답 모드를 선택."""

    def __init__(self, mode: str = "auto"):
        self.mode = mode  # "auto" | "manual"
        self._call_count = 0
        self._goal = ""
        self._pending_tasks: list[dict] = []

    def set_goal(self, goal: str) -> None:
        self._goal = goal

    def set_pending_tasks(self, tasks: list[dict]) -> None:
        self._pending_tasks = tasks

    def call(self, *, messages: list[dict], model: str, max_tokens: int = 4096) -> _LLMResponse:
        self._call_count += 1
        prompt_text = messages[-1]["content"] if messages else ""

        # Planner 호출 감지
        if "task decomposition planner" in prompt_text:
            return self._handle_planner(prompt_text, model)

        # Reviewer 호출 감지
        if "strict quality reviewer" in prompt_text:
            return self._handle_reviewer(prompt_text, model)

        # Worker (coder) 호출 — 간단한 코드 생성
        return self._handle_worker(prompt_text, model)

    def _handle_planner(self, prompt: str, model: str) -> _LLMResponse:
        if self.mode == "manual":
            return self._manual_planner(prompt, model)
        return self._auto_planner(prompt, model)

    def _handle_reviewer(self, prompt: str, model: str) -> _LLMResponse:
        if self.mode == "manual":
            return self._manual_reviewer(prompt, model)
        return self._auto_reviewer(prompt, model)

    def _handle_worker(self, prompt: str, model: str) -> _LLMResponse:
        # Worker — 목표 키워드를 분석해 맞춤형 코드 생성
        goal = self._extract_goal_from_prompt(prompt)
        think, code = self._generate_code_for_goal(goal)
        response = (
            f"<think>\n{think}\n</think>\n"
            f"<action>\n{code}\n</action>"
        )
        return _LLMResponse(
            content=response,
            prompt_tokens=100,
            completion_tokens=len(code) // 4,
            model=model,
            stop_reason="end_turn",
        )

    def _extract_goal_from_prompt(self, prompt: str) -> str:
        """프롬프트에서 목표 문장 추출."""
        for line in prompt.split("\n"):
            if "Goal:" in line or "goal:" in line:
                return line.split(":", 1)[-1].strip()
            if "목표" in line:
                return line.strip()
        return self._goal or prompt[:100]

    def _generate_code_for_goal(self, goal: str) -> tuple[str, str]:
        """목표 키워드를 분석해 실제 동작하는 Python 코드 생성."""
        g = goal.lower()

        # 분석/설계 관련 — 최우선 매칭
        if ("분석" in g or "설계" in g or "요구" in g) and ("design" in g or "분석" in g or "설계" in g or "요구" in g):
            return (
                "요구사항을 분석하고 설계 문서를 생성합니다.",
                textwrap.dedent("""\
                    requirements = {
                        "기능 요구사항": ["입력 처리", "핵심 로직 구현", "결과 출력"],
                        "비기능 요구사항": ["성능: O(n log n) 이하", "메모리: 256MB 이내"],
                        "제약 조건": ["Python 3.11+", "외부 라이브러리 미사용"],
                    }
                    print("=== 요구사항 분석 결과 ===")
                    for category, items in requirements.items():
                        print(f"\\n[{category}]")
                        for item in items:
                            print(f"  - {item}")
                    print("\\n설계 완료: 구현 준비 상태")""")
            )

        # 검증/테스트 관련 — 두번째 우선
        if "검증" in g or "테스트" in g or "verify" in g:
            return (
                "이전 단계의 결과를 검증합니다. 간단한 assert 기반 테스트를 수행합니다.",
                textwrap.dedent("""\
                    tests_passed = 0
                    tests_total = 0

                    def check(name, condition):
                        global tests_passed, tests_total
                        tests_total += 1
                        if condition:
                            tests_passed += 1
                            print(f"  PASS: {name}")
                        else:
                            print(f"  FAIL: {name}")

                    check("기본 동작 확인", True)
                    check("결과 타입 확인", isinstance(42, int))
                    check("경계값 테스트", 0 == 0)
                    print(f"\\n결과: {tests_passed}/{tests_total} 통과")""")
            )

        # 피보나치
        if "피보나치" in g or "fibonacci" in g or "fib" in g:
            return (
                "피보나치 수열을 계산합니다. 동적 프로그래밍으로 효율적으로 구현합니다.",
                textwrap.dedent("""\
                    def fibonacci(n):
                        a, b = 0, 1
                        result = []
                        for _ in range(n):
                            result.append(a)
                            a, b = b, a + b
                        return result

                    fib = fibonacci(10)
                    print(f"피보나치 수열 (10개): {fib}")
                    print(f"합계: {sum(fib)}")""")
            )

        # 정렬
        if "정렬" in g or "sort" in g:
            return (
                "정렬 알고리즘을 구현합니다. 퀵소트를 선택합니다.",
                textwrap.dedent("""\
                    def quicksort(arr):
                        if len(arr) <= 1:
                            return arr
                        pivot = arr[len(arr) // 2]
                        left = [x for x in arr if x < pivot]
                        mid = [x for x in arr if x == pivot]
                        right = [x for x in arr if x > pivot]
                        return quicksort(left) + mid + quicksort(right)

                    data = [38, 27, 43, 3, 9, 82, 10]
                    print(f"원본: {data}")
                    print(f"정렬: {quicksort(data)}")""")
            )

        # 소수
        if "소수" in g or "prime" in g:
            return (
                "에라토스테네스의 체로 소수를 찾습니다.",
                textwrap.dedent("""\
                    def sieve(n):
                        is_prime = [True] * (n + 1)
                        is_prime[0] = is_prime[1] = False
                        for i in range(2, int(n**0.5) + 1):
                            if is_prime[i]:
                                for j in range(i*i, n + 1, i):
                                    is_prime[j] = False
                        return [i for i, v in enumerate(is_prime) if v]

                    primes = sieve(50)
                    print(f"50 이하 소수: {primes}")
                    print(f"개수: {len(primes)}개")""")
            )

        # 다익스트라
        if "다익스트라" in g or "dijkstra" in g or "최단" in g:
            return (
                "다익스트라 최단경로 알고리즘을 구현합니다. heapq를 사용합니다.",
                textwrap.dedent("""\
                    import heapq

                    def dijkstra(graph, start):
                        dist = {v: float('inf') for v in graph}
                        dist[start] = 0
                        pq = [(0, start)]
                        while pq:
                            d, u = heapq.heappop(pq)
                            if d > dist[u]:
                                continue
                            for v, w in graph[u]:
                                if dist[u] + w < dist[v]:
                                    dist[v] = dist[u] + w
                                    heapq.heappush(pq, (dist[v], v))
                        return dist

                    graph = {
                        'A': [('B', 4), ('C', 2)],
                        'B': [('D', 3), ('C', 1)],
                        'C': [('B', 1), ('D', 5)],
                        'D': []
                    }
                    result = dijkstra(graph, 'A')
                    for node, cost in sorted(result.items()):
                        print(f"  A → {node}: 비용 {cost}")""")
            )

        # 팩토리얼
        if "팩토리얼" in g or "factorial" in g:
            return (
                "팩토리얼을 재귀와 반복 두 가지 방식으로 구현합니다.",
                textwrap.dedent("""\
                    def factorial_recursive(n):
                        return 1 if n <= 1 else n * factorial_recursive(n - 1)

                    def factorial_iterative(n):
                        result = 1
                        for i in range(2, n + 1):
                            result *= i
                        return result

                    for n in [5, 10, 15]:
                        print(f"{n}! = {factorial_iterative(n)}")""")
            )

        # 기본 fallback — 목표를 출력하는 코드
        return (
            f"주어진 목표 '{goal[:50]}'를 처리합니다.",
            textwrap.dedent(f"""\
                goal = '''{goal[:100]}'''
                print(f"목표: {{goal}}")
                steps = goal.split()
                print(f"키워드 {{len(steps)}}개 분석 완료")
                print("처리 결과: 성공")""")
        )

    # ── Auto mode ──────────────────────────────────────────────────

    def _auto_planner(self, prompt: str, model: str) -> _LLMResponse:
        # 목표에서 키워드를 추출해 3개 서브태스크 자동 생성 (각각 다른 키워드 포함)
        goal = self._goal or "주어진 목표"
        tasks = {
            "tasks": [
                {"id": "task-001", "goal": f"'{goal}'의 요구사항 분석 및 설계", "depends_on": []},
                {"id": "task-002", "goal": f"{goal}", "depends_on": ["task-001"]},
                {"id": "task-003", "goal": f"'{goal}'의 결과 검증 및 테스트", "depends_on": ["task-002"]},
            ]
        }
        return _LLMResponse(
            content=json.dumps(tasks, ensure_ascii=False),
            prompt_tokens=200,
            completion_tokens=150,
            model=model,
            stop_reason="end_turn",
        )

    def _auto_reviewer(self, prompt: str, model: str) -> _LLMResponse:
        review = {
            "approved": True,
            "think": "워커가 목표에 맞는 결과를 생성했습니다. 코드가 정상 실행되었고 출력이 올바릅니다.",
            "critique": "",
            "score": 0.92,
        }
        return _LLMResponse(
            content=json.dumps(review, ensure_ascii=False),
            prompt_tokens=300,
            completion_tokens=80,
            model=model,
            stop_reason="end_turn",
        )

    # ── Manual mode ────────────────────────────────────────────────

    def _manual_planner(self, prompt: str, model: str) -> _LLMResponse:
        print(f"\n  {MAGENTA}{'─' * 50}")
        print(f"  🧠 당신이 Planner입니다! 목표를 서브태스크로 분해하세요.")
        print(f"  {'─' * 50}{RESET}\n")

        tasks = []
        print(f"  {DIM}서브태스크를 입력하세요 (빈 줄로 종료):{RESET}")
        idx = 1
        while True:
            goal = input(f"  {CYAN}task-{idx:03d}{RESET} 목표: ").strip()
            if not goal:
                break
            deps_raw = input(f"         의존성 (쉼표 구분, 없으면 Enter): ").strip()
            deps = [d.strip() for d in deps_raw.split(",") if d.strip()] if deps_raw else []
            tasks.append({"id": f"task-{idx:03d}", "goal": goal, "depends_on": deps})
            idx += 1

        if not tasks:
            tasks = [
                {"id": "task-001", "goal": f"{self._goal} 수행", "depends_on": []},
            ]
            info("(기본 태스크 1개가 생성되었습니다)")

        result = json.dumps({"tasks": tasks}, ensure_ascii=False)
        return _LLMResponse(
            content=result,
            prompt_tokens=200,
            completion_tokens=len(result),
            model=model,
            stop_reason="end_turn",
        )

    def _manual_reviewer(self, prompt: str, model: str) -> _LLMResponse:
        # 프롬프트에서 태스크 목표 추출
        goal_line = ""
        for line in prompt.split("\n"):
            if line.startswith("Subtask goal:"):
                goal_line = line.replace("Subtask goal:", "").strip()
                break

        print(f"\n  {MAGENTA}{'─' * 50}")
        print(f"  🔍 당신이 Reviewer입니다! 결과를 평가하세요.")
        print(f"  대상: {goal_line}")
        print(f"  {'─' * 50}{RESET}\n")

        choice = input(f"  승인(y) / 거부(n) [{GREEN}y{RESET}]: ").strip().lower()
        approved = choice != "n"

        if approved:
            score_raw = input(f"  점수 (0.0~1.0) [{GREEN}0.9{RESET}]: ").strip()
            score = float(score_raw) if score_raw else 0.9
            think = input(f"  사유 (선택): ").strip() or "결과가 만족스럽습니다."
            critique = ""
        else:
            score_raw = input(f"  점수 (0.0~1.0) [{RED}0.3{RESET}]: ").strip()
            score = float(score_raw) if score_raw else 0.3
            critique = input(f"  개선 요구사항: ").strip() or "결과가 불충분합니다. 다시 시도하세요."
            think = input(f"  사유 (선택): ").strip() or "품질 기준 미달."

        review = {"approved": approved, "think": think, "critique": critique, "score": score}
        return _LLMResponse(
            content=json.dumps(review, ensure_ascii=False),
            prompt_tokens=300,
            completion_tokens=80,
            model=model,
            stop_reason="end_turn",
        )


class FakeSandboxResult:
    def __init__(self, exit_code=0, stdout="", stderr="", timed_out=False):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


class FakeRuntime:
    """Docker 없이 코드를 실행하는 시뮬레이터."""

    def run(self, *, image, command, mem_limit, network_disabled, timeout):
        code = command[-1] if command else ""
        try:
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                exec(code, {"__builtins__": __builtins__})
            return FakeSandboxResult(exit_code=0, stdout=buf.getvalue())
        except Exception as e:
            return FakeSandboxResult(exit_code=1, stderr=str(e))


# ── Demo 함수들 ────────────────────────────────────────────────────

def demo_session_lane():
    """Demo 1: SessionLane 자원 직렬화."""
    banner("DEMO 1: SessionLane — Per-Key 자원 직렬화", CYAN)

    from kappa.infra.session_lane import SyncSessionLane
    from kappa.config import SessionLaneConfig

    lane = SyncSessionLane(SessionLaneConfig(timeout=5.0))

    print("  SessionLane은 동일한 키에 대한 동시 접근을 직렬화합니다.")
    print(f"  현재 활성 키: {BOLD}{lane.active_keys}{RESET}\n")

    key = input(f"  잠금할 키 입력 (예: user-123): ").strip() or "user-123"

    step("Lock 획득", f"key={key}")
    with lane.lane(key):
        ok(f"'{key}' Lock 획득 성공!")
        print(f"  활성 키: {BOLD}{lane.active_keys}{RESET}")
        info("(이 블록 안에서는 같은 키로 다른 스레드가 진입할 수 없습니다)")

    ok(f"'{key}' Lock 해제됨")
    print(f"  활성 키: {BOLD}{lane.active_keys}{RESET}")

    # 멀티스레드 데모
    print(f"\n  {YELLOW}── 멀티스레드 직렬화 시연 ──{RESET}")
    import threading

    results = []

    def worker(name, k):
        with lane.lane(k):
            results.append(f"{name} 진입")
            time.sleep(0.1)
            results.append(f"{name} 완료")

    key2 = input(f"  경합시킬 키 (예: shared-resource): ").strip() or "shared-resource"
    step("2개 스레드가 같은 키로 동시 접근", f"key={key2}")

    t1 = threading.Thread(target=worker, args=("Thread-A", key2))
    t2 = threading.Thread(target=worker, args=("Thread-B", key2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print(f"\n  실행 순서:")
    for i, r in enumerate(results, 1):
        prefix = GREEN if "완료" in r else CYAN
        print(f"    {prefix}{i}. {r}{RESET}")

    ok("동일 키 → 직렬화 보장 (진입→완료→진입→완료 순서)")


def demo_jitter_backoff():
    """Demo 2: Decorrelated Jitter Backoff."""
    banner("DEMO 2: Jitter Backoff — 지수적 재시도", YELLOW)

    from kappa.infra.jitter import jitter_backoff_sync
    from kappa.config import BackoffConfig

    print("  네트워크 실패 등 일시적 오류 시 Jitter가 적용된 지수적 백오프로 재시도합니다.")
    print(f"  알고리즘: {DIM}sleep = min(cap, random(base, prev_delay × 3)){RESET}\n")

    fail_count_raw = input(f"  처음 몇 번 실패시킬까요? (1~4) [{RED}2{RESET}]: ").strip()
    fail_count = int(fail_count_raw) if fail_count_raw else 2
    fail_count = max(1, min(4, fail_count))

    attempt = 0
    delays = []

    def flaky_fn():
        nonlocal attempt
        attempt += 1
        if attempt <= fail_count:
            step(f"시도 #{attempt}", f"실패!")
            raise ConnectionError(f"일시적 네트워크 오류 (시도 {attempt})")
        step(f"시도 #{attempt}", f"성공!")
        return f"결과: 시도 {attempt}에서 성공"

    config = BackoffConfig(base_delay=0.3, max_delay=5.0, max_retries=5)

    original_sleep = time.sleep

    def tracked_sleep(secs):
        delays.append(secs)
        info(f"  ⏱ 대기: {secs:.3f}초 (jitter 적용됨)")
        original_sleep(min(secs, 0.5))  # 데모라서 실제론 짧게

    import kappa.infra.jitter as jitter_mod
    original = jitter_mod.time.sleep
    jitter_mod.time.sleep = tracked_sleep

    try:
        result = jitter_backoff_sync(flaky_fn, config=config)
        ok(f"최종 결과: {result}")
    except Exception as e:
        fail(f"모든 재시도 실패: {e}")
    finally:
        jitter_mod.time.sleep = original

    if delays:
        print(f"\n  {BOLD}대기 시간 변화:{RESET}")
        for i, d in enumerate(delays, 1):
            bar = "█" * int(d * 20)
            print(f"    재시도 {i}: {CYAN}{bar}{RESET} {d:.3f}s")


def demo_orchestrator(mode: str):
    """Demo 3: 오케스트레이터 전체 파이프라인."""
    banner("DEMO 3: Orchestrator Super-Graph 파이프라인", MAGENTA)

    from kappa.budget.gate import BudgetGate
    from kappa.budget.tracker import BudgetTracker
    from kappa.config import AgentConfig, BudgetConfig, OrchestratorConfig, TelemetryConfig
    from kappa.sandbox.executor import SandboxExecutor, SandboxConfig
    from kappa.infra.session_lane import SyncSessionLane, SessionLaneConfig
    from kappa.telemetry.manager import TelemetryManager
    from kappa.graph.orchestrator import OrchestratorGraph

    import tempfile
    tmpdir = tempfile.mkdtemp()
    telemetry_path = os.path.join(tmpdir, "demo_telemetry.jsonl")

    # Provider 설정
    provider = InteractiveProvider(mode=mode)

    # 인프라 조립
    tracker = BudgetTracker(BudgetConfig(max_total_tokens=500_000, max_cost_usd=50.0))
    gate = BudgetGate(provider=provider, tracker=tracker)
    sandbox = SandboxExecutor(runtime=FakeRuntime(), config=SandboxConfig())
    session_lane = SyncSessionLane(SessionLaneConfig(timeout=30.0))
    telemetry = TelemetryManager(TelemetryConfig(enabled=True, log_path=telemetry_path))

    orch_config = OrchestratorConfig(
        max_rejections=3,
        max_subtasks=5,
        max_parallel_workers=2,
    )

    orch = OrchestratorGraph(
        gate=gate,
        sandbox=sandbox,
        config=AgentConfig(),
        orchestrator_config=orch_config,
        session_lane=session_lane,
        telemetry=telemetry,
    )

    # 실제 LLM 사용 가능 여부 확인
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    use_real_llm = False

    if api_key and mode != "manual":
        choice_llm = input(f"  {GREEN}ANTHROPIC_API_KEY 감지!{RESET} 실제 LLM 사용? (y/n) [{GREEN}y{RESET}]: ").strip().lower()
        if choice_llm != "n":
            use_real_llm = True
            from kappa.budget.gate import AnthropicProvider
            real_provider = AnthropicProvider(api_key=api_key)
            gate = BudgetGate(provider=real_provider, tracker=tracker)
            orch = OrchestratorGraph(
                gate=gate,
                sandbox=sandbox,
                config=AgentConfig(),
                orchestrator_config=orch_config,
                session_lane=session_lane,
                telemetry=telemetry,
            )

    if use_real_llm:
        mode_label = f"{GREEN}실제 LLM (Anthropic API){RESET}"
    elif mode == "manual":
        mode_label = "수동 (당신이 Planner/Reviewer)"
    else:
        mode_label = "자동 (Mock 시뮬레이션)"
    print(f"  모드: {BOLD}{mode_label}{RESET}")
    print(f"  텔레메트리: {telemetry_path}\n")

    goal = input(f"  {BOLD}목표를 입력하세요:{RESET}\n  → ").strip()
    if not goal:
        goal = "피보나치 수열의 처음 10개를 계산하고, 합계를 구하라"
        info(f"(기본 목표 사용: {goal})")

    provider.set_goal(goal)

    print(f"\n  {CYAN}{'─' * 50}")
    print(f"  ▶ 파이프라인 실행 시작")
    print(f"  {'─' * 50}{RESET}\n")

    node_icons = {
        "planner": f"{CYAN}📋 PLANNER{RESET}",
        "dispatcher": f"{YELLOW}🚀 DISPATCHER{RESET}",
        "reviewer": f"{MAGENTA}🔍 REVIEWER{RESET}",
        "finalizer": f"{GREEN}✅ FINALIZER{RESET}",
        "failed": f"{RED}❌ FAILED{RESET}",
    }

    # stream으로 각 노드별 진행 시각화
    state = orch._initial_state(goal)
    step_num = 0

    try:
        for chunk in orch._app.stream(state):
            # LangGraph stream yields {node_name: state_update} dicts
            node_name = next(iter(chunk))
            state_update = chunk[node_name]
            step_num += 1
            icon = node_icons.get(node_name, node_name)
            print(f"\n  {'═' * 50}")
            print(f"  Step {step_num}: {icon}")
            print(f"  {'═' * 50}")

            if node_name == "planner":
                plan = state_update.get("plan", [])
                status = state_update.get("global_status", "")
                if plan:
                    print(f"\n  {BOLD}분해된 서브태스크 ({len(plan)}개):{RESET}")
                    for t in plan:
                        deps = f" ← {t['depends_on']}" if t["depends_on"] else ""
                        print(f"    {CYAN}[{t['id']}]{RESET} {t['goal']}{DIM}{deps}{RESET}")
                else:
                    fail(f"  플랜 생성 실패 (status={status})")

            elif node_name == "dispatcher":
                plan = state_update.get("plan", [])
                reviewing = [t for t in plan if t["status"] == "awaiting_review"]
                if reviewing:
                    print(f"\n  {BOLD}워커 실행 완료 → 리뷰 대기:{RESET}")
                    for t in reviewing:
                        result = t.get("result", {}) or {}
                        sr = result.get("sandbox_result") or {}
                        stdout = sr.get("stdout", "").strip() if isinstance(sr, dict) else ""
                        stderr = sr.get("stderr", "").strip() if isinstance(sr, dict) else ""
                        parsed_code = result.get("parsed_code", "")
                        status_str = result.get("status", "unknown")
                        color = GREEN if status_str == "completed" else YELLOW

                        print(f"\n    {color}[{t['id']}]{RESET} {t['goal']}")
                        print(f"    status: {color}{status_str}{RESET}")

                        if parsed_code:
                            print(f"\n    {DIM}── 생성된 코드 ──{RESET}")
                            for line in parsed_code.strip().split("\n"):
                                print(f"    {DIM}  {line}{RESET}")

                        if stdout:
                            print(f"\n    {CYAN}── 실행 결과 ──{RESET}")
                            for line in stdout.split("\n"):
                                print(f"    {CYAN}  {line}{RESET}")

                        if stderr:
                            print(f"\n    {RED}── 에러 ──{RESET}")
                            for line in stderr.split("\n"):
                                print(f"    {RED}  {line}{RESET}")

            elif node_name == "reviewer":
                plan = state_update.get("plan", [])
                records = state_update.get("telemetry_records", [])
                completed = [t for t in plan if t["status"] == "completed"]
                rejected = [t for t in plan if t["status"] == "rejected"]

                if completed:
                    print(f"\n  {GREEN}승인된 태스크:{RESET}")
                    for t in completed:
                        rec = next((r for r in records if r["task_id"] == t["id"]), {})
                        score = rec.get("score", "?")
                        print(f"    {GREEN}✓ [{t['id']}]{RESET} score={score}")

                if rejected:
                    print(f"\n  {RED}거부된 태스크:{RESET}")
                    for t in rejected:
                        rec = next((r for r in records if r["task_id"] == t["id"]), {})
                        print(f"    {RED}✗ [{t['id']}]{RESET} critique: {t.get('critique', '')}")

            elif node_name == "finalizer":
                output = state_update.get("final_output", "")
                print(f"\n  {GREEN}{BOLD}최종 출력:{RESET}")
                for line in output.split("\n"):
                    print(f"    {line}")

            elif node_name == "failed":
                fail("오케스트레이션 실패")

    except Exception as e:
        fail(f"실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

    # 텔레메트리 요약
    print(f"\n\n  {CYAN}{'─' * 50}")
    print(f"  📊 텔레메트리 요약")
    print(f"  {'─' * 50}{RESET}\n")

    try:
        summary = telemetry.summary()
        print(f"    총 기록 수:    {BOLD}{summary['total']}{RESET}")
        print(f"    평균 점수:     {BOLD}{summary['avg_score']:.2f}{RESET}")
        print(f"    성공:          {GREEN}{summary['success_count']}{RESET}")
        print(f"    거부:          {RED}{summary['rejected_count']}{RESET}")
        print(f"    거부율:        {YELLOW}{summary['rejection_rate']:.1%}{RESET}")

        records = telemetry.read_all()
        if records:
            print(f"\n    {BOLD}상세 궤적:{RESET}")
            for r in records:
                icon = f"{GREEN}✓{RESET}" if r.outcome == "success" else f"{RED}✗{RESET}"
                print(f"    {icon} [{r.task_id}] score={r.score:.2f} | {DIM}{r.think[:60]}{RESET}")
    except Exception:
        info("(텔레메트리 데이터 없음)")

    # 예산 사용량
    print(f"\n  {CYAN}{'─' * 50}")
    print(f"  💰 예산 사용량")
    print(f"  {'─' * 50}{RESET}\n")

    print(f"    사용 토큰:  {BOLD}{tracker.total_tokens:,}{RESET}")
    print(f"    사용 비용:  {BOLD}${tracker.estimated_cost_usd:.4f}{RESET}")
    print(f"    LLM 호출:  {BOLD}{tracker.call_count}회{RESET}")


# ── 메인 ───────────────────────────────────────────────────────────

def main():
    print(f"""
{CYAN}{BOLD}
  ┌─────────────────────────────────────────────────┐
  │         Kappa Phase 3 — Interactive Demo         │
  │       Multi-Agent Orchestration Pipeline          │
  └─────────────────────────────────────────────────┘{RESET}

  Phase 3에서 구현된 3가지 핵심 모듈을 직접 체험합니다:

    {CYAN}1{RESET}. SessionLane — Per-Key 자원 직렬화
    {YELLOW}2{RESET}. Jitter Backoff — 지수적 재시도
    {MAGENTA}3{RESET}. Orchestrator — 전체 파이프라인 (자동 모드)
    {MAGENTA}4{RESET}. Orchestrator — 전체 파이프라인 (수동 모드)
       {DIM}(당신이 Planner/Reviewer 역할을 직접 수행){RESET}
    {GREEN}5{RESET}. 전체 데모 순서대로 실행
    {RED}q{RESET}. 종료

  {DIM}※ ANTHROPIC_API_KEY 환경변수 설정 시 3번에서 실제 LLM을 사용할 수 있습니다.{RESET}
""")

    while True:
        choice = input(f"  선택 (1-5, q): ").strip().lower()

        if choice == "1":
            demo_session_lane()
        elif choice == "2":
            demo_jitter_backoff()
        elif choice == "3":
            demo_orchestrator("auto")
        elif choice == "4":
            demo_orchestrator("manual")
        elif choice == "5":
            demo_session_lane()
            pause()
            demo_jitter_backoff()
            pause()
            demo_orchestrator("auto")
        elif choice == "q":
            print(f"\n  {DIM}Demo 종료. 감사합니다!{RESET}\n")
            break
        else:
            info("1~5 또는 q를 입력하세요.")

        print()


if __name__ == "__main__":
    main()
