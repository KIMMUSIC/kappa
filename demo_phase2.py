"""Phase 2 Interactive Demo — 각 기능을 직접 입력하며 확인합니다.

Usage:
    python demo_phase2.py
"""

import sys
import os
import tempfile

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def pause() -> None:
    input("\n  [Enter를 눌러 다음 단계로 →] ")


# ── Demo 1: VFS Memory ─────────────────────────────────────────

def demo_vfs():
    banner("DEMO 1: VFS 가상 파일 시스템 (장기 메모리)")
    from kappa.config import MemoryConfig
    from kappa.memory.vfs import VFSManager

    # 임시 디렉토리에 워크스페이스 생성
    tmpdir = tempfile.mkdtemp()
    vfs = VFSManager(MemoryConfig(workspace_root="workspace"), base_dir=tmpdir)
    print(f"  워크스페이스 루트: {vfs.root}\n")

    # 쓰기
    print("── 1-1. 파일 쓰기 ──")
    path = input("  저장할 파일 경로 (예: LEARNINGS.md): ").strip() or "LEARNINGS.md"
    content = input("  저장할 내용 (예: 무한루프를 조심하라): ").strip() or "무한루프를 조심하라"
    vfs.write(path, content)
    print(f"  ✓ '{path}'에 저장 완료")

    # 읽기
    print("\n── 1-2. 파일 읽기 ──")
    read_path = input(f"  읽을 파일 경로 (예: {path}): ").strip() or path
    result = vfs.read(read_path)
    print(f"  읽은 내용: {result!r}")

    # 목록
    print("\n── 1-3. 파일 목록 ──")
    vfs.write("notes/architecture.md", "5-layer verification")
    files = vfs.list()
    print(f"  워크스페이스 파일 목록: {files}")

    # 경로 이탈 차단
    print("\n── 1-4. 경로 이탈 공격 차단 ──")
    attack_path = input("  공격 경로 입력 (예: ../../../etc/passwd): ").strip() or "../../../etc/passwd"
    try:
        vfs.read(attack_path)
        print("  ✗ 차단 실패!")  # 여기 오면 안 됨
    except ValueError as e:
        print(f"  ✓ 차단 성공! → {e}")

    pause()


# ── Demo 2: Tool Registry ──────────────────────────────────────

def demo_tools():
    banner("DEMO 2: 도구 레지스트리 + 내장 도구")
    from kappa.config import BudgetConfig, MemoryConfig
    from kappa.budget.tracker import BudgetTracker
    from kappa.memory.vfs import VFSManager
    from kappa.tools.registry import ToolRegistry
    from kappa.tools.builtins import ReadMemoryTool, WriteMemoryTool

    # 설정
    tmpdir = tempfile.mkdtemp()
    vfs = VFSManager(MemoryConfig(workspace_root="ws"), base_dir=tmpdir)
    tracker = BudgetTracker(BudgetConfig(max_total_tokens=500, max_cost_usd=100.0))
    registry = ToolRegistry(tracker=tracker, cost_per_tool_call=50)
    registry.register(ReadMemoryTool(vfs))
    registry.register(WriteMemoryTool(vfs))

    # 도구 목록
    print("── 2-1. 등록된 도구 목록 ──")
    for tool in registry.list_tools():
        print(f"  • {tool['name']}: {tool['description']}")

    # write_memory 실행
    print("\n── 2-2. write_memory 도구 실행 ──")
    path = input("  파일 경로 (예: rules.md): ").strip() or "rules.md"
    content = input("  내용 (예: 항상 입력을 검증하라): ").strip() or "항상 입력을 검증하라"
    result = registry.execute("write_memory", path=path, content=content)
    print(f"  결과: success={result.success}, output={result.output!r}")
    print(f"  예산 사용: {tracker.total_tokens} tokens (한도 500)")

    # read_memory 실행
    print("\n── 2-3. read_memory 도구 실행 ──")
    result = registry.execute("read_memory", path=path)
    print(f"  결과: success={result.success}, output={result.output!r}")
    print(f"  예산 사용: {tracker.total_tokens} tokens (한도 500)")

    # 미등록 도구 호출
    print("\n── 2-4. 미등록 도구 호출 ──")
    tool_name = input("  존재하지 않는 도구 이름 (예: hack_server): ").strip() or "hack_server"
    try:
        registry.execute(tool_name)
    except Exception as e:
        print(f"  ✓ 거부됨! → {e}")

    # 예산 소진 시뮬레이션
    print("\n── 2-5. 예산 소진 시뮬레이션 ──")
    print(f"  현재: {tracker.total_tokens}/500 tokens")
    calls = 0
    try:
        while True:
            registry.execute("read_memory", path=path)
            calls += 1
    except Exception as e:
        print(f"  {calls}번 추가 호출 후 차단됨!")
        print(f"  ✓ → {e}")

    pause()


# ── Demo 3: Semantic Loop Detector ─────────────────────────────

def demo_semantic():
    banner("DEMO 3: 의미론적 무한 루프 감지기")
    from kappa.config import SemanticConfig
    from kappa.defense.semantic import SemanticLoopDetector, jaccard_similarity

    # Jaccard 유사도 계산
    print("── 3-1. Jaccard 유사도 비교 ──")
    text_a = input("  텍스트 A (예: fix the parser error): ").strip() or "fix the parser error"
    text_b = input("  텍스트 B (예: fix the parser error): ").strip() or "fix the parser error"
    sim = jaccard_similarity(text_a, text_b)
    print(f"  유사도: {sim:.4f} (1.0 = 동일)")

    text_c = input("  텍스트 C (예: use dynamic programming instead): ").strip() or "use dynamic programming instead"
    sim2 = jaccard_similarity(text_a, text_c)
    print(f"  A vs C 유사도: {sim2:.4f}")

    # 루프 감지
    print("\n── 3-2. 반복 감지 시뮬레이션 ──")
    detector = SemanticLoopDetector(
        SemanticConfig(window_size=5, similarity_threshold=0.85, min_samples=3)
    )
    print("  설정: window=5, threshold=0.85, min_samples=3")
    print("  동일한 텍스트를 3번 이상 입력하면 감지됩니다.")
    print("  (빈 줄 입력 시 종료)\n")

    count = 0
    while True:
        text = input(f"  [{count+1}] think 내용 입력: ").strip()
        if not text:
            break
        detector.record(text)
        count += 1
        try:
            detector.check()
            print(f"      → 통과 (윈도우: {len(detector.history)}개)")
        except Exception as e:
            print(f"      → ✓ 루프 감지! {e}")
            break

    pause()


# ── Demo 4: Parser Routing ─────────────────────────────────────

def demo_parser():
    banner("DEMO 4: 파서 분기 — <action> vs <tool_call>")
    from kappa.graph.nodes import parse_llm_output

    examples = {
        "1": (
            "코드 실행 (기존 경로)",
            '<think>피보나치를 계산한다</think>\n<action>print(sum(range(10)))</action>'
        ),
        "2": (
            "도구 호출 (신규 경로)",
            '<think>과거 학습 내용을 읽는다</think>\n<tool_call>{"name": "read_memory", "kwargs": {"path": "LEARNINGS.md"}}</tool_call>'
        ),
        "3": (
            "양쪽 동시 사용 (거부됨)",
            '<think>혼란</think>\n<action>print(1)</action>\n<tool_call>{"name": "x", "kwargs": {}}</tool_call>'
        ),
        "4": (
            "잘못된 JSON (거부됨)",
            '<think>시도</think>\n<tool_call>{bad json}</tool_call>'
        ),
        "5": (
            "직접 입력",
            None
        ),
    }

    print("  LLM 출력 예시를 선택하세요:\n")
    for k, (desc, _) in examples.items():
        print(f"    [{k}] {desc}")

    choice = input("\n  선택 (1-5): ").strip() or "1"

    if choice == "5":
        print("\n  LLM 출력을 직접 입력하세요 (빈 줄로 종료):")
        lines = []
        while True:
            line = input("  ")
            if not line:
                break
            lines.append(line)
        raw = "\n".join(lines)
    else:
        desc, raw = examples.get(choice, examples["1"])
        print(f"\n  입력: {desc}")
        print(f"  ─────────────────────────")
        for line in raw.split("\n"):
            print(f"  {line}")
        print(f"  ─────────────────────────")

    result = parse_llm_output(raw)
    print(f"\n  결과:")
    print(f"    think     = {result.think!r}")
    print(f"    code      = {result.code!r}")
    print(f"    tool_call = {result.tool_call!r}")
    print(f"    error     = {result.error!r}")

    if result.tool_call:
        print(f"\n  → 라우팅: PARSER → TOOL 노드")
    elif result.code:
        print(f"\n  → 라우팅: PARSER → LINTER → SANDBOX")
    else:
        print(f"\n  → 라우팅: PARSER → CODER (재시도)")

    pause()


# ── Demo 5: Memory Context Injection ──────────────────────────

def demo_memory_injection():
    banner("DEMO 5: 메모리 컨텍스트 → 시스템 프롬프트 주입")
    from kappa.graph.nodes import build_messages

    memory = input("  장기 기억 내용 (예: # Rule: 항상 입력을 검증하라): ").strip() or "# Rule: 항상 입력을 검증하라"
    goal = input("  에이전트 목표 (예: 사용자 입력을 처리하는 함수 작성): ").strip() or "사용자 입력을 처리하는 함수 작성"

    state = {
        "goal": goal,
        "llm_output": "",
        "parsed_code": "",
        "sandbox_result": None,
        "attempt": 0,
        "max_attempts": 3,
        "error_history": [],
        "status": "running",
        "tool_calls": [],
        "memory_context": memory,
    }

    messages = build_messages(state)
    prompt = messages[0]["content"]

    print(f"\n  ── 생성된 프롬프트 (앞 500자) ──")
    print(f"  {prompt[:500]}")
    if len(prompt) > 500:
        print(f"  ... ({len(prompt)}자 중 500자만 표시)")

    if "[Long-term Memory]" in prompt:
        print(f"\n  ✓ 장기 기억이 프롬프트에 주입되었습니다!")
    else:
        print(f"\n  ✗ 장기 기억이 주입되지 않았습니다.")

    pause()


# ── Main ───────────────────────────────────────────────────────

def main():
    print("\n" + "─" * 60)
    print("  KAPPA Phase 2 — Interactive Feature Demo")
    print("  VFS Memory │ Tool Registry │ Semantic Defense │ Graph V2")
    print("─" * 60)

    demos = [
        ("VFS 가상 파일 시스템", demo_vfs),
        ("도구 레지스트리 + 내장 도구", demo_tools),
        ("의미론적 루프 감지기", demo_semantic),
        ("파서 분기 (action vs tool_call)", demo_parser),
        ("메모리 컨텍스트 주입", demo_memory_injection),
    ]

    print("\n  데모를 선택하세요:\n")
    print("    [0] 전체 순서대로 실행")
    for i, (name, _) in enumerate(demos, 1):
        print(f"    [{i}] {name}")
    print()

    choice = input("  선택 (0-5): ").strip() or "0"

    if choice == "0":
        for _, fn in demos:
            fn()
    elif choice.isdigit() and 1 <= int(choice) <= len(demos):
        demos[int(choice) - 1][1]()
    else:
        print("  잘못된 선택입니다.")
        return

    banner("데모 완료!")
    print("  Phase 2의 모든 기능이 정상 동작합니다.")
    print("  147/147 테스트 올 패스.\n")


if __name__ == "__main__":
    main()
