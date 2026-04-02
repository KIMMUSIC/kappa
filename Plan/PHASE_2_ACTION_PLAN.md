# [PHASE_2] 인지적 자율성 확보: VFS 메모리, MCP 도구 및 의미론적 방어선 구축

## 1. Phase 2의 목표 (Objective)
Phase 1에서 완벽하게 구축된 '가드레일 기반 결정론적 자가 치유 루프' 내부에, 에이전트가 단기 컨텍스트의 한계를 극복하고 외부 세계와 소통할 수 있도록 **가상 파일 시스템(VFS) 장기 메모리**와 **MCP(Model Context Protocol) 기반 도구(Tools) 인터페이스**를 이식한다.
모든 확장은 Phase 1의 예산 통제(Budget)와 샌드박스 격리(Sandbox) 원칙을 최우선으로 준수해야 한다.

## 2. 작업 지시 사항 (Action Items)

### Task 1: 가상 파일 시스템(VFS) 기반 장기 메모리 (`src/kappa/memory/`)
장기 실행 시 핵심 지식을 잃어버리는 '중간 상실(Lost in the middle)' 현상을 방지한다.
*   **요구사항:**
    *   `vfs.py`를 생성하여 에이전트 전용 격리된 작업 디렉토리(예: `.kappa_workspace/`)로 파일 접근을 제한하는 VFSManager를 구현할 것. (경로 이탈 `../` 공격 원천 차단)
    *   에이전트가 과거의 실패 교훈이나 중요한 아키텍처 규칙을 `LEARNINGS.md` 등의 마크다운 파일에 기록(Write)하고 검색(Read)할 수 있는 인터페이스 제공.
    *   그래프 실행 시 `build_messages()` 함수에서 VFS의 핵심 파일 내용을 시스템 프롬프트(System Context)로 자동 주입할 것.

### Task 2: MCP 도구 레지스트리 및 브릿지 (`src/kappa/tools/`)
외부 도구를 플러그 앤 플레이(Plug-and-play) 방식으로 연결할 수 있는 인프라를 구축한다.
*   **요구사항:**
    *   `mcp_client.py` (또는 `registry.py`)를 생성하여 표준화된 도구 규격(BaseTool 프로토콜)으로 외부 도구를 등록하고 실행할 수 있는 브릿지 인터페이스 구현.
    *   초기 검증용으로 VFS를 조작할 수 있는 `read_memory`, `write_memory` 시스템 도구를 우선 구현할 것.
    *   **[핵심 제약]:** 에이전트가 도구를 호출하는 과정에서 발생하는 모든 통신 및 실행 비용은 반드시 Phase 1의 `BudgetGate`를 거쳐 추적되어야 한다.

### Task 3: 의미론적 무한 루프 감지기 (`src/kappa/defense/`)
GEODE 아키텍처의 핵심인 '의미론적 유사성 분석(Semantic Similarity Analysis)' 방어 기제를 도입하여 목표 표류(Goal Drift)를 막는다.
*   **요구사항:**
    *   `semantic.py`를 생성하여 에이전트의 최근 N개 `<think>` 내용이나 도구 호출 인자(Arguments)의 텍스트 유사도(TF-IDF, Jaccard 등 경량 알고리즘 우선)를 계산.
    *   동일한 헛발질이 임계치 이상 반복되면 `SemanticLoopException`을 발생시켜 조기 차단.

### Task 4: LangGraph 자가 치유 루프 고도화 (Graph V2)
Phase 1의 루프(`CODER -> PARSER -> LINTER -> SANDBOX`)를 확장하여 도구 분기를 지원한다.
*   **요구사항:**
    *   `AgentState`에 `tool_calls`, `memory_context` 필드를 추가.
    *   기존 `_parser_node`가 파이썬 코드 실행(`sandbox`)과 도구 호출(`tool`)을 명확히 구분하여 파싱하도록 로직 고도화.
    *   LangGraph에 `_tool_node`를 추가하고, 파싱 결과에 따라 `sandbox_node`와 `tool_node`로 향하는 **조건부 엣지(Conditional Edge)**를 분리할 것. 도구 실행 결과 역시 자가 치유 루프를 타야 함.

## 3. 완료 검증 기준 (Definition of Done)
1. **Memory Test:** 에이전트가 VFS에 규칙을 기록하게 한 뒤, 세션을 초기화하고 실행했을 때 해당 규칙을 기억하여 적용하는가?
2. **Tool Routing Test:** 에이전트가 목적에 따라 샌드박스 코드 실행과 도구 호출을 정확히 분기하여 사용하는가?
3. **Semantic Defense Test:** 의도적으로 똑같은 오답 도구 호출을 반복하게 했을 때, `max_attempts` 도달 전에 '의미론적 감지기'가 루프를 멈추는가?
4. **Regression 방어:** Phase 1에서 작성한 47개의 모든 테스트 코드가 단 하나도 깨지지 않고(100% PASS) 통과해야 한다.

---
**🚨 [AI 에이전트 행동 지침] 🚨**
절대 즉시 코드를 작성하거나 수정하지 마라.
Phase 1의 기존 아키텍처와 완벽하게 호환되도록, Task 1~4를 위한 **업데이트될 디렉토리/클래스 구조안**과 `_tool_node`가 추가된 **새로운 LangGraph 흐름도(ASCII Art)**를 먼저 브리핑하고 사용자의 승인(Approve)을 대기하라.