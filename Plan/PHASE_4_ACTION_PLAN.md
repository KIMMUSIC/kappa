# [PHASE_4] 프로덕션 도입: MCP 연동, 도메인 RAG 및 CLI 제어 평면

## 1. Phase 4의 목표 (Objective)
Phase 1~3를 통해 완벽한 통제력과 확장성을 갖춘 '거대 자율 에이전트 하네스(Harness)' 코어 엔진이 완성되었다.
본 단계에서는 이 범용 엔진을 **실제 비즈니스 목표(Domain)**에 투입하기 위해, 기획 문서의 로드맵에 명시된 **표준 MCP(Model Context Protocol) 브릿지**와 **도메인 특화 RAG** 파이프라인을 구축한다. 
최종적으로 사용자 친화적인 대화형 CLI(Command Line Interface)를 얹어, 인간 조작자(Human-in-the-loop)의 통제 아래 리얼 LLM 기반의 End-to-End 실전 시나리오를 구동한다.

## 2. 작업 지시 사항 (Action Items)

### Task 1: 표준 MCP (Model Context Protocol) 클라이언트 브릿지 (`src/kappa/tools/mcp.py`)
에이전트가 외부 SaaS, 사내 DB, GitHub 등 현실 세계의 소프트웨어와 표준화된 방식으로 통신할 수 있게 만든다.
*   **요구사항:**
    *   외부 MCP 서버가 제공하는 도구(Tools) 카탈로그를 읽어와, Phase 2에서 구축한 `ToolRegistry`의 `BaseTool` 규격에 맞게 동적으로 래핑(Wrapping)하는 어댑터(Adapter) 구현.
    *   모든 MCP 도구 호출 역시 반드시 `BudgetGate`의 예산 추적 및 통제를 받도록 설계할 것.

### Task 2: 목표 도메인 특화 RAG 파이프라인 (`src/kappa/rag/`)
범용 LLM의 한계를 극복하고 심도 있는 전문 지식을 에이전트에게 공급하기 위한 검색 증강 생성(RAG) 도구를 주입한다.
*   **요구사항:**
    *   전문 문서(마크다운, PDF 등)를 청킹하고 임베딩하여 가벼운 벡터 저장소(예: ChromaDB 또는 FAISS)에 인덱싱하는 `RAGManager` 구현.
    *   에이전트나 Reviewer가 쿼리를 던지면 관련 지식을 검색해 반환하는 `KnowledgeSearchTool`을 만들어 `ToolRegistry`에 내장.

### Task 3: Interactive CLI 및 Human-in-the-loop (`src/kappa/cli.py`)
사용자(Operator)가 오케스트레이터에게 편하게 목표를 지시하고 실시간으로 통제할 수 있는 진입점을 만든다.
*   **요구사항:**
    *   `Rich` 라이브러리 등을 활용하여 Planner의 분해 과정, Dispatcher의 병렬 워커 할당, Reviewer의 비평 상태를 터미널에 아름답게 시각화할 것.
    *   **승인 개입(Human-in-the-loop):** 워커가 파괴적인 시스템 명령을 실행하려 하거나, 예산의 80% 이상 도달 등 민감한 상황에서 시스템을 일시 정지(Interrupt)하고 사용자에게 `[Y/N]` 승인을 묻는 로직 구현.

## 3. 완료 검증 기준 (Definition of Done)
1. **MCP & RAG Test:** MCP 브릿지와 RAG 검색 도구가 기존 자가 치유 루프에 완벽히 호환되며 성공적으로 실행되는가?
2. **CLI & HITL Test:** CLI 상태 시각화 UI가 멈춤 없이 렌더링되며, 민감한 작업 수행 전 인간의 승인을 정상적으로 대기하고 반려(N) 시 안전하게 취소되는가?
3. **Absolute No Regression:** 기존 250개의 코어 테스트는 단 하나도 깨지지 않고 **100% 통과(ALL GREEN)**해야 한다.

---
**🚨 [AI 에이전트 행동 지침] 🚨**
지금까지 증명한 완벽한 '개방-폐쇄 원칙(OCP)'과 '순수 추가(Additive-Only)' 원칙을 Phase 4에서도 절대적으로 고수하라.
단 한 줄의 코드도 즉시 작성하지 말고, 나에게 다음 세 가지를 브리핑하고 승인을 대기하라.
1. 외부 MCP 도구들을 우리의 깐깐한 `ToolRegistry`에 우회 없이 안전하게 안착시키기 위한 어댑터(Adapter) 아키텍처 설계안.
2. Human-in-the-loop(승인 개입)가 기존 LangGraph 상태 머신 내에서 기술적으로 어느 부분에 어떻게 삽입될 것인지에 대한 전략 (기존 250개 테스트를 깨지 않기 위한 방안).
3. 터미널 인터페이스(`cli.py`)에서 `OrchestratorGraph`의 상태 변화를 실시간으로 어떻게 후킹(Hooking)하여 화면에 렌더링할 것인지에 대한 UI 스트리밍 전략.