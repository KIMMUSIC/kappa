# [PHASE_3] 다중 에이전트 오케스트레이션(Science of Scaling) 및 Agent-RRM 파이프라인

## 1. Phase 3의 목표 (Objective)
Phase 2에서 완성된 완벽한 단일 자가 치유 에이전트를 '실무자(Worker)' 노드로 삼아, 이를 통제하는 **'중앙 집중식 오케스트레이터(Centralized Orchestrator)'**를 도입한다. 
구글 리서치의 확장성 원리에 따라 무분별한 에이전트 병렬 배치를 금지하고, 작업 분해(Task Decomposition)와 품질 게이트(Quality Gate)를 주도하는 매니저 계층을 둔다. 
또한, 다중 실행 환경의 인프라 붕괴를 막는 **SessionLane(키 기반 직렬화)**과, 에이전트 스스로 진화하게 만드는 **Agent-RRM(추론 보상 모델) 텔레메트리** 파이프라인을 구축한다.

## 2. 작업 지시 사항 (Action Items)

### Task 1: 인프라 회복탄력성 - SessionLane & Jitter 백오프 (`src/kappa/infra/`)
여러 Worker가 동시에 실행되며 API 및 자원(VFS, 예산)을 경합할 때 시스템이 마비되는 것을 방어한다.
*   **요구사항:**
    *   `session_lane.py`: 작업의 고유 식별 키(Key)를 기반으로 작동하는 세마포어(Per-key serialization) 메커니즘을 구현. 동일한 리소스(예: 같은 파일) 접근 시 충돌 없이 직렬화 대기열을 형성하고, 독립 자원은 완전 병렬 실행되도록 제어.
    *   `jitter.py`: 여러 워커가 동시에 LLM API Rate Limit(429)에 걸렸을 때 발생하는 'Thundering Herd' 현상을 막기 위해, 난수 기반의 변동성을 부여하는 지수적 백오프(Decorrelated Jitter) 로직을 통신부에 적용.

### Task 2: 중앙 집중식 오케스트레이터(Quality Gate) 구축 (`src/kappa/graph/orchestrator.py`)
거대 목표를 서브 태스크로 분해하고, Phase 2의 `SelfHealingGraph`를 하위 워커(Sub-graph)로 통제하는 상위 제어 평면(Super-Graph)을 만든다.
*   **요구사항:**
    *   `OrchestratorState` 스키마 설계: 메인 목표, 분해된 계획(Plan), 완료된 작업, 글로벌 상태 등.
    *   **Planner Node:** 메인 목표의 의존성을 분석하여 병렬/순차 실행이 가능한 하위 과제로 분할(Task Decomposition).
    *   **Dispatcher Node:** 분할된 과제를 `SelfHealingGraph` 워커들에게 위임. (다중 실행 시 반드시 SessionLane 인프라를 거쳐 통제)
    *   **Reviewer (Quality Gate) Node:** 워커들의 산출물을 취합 및 검증. 오류나 환각 발견 시 결과를 시스템에 병합하지 않고, 비평(Critique)과 함께 워커에게 즉시 반려(Reject) 및 재작업 지시.

### Task 3: Agent-RRM 텔레메트리 파이프라인 (`src/kappa/telemetry/`)
에이전트의 단순한 성공/실패가 아니라, '추론 궤적' 자체를 평가하고 저장하는 피드백 시스템을 구축한다.
*   **요구사항:**
    *   작업 완료 또는 Reviewer 반려 시, 옵저버 패턴(Observer Pattern)을 통해 에이전트의 궤적을 JSONL 포맷으로 기록하는 `TelemetryManager` 구현.
    *   반드시 저장되어야 할 3가지 핵심 메타데이터:
        1. `<think>`: 워커의 명시적 추론 흔적 (사고 과정)
        2. `<critique>`: Reviewer 노드가 생성한 집중 비평 텍스트
        3. `<score>`: 프로세스 종합 스칼라 점수 (0.0 ~ 1.0)

## 3. 완료 검증 기준 (Definition of Done)
1. **Concurrency Test:** 다중 스레드 환경에서 SessionLane과 Jitter를 통해 리소스 충돌(Race Condition)과 API 병목이 안전하게 직렬화/백오프 되는가?
2. **Orchestration Test:** 복합 과제 주입 시 Planner가 2개 이상의 작업으로 분할하고, 다중 Worker가 처리한 뒤, Reviewer가 최종 검수를 통해 단일 결과물을 완성해 내는가?
3. **Telemetry Test:** 작업 종료 후 텔레메트리 로그 파일에 `think`, `critique`, `score` 메타데이터가 정확히 누적 기록되는가?
4. **No Regression 방어 (가장 중요):** Phase 1, 2에서 달성한 **147개의 모든 테스트 코드가 단 하나의 실패 없이 100% 통과(ALL GREEN)**해야 한다.

---
**🚨 [AI 에이전트 행동 지침] 🚨**
단 한 줄의 코드도 즉시 작성하거나 기존 코드를 수정하지 마라.
Phase 3 통합을 위해 다음 세 가지를 나에게 먼저 브리핑하고 승인(Approve)을 대기하라.
1. 추가될 오케스트레이터, SessionLane, 텔레메트리 패키지의 디렉토리 및 클래스 설계안
2. **상위 매니저(Planner/Reviewer)와 하위 워커(Phase 2 Graph)가 결합된 계층형 다중 에이전트 LangGraph 흐름도 (ASCII Art)**
3. 기존 147개 테스트를 절대 훼손하지 않기 위한(역호환성 100% 유지) 아키텍처 보호 전략