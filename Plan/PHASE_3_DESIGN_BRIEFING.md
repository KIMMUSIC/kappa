# Phase 3 설계안 브리핑 — 다중 에이전트 오케스트레이션 및 Agent-RRM 인프라

## 1. 디렉토리 및 클래스 설계안

### 1-A. 새 패키지 구조

```
src/kappa/
├── infra/                          ← [NEW] 인프라 회복탄력성 패키지
│   ├── __init__.py                 # SessionLane, jitter_backoff 공개
│   ├── session_lane.py             # Per-key 세마포어 직렬화
│   └── jitter.py                   # Decorrelated Jitter 지수적 백오프
│
├── graph/
│   ├── __init__.py                 # 기존 + OrchestratorGraph 추가 공개
│   ├── graph.py                    # SelfHealingGraph (수정 없음 — 완전 보존)
│   ├── nodes.py                    # 기존 그대로
│   ├── state.py                    # 기존 그대로
│   └── orchestrator.py             ← [NEW] 상위 오케스트레이터 Super-Graph
│
├── telemetry/                      ← [NEW] Agent-RRM 텔레메트리 패키지
│   ├── __init__.py                 # TelemetryManager, TrajectoryRecord 공개
│   └── manager.py                  # 옵저버 패턴 JSONL 로거
│
├── budget/                         # 수정 없음
├── sandbox/                        # 수정 없음
├── tools/                          # 수정 없음
├── memory/                         # 수정 없음
├── defense/                        # 수정 없음
├── config.py                       # OrchestratorConfig, TelemetryConfig 추가
└── exceptions.py                   # OrchestratorError, SessionLaneTimeout 추가
```

### 1-B. 클래스 상세 설계

#### `src/kappa/infra/session_lane.py` — SessionLane

```python
class SessionLane:
    """Per-key 세마포어 기반 자원 접근 직렬화.
    
    동일 key(예: 파일 경로, API endpoint)로 진입하는 요청은 직렬 대기열을 형성.
    서로 다른 key는 완전 병렬 실행. 일정 시간 내 락 획득 실패 시 SessionLaneTimeout.
    """
    def __init__(self, timeout: float = 30.0) -> None
        # self._locks: dict[str, asyncio.Lock] — per-key 락 맵
        # self._timeout: float — 락 획득 대기 타임아웃
    
    async def acquire(self, key: str) -> None
        # key별 Lock 생성(lazy) 후 timeout 내 획득 시도
        # 실패 시 SessionLaneTimeout 발생
    
    async def release(self, key: str) -> None
        # 해당 key의 Lock 해제

    @asynccontextmanager
    async def lane(self, key: str)
        # async with session_lane.lane("file:config.py"):  형태로 사용
        # acquire → yield → release (try/finally)
```

> **설계 판단**: `asyncio.Lock`을 사용하되, 동기 환경 호환을 위해 내부에 `threading.Lock` 기반의 `SyncSessionLane`도 제공. 현재 `SelfHealingGraph`가 동기(`invoke`) 방식이므로, Orchestrator 레벨에서는 `concurrent.futures.ThreadPoolExecutor` + `SyncSessionLane` 조합으로 병렬성 확보.

#### `src/kappa/infra/jitter.py` — Decorrelated Jitter 백오프

```python
@dataclass(frozen=True)
class BackoffConfig:
    base_delay: float = 1.0       # 초기 대기 시간(초)
    max_delay: float = 60.0       # 최대 대기 시간 캡
    max_retries: int = 5          # 최대 재시도 횟수

async def jitter_backoff(
    fn: Callable[..., T],
    *args,
    config: BackoffConfig | None = None,
    retryable: Callable[[Exception], bool] | None = None,
    **kwargs
) -> T:
    """Decorrelated Jitter 지수적 백오프로 fn을 재시도.
    
    AWS 권장 알고리즘:  sleep = min(cap, random_between(base, prev_sleep * 3))
    retryable 함수로 재시도 대상 예외를 필터링 (기본: HTTP 429, 5xx).
    max_retries 초과 시 마지막 예외를 그대로 전파.
    """

def jitter_backoff_sync(fn, *args, config=None, retryable=None, **kwargs) -> T:
    """동기 버전 — ThreadPoolExecutor 환경용."""
```

#### `src/kappa/graph/orchestrator.py` — OrchestratorGraph

```python
class SubTask(TypedDict):
    """Planner가 분해한 하위 과제 단위."""
    id: str                    # 고유 식별자 (예: "task-001")
    goal: str                  # 워커에게 전달할 목표
    depends_on: list[str]      # 선행 과제 ID (DAG 의존성)
    status: str                # pending | running | completed | rejected
    result: dict | None        # 워커 산출물 (AgentState 스냅샷)
    critique: str              # Reviewer 비평 (반려 시)
    attempts: int              # 재작업 횟수

class OrchestratorState(TypedDict):
    """오케스트레이터 Super-Graph의 상태 스키마."""
    main_goal: str             # 최상위 목표
    plan: list[SubTask]        # 분해된 작업 목록 (DAG)
    completed: list[str]       # 완료된 task ID 목록
    rejected_count: int        # 총 반려 횟수
    max_rejections: int        # 반려 상한 (무한 루프 방지)
    global_status: str         # planning | dispatching | reviewing | done | failed
    final_output: str          # 최종 통합 산출물
    telemetry_records: list[dict]  # 궤적 로그 버퍼

class OrchestratorGraph:
    """계층형 다중 에이전트 오케스트레이터 (Super-Graph).
    
    Planner → Dispatcher → [Worker(SelfHealingGraph)...] → Reviewer 
    순환 루프로 구성. Quality Gate(Reviewer)에서 반려되면 해당 워커만 재실행.
    """
    def __init__(
        self,
        gate: BudgetGate,
        sandbox: SandboxExecutor,
        config: AgentConfig | None = None,
        orchestrator_config: OrchestratorConfig | None = None,
        registry: ToolRegistry | None = None,
        detector: SemanticLoopDetector | None = None,
        telemetry: TelemetryManager | None = None,
        session_lane: SyncSessionLane | None = None,
    ) -> None

    # ── 노드 구현 ──

    def _planner_node(self, state: OrchestratorState) -> dict:
        """LLM에게 main_goal의 의존성 분석 → SubTask DAG 생성 요청.
        별도 시스템 프롬프트로 JSON 포맷 Plan 출력을 강제."""
    
    def _dispatcher_node(self, state: OrchestratorState) -> dict:
        """DAG에서 의존성 충족된 pending 태스크를 식별,
        ThreadPoolExecutor로 병렬 위임. SessionLane으로 자원 충돌 방지.
        각 워커는 SelfHealingGraph.run() 호출."""
    
    def _reviewer_node(self, state: OrchestratorState) -> dict:
        """완료된 워커 산출물을 LLM이 검수. 
        합격 → completed로 이동, 불합격 → critique 첨부 + status='rejected'.
        텔레메트리 레코드 생성 (think/critique/score)."""
    
    def _finalizer_node(self, state: OrchestratorState) -> dict:
        """모든 SubTask 완료 후 결과물 통합, final_output 생성."""
    
    # ── 라우팅 ──

    def _route_after_review(self, state) -> str:
        """all completed → finalizer | rejected 존재 → dispatcher | max_rejections → failed"""
    
    # ── 공개 API ──

    def run(self, goal: str) -> OrchestratorState
    def stream(self, goal: str) -> Generator
```

> **핵심 설계 원칙**: `OrchestratorGraph`는 `SelfHealingGraph`를 **내부적으로 인스턴스화하여 `.run()` 호출**하는 방식. 기존 `SelfHealingGraph` 클래스의 코드를 **단 한 줄도 수정하지 않는다**.

#### `src/kappa/telemetry/manager.py` — TelemetryManager

```python
@dataclass(frozen=True)
class TrajectoryRecord:
    """단일 워커 실행 궤적의 구조화된 기록."""
    task_id: str               # SubTask.id
    worker_goal: str           # 워커에게 주어진 목표
    think: str                 # <think> 블록 내용 (추론 궤적)
    critique: str              # Reviewer 비평 텍스트 (성공 시 빈 문자열)
    score: float               # 프로세스 종합 점수 (0.0 ~ 1.0)
    outcome: str               # success | rejected | error
    token_usage: int           # 이 궤적에서 소비한 토큰 수
    timestamp: str             # ISO 8601

class TelemetryManager:
    """Agent-RRM 텔레메트리 수집기 (옵저버 패턴).
    
    워커 완료/반려 이벤트를 구독하여 TrajectoryRecord를 JSONL 파일에 누적 기록.
    """
    def __init__(self, log_path: Path | str = ".kappa_telemetry/trajectories.jsonl")
    
    def record(self, trajectory: TrajectoryRecord) -> None:
        """궤적을 JSONL 파일에 한 줄 추가. thread-safe (Lock 보호)."""
    
    def read_all(self) -> list[TrajectoryRecord]:
        """전체 궤적 로그를 메모리로 로드."""
    
    def summary(self) -> dict:
        """통계 요약: 총 기록 수, 평균 score, 반려율 등."""
```

#### `src/kappa/config.py` — 추가 설정

```python
@dataclass(frozen=True)
class OrchestratorConfig:
    max_rejections: int = 3           # Reviewer 반려 상한
    max_subtasks: int = 10            # Planner가 생성 가능한 최대 과제 수
    max_parallel_workers: int = 3     # 동시 실행 워커 수
    planner_model: str = "claude-sonnet-4-20250514"
    reviewer_model: str = "claude-sonnet-4-20250514"

@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    log_path: str = ".kappa_telemetry/trajectories.jsonl"
```

#### `src/kappa/exceptions.py` — 추가 예외

```python
class OrchestratorError(KappaError):
    """오케스트레이터 수준 실패 (Planning 불가, 반려 상한 초과 등)."""

class SessionLaneTimeout(KappaError):
    """SessionLane 락 획득 타임아웃."""
```

---

## 2. 계층형 다중 에이전트 LangGraph 흐름도 (ASCII Art)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ORCHESTRATOR SUPER-GRAPH (LangGraph)                      ║
║                                                                              ║
║  ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐  ║
║  │ PLANNER  │────▶│  DISPATCHER  │────▶│   REVIEWER   │────▶│ FINALIZER  │  ║
║  │  (LLM)   │     │              │     │ Quality Gate │     │            │  ║
║  └──────────┘     └──────┬───────┘     └──────┬───────┘     └─────┬──────┘  ║
║       │                  │                    │                    │         ║
║       │    ┌─────────────┼─────────────┐      │                    │         ║
║       │    │  SessionLane + Jitter     │      │                 ╔══▼══╗      ║
║       │    │  ┌─────────┬─────────┐    │      │                 ║ END ║      ║
║       │    │  ▼         ▼         ▼    │      │                 ╚═════╝      ║
║       │    │┌─────┐  ┌─────┐  ┌─────┐ │      │                              ║
║       │    ││ W-1 │  │ W-2 │  │ W-N │ │      │                              ║
║       │    │└──┬──┘  └──┬──┘  └──┬──┘ │      │                              ║
║       │    │   │        │        │     │      │                              ║
║       │    └───┼────────┼────────┼─────┘      │                              ║
║       │        └────────┴────────┘            │                              ║
║       │            결과 취합                   │                              ║
║       │                ▼                      │                              ║
║       │         ┌─────────────┐               │                              ║
║       │         │  REVIEWER   │◀──────────────┘                              ║
║       │         │ 합격/반려?  │                                              ║
║       │         └──────┬──────┘                                              ║
║       │           ┌────┴─────┐                                               ║
║       │      합격 ▼     반려 ▼                                               ║
║       │    completed   ┌──────────┐                                          ║
║       │                │critique +│──▶ DISPATCHER (해당 태스크만 재위임)      ║
║       │                │re-reject │                                          ║
║       │                └──────────┘                                          ║
║       │                                                                      ║
║  rejected_count ≥ max_rejections ──────────────▶ FAILED (안전 종료)          ║
║                                                                              ║
║  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐    ║
║  │                    TELEMETRY OBSERVER                                │    ║
║  │  Reviewer 합격/반려 이벤트마다 TrajectoryRecord를 JSONL에 기록       │    ║
║  │  { think | critique | score }                                        │    ║
║  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════╗
║              WORKER SUB-GRAPH (기존 SelfHealingGraph — 수정 없음)     ║
║                                                                      ║
║  coder ──▶ parser ──┬── <action> ──▶ linter ──┬── ok ──▶ sandbox ──┐ ║
║                     │                         │                    │ ║
║                     ├── <tool_call> ──▶ tool ─┤                    │ ║
║                     │                         │                    │ ║
║                     └── parse_error ──────────┴────────────────────┤ ║
║                                                                    │ ║
║                          attempt < max ◀───────────────────────────┘ ║
║                          attempt ≥ max ──▶ return AgentState         ║
╚═══════════════════════════════════════════════════════════════════════╝

Super-Graph 노드 흐름 요약:
  PLANNER ──▶ DISPATCHER ──▶ [Workers...] ──▶ REVIEWER ──┬──▶ FINALIZER ──▶ END
                   ▲                                     │
                   └─────── rejected (critique) ─────────┘
```

---

## 3. 기존 147개 테스트 역호환성 100% 유지 전략

### 전략 A: "순수 추가(Additive-Only)" 원칙

| 기존 모듈 | 변경 정책 |
|---|---|
| `graph/graph.py` | **수정 0줄.** `SelfHealingGraph`는 완전히 보존 |
| `graph/nodes.py` | **수정 0줄.** 파서, 린터, 메시지 빌더 보존 |
| `graph/state.py` | **수정 0줄.** `AgentState` TypedDict 보존 |
| `budget/` | **수정 0줄.** `BudgetTracker`, `BudgetGate` 보존 |
| `sandbox/` | **수정 0줄.** `SandboxExecutor` 보존 |
| `tools/` | **수정 0줄.** `ToolRegistry`, 빌트인 보존 |
| `memory/` | **수정 0줄.** `VFSManager` 보존 |
| `defense/` | **수정 0줄.** `SemanticLoopDetector` 보존 |
| `config.py` | **추가만.** `OrchestratorConfig`, `TelemetryConfig` dataclass 추가. 기존 5개 config 클래스 미변경 |
| `exceptions.py` | **추가만.** `OrchestratorError`, `SessionLaneTimeout` 추가. 기존 5개 예외 클래스 미변경 |
| `graph/__init__.py` | **추가만.** `OrchestratorGraph` export 추가. 기존 `SelfHealingGraph`, `AgentState` export 보존 |

### 전략 B: "새 파일에만 새 코드" 격리

- `infra/session_lane.py`, `infra/jitter.py` → 완전 신규
- `graph/orchestrator.py` → 완전 신규
- `telemetry/manager.py` → 완전 신규
- 기존 8개 테스트 파일에는 **단 한 줄도 추가/수정/삭제하지 않음**

### 전략 C: "의존성 단방향 강제"

```
orchestrator.py ──depends──▶ SelfHealingGraph   (import해서 .run() 호출)
orchestrator.py ──depends──▶ SessionLane         (인프라 계층)
orchestrator.py ──depends──▶ TelemetryManager    (옵저버)
orchestrator.py ──depends──▶ BudgetGate          (기존 것을 그대로 주입)

SelfHealingGraph ──────────▶ (기존 의존성만 유지, 오케스트레이터 몰라도 됨)
```

**역방향 의존성 금지**: `SelfHealingGraph`가 `OrchestratorGraph`를 import하는 일은 절대 없음. 오케스트레이터는 기존 워커를 **블랙박스**로 소비할 뿐이다.

### 전략 D: TDD 기반 단계별 검증

| 단계 | 작업 | 검증 |
|---|---|---|
| Task 1 커밋 후 | `infra/` 패키지 신규 작성 | `pytest` 전체 실행 → 147 + 신규 N개 = All Green |
| Task 2 커밋 후 | `orchestrator.py` 신규 작성 | `pytest` 전체 실행 → 147 + 신규 M개 = All Green |
| Task 3 커밋 후 | `telemetry/` 패키지 신규 작성 | `pytest` 전체 실행 → 147 + 신규 K개 = All Green |
| 최종 통합 | 전체 연동 테스트 | `pytest` 전체 실행 → 147 + 모든 신규 = All Green |

매 Task마다 `pytest`를 실행하여 기존 147개가 단 하나도 깨지지 않는 것을 증명한 뒤에만 다음 Task로 진행한다.

---

**요약**: 기존 코드를 **한 줄도 수정하지 않고**, 새 패키지(`infra/`, `telemetry/`)와 새 파일(`orchestrator.py`)만 추가하며, `config.py`와 `exceptions.py`에는 기존 클래스를 건드리지 않는 **순수 추가(append)** 만 수행한다. `SelfHealingGraph`를 블랙박스 워커로 소비하는 단방향 의존성으로 계층을 분리하여, 147개 테스트의 역호환성을 구조적으로 보장한다.
