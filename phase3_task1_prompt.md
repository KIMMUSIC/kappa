제안해 준 Phase 3 설계안, 계층형 Super-Graph 흐름도, 그리고 100% 역호환성을 위한 4가지 전략(Additive-Only, 단방향 의존성 등)을 꼼꼼히 확인했다.

기존 코드를 단 한 줄도 수정하지 않고 새로운 상위 제어 평면을 얹는 설계 사상은 완벽한 개방-폐쇄 원칙(OCP)의 교과서다. 또한 동기 환경의 한계를 `ThreadPoolExecutor`와 `SyncSessionLane`으로 우아하게 돌파한 점과 텔레메트리에 옵저버 패턴을 적용한 점이 매우 인상적이다. 

설계안을 100% 전면 승인(Approve)한다!

[작업 실행 지시사항]
네가 약속한 '전략 D(TDD 기반 단계별 검증)'에 따라, 먼저 **[Task 1: 인프라 회복탄력성 - SessionLane & Jitter 백오프]** 관련 코드만 우선 작성해라. (Task 2 오케스트레이터나 Task 3 텔레메트리는 아직 절대 시작하지 마라)

1. `src/kappa/config.py` 및 `src/kappa/exceptions.py` 에 필요한 설정과 예외를 순수 추가(Append)해라.
2. `src/kappa/infra/session_lane.py` 와 `src/kappa/infra/jitter.py` 를 설계안대로 작성해라.
   - 다중 스레드(ThreadPoolExecutor) 환경과 비동기(Async) 환경 모두에서 안전하게 동작하도록 동기화 락(Lock) 처리에 각별히 신경 쓸 것.
3. Task 1 코드를 모두 작성했다면, 해당 인프라 모듈을 검증하는 테스트 코드(`tests/test_infra.py` 등)를 작성하고 스스로 실행해라.
   - SessionLane 검증: 여러 워커(스레드/비동기 태스크)가 동일한 키(key)에 동시 접근했을 때 경합(Race Condition) 없이 완벽히 직렬화(대기)되는가? 서로 다른 키에 대해서는 완전히 병렬로 실행되는가? 타임아웃 처리는 정상적인가?
   - Jitter 검증: Thundering Herd 현상을 막기 위해 재시도 대기 시간에 난수(Decorrelated Jitter)가 제대로 부여되며 지수 백오프가 동작하는가?
4. **[핵심 검증]:** 신규 작성한 인프라 테스트가 통과하는지 확인한 직후, `pytest`를 전체 실행하여 **기존 147개 테스트 역시 단 하나도 깨지지 않고 100% 통과(All Green)하는지 반드시 확인**해라.

테스트 결과가 모두 완벽하게 통과하면 그 결과를 나에게 브리핑해라. 결과를 확인한 뒤, 이 프로젝트의 거대 두뇌인 Task 2(Orchestrator) 진행을 승인하겠다. 즉시 Task 1 구현을 시작해라!