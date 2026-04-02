제안한 디렉토리 구조와 3단계 핵심 컴포넌트 설계안을 꼼꼼히 확인했다. 

Phase 1의 핵심 철학인 '결정론적 통제'와 '격리'를 완벽하게 이해했으며, 특히 샌드박스의 로우레벨 제약 조건(network=none 등)이나 LangGraph 노드의 조건부 엣지 설계가 아주 훌륭하다. 

설계안을 100% 승인(Approve)한다.

[LLM 프로바이더 선정]
LLM 프로바이더는 기본적으로 `Anthropic`를 활용하고, 모델은 `claude-3.5-sonnet`를 기준으로 세팅해라. (추후 다른 벤더로 교체가 쉽도록 인터페이스 추상화에 신경 쓸 것)

[작업 실행 지시사항]
안정적인 구현을 위해 절대 한 번에 전체 코드를 다 작성하지 마라. 다음 순서에 따라 엄격하게 쪼개서(TDD 방식) 진행해라.

1. 먼저 `pyproject.toml` (또는 `requirements.txt`) 및 `.env.example`을 생성하여 기초 의존성 환경을 세팅해라.
2. 그 다음 **[Task 1 (예산 게이트 및 회로 차단기)]** 에 해당하는 아래 파일들만 우선 작성해라.
   - `src/kappa/config.py`
   - `src/kappa/exceptions.py`
   - `src/kappa/budget/tracker.py`
   - `src/kappa/budget/gate.py`
   - `tests/test_budget.py`
3. Task 1 코드를 모두 작성했다면, 반드시 `pytest tests/test_budget.py`를 실행하여 의도적으로 예산을 초과시켰을 때 `BudgetExceededException`이 정상적으로 발생하며 강제 셧다운되는지 스스로 테스트하고, 그 결과를 나에게 브리핑해라.

Task 1의 테스트 통과가 확인되면 내가 Task 2 진행을 승인하겠다. 즉시 환경 세팅과 Task 1 코드 작성을 시작해라!