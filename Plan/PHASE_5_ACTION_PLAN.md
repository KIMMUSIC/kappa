# [PHASE_5] 프로덕션 런칭: 패키징 및 실전 엔트리포인트(Entrypoint) 구축

## 1. Phase 5의 목표 (Objective)
Kappa Harness OS의 코어 엔진부터 터미널 대시보드까지 모든 개발이 성공적으로 완료되었다.
본 단계에서는 테스트 환경의 Fake/Mock 객체들을 걷어내고, 실제 LLM, Docker, MCP 서버와 통신하는 **프로덕션 진입점(main.py)**을 조립(Wiring)한다. 또한 누구나 이 시스템을 쉽게 이해하고 구동할 수 있도록 **최종 문서화(README.md)**를 완벽하게 작성하여 대장정을 마무리한다.

## 2. 작업 지시 사항 (Action Items)

### Task 1: 실전 엔트리포인트 조립 (`src/kappa/main.py`)
전체 컴포넌트를 엮어 CLI 대시보드를 띄우는 실행 스크립트를 작성한다.
*   **요구사항:**
    *   `.env` 파일 로드 및 각종 Config 객체 초기화.
    *   실제 `OpenAIProvider`(또는 `AnthropicProvider`), `DockerRuntime`, 실제 `VectorStore`, `MCPBridge` 인스턴스 생성.
    *   `OrchestratorGraph`에 위 인스턴스들과 `HITLInterceptor`를 주입(Dependency Injection).
    *   `argparse`를 사용해 터미널에서 `--goal`을 입력받아 `run_dashboard()`를 실행.

### Task 2: 마스터 README 및 환경변수 템플릿 (`README.md`, `.env.example`)
*   **요구사항:**
    *   `.env.example`: 시스템 구동에 필요한 API 키, Docker 설정, 예산 한도 등의 템플릿과 주석 작성.
    *   `README.md`: 프로젝트 비전(결정론적 하네스), 9-Pillars 핵심 컴포넌트, 아키텍처 다이어그램(Mermaid/ASCII), **설치 및 Quick Start (실행 명령어 예시)**를 완벽히 정리.

### Task 3: 첫 실전 시나리오(Goal) 제안
*   **요구사항:**
    *   이 거대한 시스템의 강력함(RAG, MCP, 샌드박스 자가치유 코딩, HITL)을 한 번에 체감할 수 있는 훌륭한 "첫 번째 실전 프롬프트(Goal)"를 기획하여 제안할 것.

## 3. 완료 검증 기준 (Definition of Done)
1. `python src/kappa/main.py --goal "..."` 명령어 실행 시, 실제 네트워크를 타고 대시보드가 정상적으로 구동될 준비가 되었는가?
2. `README.md`를 읽고 제3의 개발자가 시스템의 철학과 구조를 명확히 이해할 수 있는가?
3. 모든 작업 완료 후에도 기존 394개 테스트가 **100% 통과(ALL GREEN)** 상태를 유지하는가?