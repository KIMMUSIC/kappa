# Kappa Project Instructions

## Project
- Python 3.11+ | pytest | LangGraph
- Source: `src/kappa/` | Tests: `tests/` | Docs: `docs/`

## Documentation Auto-Update
코드 변경(src/, tests/) 후에는 반드시 `docs/index.html`의 관련 섹션도 함께 갱신할 것.

| 변경 유형 | 갱신 대상 |
|---|---|
| 테스트 수 변경 | Test Results 섹션 숫자 + 총합 (47/47 등) |
| 새 모듈 추가 | Module Cards + Architecture SVG 다이어그램 |
| 새 Task/Phase 완료 | Development Timeline 섹션에 추가 |
| GEODE 파이프라인 변경 | Pipeline 섹션 레이어 갱신 |
| 노드/엣지 추가·삭제 | Architecture SVG 플로우 다이어그램 반영 |

## Color Palette (docs only)
`#f0b866` `#4ade80` `#60a5fa` `#818cf8` `#f87171` + `#08080a`~`#1c1c20` dark bg + white/gray만 사용.
