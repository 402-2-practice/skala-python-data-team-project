# 5인 GitHub 병렬 협업 구조

## 1. 역할 분담

역할은 작업 단계가 아니라 변경 파일의 소유권으로 나눈다. 공통 계약이 정해진 뒤 다섯 담당자가 동시에 개발할 수 있다.

| 담당 | 책임 | 주 변경 파일 | 완료 조건 |
|---|---|---|---|
| 윤찬웅 데이터·EDA | Pandas/Polars 비교, 정제, 결측·중복, 기술통계 | `src/data.py`, `src/eda.py` | 정제 CSV와 EDA 표 생성 |
| 정민규 통계·PSM | 상관계수, Welch t-test, 매칭, 균형 진단 | `src/statistics.py` | 매칭 전후 결과와 해석 생성 |
| 이서현 시각화·보고서 | Seaborn, Plotly, report.md 자동화 | `src/visualization.py`, `src/report.py` | PNG·HTML·Markdown 생성 |
| 원광식 ML 모델링 | sklearn Pipeline 학습, 평가지표, 모델 저장 | `src/modeling.py` | joblib·지표(Accuracy/Precision/Recall/F1/ROC-AUC) 생성 |
| 고동민 테스트·통합·문서 QA | pytest 단위테스트, CI 강화, 전체 실행 검증, README/문서 일관성, 제출 준비 | `tests/`, `main.py`, `.github/workflows/ci.yml`, `README.md` | pytest 전체 통과 + 전체 파이프라인 처음부터 성공 + 제출물 정리 |

원광식은 모델 구축에 집중하고, 고동민은 다른 사람의 구현을 대신하는 사람이 아니라 각 모듈이 약속한 계약대로 동작하는지 테스트로 증명하고 전체 실행과 문서 일관성을 확인하는 담당자다. 발표 자료와 최종 해석은 다섯 명이 공동 검토한다.

## 2. 공통 기능 소유권

`src/config.py`와 `src/data.py`의 반환 열은 모든 작업에 영향을 주는 공통 계약이다.

- 윤찬웅이 기본 소유자다.
- 열 이름을 추가·삭제·변경하는 PR은 최소 두 명이 리뷰한다.
- 공통 함수가 완성될 때까지 기다리지 않도록 함수 이름과 반환값을 먼저 합의한다.
- 담당 모듈 내부 함수는 각 담당자가 자유롭게 나눌 수 있다.

## 3. 브랜치

초기 뼈대를 `main`에 한 번 반영한 후 아래 브랜치를 동시에 만든다.

```text
feat/data-eda
feat/statistics-psm
feat/visualization-report
feat/modeling
feat/testing-qa
```

추가 수정은 짧은 브랜치를 사용한다.

```text
fix/income-label-cleaning
fix/psm-balance-check
docs/presentation-script
```

한 브랜치에서 여러 담당 영역을 동시에 수정하지 않는다. 그래야 PR이 작아지고 충돌이 줄어든다.

## 4. 병렬 작업 Flow

```text
main: 초기 폴더·인터페이스 합의
  ├─ 윤찬웅 feat/data-eda ───────────┐
  ├─ 정민규 feat/statistics-psm ────┤
  ├─ 이서현 feat/visualization-report ┤→ 각각 PR·리뷰·merge
  └─ 원광식 feat/modeling ───────────┘
                                      ↓
                    고동민 feat/testing-qa (각 모듈 단위테스트 작성)
                                      ↓
                          integration/final-check (고동민 담당)
                                      ↓
                         전체 실행·report 확인·발표 준비
```

각 담당자는 `main`에서 브랜치를 만들고 자기 모듈을 개발한다. 다른 담당자의 결과 파일이 아직 없으면 문서에 정한 열 이름과 출력 파일명을 기준으로 작은 샘플 데이터나 임시 결과를 사용한다. 고동민은 윤찬웅·정민규·이서현·원광식의 PR이 하나씩 merge될 때마다 해당 모듈의 테스트를 추가해나갈 수 있으며, 모든 PR이 끝나길 기다릴 필요는 없다.

## 5. Commit 규칙

커밋 한 개에는 한 가지 의도만 담는다.

```text
feat: add pandas and polars load comparison
feat: add propensity score matching
feat: add interactive education chart
test: add income label cleaning test
fix: handle trailing period in income labels
docs: explain causal interpretation limits
```

`수정`, `작업 완료`, `최종` 같은 메시지는 변경 의도를 알기 어려우므로 사용하지 않는다.

## 6. Pull Request 규칙

PR을 만들기 전에:

1. 최신 `main`을 자기 브랜치에 반영한다.
2. 담당 단계가 단독 실행되는지 확인한다.
3. 변경한 출력 파일과 실행 결과를 PR에 적는다.
4. 데이터 정의나 공통 함수 변경 여부를 표시한다.
5. 최소 한 명의 승인을 받은 후 merge한다.

권장 PR 크기는 하나의 담당 기능 또는 하나의 그래프 단위다. 큰 PR 하나를 마지막에 올리는 방식은 리뷰와 충돌 해결이 어렵다.

## 7. Merge 순서

인터페이스가 고정되어 있으므로 기능 PR은 원칙적으로 순서 없이 merge할 수 있다. 단, `report.py`는 통계와 모델의 JSON 산출물 이름에 의존하므로 다음을 확인한다.

- `eda_summary.json`
- `welch_ttest.json`
- `psm_result.json`
- `psm_sensitivity_result.json`
- `model_metrics.json`

마지막에는 고동민이 `integration/final-check` 브랜치에서 처음부터 전체 실행한다. 이 브랜치에는 새 분석 기능을 넣지 않고 연결 오류만 수정한다.

## 8. 충돌 방지 규칙

- 모든 사람이 `main.py`를 수정하지 않는다. 단계 연결 변경은 고동민에게 요청한다.
- 모든 사람이 `data.py`를 수정하지 않는다. 데이터 계약 변경은 윤찬웅의 PR에서 처리한다.
- 원본 `adult.csv`, 생성 모델, 대용량 출력 파일은 Git에 커밋하지 않는다.
- 그래프 파일 이름과 테이블 파일 이름은 문서에 정한 이름을 유지한다.
- PR에서 자동 포맷 때문에 담당 외 파일 전체가 바뀌지 않도록 한다.

## 9. 완료 정의

- 새 가상환경에서 `pip install -r requirements.txt`가 성공한다.
- `python main.py` 한 번으로 전처리부터 report.md까지 생성된다.
- `pytest`가 전부 통과한다.
- Pandas와 Polars 결과 행 수가 일치한다.
- Seaborn PNG와 Plotly HTML이 각각 하나 이상 생성된다.
- t-test의 p-value와 해석이 report에 포함된다.
- PSM 결과를 인과관계의 확정적 증명으로 표현하지 않는다.
- 모델 Accuracy, Precision, Recall, F1, ROC-AUC가 생성된다.
- `income_pipeline.joblib`이 생성된다.