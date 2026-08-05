# 대학 학위, 정말 값어치가 있을까?

Adult Census 데이터에서 대학 학위 보유와 고소득(연 5만 달러 초과)의 관계를 분석한다. 단순 집단 비교, Welch t-test, 성향점수매칭(PSM), 고소득 예측 모델을 하나의 재현 가능한 파이프라인으로 실행한다.

## 연구 질문

> 관측된 배경 특성이 비슷한 사람들을 비교했을 때 대학 학위 보유와 고소득 여부 사이의 연관성이 남아 있는가?

이 프로젝트는 관찰 데이터 분석이다. PSM으로 관측된 교란요인을 조정하지만 측정되지 않은 교란요인은 제거할 수 없다. 따라서 "대학 학위가 소득 증가의 원인임을 증명했다"가 아니라 "관측된 조건을 조정한 후에도 연관성이 남았다/남지 않았다"고 해석한다.

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`adult.csv`를 `data/raw/adult.csv`에 둔 다음 실행한다.

```bash
python main.py
```

개별 단계만 실행할 수도 있다.

```bash
python main.py --stage eda
python main.py --stage statistics
python main.py --stage visualization
python main.py --stage model
python main.py --stage report
```

다른 위치의 데이터를 사용할 때:

```bash
python main.py --data /path/to/adult.csv
```

## 테스트

```bash
pytest
```

각 담당 모듈이 약속한 산출물 계약(반환값, 생성 파일)을 합성 데이터로 검증한다. `tests/conftest.py`가
매 테스트마다 출력 경로를 임시 디렉터리로 격리하므로, 테스트를 실행해도 `outputs/`나 `report.md`의
실제 실행 결과는 덮어써지지 않는다.

## 파이프라인

```text
adult.csv
   ↓
공통 로딩·정제 (Pandas/Polars 비교, ?, 공백, 중복 처리)
   ↓
adult_cleaned.csv + high_income + college_degree
   ├── EDA ─────────────→ 요약표
   ├── 통계·PSM ────────→ t-test, 매칭 표본, 효과 차이
   ├── 시각화 ──────────→ Seaborn PNG, Plotly HTML (학위·소득·PSM 주제)
   └── ML Pipeline ─────→ 평가 지표, joblib 모델
        └── 모델 진단 시각화 → ROC curve, confusion matrix, 성능 지표 막대그래프
                            ↓
                         report.md
```

## 폴더 구조

```text
.
├── .github/
│   └── workflows/ci.yml       # 문법 검사 + pytest 실행
├── data/
│   ├── raw/                   # 수정하지 않는 원본
│   └── processed/              # 공통 정제 데이터
├── docs/
│   ├── ANALYSIS_DESIGN.md              # 변수 정의와 해석 원칙
│   ├── TEAM_WORKFLOW.md                # 역할, 브랜치, PR 규칙
│   ├── MODEL_SELECTION_LOG.md          # 모델 채택 실험 기록
│   ├── ML_MODELING_SUMMARY.md          # ML 모델링 결과 요약
│   └── STATISTICAL_ANALYSIS_SUMMARY.md # 통계·PSM 분석 요약
├── outputs/
│   ├── figures/                # PNG, HTML
│   ├── tables/                 # CSV, JSON
│   └── models/                 # joblib
├── src/
│   ├── config.py                # 공통 경로와 상수
│   ├── data.py                  # 로딩·정제·파생변수
│   ├── eda.py                   # EDA
│   ├── statistics.py            # t-test·PSM
│   ├── visualization.py         # 주제 중심 시각화 (Seaborn·Plotly)
│   ├── modeling.py              # sklearn Pipeline
│   ├── model_visualization.py   # 모델 진단 시각화 (ROC·confusion matrix)
│   └── report.py                # report.md 자동 생성
├── tests/                      # pytest 단위테스트
├── main.py                     # 전체 실행 진입점
├── pyproject.toml              # pytest 설정
├── requirements.txt
└── README.md
```

## 공통 데이터 계약

모든 담당자는 `src.data.load_and_clean()`의 결과를 사용한다. 각 모듈에서 데이터를 다시 정제하지 않는다.

- `high_income`: `>50K`이면 1, `<=50K`이면 0
- `college_degree`: Bachelors, Masters, Prof-school, Doctorate이면 1
- `education` 문자열의 공백과 마지막 마침표 제거
- `?`는 결측값으로 변환
- 완전 중복 행 제거
- `fnlwgt`는 개인 소득 특성이 아닌 표본 가중치이므로 예측 입력에서 제외
- `education-num`은 `education`과 중복되므로 예측 입력에서 제외

팀 협업 규칙은 `docs/TEAM_WORKFLOW.md`에서 확인한다.

