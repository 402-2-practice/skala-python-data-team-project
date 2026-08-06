"""분석 산출물을 읽어 report.md를 자동 생성한다.
담당: 이서현 (시각화·보고서)
"""

from __future__ import annotations

from datetime import datetime
import json

import pandas as pd

from src.config import REPORT_PATH, TABLE_DIR


def _read_json(filename: str) -> dict:
    return json.loads((TABLE_DIR / filename).read_text(encoding="utf-8"))


def _read_correlations() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "correlations.csv", index_col=0)


def generate_report() -> None:
    eda = _read_json("eda_summary.json")
    benchmark = _read_json("data_engine_benchmark.json")
    test = _read_json("welch_ttest.json")
    psm = _read_json("psm_result.json")
    sensitivity = _read_json("psm_sensitivity_result.json")
    model = _read_json("model_metrics.json")
    correlations = _read_correlations()

    # high_income과의 상관계수만 뽑아 절댓값 큰 순으로 정렬 — 8x8 전체 행렬을
    # 표로 나열하면 가독성이 떨어지므로, 목표변수와의 관계로 좁혀서 보여준다.
    # 전체 조합은 correlation_heatmap.png로 별도 확인할 수 있다.
    income_correlation = (
        correlations["high_income"].drop("high_income").sort_values(key=abs, ascending=False)
    )
    correlation_rows = "\n".join(
        f"| {name} | {value:.3f} |" for name, value in income_correlation.items()
    )

    significance = "통계적으로 유의했다" if test["significant_at_0_05"] else "통계적으로 유의하지 않았다"
    report = f"""# 대학 학위, 정말 값어치가 있을까?

생성 시각: {datetime.now():%Y-%m-%d %H:%M}

## 연구 질문

관측된 배경 특성이 유사한 사람들을 비교했을 때 대학 학위 보유와 고소득(연 5만 달러 초과) 사이의 연관성이 남아 있는지 분석한다.

## 데이터

- 정제 후 표본: {eda['dataset']['rows']:,}명
- 대학 학위 보유 비율: {eda['college_degree_distribution']['college_degree_rate'] * 100:.2f}%
- 전체 고소득 비율: {eda['target_distribution']['high_income_rate'] * 100:.2f}%
- Pandas 로딩 시간: {benchmark['pandas_median_seconds']:.6f}초
- Polars 로딩 시간: {benchmark['polars_median_seconds']:.6f}초

### 학력별 고소득률

[인터랙티브 차트 열기](outputs/figures/education_income_rate.html)

## 상관관계

수치형 변수 간 피어슨 상관계수를 계산했다. 고소득 여부(high_income)와의 상관은 다음과 같다.

| 변수 | 상관계수 |
|---|---|
{correlation_rows}

`education-num`(교육 연수)과 `college_degree`(대학 학위 보유)가 고소득과 가장 강하게 연관되어 있다. 다만 상관계수는 선형적 연관성일 뿐 인과관계를 의미하지 않으며, 다른 변수와의 교란 가능성은 이어지는 매칭 분석에서 다룬다. 전체 변수 조합의 상관관계는 히트맵으로 확인할 수 있다.

![상관관계 히트맵](outputs/figures/correlation_heatmap.png)

## 매칭 전 비교

- 비학위 집단 고소득률: {test['no_degree_mean']:.2%}
- 학위 집단 고소득률: {test['degree_mean']:.2%}
- 단순 비율 차이: {test['mean_difference'] * 100:.2f}%p
- p-value: {test['p_value']:.6g}

Welch t-test에서 두 집단의 평균 차이는 {significance}. 다만 이 결과만으로 인과관계를 주장할 수는 없다.

## 성향점수매칭 이후 비교

- 매칭 쌍: {psm['matched_pairs']:,}쌍
- 비학위 집단 고소득률: {psm['matched_no_degree_rate']:.2%}
- 학위 집단 고소득률: {psm['matched_degree_rate']:.2%}
- 매칭 후 비율 차이: {psm['matched_rate_difference'] * 100:.2f}%p
- p-value: {psm['p_value']:.6g}
- 최대 SMD(매칭 전 → 후): {psm['max_smd_before']:.3f} → {psm['max_smd_after']:.3f}

PSM은 교육 이전에 정해진 것으로 볼 수 있는 age, sex, race, native-country를 이용했다. 관측된 변수에 대해서만 집단을 비슷하게 만들며, 측정되지 않은 가정환경, 능력, 지역, 교육비 등의 교란요인은 통제하지 못한다. 따라서 결과는 인과효과의 확정적 증명이 아니라 조건부 연관성으로 해석한다.

현재 직업과 근무시간은 교육의 결과로 달라질 수 있는 매개변수이므로 주 매칭 변수에서 제외했다. 이 변수까지 통제한 분석은 교육이 직업과 근무 형태를 통해 소득에 영향을 주는 경로를 제거한 별도의 민감도 분석으로만 해석해야 한다.

## 직업·근무시간 포함 민감도 분석

- 비학위 집단 고소득률: {sensitivity['matched_no_degree_rate']:.2%}
- 학위 집단 고소득률: {sensitivity['matched_degree_rate']:.2%}
- 조건부 비율 차이: {sensitivity['matched_rate_difference'] * 100:.2f}%p
- p-value: {sensitivity['p_value']:.6g}
- 최대 SMD(매칭 전 → 후): {sensitivity['max_smd_before']:.3f} → {sensitivity['max_smd_after']:.3f}

이 결과는 현재 직업과 근무시간이 같은 사람 사이에서 남는 학위의 조건부 연관성을 뜻한다. 주 PSM과 차이가 크다면 직업·근무시간이 학위와 소득을 연결하는 경로일 가능성을 제시하지만, 별도의 매개효과 분석 없이 그 경로를 확정하지 않는다.

## 예측 모델

- Accuracy: {model['accuracy']:.3f}
- Precision: {model['precision']:.3f}
- Recall: {model['recall']:.3f}
- F1: {model['f1']:.3f}
- ROC-AUC: {model['roc_auc']:.3f}

예측 모델은 고소득 여부를 예측하는 보조 분석이다. 예측력이 높은 것과 대학 학위의 인과효과가 크다는 것은 서로 다른 주장이다.

### 혼동행렬

![모델 혼동행렬](outputs/figures/model_confusion_matrix.png)

테스트 데이터의 실제 소득 등급과 모델이 예측한 소득 등급을 비교한다.

### ROC 곡선

![모델 ROC 곡선](outputs/figures/model_roc_curve.png)

분류 임계값에 따른 거짓 양성률과 참 양성률의 관계를 보여준다.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n[리포트] 생성 완료: {REPORT_PATH}")
