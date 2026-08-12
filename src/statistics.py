"""
Date: 05AUG2026
담당: 정민규 (통계·PSM)
Purpose: 대학 학위 보유 여부와 고소득 여부의 관계를 통계적으로 검정하고,
         성향점수매칭(PSM)으로 관측된 집단 차이를 조정한다.

Main Tasks:
    1. 학위·비학위 집단의 고소득률과 효과크기를 비교한다.
    2. 과제 요구사항인 Welch t-test를 수행하고 p-value를 해석한다.
    3. 교육 이전 특성을 이용한 주 PSM을 수행한다.
    4. 직업·근무시간을 포함한 민감도 PSM을 수행한다.
    5. 매칭 전후 SMD를 계산해 공변량 균형을 검증한다.
    6. 통계 결과를 JSON·CSV로 저장해 시각화와 보고서에서 재사용한다.

Caution:
    이 분석은 관찰 데이터에 기반한다. PSM은 데이터에 존재하는 변수만 조정하므로
    결과를 대학 학위의 확정적인 인과효과로 해석하지 않는다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import t as student_t
from scipy.stats import ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TABLE_DIR


# 교육의 총효과를 목표로 하므로 occupation, hours-per-week처럼 교육 이후에
# 결정될 수 있는 변수(매개변수)는 성향점수 계산에서 제외한다.
PSM_NUMERIC_COVARIATES = ["age"]
PSM_CATEGORICAL_COVARIATES = ["sex", "race", "native-country"]
SENSITIVITY_NUMERIC_COVARIATES = ["age", "hours-per-week"]
SENSITIVITY_CATEGORICAL_COVARIATES = [
    "sex",
    "race",
    "native-country",
    "occupation",
]

BALANCE_THRESHOLD = 0.1
MATCH_CANDIDATE_COUNT = 500


# ============================================================
# TASK 1. 통계 분석 입력 검증
# ============================================================

def _validate_columns(df: pd.DataFrame, required: list[str], analysis_name: str) -> None:
    """분석에 필요한 열이 모두 존재하는지 확인한다.

    잘못된 데이터가 통계 함수 내부에서 모호한 오류를 일으키는 것을 방지하기 위한
    공통 검증 함수다.
    """
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{analysis_name}에 필요한 열이 없습니다: {missing}")


def _validate_binary_groups(df: pd.DataFrame, treatment: str, outcome: str) -> None:
    """처리변수와 결과변수가 PSM에 필요한 0/1 구조인지 확인한다."""
    treatment_series = pd.to_numeric(
        df[treatment].dropna(),
        errors="raise",
    )

    outcome_series = pd.to_numeric(
        df[outcome].dropna(),
        errors="raise",
        )
    treatment_values = set(treatment_series.unique())
    outcome_values = set(outcome_series.unique())
    if treatment_values != {0, 1}:
        raise ValueError(f"{treatment}은 0과 1 두 집단을 모두 포함해야 합니다: {treatment_values}")
    if not outcome_values.issubset({0, 1}):
        raise ValueError(f"{outcome}은 0/1 이진 결과여야 합니다: {outcome_values}")


def _difference_confidence_interval(
    degree: pd.Series,
    no_degree: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Welch 방식으로 학위·비학위 집단 평균 차이의 신뢰구간을 계산한다.

    `high_income`이 0/1 변수이므로 집단 평균은 고소득률이며, 반환되는 구간은
    두 집단 고소득률 차이의 95% 신뢰구간으로 해석한다.
    """
    n_degree, n_no_degree = len(degree), len(no_degree)
    var_degree, var_no_degree = degree.var(ddof=1), no_degree.var(ddof=1)
    term_degree = var_degree / n_degree
    term_no_degree = var_no_degree / n_no_degree
    standard_error = float(np.sqrt(term_degree + term_no_degree))
    difference = float(degree.mean() - no_degree.mean())

    if standard_error == 0:
        return difference, difference

    numerator = (term_degree + term_no_degree) ** 2
    denominator = (term_degree**2 / (n_degree - 1)) + (term_no_degree**2 / (n_no_degree - 1))
    degrees_of_freedom = numerator / denominator
    critical = float(student_t.ppf(1 - (1 - confidence) / 2, degrees_of_freedom))
    return difference - critical * standard_error, difference + critical * standard_error


def _odds_ratio(degree: pd.Series, no_degree: pd.Series) -> float:
    """학위 집단과 비학위 집단의 고소득 오즈비를 계산한다.

    분할표에 빈 셀이 있으면 0으로 나누는 문제를 막기 위해 각 셀에 0.5를 더하는
    Haldane-Anscombe 보정을 적용한다.
    """
    degree_high = float(degree.sum())
    degree_low = float(len(degree) - degree_high)
    no_degree_high = float(no_degree.sum())
    no_degree_low = float(len(no_degree) - no_degree_high)
    cells = np.array([degree_high, degree_low, no_degree_high, no_degree_low])
    if (cells == 0).any():
        degree_high, degree_low, no_degree_high, no_degree_low = cells + 0.5
    return float((degree_high * no_degree_low) / (degree_low * no_degree_high))


# ============================================================
# TASK 2. 매칭 전 집단 비교 및 Welch t-test
# ============================================================

def welch_test(df: pd.DataFrame, treatment: str = "college_degree", outcome: str = "high_income") -> dict:
    """학위·비학위 집단의 고소득률 차이를 Welch t-test로 검정한다.

    Args:
        df: `college_degree`와 결과변수를 포함한 정제 데이터.
        outcome: 비교할 0/1 결과변수. 기본값은 `high_income`이다.

    Returns:
        표본 수, 집단별 고소득률, 비율 차이, 95% 신뢰구간, 위험비,
        오즈비, Cohen's h, t 통계량과 p-value를 담은 딕셔너리.

    Notes:
        과제 요구사항에 따라 이진 결과에 `ttest_ind(equal_var=False)`를 사용한다.
        이 결과는 매칭 전 단순 비교이므로 인과효과로 해석하지 않는다.
    """
    _validate_columns(df, [treatment, outcome], "Welch t-test")
    _validate_binary_groups(df, treatment, outcome)
    control = df.loc[df[treatment] == 0, outcome].dropna().astype(float)
    treated = df.loc[df[treatment] == 1, outcome].dropna().astype(float)
    if len(control) < 2 or len(treated) < 2:
        raise ValueError("Welch t-test에는 집단별로 최소 2개 표본이 필요합니다.")

    statistic, p_value = ttest_ind(treated, control, equal_var=False)
    difference = float(treated.mean() - control.mean())
    ci_low, ci_high = _difference_confidence_interval(treated, control)
    no_degree_rate = float(control.mean())
    degree_rate = float(treated.mean())
    risk_ratio = float(degree_rate / no_degree_rate) if no_degree_rate > 0 else None
    cohens_h = float(
        2 * np.arcsin(np.sqrt(degree_rate))
        - 2 * np.arcsin(np.sqrt(no_degree_rate))
    )
    significant = bool(np.isfinite(p_value) and p_value < 0.05)
    result = {
        "outcome": outcome,
        "treatment": treatment,
        "outcome": outcome,

        "treated_n": int(len(treated)),
        "control_n": int(len(control)),

        "treated_mean": float(
            treated.mean()
        ),

        "control_mean": float(
            control.mean()
        ),
        "mean_difference": difference,
        "difference_ci_95_low": float(ci_low),
        "difference_ci_95_high": float(ci_high),
        "risk_ratio": risk_ratio,
        "odds_ratio": _odds_ratio(treated, control),
        "cohens_h": cohens_h,
        "t_statistic": float(statistic),
        "p_value": float(p_value),
        "p_value_display": "< 1e-300" if p_value == 0 else f"{p_value:.6g}",
        "significant_at_0_05": significant,
        "interpretation": (
            "두 집단의 고소득률 차이는 통계적으로 유의하다. "
            "다만 이 매칭 전 비교만으로 대학 학위의 인과효과를 뜻하지 않는다."
            if significant
            else "두 집단의 고소득률 차이가 통계적으로 유의하다는 근거가 부족하다."
        ),
        "method_note": (
            "과제 요구사항에 따라 0/1 결과에 Welch t-test를 적용했다. "
            "평균은 고소득 비율이므로 평균 차이는 비율 차이로 해석한다."
        ),
    }
    (TABLE_DIR / "welch_ttest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


# ============================================================
# TASK 3. 매칭 전후 공변량 균형 진단
# ============================================================

def _smd_table(
    before: pd.DataFrame,
    after: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    """각 공변량의 매칭 전후 절대 표준화 평균차이(SMD)를 계산한다.

    범주형 변수는 더미변수로 변환하며, 일반적인 기준에 따라 절대 SMD가 0.1
    미만이면 처리집단과 대조집단의 균형이 양호한 것으로 판단한다.
    """

    def calculate(sample: pd.DataFrame) -> pd.Series:
        encoded = pd.get_dummies(
            sample[["college_degree", *covariates]],
            columns=[
                column
                for column in covariates
                if not pd.api.types.is_numeric_dtype(sample[column])
            ],
            dummy_na=True,
            dtype=float,
        )
        treated = encoded[encoded["college_degree"] == 1].drop(columns="college_degree")
        control = encoded[encoded["college_degree"] == 0].drop(columns="college_degree")
        treated_mean = treated.mean()
        control_mean = control.mean()

        mean_diff = treated_mean - control_mean
        pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        smd = mean_diff.abs() / pooled_sd

        zero_sd = pooled_sd == 0

        smd.loc[
            zero_sd & (mean_diff.abs() == 0)
        ] = 0.0

        smd.loc[
            zero_sd & (mean_diff.abs() > 0)
        ] = np.inf

        return smd

    return (
        pd.concat(
            [calculate(before).rename("smd_before"), calculate(after).rename("smd_after")],
            axis=1,
        )
        .fillna(0)
        .rename_axis("covariate")
        .reset_index()
        .sort_values("smd_after", ascending=False)
    )


# ============================================================
# TASK 4. 성향점수 계산 및 1:1 최근접 이웃 매칭
# ============================================================

def propensity_score_matching(
    df: pd.DataFrame,
    treatment: str = "college_degree",
    outcome: str = "high_income",
    numeric_covariates: list[str] | None = None,
    categorical_covariates: list[str] | None = None,
    with_replacement: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """성향점수를 이용해 학위 집단과 유사한 비학위 대조군을 매칭한다.

    Args:
        df: 처리변수, 결과변수와 매칭 공변량을 포함한 정제 데이터.
        numeric_covariates: 성향점수에 사용할 수치형 공변량.
        categorical_covariates: 성향점수에 사용할 범주형 공변량.
        with_replacement: True면 같은 대조군의 반복 매칭을 허용한다. 기본값은
            표본 독립성과 해석 가능성을 높이기 위해 False다.

    Returns:
        매칭된 처리·대조 표본 DataFrame과 매칭 품질·고소득률 차이 요약.

    Method:
        1. 로지스틱 회귀로 대학 학위 보유 성향점수를 계산한다.
        2. 두 집단의 공통지지 영역 밖에 있는 표본을 제외한다.
        3. logit 성향점수 표준편차의 0.2를 caliper로 사용한다.
        4. 기본적으로 대조군 중복 없는 1:1 최근접 이웃 매칭을 수행한다.
        5. 매칭 전후 절대 SMD를 비교해 관측 공변량 균형을 확인한다.

    Caution:
        균형 기준을 충족해도 관측되지 않은 교란요인은 통제되지 않는다.
    """
    numeric_covariates = list(numeric_covariates or PSM_NUMERIC_COVARIATES)
    categorical_covariates = list(categorical_covariates or PSM_CATEGORICAL_COVARIATES)
    columns = [
        treatment,
        outcome,
        *numeric_covariates,
        *categorical_covariates,
    ]
    _validate_columns(df, columns, "성향점수매칭")
    _validate_binary_groups(df, treatment, outcome)
    analysis = df[columns].dropna(subset=[treatment, outcome]).copy()
    # sklearn 버전별 pd.NA 처리 차이를 피하기 위해 범주형 결측치를 np.nan으로 통일한다.
    for column in categorical_covariates:
        analysis[column] = analysis[column].astype(object).where(analysis[column].notna(), np.nan)

    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric_covariates),
            ("categorical", categorical_pipe, categorical_covariates),
        ]
    )
    propensity_model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
        ]
    )

    covariates = numeric_covariates + categorical_covariates
    propensity_model.fit(analysis[covariates], analysis[treatment])
    analysis["propensity_score"] = propensity_model.predict_proba(analysis[covariates])[:, 1]

    treated = analysis[analysis[treatment] == 1].copy()
    control = analysis[analysis[treatment] == 0].copy()
    treated_before_support = len(treated)
    control_before_support = len(control)

    # 두 집단이 실제로 비교 가능한 성향점수 범위만 남긴다.
    lower = max(treated["propensity_score"].min(), control["propensity_score"].min())
    upper = min(treated["propensity_score"].max(), control["propensity_score"].max())
    treated = treated[treated["propensity_score"].between(lower, upper)]
    control = control[control["propensity_score"].between(lower, upper)]

    clipped = analysis["propensity_score"].clip(1e-6, 1 - 1e-6)
    analysis["propensity_logit"] = logit(clipped)
    treated = analysis.loc[treated.index].copy()
    control = analysis.loc[control.index].copy()
    caliper = 0.2 * float(analysis["propensity_logit"].std())
    neighbor_count = 1 if with_replacement else min(MATCH_CANDIDATE_COUNT, len(control))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count)
    neighbors.fit(control[["propensity_logit"]])
    distances, indices = neighbors.kneighbors(treated[["propensity_logit"]])

    pairs = []
    used_control_positions: set[int] = set()
    # 매칭이 어려운 처리대상부터 배정해 공통지지 경계에 있는 표본의 탈락을 줄인다.
    treated_order = np.argsort(distances[:, 0])[::-1]
    for treated_pos in treated_order:
        selected_control_pos = None
        selected_distance = None
        for distance, control_pos in zip(distances[treated_pos], indices[treated_pos]):
            control_pos = int(control_pos)
            if distance > caliper:
                break
            if with_replacement or control_pos not in used_control_positions:
                selected_control_pos = control_pos
                selected_distance = float(distance)
                break

        if selected_control_pos is not None:
            if not with_replacement:
                used_control_positions.add(selected_control_pos)
            treated_row = treated.iloc[int(treated_pos)]
            control_row = control.iloc[selected_control_pos]
            pairs.extend(
                [
                    {
                        **treated_row.to_dict(),
                        "source_index": treated_row.name,
                        "match_distance": selected_distance,
                        "pair_id": len(pairs) // 2,
                        "matched_role": "treated",
                    },
                    {
                        **control_row.to_dict(),
                        "source_index": control_row.name,
                        "match_distance": selected_distance,
                        "pair_id": len(pairs) // 2,
                        "matched_role": "control",
                    },
                ]
            )

    matched = pd.DataFrame(pairs)
    if matched.empty:
        raise RuntimeError("caliper 안에서 매칭된 표본이 없습니다.")

    matched_degree = matched.loc[matched["matched_role"] == "treated", "high_income"].astype(float)
    matched_control = matched.loc[matched["matched_role"] == "control", "high_income"].astype(float)
    paired = (
        matched
        .pivot(
            index="pair_id",
            columns="matched_role",
            values=outcome,
        )
        .dropna()
    )
    
    from scipy.stats import ttest_rel

    statistic, p_value = ttest_rel(
        paired["treated"].astype(float),
        paired["control"].astype(float),
    )
    matched_pairs = int(len(matched) / 2)
    control_sources = matched.loc[matched["matched_role"] == "control", "source_index"]
    unique_controls = int(control_sources.nunique())
    max_control_reuse = int(control_sources.value_counts().max())
    result = {
        "method": (
            "1:1 nearest-neighbor PSM with replacement and caliper"
            if with_replacement
            else "1:1 greedy nearest-neighbor PSM without replacement and caliper"
        ),
        "matching_with_replacement": with_replacement,
        "nearest_control_candidates": int(neighbor_count),
        "estimand": "ATT-like matched risk difference among retained degree holders",
        "covariates": covariates,
        "treated_before_common_support": int(treated_before_support),
        "control_before_common_support": int(control_before_support),
        "treated_in_common_support": int(len(treated)),
        "control_in_common_support": int(len(control)),
        "matched_pairs": matched_pairs,
        "treated_retention_rate": float(matched_pairs / treated_before_support),
        "unique_matched_controls": unique_controls,
        "control_reuse_rate": float(1 - unique_controls / matched_pairs),
        "max_control_reuse_count": max_control_reuse,
        "mean_match_distance": float(
            matched.loc[matched["matched_role"] == "treated", "match_distance"].mean()
        ),
        "common_support_lower": float(lower),
        "common_support_upper": float(upper),
        "caliper": float(caliper),
        "matched_no_degree_rate": float(matched_control.mean()),
        "matched_degree_rate": float(matched_degree.mean()),
        "matched_rate_difference": float(matched_degree.mean() - matched_control.mean()),
        "p_value": float(p_value),
        "p_value_display": "< 1e-300" if p_value == 0 else f"{p_value:.6g}",
        "t_statistic": float(statistic),
        "inference_warning": (
            "대조군 재사용으로 매칭쌍이 독립이 아닐 수 있다. "
            if with_replacement
            else "매칭으로 선택된 표본에 대한 검정이므로 일반 모집단 추론에는 한계가 있다. "
        )
        + (
            "매칭 후 t-test는 과제 요구에 따른 보조 결과이며, "
            "효과크기와 균형 진단을 함께 해석한다."
        ),
    }

    balance = _smd_table(analysis, matched, covariates)
    result["max_smd_before"] = float(balance["smd_before"].max())
    result["max_smd_after"] = float(balance["smd_after"].max())
    result["imbalanced_covariates_before"] = int(
        (balance["smd_before"] >= BALANCE_THRESHOLD).sum()
    )
    result["imbalanced_covariates_after"] = int(
        (balance["smd_after"] >= BALANCE_THRESHOLD).sum()
    )
    result["balanced_under_0_1"] = bool(
        (balance["smd_after"] < BALANCE_THRESHOLD).all()
    )
    result["balance_interpretation"] = (
        "모든 공변량의 매칭 후 절대 SMD가 0.1 미만으로, 관측 공변량 균형 기준을 충족했다."
        if result["balanced_under_0_1"]
        else "일부 공변량의 매칭 후 절대 SMD가 0.1 이상이므로 인과적 해석을 보류해야 한다."
    )

    return matched, balance, result


# ============================================================
# TASK 5. 통계 파이프라인 실행 및 산출물 저장
# ============================================================

def run_statistics(df: pd.DataFrame) -> tuple[dict, dict]:
    """매칭 전 비교, 주 PSM, 민감도 PSM을 순서대로 실행한다.

    주 PSM은 교육 이전 특성인 age, sex, race, native-country를 사용한다.
    민감도 분석은 직업과 근무시간을 추가하지만, 두 변수는 교육 이후 달라질 수
    있으므로 결과를 교육의 총효과가 아닌 조건부 연관성으로 해석한다.

    Returns:
        기존 `report.py` 계약을 유지하기 위해 Welch t-test 결과와 주 PSM 결과를
        튜플로 반환한다. 민감도 결과를 포함한 전체 요약은
        `statistics_summary.json`에 별도로 저장한다.
    """
    numeric = df.select_dtypes(include="number")
    numeric.corr().to_csv(TABLE_DIR / "correlations.csv")

    """
    Date: 06AUG2026
    Author: Jung Minkyu
    Purpose: fix: 상관계수 행렬 콘솔에 출력하기
    """
    print("\n[통계] 수치형 변수 간 상관계수 행렬:")
    print(numeric.corr().to_string())

    test_result = welch_test(df, treatment="college_degree", outcome="high_income",)
    _, _, psm_result = propensity_score_matching(df, treatment="college_degree", outcome="high_income")
    _, _, sensitivity_result = propensity_score_matching(
        df,
        treatment="college_degree",
        outcome="high_income",
        numeric_covariates=SENSITIVITY_NUMERIC_COVARIATES,
        categorical_covariates=SENSITIVITY_CATEGORICAL_COVARIATES,
    )
    combined_summary = {
        "analysis_question": (
            "관측된 배경 특성을 조정한 뒤에도 대학 학위 보유와 고소득 여부의 "
            "연관성이 남아 있는가?"
        ),
        "crude_comparison": test_result,
        "primary_psm": psm_result,
        "sensitivity_psm": sensitivity_result,
        "causal_caution": (
            "PSM은 관측된 변수만 조정한다. 능력, 가정환경, 지역, 교육비처럼 "
            "데이터에 없는 교란요인은 통제할 수 없으므로 인과관계의 확정적 증명이 아니다."
        ),
        "ml_interpretation_caution": (
            "예측 모델의 permutation importance 순위는 예측 기여도이며 인과효과 순위가 아니다. "
            "학력의 중요도가 1위가 아니어도 PSM 분석의 필요성이나 결과와 모순되지 않는다."
        ),
    }
    (TABLE_DIR / "statistics_summary.json").write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[통계] 매칭 전 Welch t-test")
    print(pd.Series(test_result).to_string())
    print("\n[통계] PSM 이후 결과")
    print(pd.Series(psm_result).to_string())
    print("\n[통계] 직업·근무시간 포함 민감도 분석")
    print(pd.Series(sensitivity_result).to_string())
    return test_result, psm_result
