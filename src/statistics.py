"""기술통계, Welch t-test, 성향점수매칭(PSM)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import logit
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


def welch_test(df: pd.DataFrame, outcome: str = "high_income") -> dict:
    no_degree = df.loc[df["college_degree"] == 0, outcome].dropna().astype(float)
    degree = df.loc[df["college_degree"] == 1, outcome].dropna().astype(float)
    statistic, p_value = ttest_ind(degree, no_degree, equal_var=False)
    result = {
        "outcome": outcome,
        "no_degree_mean": float(no_degree.mean()),
        "degree_mean": float(degree.mean()),
        "mean_difference": float(degree.mean() - no_degree.mean()),
        "t_statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05),
    }
    (TABLE_DIR / "welch_ttest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _smd_table(
    before: pd.DataFrame,
    after: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    """각 공변량의 매칭 전후 표준화 평균차이(SMD)를 계산한다."""

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
        pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        smd = (treated.mean() - control.mean()) / pooled_sd.replace(0, np.nan)
        return smd.fillna(0).abs()

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


def propensity_score_matching(
    df: pd.DataFrame,
    numeric_covariates: list[str] | None = None,
    categorical_covariates: list[str] | None = None,
    output_prefix: str = "psm",
) -> tuple[pd.DataFrame, dict]:
    numeric_covariates = numeric_covariates or PSM_NUMERIC_COVARIATES
    categorical_covariates = categorical_covariates or PSM_CATEGORICAL_COVARIATES
    columns = [
        "college_degree",
        "high_income",
        *numeric_covariates,
        *categorical_covariates,
    ]
    analysis = df[columns].dropna(subset=["college_degree", "high_income"]).copy()
    # 일부 sklearn 버전은 pandas의 pd.NA를 직접 처리하지 못하므로 np.nan으로 통일한다.
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
    propensity_model.fit(analysis[covariates], analysis["college_degree"])
    analysis["propensity_score"] = propensity_model.predict_proba(analysis[covariates])[:, 1]

    treated = analysis[analysis["college_degree"] == 1].copy()
    control = analysis[analysis["college_degree"] == 0].copy()

    # 공통지지 영역 밖의 표본을 제거한다.
    lower = max(treated["propensity_score"].min(), control["propensity_score"].min())
    upper = min(treated["propensity_score"].max(), control["propensity_score"].max())
    treated = treated[treated["propensity_score"].between(lower, upper)]
    control = control[control["propensity_score"].between(lower, upper)]

    clipped = analysis["propensity_score"].clip(1e-6, 1 - 1e-6)
    analysis["propensity_logit"] = logit(clipped)
    treated = analysis.loc[treated.index].copy()
    control = analysis.loc[control.index].copy()
    caliper = 0.2 * float(analysis["propensity_logit"].std())
    neighbors = NearestNeighbors(n_neighbors=1)
    neighbors.fit(control[["propensity_logit"]])
    distances, indices = neighbors.kneighbors(treated[["propensity_logit"]])

    pairs = []
    for treated_pos, (distance, control_pos) in enumerate(zip(distances[:, 0], indices[:, 0])):
        if distance <= caliper:
            treated_row = treated.iloc[treated_pos]
            control_row = control.iloc[control_pos]
            pairs.extend(
                [
                    {
                        **treated_row.to_dict(),
                        "pair_id": len(pairs) // 2,
                        "matched_role": "degree",
                    },
                    {
                        **control_row.to_dict(),
                        "pair_id": len(pairs) // 2,
                        "matched_role": "no_degree",
                    },
                ]
            )

    matched = pd.DataFrame(pairs)
    if matched.empty:
        raise RuntimeError("caliper 안에서 매칭된 표본이 없습니다.")

    matched_degree = matched.loc[matched["matched_role"] == "degree", "high_income"].astype(float)
    matched_control = matched.loc[matched["matched_role"] == "no_degree", "high_income"].astype(float)
    statistic, p_value = ttest_ind(matched_degree, matched_control, equal_var=False)
    result = {
        "method": "1:1 nearest-neighbor PSM with replacement and caliper",
        "covariates": covariates,
        "matched_pairs": int(len(matched) / 2),
        "common_support_lower": float(lower),
        "common_support_upper": float(upper),
        "caliper": float(caliper),
        "matched_no_degree_rate": float(matched_control.mean()),
        "matched_degree_rate": float(matched_degree.mean()),
        "matched_rate_difference": float(matched_degree.mean() - matched_control.mean()),
        "p_value": float(p_value),
        "t_statistic": float(statistic),
    }

    balance = _smd_table(analysis, matched, covariates)
    result["max_smd_before"] = float(balance["smd_before"].max())
    result["max_smd_after"] = float(balance["smd_after"].max())
    result["balanced_under_0_1"] = bool((balance["smd_after"] < 0.1).all())

    matched.to_csv(TABLE_DIR / f"{output_prefix}_matched_sample.csv", index=False)
    balance.to_csv(TABLE_DIR / f"{output_prefix}_balance.csv", index=False)
    (TABLE_DIR / f"{output_prefix}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return matched, result


def run_statistics(df: pd.DataFrame) -> tuple[dict, dict]:
    numeric = df.select_dtypes(include="number")
    numeric.corr().to_csv(TABLE_DIR / "correlations.csv")
    test_result = welch_test(df)
    _, psm_result = propensity_score_matching(df, output_prefix="psm")
    _, sensitivity_result = propensity_score_matching(
        df,
        numeric_covariates=SENSITIVITY_NUMERIC_COVARIATES,
        categorical_covariates=SENSITIVITY_CATEGORICAL_COVARIATES,
        output_prefix="psm_sensitivity",
    )
    print("\n[통계] 매칭 전 Welch t-test")
    print(pd.Series(test_result).to_string())
    print("\n[통계] PSM 이후 결과")
    print(pd.Series(psm_result).to_string())
    print("\n[통계] 직업·근무시간 포함 민감도 분석")
    print(pd.Series(sensitivity_result).to_string())
    return test_result, psm_result
