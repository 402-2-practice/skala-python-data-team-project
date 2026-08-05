"""src.statistics의 Welch t-test와 성향점수매칭(PSM)을 합성 데이터로 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics import propensity_score_matching, welch_test


def test_welch_test_detects_clear_income_gap():
    df = pd.DataFrame(
        {
            "college_degree": [1] * 20 + [0] * 20,
            "high_income": [1] * 16 + [0] * 4 + [1] * 4 + [0] * 16,
        }
    )

    result = welch_test(df)

    assert result["degree_mean"] > result["no_degree_mean"]
    assert result["p_value"] < 0.05
    assert result["significant_at_0_05"] is True


def test_welch_test_reports_no_gap_for_identical_groups():
    df = pd.DataFrame(
        {
            "college_degree": [1] * 20 + [0] * 20,
            "high_income": ([1] * 10 + [0] * 10) * 2,
        }
    )

    result = welch_test(df)

    assert result["mean_difference"] == 0
    assert result["significant_at_0_05"] is False


def test_propensity_score_matching_balances_covariates():
    rng = np.random.default_rng(42)
    n = 200
    age = rng.integers(20, 60, size=n)
    sex = rng.choice(["Male", "Female"], size=n)
    race = rng.choice(["White", "Black"], size=n)
    country = rng.choice(["United-States", "Mexico"], size=n)
    # 배경(age)이 클수록 학위를 가질 확률이 높고, 학위가 있으면 고소득 확률도 높게 만든다.
    college_degree = (age + rng.normal(0, 5, size=n) > 45).astype(int)
    high_income = ((college_degree == 1) & (rng.random(n) < 0.6)) | (
        (college_degree == 0) & (rng.random(n) < 0.2)
    )

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "race": race,
            "native-country": country,
            "college_degree": college_degree,
            "high_income": high_income.astype(int),
        }
    )

    matched, result = propensity_score_matching(df, output_prefix="test_psm")

    assert result["matched_pairs"] > 0
    assert len(matched) == result["matched_pairs"] * 2
    # 매칭 후에는 처리/대조 집단의 공변량 불균형(SMD)이 매칭 전보다 작거나 같아야 한다.
    assert result["max_smd_after"] <= result["max_smd_before"]
