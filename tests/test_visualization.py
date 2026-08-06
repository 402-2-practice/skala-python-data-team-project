"""src.visualization.create_visualizations()가 요구 산출물(PNG·HTML)을 생성하는지 검증한다."""

from __future__ import annotations

import pandas as pd

import src.visualization as visualization
from src.visualization import create_visualizations


def test_create_visualizations_writes_expected_figures():
    df = pd.DataFrame(
        {
            "high_income": [0, 1, 1, 1, 0, 0],
            "education": ["HS-grad", "HS-grad", "Bachelors", "Bachelors", "HS-grad", "Masters"],
            "education-num": [9, 9, 13, 13, 9, 14],
            "age": [22, 35, 41, 29, 50, 38],
            "income": ["<=50K", ">50K", ">50K", ">50K", "<=50K", "<=50K"],
            "capital-gain": [0, 0, 5000, 0, 0, 2000],
            "race": ["White", "White", "Black", "White", "Asian-Pac-Islander", "White"],
            "sex": ["Male", "Female", "Male", "Female", "Male", "Female"],
        }
    )

    # PSM 산출물(welch_ttest.json/psm_result.json 등)이 없어도 create_visualizations는
    # 경고만 출력하고 넘어가야 한다 — statistics 단계 없이 시각화만 단독 실행하는 경우를 검증.
    create_visualizations(df)

    expected_files = [
        "education_income_rate.html",
        "age_distribution_by_income.png",
        "capital_gain_indicator_income_rate.png",
        "race_sex_income_rate.html",
    ]
    for filename in expected_files:
        path = visualization.FIGURE_DIR / filename
        assert path.exists() and path.stat().st_size > 0
