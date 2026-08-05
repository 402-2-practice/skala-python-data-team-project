"""src.visualization.create_visualizations()가 요구 산출물(PNG·HTML)을 생성하는지 검증한다."""

from __future__ import annotations

import pandas as pd

import src.visualization as visualization
from src.visualization import create_visualizations


def test_create_visualizations_writes_png_and_html():
    df = pd.DataFrame(
        {
            "college_degree": [0, 0, 1, 1, 0, 1],
            "high_income": [0, 1, 1, 1, 0, 0],
            "education": ["HS-grad", "HS-grad", "Bachelors", "Bachelors", "HS-grad", "Masters"],
            "education-num": [9, 9, 13, 13, 9, 14],
        }
    )

    create_visualizations(df)

    png_path = visualization.FIGURE_DIR / "degree_income_rate.png"
    html_path = visualization.FIGURE_DIR / "education_income_rate.html"
    assert png_path.exists() and png_path.stat().st_size > 0
    assert html_path.exists() and html_path.stat().st_size > 0
