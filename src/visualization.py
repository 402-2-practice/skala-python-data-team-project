"""Seaborn 정적 시각화와 Plotly 인터랙티브 시각화."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.config import FIGURE_DIR


def create_visualizations(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    degree_rate = (
        df.groupby("college_degree", observed=True)["high_income"]
        .mean()
        .mul(100)
        .rename("high_income_rate")
        .reset_index()
    )
    degree_rate["degree_label"] = degree_rate["college_degree"].map(
        {0: "No college degree", 1: "College degree"}
    )

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=degree_rate, x="degree_label", y="high_income_rate", hue="degree_label", legend=False)
    ax.set(title="High-income rate by college degree", xlabel="Degree group", ylabel="High-income rate (%)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "degree_income_rate.png", dpi=160)
    plt.close()

    education_rate = (
        df.groupby(["education", "education-num"], observed=True)["high_income"]
        .agg(["mean", "size"])
        .reset_index()
        .sort_values("education-num")
    )
    education_rate["high_income_rate"] = education_rate["mean"] * 100
    fig = px.bar(
        education_rate,
        x="education",
        y="high_income_rate",
        hover_data=["size", "education-num"],
        title="Interactive high-income rate by education",
        labels={"education": "Education", "high_income_rate": "High-income rate (%)"},
    )
    fig.write_html(FIGURE_DIR / "education_income_rate.html", include_plotlyjs="cdn")

