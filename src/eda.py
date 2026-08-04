"""기초 EDA와 품질 점검."""

from __future__ import annotations

import json

import pandas as pd

from src.config import TABLE_DIR


def run_eda(df: pd.DataFrame, load_comparison: dict) -> dict:
    summary = {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows_after_cleaning": int(df.duplicated().sum()),
        "college_degree_count": int(df["college_degree"].sum()),
        "college_degree_pct": round(df["college_degree"].mean() * 100, 2),
        "high_income_pct": round(df["high_income"].mean() * 100, 2),
        **load_comparison,
    }

    missing = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique_count": df.nunique(dropna=True),
        }
    ).sort_values("missing_pct", ascending=False)

    degree_summary = (
        df.groupby("college_degree", observed=True)
        .agg(
            sample_size=("high_income", "size"),
            high_income_count=("high_income", "sum"),
            high_income_rate=("high_income", "mean"),
            mean_age=("age", "mean"),
            mean_work_hours=("hours-per-week", "mean"),
        )
        .reset_index()
    )
    degree_summary["high_income_rate"] *= 100

    df.describe(include="all").T.to_csv(TABLE_DIR / "descriptive_statistics.csv")
    missing.to_csv(TABLE_DIR / "missing_summary.csv")
    degree_summary.to_csv(TABLE_DIR / "degree_summary.csv", index=False)
    (TABLE_DIR / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[EDA] 데이터 요약")
    print(pd.Series(summary).to_string())
    print("\n[EDA] 대학 학위별 고소득률")
    print(degree_summary.round(2).to_string(index=False))
    return summary

