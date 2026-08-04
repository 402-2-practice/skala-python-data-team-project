"""공통 데이터 계약: 로딩, 정제, 파생변수 생성."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.config import COLLEGE_DEGREES


def load_pandas(path: Path) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    df = pd.read_csv(
        path,
        skipinitialspace=True,
        na_values=["?", " ?", "", "NA", "N/A"],
    )
    return df, time.perf_counter() - start


def load_polars(path: Path):
    """Polars는 이 함수 안에서 import해 의존성 오류 위치를 명확히 한다."""
    import polars as pl

    start = time.perf_counter()
    df = pl.read_csv(path, null_values=["?", " ?", "", "NA", "N/A"])
    return df, time.perf_counter() - start


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned.columns = cleaned.columns.str.strip()
    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.rstrip(".")
        cleaned[column] = cleaned[column].replace({"?": pd.NA, "": pd.NA})

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    required = {"education", "income"}
    missing = required - set(cleaned.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")

    income_map = {"<=50K": 0, ">50K": 1}
    cleaned["high_income"] = cleaned["income"].map(income_map).astype("Int64")
    if cleaned["high_income"].isna().any():
        unknown = cleaned.loc[cleaned["high_income"].isna(), "income"].unique().tolist()
        raise ValueError(f"알 수 없는 income 값이 있습니다: {unknown}")

    cleaned["college_degree"] = cleaned["education"].isin(COLLEGE_DEGREES).astype(int)
    return cleaned


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict[str, float | int]]:
    raw, pandas_seconds = load_pandas(path)
    polars_df, polars_seconds = load_polars(path)
    cleaned = clean_data(raw)

    load_comparison = {
        "pandas_rows": len(raw),
        "polars_rows": polars_df.height,
        "pandas_seconds": round(pandas_seconds, 6),
        "polars_seconds": round(polars_seconds, 6),
    }
    return cleaned, load_comparison

