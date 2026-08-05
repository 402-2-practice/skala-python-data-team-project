from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median
from typing import Literal

import pandas as pd
import polars as pl

from .config import (
    ADULT_COLUMNS,
    COLLEGE_DEGREES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TABLE_DIR,
    ensure_directories,
)

"""Adult Income 데이터 로딩, 정제 및 Pandas/Polars 성능 비교."""

"""main.py에서 사용 시 : 

cleaned_df, comparison = run_data_pipeline()

반환 값 : 
cleaned_df : 정제된 pandas.DataFrame
comparison : Pandas/Polars 성능 비교 딕셔너리
"""

Backend = Literal["auto", "pandas", "polars"]

BENCHMARK_RESULT_PATH = (
    TABLE_DIR / "data_engine_benchmark.json"
)

VALID_INCOME_LABELS = {
    "<=50K",
    ">50K",
}


# ============================================================
# 공통 검증
# ============================================================

def _validate_data_path(path: Path) -> None:
    """입력 데이터 경로가 유효한지 확인한다."""

    if not path.exists():
        raise FileNotFoundError(
            f"Adult 데이터 파일을 찾을 수 없습니다: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"데이터 경로가 파일이 아닙니다: {path}"
        )


def _has_header(path: Path) -> bool:
    """CSV 첫 행이 열 이름인지 확인한다."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as file:
        first_line = file.readline()

    if not first_line:
        raise ValueError(
            f"데이터 파일이 비어 있습니다: {path}"
        )

    first_value = (
        first_line
        .split(",", maxsplit=1)[0]
        .strip()
        .lower()
    )

    return first_value == "age"


def _validate_columns(
    columns: list[str],
) -> None:
    """Adult 데이터의 필수 열이 존재하는지 검사한다."""

    duplicate_columns = [
        column
        for column in set(columns)
        if columns.count(column) > 1
    ]

    if duplicate_columns:
        raise ValueError(
            "중복된 열 이름이 있습니다: "
            f"{sorted(duplicate_columns)}"
        )

    missing_columns = [
        column
        for column in ADULT_COLUMNS
        if column not in columns
    ]

    unexpected_columns = [
        column
        for column in columns
        if column not in ADULT_COLUMNS
    ]

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Adult 데이터 열 구성이 올바르지 않습니다. "
            f"누락 열={missing_columns}, "
            f"예상하지 못한 열={unexpected_columns}"
        )


# ============================================================
# 데이터 로드
# ============================================================

def load_pandas(
    path: str | Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Adult 데이터를 Pandas DataFrame으로 불러온다."""

    path = Path(path)
    _validate_data_path(path)

    has_header = _has_header(path)

    read_options: dict[str, object] = {
        "na_values": ["?", " ?"],
        "skipinitialspace": True,
    }

    if has_header:
        df = pd.read_csv(
            path,
            **read_options,
        )
    else:
        df = pd.read_csv(
            path,
            header=None,
            names=ADULT_COLUMNS,
            **read_options,
        )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    _validate_columns(
        list(df.columns)
    )

    return df


def load_polars(
    path: str | Path = RAW_DATA_PATH,
) -> pl.DataFrame:
    """Adult 데이터를 Polars DataFrame으로 불러온다."""

    path = Path(path)
    _validate_data_path(path)

    has_header = _has_header(path)

    read_options: dict[str, object] = {
        "has_header": has_header,
        "null_values": ["?", " ?"],
        "infer_schema_length": 10_000,
    }

    if not has_header:
        read_options["new_columns"] = ADULT_COLUMNS

    df = pl.read_csv(
        path,
        **read_options,
    )

    rename_mapping = {
        column: column.strip()
        for column in df.columns
        if column != column.strip()
    }

    if rename_mapping:
        df = df.rename(rename_mapping)

    _validate_columns(df.columns)

    return df


# ============================================================
# Pandas 정제
# ============================================================

def _normalize_pandas_strings(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Pandas 문자열 열의 공백과 소득 마침표를 정리한다."""

    result = df.copy()

    string_columns = result.select_dtypes(
        include=["object", "string"],
    ).columns

    for column in string_columns:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
        )

    # adult.test 형식에서 income 끝에 붙는 마침표만 제거
    result["income"] = (
        result["income"]
        .str.rstrip(".")
    )

    result[string_columns] = (
        result[string_columns]
        .replace(
            {
                "?": pd.NA,
                "": pd.NA,
            }
        )
    )

    return result


def _valid_pandas_rows(
    df: pd.DataFrame,
) -> pd.Series:
    """논리적으로 유효한 행을 판별한다."""

    return (
        df["age"].between(1, 120)
        & df["hours-per-week"].between(1, 168)
        & df["education-num"].between(1, 20)
        & df["fnlwgt"].gt(0)
        & df["capital-gain"].ge(0)
        & df["capital-loss"].ge(0)
        & df["income"].isin(VALID_INCOME_LABELS)
    )


def clean_with_pandas(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pandas로 Adult 데이터를 정제한다."""

    result = _normalize_pandas_strings(df)

    initial_rows = len(result)

    rows_before_missing = len(result)

    result = (
        result
        .dropna()
        .reset_index(drop=True)
    )

    missing_removed = (
        rows_before_missing - len(result)
    )

    rows_before_duplicates = len(result)

    result = (
        result
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicate_removed = (
        rows_before_duplicates - len(result)
    )

    valid_mask = _valid_pandas_rows(result)

    invalid_removed = int(
        (~valid_mask).sum()
    )

    result = (
        result
        .loc[valid_mask]
        .copy()
        .reset_index(drop=True)
    )

    result["high_income"] = (
        result["income"]
        .eq(">50K")
        .astype("int8")
    )

    result["college_degree"] = (
        result["education"]
        .isin(COLLEGE_DEGREES)
        .astype("int8")
    )

    cleaning_info = {
        "initial_rows": initial_rows,
        "missing_removed": missing_removed,
        "duplicate_removed": duplicate_removed,
        "invalid_removed": invalid_removed,
        "rows_after_cleaning": len(result),
        "columns_after_cleaning": len(result.columns),
    }

    return result, cleaning_info


# ============================================================
# Polars 정제
# ============================================================

def _normalize_polars_strings(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Polars 문자열 열의 공백과 소득 마침표를 정리한다."""

    string_columns = [
        column
        for column, dtype in df.schema.items()
        if dtype == pl.String
    ]

    result = df.with_columns(
        [
            pl.col(column)
            .str.strip_chars()
            .alias(column)
            for column in string_columns
        ]
    )

    result = result.with_columns(
        pl.col("income")
        .str.strip_chars_end(".")
        .alias("income")
    )

    result = result.with_columns(
        [
            pl.when(
                pl.col(column).is_in(["?", ""])
            )
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
            for column in string_columns
        ]
    )

    return result


def _valid_polars_rows() -> pl.Expr:
    """논리적으로 유효한 Polars 행 조건을 반환한다."""

    return (
        (pl.col("age") >= 1)
        & (pl.col("age") <= 120)
        & (pl.col("hours-per-week") >= 1)
        & (pl.col("hours-per-week") <= 168)
        & (pl.col("education-num") >= 1)
        & (pl.col("education-num") <= 20)
        & (pl.col("fnlwgt") > 0)
        & (pl.col("capital-gain") >= 0)
        & (pl.col("capital-loss") >= 0)
        & pl.col("income").is_in(
            list(VALID_INCOME_LABELS)
        )
    )


def clean_with_polars(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Polars로 Adult 데이터를 정제한다."""

    result = _normalize_polars_strings(df)

    initial_rows = result.height

    rows_before_missing = result.height

    result = result.drop_nulls()

    missing_removed = (
        rows_before_missing - result.height
    )

    rows_before_duplicates = result.height

    result = result.unique(
        maintain_order=True,
    )

    duplicate_removed = (
        rows_before_duplicates - result.height
    )

    rows_before_invalid = result.height

    result = result.filter(
        _valid_polars_rows()
    )

    invalid_removed = (
        rows_before_invalid - result.height
    )

    result = result.with_columns(
        [
            (
                pl.col("income") == ">50K"
            )
            .cast(pl.Int8)
            .alias("high_income"),

            pl.col("education")
            .is_in(list(COLLEGE_DEGREES))
            .cast(pl.Int8)
            .alias("college_degree"),
        ]
    )

    cleaning_info = {
        "initial_rows": initial_rows,
        "missing_removed": missing_removed,
        "duplicate_removed": duplicate_removed,
        "invalid_removed": invalid_removed,
        "rows_after_cleaning": result.height,
        "columns_after_cleaning": result.width,
    }

    return result, cleaning_info


# ============================================================
# 결과 변환 및 일치 검증
# ============================================================

def _polars_to_pandas(
    df: pl.DataFrame,
) -> pd.DataFrame:
    """추가 변환 라이브러리 없이 Polars를 Pandas로 바꾼다."""

    return pd.DataFrame(
        df.to_dict(as_series=False)
    )


def validate_cleaning_results(
    pandas_df: pd.DataFrame,
    polars_df: pl.DataFrame,
) -> None:
    """Pandas와 Polars 정제 결과가 같은지 검사한다."""

    converted_polars = (
        _polars_to_pandas(polars_df)
        .reset_index(drop=True)
    )

    pandas_result = (
        pandas_df
        .reset_index(drop=True)
    )

    try:
        pd.testing.assert_frame_equal(
            pandas_result,
            converted_polars,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(
            "Pandas와 Polars의 정제 결과가 다릅니다. "
            "성능 비교 전에 정제 계약을 확인해야 합니다."
        ) from error


# ============================================================
# 엔진 실행 및 성능 비교
# ============================================================

def _run_pandas_pipeline(
    path: Path,
) -> tuple[
    pd.DataFrame,
    float,
    dict[str, int],
]:
    """Pandas 로딩과 정제의 전체 시간을 측정한다."""

    start = time.perf_counter()

    raw_df = load_pandas(path)

    cleaned_df, cleaning_info = (
        clean_with_pandas(raw_df)
    )

    elapsed = (
        time.perf_counter() - start
    )

    return (
        cleaned_df,
        elapsed,
        cleaning_info,
    )


def _run_polars_pipeline(
    path: Path,
) -> tuple[
    pl.DataFrame,
    float,
    dict[str, int],
]:
    """Polars 로딩과 정제의 전체 시간을 측정한다."""

    start = time.perf_counter()

    raw_df = load_polars(path)

    cleaned_df, cleaning_info = (
        clean_with_polars(raw_df)
    )

    elapsed = (
        time.perf_counter() - start
    )

    return (
        cleaned_df,
        elapsed,
        cleaning_info,
    )


def benchmark_cleaners(
    path: str | Path = RAW_DATA_PATH,
    repeats: int = 3,
) -> tuple[
    pd.DataFrame,
    pl.DataFrame,
    dict[str, object],
]:
    """Pandas와 Polars의 전체 로딩·정제 성능을 비교한다."""

    path = Path(path)
    _validate_data_path(path)

    if repeats < 1:
        raise ValueError(
            "repeats는 1 이상이어야 합니다."
        )

    pandas_times: list[float] = []
    polars_times: list[float] = []

    pandas_cleaned: pd.DataFrame | None = None
    polars_cleaned: pl.DataFrame | None = None

    pandas_info: dict[str, int] = {}
    polars_info: dict[str, int] = {}

    for _ in range(repeats):
        (
            pandas_cleaned,
            pandas_elapsed,
            pandas_info,
        ) = _run_pandas_pipeline(path)

        pandas_times.append(
            pandas_elapsed
        )

        (
            polars_cleaned,
            polars_elapsed,
            polars_info,
        ) = _run_polars_pipeline(path)

        polars_times.append(
            polars_elapsed
        )

    if pandas_cleaned is None or polars_cleaned is None:
        raise RuntimeError(
            "성능 비교 결과를 생성하지 못했습니다."
        )

    validate_cleaning_results(
        pandas_cleaned,
        polars_cleaned,
    )

    pandas_median = median(
        pandas_times
    )

    polars_median = median(
        polars_times
    )

    selected_engine = (
        "pandas"
        if pandas_median <= polars_median
        else "polars"
    )

    comparison: dict[str, object] = {
        "repeats": repeats,
        "results_match": True,
        "pandas_seconds": [
            round(value, 6)
            for value in pandas_times
        ],
        "polars_seconds": [
            round(value, 6)
            for value in polars_times
        ],
        "pandas_median_seconds": round(
            pandas_median,
            6,
        ),
        "polars_median_seconds": round(
            polars_median,
            6,
        ),
        "selected_engine": selected_engine,
        "pandas_cleaning_info": pandas_info,
        "polars_cleaning_info": polars_info,
    }

    return (
        pandas_cleaned,
        polars_cleaned,
        comparison,
    )


# ============================================================
# 저장
# ============================================================

def save_processed_data(
    df: pd.DataFrame,
) -> None:
    """공통 정제 데이터를 CSV로 저장한다."""

    ensure_directories()

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )


def save_benchmark_result(
    comparison: dict[str, object],
) -> None:
    """Pandas/Polars 비교 결과를 JSON으로 저장한다."""

    ensure_directories()

    with BENCHMARK_RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 전체 데이터 파이프라인
# ============================================================

def run_data_pipeline(
    path: str | Path = RAW_DATA_PATH,
    backend: Backend = "auto",
    benchmark_repeats: int = 3,
    save_output: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    데이터를 정제하고 Pandas DataFrame으로 반환한다.

    backend:
        auto:
            Pandas와 Polars를 모두 실행하고 더 빠른 엔진 선택

        pandas:
            Pandas만 실행

        polars:
            Polars만 실행
    """

    path = Path(path)

    if backend == "auto":
        (
            pandas_df,
            polars_df,
            comparison,
        ) = benchmark_cleaners(
            path=path,
            repeats=benchmark_repeats,
        )

        if comparison["selected_engine"] == "pandas":
            cleaned_df = pandas_df
        else:
            cleaned_df = _polars_to_pandas(
                polars_df
            )

    elif backend == "pandas":
        (
            cleaned_df,
            elapsed,
            cleaning_info,
        ) = _run_pandas_pipeline(path)

        comparison = {
            "selected_engine": "pandas",
            "pandas_median_seconds": round(
                elapsed,
                6,
            ),
            "pandas_cleaning_info": cleaning_info,
        }

    elif backend == "polars":
        (
            polars_df,
            elapsed,
            cleaning_info,
        ) = _run_polars_pipeline(path)

        cleaned_df = _polars_to_pandas(
            polars_df
        )

        comparison = {
            "selected_engine": "polars",
            "polars_median_seconds": round(
                elapsed,
                6,
            ),
            "polars_cleaning_info": cleaning_info,
        }

    else:
        raise ValueError(
            "backend는 'auto', 'pandas', "
            "'polars' 중 하나여야 합니다."
        )

    cleaned_df = (
        cleaned_df
        .reset_index(drop=True)
    )

    if save_output:
        save_processed_data(cleaned_df)
        save_benchmark_result(comparison)

    return cleaned_df, comparison


# ============================================================
# 다른 모듈에서 사용하는 공통 인터페이스
# ============================================================

def load_and_clean(
    path: str | Path = RAW_DATA_PATH,
    backend: Backend = "auto",
    save_output: bool = True,
) -> pd.DataFrame :
    """
    팀의 모든 분석 모듈이 사용하는 공통 함수.

    반환형은 항상 Pandas DataFrame이다.
    """

    cleaned_df,__ = run_data_pipeline(
        path=path,
        backend=backend,
        save_output=save_output,
    )

    return cleaned_df
