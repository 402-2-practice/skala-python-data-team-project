"""Adult Census 정제 데이터에 대한 탐색적 데이터 분석(EDA).

담당: 윤찬웅 (데이터·EDA)

이 모듈은 결측 행, 중복 행 및 유효 범위 밖의 행을 직접 제거하지 않는다.

EDA는 ``src.data.load_before_cleaning()``이 반환하는 데이터를 사용한다.
이 데이터는 문자열과 결측 표현만 정규화된 상태이며,
결측·중복·유효성 기준에 따른 행 삭제는 수행되지 않은 상태다.

main.py에서 사용 시:

eda_result = run_eda(
    data_path=RAW_DATA_PATH,
)

반환 값 : 

eda_result["summary"]
eda_result["tables"]
eda_result["output_paths"]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DESCRIPTIVE_STATS_PATH,
    DUPLICATE_RESULT_PATH,
    EDA_SUMMARY_PATH,
    MISSING_VALUES_PATH,
    RAW_DATA_PATH,
    TABLE_DIR,
    ensure_directories,
)
from src.data import load_before_cleaning, Backend

# config.py에 아직 별도 상수가 없으므로 EDA 전용 산출물은 여기서 정의한다.
# 팀에서 경로를 공통 관리하기로 합의하면 config.py로 옮겨도 된다.
COLLEGE_INCOME_SUMMARY_PATH = TABLE_DIR / "college_income_summary.csv"
EDUCATION_INCOME_SUMMARY_PATH = TABLE_DIR / "education_income_summary.csv"
CATEGORICAL_SUMMARY_PATH = TABLE_DIR / "categorical_summary.csv"
NUMERIC_GROUP_SUMMARY_PATH = TABLE_DIR / "numeric_group_summary.csv"

REQUIRED_COLUMNS = {
    "education",
    "income",
    "high_income",
    "college_degree",
}

COLLEGE_GROUP_LABELS = {
    0: "No college degree",
    1: "College degree",
}


# ============================================================
# 입력 검증
# ============================================================

def validate_eda_input(df: pd.DataFrame) -> None:
    """EDA에 필요한 공통 데이터 계약을 검증한다."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "EDA 입력은 pandas.DataFrame이어야 합니다. "
            f"현재 타입: {type(df).__name__}"
        )

    if df.empty:
        raise ValueError("EDA 입력 데이터가 비어 있습니다.")

    duplicate_columns = (
        pd.Index(df.columns)
        .duplicated(keep=False)
    )

    if duplicate_columns.any():
        duplicated = sorted(
            set(
                pd.Index(df.columns)[duplicate_columns]
                .astype(str)
                .tolist()
            )
        )
        raise ValueError(
            f"EDA 입력에 중복 열 이름이 있습니다: {duplicated}"
        )

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "EDA에 필요한 열이 없습니다: "
            f"{missing_columns}"
        )


    for column in (
        "high_income",
        "college_degree",
    ):
        invalid_mask = (
            df[column].notna()
            & ~df[column].isin([0, 1])
        )

        if invalid_mask.any():
            invalid_values = (
                df.loc[
                    invalid_mask,
                    column,
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"{column}은 0과 1 또는 결측값만 "
                f"포함해야 합니다. 현재 값: {invalid_values}"
            )

# ============================================================
# 기본 데이터 품질 표
# ============================================================

def build_missing_values_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """열별 결측치 개수와 비율을 계산한다."""

    total_rows = len(df)

    result = pd.DataFrame(
        {
            "column": df.columns.astype(str),
            "dtype": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "missing_count": [
                int(value)
                for value in df.isna().sum()
            ],
        }
    )

    if total_rows:
        result["missing_rate"] = (
            result["missing_count"] / total_rows
        )
    else:
        result["missing_rate"] = 0.0

    return result.sort_values(
        ["missing_count", "column"],
        ascending=[False, True],
        ignore_index=True,
    )


def build_duplicate_result(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """완전 중복 행의 현재 상태를 한 행의 표로 만든다."""

    duplicate_rows = int(
        df.duplicated().sum()
    )

    row_count = len(df)

    return pd.DataFrame(
        [
            {
                "row_count": row_count,
                "duplicate_rows": duplicate_rows,
                "duplicate_rate": (
                    duplicate_rows / row_count
                    if row_count
                    else 0.0
                ),
            }
        ]
    )


# ============================================================
# 기술통계
# ============================================================

def build_descriptive_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """수치형 열의 기술통계를 생성한다."""

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "q1",
                "median",
                "q3",
                "max",
                "missing_count",
            ]
        )

    stats = (
        numeric_df
        .describe(
            percentiles=[0.25, 0.50, 0.75]
        )
        .transpose()
        .reset_index()
        .rename(
            columns={
                "index": "column",
                "25%": "q1",
                "50%": "median",
                "75%": "q3",
            }
        )
    )

    stats["missing_count"] = (
        stats["column"]
        .map(numeric_df.isna().sum())
        .astype(int)
    )

    column_order = [
        "column",
        "count",
        "mean",
        "std",
        "min",
        "q1",
        "median",
        "q3",
        "max",
        "missing_count",
    ]

    return stats[column_order]


def build_categorical_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """범주형 열의 범주별 빈도와 비율을 계산한다."""

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    records: list[dict[str, Any]] = []

    for column in categorical_columns:
        counts = (
            df[column]
            .astype("string")
            .value_counts(
                dropna=False
            )
        )

        for category, count in counts.items():
            records.append(
                {
                    "column": column,
                    "category": (
                        "<NA>"
                        if pd.isna(category)
                        else str(category)
                    ),
                    "count": int(count),
                    "proportion": (
                        float(count / len(df))
                        if len(df)
                        else 0.0
                    ),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "column",
                "category",
                "count",
                "proportion",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["column", "count", "category"],
            ascending=[True, False, True],
            ignore_index=True,
        )
    )


# ============================================================
# 연구 질문 중심 EDA
# ============================================================

def build_college_income_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """대학 학위 보유 여부별 표본 수와 고소득률을 계산한다."""

    summary = (
        df.groupby(
            "college_degree",
            observed=True,
        )
        .agg(
            sample_size=(
                "high_income",
                "count",
            ),
            high_income_count=(
                "high_income",
                "sum",
            ),
            high_income_rate=(
                "high_income",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "college_degree",
            ignore_index=True,
        )
    )

    summary["college_group"] = (
        summary["college_degree"]
        .map(COLLEGE_GROUP_LABELS)
        .fillna("Unknown")
    )

    if "age" in df.columns:
        age_summary = (
            df.groupby(
                "college_degree",
                observed=True,
            )["age"]
            .agg(
                mean_age="mean",
                median_age="median",
            )
            .reset_index()
        )
        summary = summary.merge(
            age_summary,
            on="college_degree",
            how="left",
        )

    if "hours-per-week" in df.columns:
        hours_summary = (
            df.groupby(
                "college_degree",
                observed=True,
            )["hours-per-week"]
            .agg(
                mean_hours_per_week="mean",
                median_hours_per_week="median",
            )
            .reset_index()
        )
        summary = summary.merge(
            hours_summary,
            on="college_degree",
            how="left",
        )

    preferred_order = [
        "college_degree",
        "college_group",
        "sample_size",
        "high_income_count",
        "high_income_rate",
        "mean_age",
        "median_age",
        "mean_hours_per_week",
        "median_hours_per_week",
    ]

    existing_columns = [
        column
        for column in preferred_order
        if column in summary.columns
    ]

    return summary[existing_columns]


def build_education_income_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """세부 학력별 표본 수와 고소득률을 계산한다."""

    aggregation: dict[str, tuple[str, str]] = {
        "sample_size": (
            "high_income",
            "count",
        ),
        "high_income_count": (
            "high_income",
            "sum",
        ),
        "high_income_rate": (
            "high_income",
            "mean",
        ),
        "college_degree": (
            "college_degree",
            "max",
        ),
    }

    summary = (
        df.groupby(
            "education",
            observed=True,
        )
        .agg(**aggregation)
        .reset_index()
    )

    if "education-num" in df.columns:
        education_order = (
            df.groupby(
                "education",
                observed=True,
            )["education-num"]
            .median()
            .rename("education_num")
            .reset_index()
        )

        summary = summary.merge(
            education_order,
            on="education",
            how="left",
        )

        summary = summary.sort_values(
            [
                "education_num",
                "high_income_rate",
                "education",
            ],
            ascending=[
                True,
                True,
                True,
            ],
            ignore_index=True,
        )
    else:
        summary = summary.sort_values(
            [
                "high_income_rate",
                "education",
            ],
            ascending=[
                True,
                True,
            ],
            ignore_index=True,
        )

    return summary


def build_numeric_group_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """학위 집단별 주요 수치형 변수의 평균·중앙값을 계산한다."""

    excluded_columns = {
        "college_degree",
        "high_income",
    }

    numeric_columns = [
        column
        for column in df.select_dtypes(
            include="number"
        ).columns
        if column not in excluded_columns
    ]

    records: list[dict[str, Any]] = []

    for column in numeric_columns:
        grouped = df.groupby(
            "college_degree",
            observed=True,
        )[column]

        for group_value, values in grouped:
            records.append(
                {
                    "college_degree": int(group_value),
                    "college_group": (
                        COLLEGE_GROUP_LABELS.get(
                            int(group_value),
                            "Unknown",
                        )
                    ),
                    "variable": column,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "college_degree",
                "college_group",
                "variable",
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            [
                "variable",
                "college_degree",
            ],
            ignore_index=True,
        )
    )


# ============================================================
# 보고서용 요약
# ============================================================

def _group_row(
    college_summary: pd.DataFrame,
    group_value: int,
) -> pd.Series | None:
    """학위 집단 요약에서 특정 집단의 행을 찾는다."""

    matched = college_summary.loc[
        college_summary["college_degree"]
        == group_value
    ]

    if matched.empty:
        return None

    return matched.iloc[0]


def build_eda_summary(
    df: pd.DataFrame,
    missing_table: pd.DataFrame,
    duplicate_table: pd.DataFrame,
    college_summary: pd.DataFrame,
    education_summary: pd.DataFrame,
) -> dict[str, Any]:
    """report.py가 읽을 수 있는 EDA JSON 요약을 만든다."""

    degree_row = _group_row(
        college_summary,
        1,
    )
    non_degree_row = _group_row(
        college_summary,
        0,
    )

    degree_rate = (
        float(degree_row["high_income_rate"])
        if degree_row is not None
        else None
    )

    non_degree_rate = (
        float(
            non_degree_row[
                "high_income_rate"
            ]
        )
        if non_degree_row is not None
        else None
    )

    rate_difference = (
        degree_rate - non_degree_rate
        if (
            degree_rate is not None
            and non_degree_rate is not None
        )
        else None
    )

    total_rows = len(df)
    valid_income_count = int(
        df["high_income"].count()
    )

    high_income_count = int(
        df["high_income"]
        .eq(1)
        .sum()
    )

    low_income_count = int(
        df["high_income"]
        .eq(0)
        .sum()
    )

    missing_income_count = (
        total_rows - valid_income_count
    )
    
    valid_degree_count = int(
        df["college_degree"].count()
    )

    college_degree_count = int(
        df["college_degree"]
        .eq(1)
        .sum()
    )

    no_college_degree_count = int(
        df["college_degree"]
        .eq(0)
        .sum()
    )

    missing_degree_count = (
        total_rows - valid_degree_count
    )

    top_education_records = (
        education_summary
        .sort_values(
            [
                "high_income_rate",
                "sample_size",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(5)
        .to_dict(
            orient="records"
        )
    )

    return {
        "generated_at_utc": (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        ),
        "dataset": {
            "rows": total_rows,
            "columns": int(
                len(df.columns)
            ),
            "column_names": [
                str(column)
                for column in df.columns
            ],
        },
        "data_quality_before_cleaning": {
            "missing_cells": int(
                missing_table[
                    "missing_count"
                ].sum()
            ),
            "columns_with_missing_values": int(
                (
                    missing_table[
                        "missing_count"
                    ]
                    > 0
                ).sum()
            ),
            "duplicate_rows": int(
                duplicate_table.loc[
                    0,
                    "duplicate_rows",
                ]
            ),
        },
        "target_distribution": {
            "valid_income_count": (
                valid_income_count
            ),
            "missing_income_count": (
                missing_income_count
            ),
            "high_income_count": (
                high_income_count
            ),
            "high_income_rate": (
                high_income_count
                / valid_income_count
                if valid_income_count
                else 0.0
            ),
            "low_income_count": (
                low_income_count
            ),
        },

        "college_degree_distribution": {
            "valid_degree_count": (
                valid_degree_count
            ),
            "missing_degree_count": (
                missing_degree_count
            ),
            "college_degree_count": (
                college_degree_count
            ),
            "college_degree_rate": (
                college_degree_count
                / valid_degree_count
                if valid_degree_count
                else 0.0
            ),
            "no_college_degree_count": (
                no_college_degree_count
            ),
        },
        
        "unadjusted_association": {
            "college_degree_high_income_rate": (
                degree_rate
            ),
            "no_college_degree_high_income_rate": (
                non_degree_rate
            ),
            "high_income_rate_difference": (
                rate_difference
            ),
            "interpretation": (
                "이 차이는 조정 전 단순 집단 비교이며, "
                "대학 학위의 인과효과로 해석할 수 없다."
            ),
        },
        "top_education_by_high_income_rate": (
            top_education_records
        ),
        "analysis_note": (
            "EDA는 관측된 분포와 조정 전 연관성을 요약한다. "
            "Welch t-test와 PSM 결과는 statistics.py에서 별도로 생성한다."
        ),
    }


def build_flat_eda_summary(
    summary: dict[str, Any],
) -> pd.DataFrame:
    """CSV용 핵심 EDA 요약을 한 행으로 평탄화한다."""

    dataset = summary["dataset"]
    quality = summary[
        "data_quality_before_cleaning"
    ]
    target = summary[
        "target_distribution"
    ]
    degree = summary[
        "college_degree_distribution"
    ]
    association = summary[
        "unadjusted_association"
    ]

    return pd.DataFrame(
        [
            {
                "rows": dataset["rows"],
                "columns": dataset["columns"],
                "missing_cells": (
                    quality["missing_cells"]
                ),
                "duplicate_rows": (
                    quality["duplicate_rows"]
                ),
                "high_income_count": (
                    target["high_income_count"]
                ),
                "high_income_rate": (
                    target["high_income_rate"]
                ),
                "college_degree_count": (
                    degree[
                        "college_degree_count"
                    ]
                ),
                "college_degree_rate": (
                    degree[
                        "college_degree_rate"
                    ]
                ),
                "college_degree_high_income_rate": (
                    association[
                        "college_degree_high_income_rate"
                    ]
                ),
                "no_college_degree_high_income_rate": (
                    association[
                        "no_college_degree_high_income_rate"
                    ]
                ),
                "high_income_rate_difference": (
                    association[
                        "high_income_rate_difference"
                    ]
                ),
            }
        ]
    )


# ============================================================
# JSON 저장 보조
# ============================================================

def _json_default(value: Any) -> Any:
    """NumPy/Pandas 스칼라를 JSON 기본형으로 변환한다."""

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        return value.item()

    raise TypeError(
        f"{type(value).__name__}은 JSON으로 직렬화할 수 없습니다."
    )


def _eda_summary_paths() -> tuple[Path, Path]:
    """
    config.py의 EDA JSON 경로를 기준으로
    JSON 요약과 보조 CSV 요약 경로를 생성한다.
    """

    configured_path = Path(
        EDA_SUMMARY_PATH
    )

    if configured_path.suffix.lower() == ".json":
        json_path = configured_path
        csv_path = configured_path.with_suffix(
            ".csv"
        )
    else:
        csv_path = configured_path
        json_path = configured_path.with_suffix(
            ".json"
        )

    return json_path, csv_path


# ============================================================
# 저장
# ============================================================

def save_eda_outputs(
    *,
    summary: dict[str, Any],
    missing_table: pd.DataFrame,
    duplicate_table: pd.DataFrame,
    descriptive_stats: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    college_summary: pd.DataFrame,
    education_summary: pd.DataFrame,
    numeric_group_summary: pd.DataFrame,
) -> dict[str, Path]:
    """EDA 결과 파일을 저장하고 생성 경로를 반환한다."""

    ensure_directories()

    summary_json_path, summary_csv_path = (
        _eda_summary_paths()
    )

    summary_json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )

    build_flat_eda_summary(
        summary
    ).to_csv(
        summary_csv_path,
        index=False,
    )

    missing_table.to_csv(
        MISSING_VALUES_PATH,
        index=False,
    )

    duplicate_table.to_csv(
        DUPLICATE_RESULT_PATH,
        index=False,
    )

    descriptive_stats.to_csv(
        DESCRIPTIVE_STATS_PATH,
        index=False,
    )

    categorical_summary.to_csv(
        CATEGORICAL_SUMMARY_PATH,
        index=False,
    )

    college_summary.to_csv(
        COLLEGE_INCOME_SUMMARY_PATH,
        index=False,
    )

    education_summary.to_csv(
        EDUCATION_INCOME_SUMMARY_PATH,
        index=False,
    )

    numeric_group_summary.to_csv(
        NUMERIC_GROUP_SUMMARY_PATH,
        index=False,
    )

    return {
        "eda_summary_json": (
            summary_json_path
        ),
        "eda_summary_csv": (
            summary_csv_path
        ),
        "missing_values": Path(
            MISSING_VALUES_PATH
        ),
        "duplicate_result": Path(
            DUPLICATE_RESULT_PATH
        ),
        "descriptive_stats": Path(
            DESCRIPTIVE_STATS_PATH
        ),
        "categorical_summary": (
            CATEGORICAL_SUMMARY_PATH
        ),
        "college_income_summary": (
            COLLEGE_INCOME_SUMMARY_PATH
        ),
        "education_income_summary": (
            EDUCATION_INCOME_SUMMARY_PATH
        ),
        "numeric_group_summary": (
            NUMERIC_GROUP_SUMMARY_PATH
        ),
    }


# ============================================================
# 전체 EDA 파이프라인
# ============================================================

def run_eda(
    df: pd.DataFrame | None = None,
    *,
    data_path: str | Path = RAW_DATA_PATH,
    save_output: bool = True,
) -> dict[str, Any]:
    """
    행 제거 전 Adult 데이터를 사용해 EDA를 수행한다.

    df가 전달되지 않으면 문자열과 결측 표현만 정규화한
    원본 데이터를 불러온다.
    """

    if df is None:
        df = load_before_cleaning(
            path=data_path,
        )
    else:
        # 호출한 쪽의 DataFrame을 변경하지 않는다.
        df = df.copy()

    validate_eda_input(df)

    missing_table = (
        build_missing_values_table(df)
    )

    duplicate_table = (
        build_duplicate_result(df)
    )

    descriptive_stats = (
        build_descriptive_stats(df)
    )

    categorical_summary = (
        build_categorical_summary(df)
    )

    college_summary = (
        build_college_income_summary(df)
    )

    education_summary = (
        build_education_income_summary(df)
    )

    numeric_group_summary = (
        build_numeric_group_summary(df)
    )

    summary = build_eda_summary(
        df=df,
        missing_table=missing_table,
        duplicate_table=duplicate_table,
        college_summary=college_summary,
        education_summary=education_summary,
    )

    output_paths: dict[str, Path] = {}

    if save_output:
        output_paths = save_eda_outputs(
            summary=summary,
            missing_table=missing_table,
            duplicate_table=duplicate_table,
            descriptive_stats=descriptive_stats,
            categorical_summary=categorical_summary,
            college_summary=college_summary,
            education_summary=education_summary,
            numeric_group_summary=(
                numeric_group_summary
            ),
        )

    return {
        "summary": summary,
        "tables": {
            "missing_values": missing_table,
            "duplicate_result": duplicate_table,
            "descriptive_stats": descriptive_stats,
            "categorical_summary": (
                categorical_summary
            ),
            "college_income_summary": (
                college_summary
            ),
            "education_income_summary": (
                education_summary
            ),
            "numeric_group_summary": (
                numeric_group_summary
            ),
        },
        "output_paths": output_paths,
    }
