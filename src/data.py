"""Adult Income 데이터 로딩, 정제 및 Pandas/Polars 성능 비교"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from config import (
    ADULT_COLUMNS,
    COLLEGE_DEGREES,
    PROCESSED_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)


# ============================================================
# 1. Pandas 데이터 로딩
# ============================================================

def load_data_pandas(
    path: Path = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, float]:

    start = time.perf_counter()

    df = pd.read_csv(
        path,
        na_values="?",
        skipinitialspace=True,
    )

    elapsed = time.perf_counter() - start

    return df, elapsed


# ============================================================
# 2. Polars 데이터 로딩
# ============================================================

def load_data_polars(
    path: Path = RAW_DATA_PATH,
):

    import polars as pl

    start = time.perf_counter()

    df = pl.read_csv(
        path,
        null_values="?",
    )

    elapsed = time.perf_counter() - start

    return df, elapsed


# ============================================================
# 3. 데이터 정제
# ============================================================

def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    cleaned = df.copy()

    # 컬럼명 공백 제거
    cleaned.columns = (
        cleaned.columns
        .astype(str)
        .str.strip()
    )

    # 문자열 데이터 정리
    object_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in object_columns:

        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .str.rstrip(".")
        )

        cleaned[column] = (
            cleaned[column]
            .replace(
                {
                    "": pd.NA,
                    "NA": pd.NA,
                    "N/A": pd.NA,
                }
            )
        )

    # 중복 제거
    cleaned = (
        cleaned
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # 필수 컬럼 확인
    required_columns = {
        "education",
        "income",
    }

    missing_columns = (
        required_columns
        - set(cleaned.columns)
    )

    if missing_columns:

        raise ValueError(
            f"필수 열이 없습니다: "
            f"{sorted(missing_columns)}"
        )

    # 소득 0/1 변수 생성
    income_map = {
        "<=50K": 0,
        ">50K": 1,
    }

    cleaned["high_income"] = (
        cleaned["income"]
        .map(income_map)
        .astype("Int64")
    )

    # 잘못된 income 값 확인
    if cleaned["high_income"].isna().any():

        unknown_income = (
            cleaned.loc[
                cleaned["high_income"].isna(),
                "income",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"알 수 없는 income 값이 있습니다: "
            f"{unknown_income}"
        )

    # 대학 학위 여부
    cleaned["college_degree"] = (
        cleaned["education"]
        .isin(COLLEGE_DEGREES)
        .astype(int)
    )

    return cleaned


# ============================================================
# 4. Pandas / Polars 우수 도구 선택
# ============================================================

def select_best_tool(
    pandas_seconds: float,
    polars_seconds: float,
) -> str:
    """
    Pandas와 Polars의 데이터 로딩 시간을 비교하여
    더 빠른 도구를 반환합니다.
    """

    if pandas_seconds < polars_seconds:

        return "pandas"

    elif polars_seconds < pandas_seconds:

        return "polars"

    return "same"


# ============================================================
# 5. 정제 데이터 저장
# ============================================================

def save_processed_data(
    df: pd.DataFrame,
) -> None:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print(
        f"\n정제 데이터 저장 완료: "
        f"{PROCESSED_DATA_PATH}"
    )


# ============================================================
# 6. 전체 데이터 파이프라인
# ============================================================

def run_data_pipeline(
    path: Path = RAW_DATA_PATH,
) -> tuple[
    pd.DataFrame,
    dict[str, float | int | str],
]:
    """
    전체 데이터 파이프라인 실행.

    반환값:
        cleaned_df
            정제된 Pandas DataFrame

        comparison
            Pandas / Polars 비교 결과
            및 더 빠른 도구 정보
    """

    print("\n" + "=" * 60)
    print("[DATA] 데이터 로딩 및 정제")
    print("=" * 60)

    # ========================================================
    # 1. Pandas 로딩
    # ========================================================

    pandas_df, pandas_seconds = (
        load_data_pandas(path)
    )

    print("\n[Pandas]")

    print(
        f"데이터 크기: "
        f"{pandas_df.shape}"
    )

    print(
        f"로딩 시간: "
        f"{pandas_seconds:.6f}초"
    )

    # ========================================================
    # 2. Polars 로딩
    # ========================================================

    polars_df, polars_seconds = (
        load_data_polars(path)
    )

    print("\n[Polars]")

    print(
        f"데이터 크기: "
        f"{polars_df.shape}"
    )

    print(
        f"로딩 시간: "
        f"{polars_seconds:.6f}초"
    )

    # ========================================================
    # 3. 데이터 크기 비교
    # ========================================================

    print("\n[Pandas vs Polars]")

    if pandas_df.shape == polars_df.shape:

        print(
            "Pandas와 Polars의 "
            "데이터 크기가 일치합니다."
        )

    else:

        print(
            "주의: Pandas와 Polars의 "
            "데이터 크기가 다릅니다."
        )

    # ========================================================
    # 4. 더 빠른 도구 선택
    # ========================================================

    best_tool = select_best_tool(
        pandas_seconds,
        polars_seconds,
    )

    print("\n[성능 비교 결과]")

    if best_tool == "pandas":

        print(
            "Pandas가 더 빠른 데이터 로딩 성능을 보였습니다."
        )

    elif best_tool == "polars":

        print(
            "Polars가 더 빠른 데이터 로딩 성능을 보였습니다."
        )

    else:

        print(
            "Pandas와 Polars의 성능이 동일합니다."
        )

    # ========================================================
    # 5. 정제 전 상태 확인
    # ========================================================

    before_rows = len(
        pandas_df
    )

    missing_before = (
        pandas_df
        .isnull()
        .sum()
        .sum()
    )

    duplicate_before = (
        pandas_df
        .duplicated()
        .sum()
    )

    print("\n[정제 전]")

    print(
        f"데이터: "
        f"{before_rows:,}건"
    )

    print(
        f"결측치 셀: "
        f"{missing_before:,}개"
    )

    print(
        f"중복 행: "
        f"{duplicate_before:,}건"
    )

    # ========================================================
    # 6. 데이터 정제
    # ========================================================

    cleaned_df = clean_data(
        pandas_df
    )

    # ========================================================
    # 7. 정제 후 상태 확인
    # ========================================================

    after_rows = len(
        cleaned_df
    )

    missing_after = (
        cleaned_df
        .isnull()
        .sum()
        .sum()
    )

    duplicate_after = (
        cleaned_df
        .duplicated()
        .sum()
    )

    print("\n[정제 후]")

    print(
        f"데이터: "
        f"{after_rows:,}건"
    )

    print(
        f"결측치 셀: "
        f"{missing_after:,}개"
    )

    print(
        f"중복 행: "
        f"{duplicate_after:,}건"
    )

    # ========================================================
    # 8. 정제 데이터 저장
    # ========================================================

    save_processed_data(
        cleaned_df
    )

    # ========================================================
    # 9. 비교 결과 생성
    # ========================================================

    comparison = {

        "pandas_rows":
            pandas_df.shape[0],

        "pandas_columns":
            pandas_df.shape[1],

        "polars_rows":
            polars_df.shape[0],

        "polars_columns":
            polars_df.shape[1],

        "pandas_seconds":
            round(
                pandas_seconds,
                6,
            ),

        "polars_seconds":
            round(
                polars_seconds,
                6,
            ),

        "best_tool":
            best_tool,
    }

    return (
        cleaned_df,
        comparison,
    )


# ============================================================
# 10. 다른 Python 파일에서 사용할 함수
# ============================================================

def get_best_data_tool() -> str:
    """
    Pandas와 Polars의 성능을 비교하고
    더 빠른 도구의 이름을 반환합니다.

    다른 Python 파일에서 import하여 사용할 수 있습니다.

    예:
        from src.data import get_best_data_tool

        best_tool = get_best_data_tool()
    """

    _, comparison = run_data_pipeline()

    return comparison["best_tool"]


# ============================================================
# pandas/polars 비교 결과 출력을 위해 main.py에서 아래 코드 실행 요구
# ============================================================

# if __name__ == "__main__":

#     cleaned_df, comparison = (
#         run_data_pipeline()
#     )

#     print("\n" + "=" * 60)
#     print("[최종 결과]")
#     print("=" * 60)

#     print(
#         f"가장 빠른 데이터 처리 도구: "
#         f"{comparison['best_tool']}"
#     )

#     print(
#         f"\nPandas 로딩 시간: "
#         f"{comparison['pandas_seconds']}초"
#     )

#     print(
#         f"Polars 로딩 시간: "
#         f"{comparison['polars_seconds']}초"
#     )
