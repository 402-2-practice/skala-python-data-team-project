"""Adult Income 데이터 로딩, 정제 및 Pandas/Polars 성능 비교."""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.config import (
    ADULT_COLUMNS,
    CSV_HAS_HEADER,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    ensure_directories,
)

VALID_INCOME_LABELS = frozenset{
    "<=50K",">50K",
}

# CSV에서 숫자로 해석되어야 하는 원본 변수.
# 변환할 수 없는 값은 결측값으로 처리한다.
NUMERIC_COLUMNS = (
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
)

# ============================================================
# 입력 검증
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

def _validate_columns(columns: list[str],) -> None:
    """Adult 데이터의 필수 열이 존재하는지 검사한다."""

    duplicate_columns = sorted(
        {
            column
            for column in columns
            if columns.count(column) > 1
        }
    )

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

def load_raw_data(path: str | Path = RAW_DATA_PATH,) -> pd.DataFrame:
    """Adult 데이터를 Pandas DataFrame으로 불러온다."""

    path = Path(path)
    _validate_data_path(path)

    read_options: dict[str, object] = {
        "na_values": ["?", " ?"],
        "skipinitialspace": True,
    }

    if CSV_HAS_HEADER :
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

    _validate_columns(list(df.columns))

    return df

# ============================================================
# 공통 정규화
# ============================================================

def _normalize_strings(df: pd.DataFrame,) -> pd.DataFrame:
    """문자열 열의 공백과 결측 표현을 정규화한다"""

    result = df.copy()

    string_columns = result.select_dtypes(include=["object", "string"],).columns

    for column in string_columns:
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "?" : pd.NA,
                    "" : pd.NA,
                }
            )
        )

    # adult.test 형식에서는 income 값 뒤에 마침표
    result[RAW_TARGET_COLUMN] = (
        result[RAW_TARGET_COLUMN]
        .str.rstrip(".")
    )

    return result

def _coerce_numeric_columns(df: pd.DataFrame,) -> pd.DataFrame:
    """숫자형 원본 변수를 숫자로 변환하고 변환 실패는 결측값으로 처리한다."""

    result = df.copy()

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result

# ============================================================
# 유효성 검증
# ============================================================

def _valid_rows(df: pd.DataFrame,) -> pd.Series:
    """값이 존재하는 경우에만 논리적 허용 범위를 검사한다.

    결측값은 공통 정제 단계에서 제거하지 않는다.
    실제 분석에 필요한 열의 결측치는 prepare_analysis_data()에서
    해당 분석을 수행할 때 제거한다.
    """

    return (
        (
            df["age"].isna()
            | df["age"].between(1, 120)
        )
        & (
            df["hours-per-week"].isna()
            | df["hours-per-week"].between(1, 168)
        )
        & (
            df["education-num"].isna()
            | df["education-num"].between(1, 20)
        )
        & (
            df["fnlwgt"].isna()
            | df["fnlwgt"].gt(0)
        )
        & (
            df["capital-gain"].isna()
            | df["capital-gain"].ge(0)
        )
        & (
            df["capital-loss"].isna()
            | df["capital-loss"].ge(0)
        )
        & (
            df[RAW_TARGET_COLUMN].isna()
            | df[RAW_TARGET_COLUMN].isin(
                VALID_INCOME_LABELS
            )
        )
    )

# ============================================================
# 데이터 정제
# ============================================================

def clean_data(df: pd.DataFrame,) -> tuple[pd.DataFrame, dict[str, int]]:
    """Adult 원본 데이터를 서비스에서 사용할 공통 형태로 정제한다.

    공통 단계에서는:
        - 문자열 표현을 정규화한다.
        - 숫자 변환 실패를 결측값으로 처리한다.
        - 완전 중복 행을 제거한다.
        - 논리적으로 유효하지 않은 값을 가진 행을 제거한다.
        - income으로부터 high_income을 생성한다.

    결측값이 존재한다는 이유만으로 행 전체를 제거하지 않는다.
    """

    if not isinstance(df,pd.DataFrame,):
        raise TypeError(
            "clean_data() 입력은 pandas.DataFrame이어야 합니다."
        )

    if df.empty:
        raise ValueError(
            "정제할 데이터가 비어 있습니다."
        )
         
    result = _normalize_strings(df)
    result = _coerce_numeric_columns(result)

    initial_rows = len(result)

    rows_with_missing = int(
    result.isna().any(axis=1).sum()
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

    valid_mask = _valid_rows(result)

    invalid_removed = int(
        (~valid_mask).sum()
    )

    result = (
        result
        .loc[valid_mask]
        .copy()
        .reset_index(drop=True)
    )

    # 원본 income이 결측이면 파생 target도 결측으로 유지한다.
    result[TARGET_COLUMN] = (
        result[RAW_TARGET_COLUMN]
        .map(
            {
                "<=50K": 0,
                ">50K": 1,
            }
        )
        .astype("Int8")
    )

    cleaning_info = {
        "initial_rows": int(initial_rows),
        "rows_with_missing": rows_with_missing,
        "duplicate_removed": int(duplicate_removed),
        "invalid_removed": invalid_removed,
        "rows_after_cleaning": int(len(result)),
        "columns_after_cleaning": int(len(result.columns)),
    }

    return result, cleaning_info

# ============================================================
# 분석별 결측치 처리
# ============================================================

def prepare_analysis_data(
    df: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """해당 분석에서 실제로 사용하는 열에 대해서만 결측 행을 제거한다."""

    if not isinstance(df,pd.DataFrame,):
        raise TypeError(
            "분석 입력은 pandas.DataFrame이어야 합니다."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "분석에 필요한 열이 없습니다: "
            f"{missing_columns}"
        )
    
    # 같은 열이 여러 번 전달되더라도 한 번만 선택한다.
    unique_columns = list(
        dict.fromkeys(required_columns)
    )

    return (
        df[unique_columns]
        .dropna()
        .copy()
        .reset_index(drop=True)
    )

# ============================================================
# EDA용 데이터
# ============================================================

def load_before_cleaning(path: str | Path = RAW_DATA_PATH,) -> pd.DataFrame:
    """행을 제거하기 전의 Adult 데이터를 EDA용으로 반환한다.

    문자열과 결측 표현은 통일하지만,
    결측·중복·유효 범위에 따른 행 삭제는 수행하지 않는다.
    """

    result = load_raw_data(path)
    result = _normalize_strings(result)
    result = _coerce_numeric_columns(result)

    result[TARGET_COLUMN] = (
        result[RAW_TARGET_COLUMN]
        .map(
            {
                "<=50K": 0,
                ">50K": 1,
            }
        )
        .astype("Int8")
    )

    return result

# ============================================================
# 정제 데이터 저장
# ============================================================

def save_processed_data(df: pd.DataFrame,) -> None:
    """정제 데이터를 CSV로 저장한다."""

    ensure_directories()

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

# ============================================================
# 공통 인터페이스
# ============================================================

# EDA·통계·시각화·모델링 모듈에 동일한 정제 Pandas DataFrame을 제공하는 공통 인터페이스다.
def load_and_clean(
    path: str | Path = RAW_DATA_PATH,
    *,
    save_output: bool = False,
) -> pd.DataFrame :

    raw_df = load_raw_data(path)
    cleaned_df,_ = clean_data(raw_df)

    if save_output : 
        save_processed_data(cleaned_df)

    return cleaned_df
