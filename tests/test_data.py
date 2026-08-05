"""src.data.clean_with_pandas()가 지키기로 약속한 정제 계약을 검증한다."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import clean_with_pandas


def _sample_frame(n: int = 2, **overrides: list) -> pd.DataFrame:
    base = {
        "age": [39, 45],
        "workclass": ["Private", "Private"],
        "fnlwgt": [77516, 83311],
        "education": ["Bachelors", "HS-grad"],
        "education-num": [13, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "income": ["<=50K", ">50K"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_income_is_mapped_to_binary_high_income():
    cleaned, _ = clean_with_pandas(_sample_frame())

    assert cleaned.loc[cleaned["income"] == "<=50K", "high_income"].tolist() == [0]
    assert cleaned.loc[cleaned["income"] == ">50K", "high_income"].tolist() == [1]


def test_trailing_period_and_whitespace_are_stripped_before_mapping():
    # 원본 adult.test 파일은 " >50K." 처럼 마침표가 붙어 있는 경우가 있다.
    cleaned, _ = clean_with_pandas(_sample_frame(income=[" <=50K.", " >50K."]))

    assert cleaned["high_income"].tolist() == [0, 1]


def test_college_degree_flag_matches_config_list():
    cleaned, _ = clean_with_pandas(
        _sample_frame(education=["Bachelors", "HS-grad"], income=["<=50K", "<=50K"])
    )

    assert cleaned.loc[cleaned["education"] == "Bachelors", "college_degree"].tolist() == [1]
    assert cleaned.loc[cleaned["education"] == "HS-grad", "college_degree"].tolist() == [0]


def test_question_mark_rows_are_dropped_as_missing():
    # clean_with_pandas는 "?"를 결측치로 바꾼 뒤 dropna()로 제거하므로,
    # 결과 프레임에는 해당 행 자체가 남지 않는다 (컬럼에 NaN이 남는 게 아님).
    cleaned, cleaning_info = clean_with_pandas(_sample_frame(workclass=["?", "Private"]))

    assert len(cleaned) == 1
    assert cleaned["workclass"].tolist() == ["Private"]
    assert cleaning_info["missing_removed"] == 1


def test_duplicate_rows_are_removed():
    df = pd.concat([_sample_frame(), _sample_frame()], ignore_index=True)

    cleaned, cleaning_info = clean_with_pandas(df)

    assert len(cleaned) == 2
    assert cleaning_info["duplicate_removed"] == 2


def test_missing_required_column_raises_key_error():
    # clean_with_pandas는 load_pandas()가 이미 열 구성을 검증했다고 전제하므로,
    # income처럼 내부에서 바로 참조하는 필수 열이 없으면 KeyError로 즉시 실패한다.
    df = _sample_frame().drop(columns=["income"])

    with pytest.raises(KeyError):
        clean_with_pandas(df)


def test_unknown_income_value_is_filtered_out_as_invalid():
    # 알 수 없는 income 값은 예외를 던지지 않고, 유효하지 않은 행으로 집계되어 제거된다.
    cleaned, cleaning_info = clean_with_pandas(
        _sample_frame(income=["<=50K", "unexpected-value"])
    )

    assert cleaned["income"].tolist() == ["<=50K"]
    assert cleaning_info["invalid_removed"] == 1
