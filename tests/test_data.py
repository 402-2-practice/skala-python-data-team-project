"""src.data.clean_data()가 지키기로 약속한 공통 데이터 계약을 검증한다."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import clean_data


def _sample_frame(**overrides: list) -> pd.DataFrame:
    base = {
        "age": [39, 45],
        "education": ["Bachelors", "HS-grad"],
        "income": ["<=50K", ">50K"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_income_is_mapped_to_binary_high_income():
    cleaned = clean_data(_sample_frame())

    assert cleaned.loc[cleaned["income"] == "<=50K", "high_income"].tolist() == [0]
    assert cleaned.loc[cleaned["income"] == ">50K", "high_income"].tolist() == [1]


def test_trailing_period_and_whitespace_are_stripped_before_mapping():
    # 원본 adult.test 파일은 " >50K." 처럼 마침표가 붙어 있는 경우가 있다.
    cleaned = clean_data(_sample_frame(income=[" <=50K.", " >50K."]))

    assert cleaned["high_income"].tolist() == [0, 1]


def test_college_degree_flag_matches_config_list():
    cleaned = clean_data(
        _sample_frame(education=["Bachelors", "HS-grad"], income=["<=50K", "<=50K"])
    )

    assert cleaned.loc[cleaned["education"] == "Bachelors", "college_degree"].tolist() == [1]
    assert cleaned.loc[cleaned["education"] == "HS-grad", "college_degree"].tolist() == [0]


def test_question_mark_is_converted_to_missing():
    cleaned = clean_data(_sample_frame(workclass=["?", "Private"]))

    assert cleaned["workclass"].isna().tolist() == [True, False]


def test_duplicate_rows_are_removed():
    df = pd.concat([_sample_frame(), _sample_frame()], ignore_index=True)

    cleaned = clean_data(df)

    assert len(cleaned) == 2


def test_missing_required_column_raises_value_error():
    df = _sample_frame().drop(columns=["income"])

    with pytest.raises(ValueError):
        clean_data(df)


def test_unknown_income_value_raises_value_error():
    df = _sample_frame(income=["<=50K", "unexpected-value"])

    with pytest.raises(ValueError):
        clean_data(df)
