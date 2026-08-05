"""로컬 전용 소규모 테스트 데이터 생성.

src/data.py가 현재 깨져있어(KNOWN_ISSUES.md 참고) import조차 안 되므로,
같은 정제 로직을 여기서 직접 재현해 modeling.py를 독립적으로 테스트할 수
있는 샘플을 만든다. data/processed/*.csv는 이미 .gitignore에 걸려있어
git에는 올라가지 않는다 — 로컬 개발용일 뿐, 실제 제출/실행은 정식 파이프라인
(data.py 결함 수정 후 main.py)으로 한다.

사용법:
    python scripts/make_local_test_sample.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "adult.csv"
SAMPLE_PATH = ROOT_DIR / "data" / "processed" / "adult_test_sample.csv"
SAMPLE_SIZE = 1000
RANDOM_STATE = 42

COLLEGE_DEGREES = ["Bachelors", "Masters", "Prof-school", "Doctorate"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """src/data.py의 clean_data()와 동일한 규칙 (결함 없는 부분만 재현)."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.astype(str).str.strip()

    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.rstrip(".")
        cleaned[column] = cleaned[column].replace({"?": pd.NA, "": pd.NA})

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    income_map = {"<=50K": 0, ">50K": 1}
    cleaned["high_income"] = cleaned["income"].map(income_map).astype("Int64")
    cleaned = cleaned.dropna(subset=["high_income"])
    cleaned["college_degree"] = cleaned["education"].isin(COLLEGE_DEGREES).astype(int)
    return cleaned.dropna().reset_index(drop=True)


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"원본 데이터가 없습니다: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH, na_values="?", skipinitialspace=True)
    cleaned = clean(raw)

    per_class = SAMPLE_SIZE // 2
    parts = [
        group.sample(n=min(len(group), per_class), random_state=RANDOM_STATE)
        for _, group in cleaned.groupby("high_income")
    ]
    sample = pd.concat(parts, ignore_index=True)

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    print(f"테스트 샘플 생성 완료: {SAMPLE_PATH} (행 {len(sample)}개, high_income 비율 {sample['high_income'].mean():.2%})")


if __name__ == "__main__":
    main()
