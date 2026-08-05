"""프로젝트 전역 공통 경로·상수.

모든 모듈은 출력 경로나 컬럼 목록을 직접 하드코딩하지 않고 이 파일의 값을
import해서 쓴다. 경로를 옮기거나 파일명을 바꿀 때 여기 한 곳만 고치면 되게
하기 위함이다.
"""

from pathlib import Path


# ============================================================
# 프로젝트 기본 경로
# ============================================================

#config.py가 src 폴더 안에 있다는 전제에서 프로젝트 루트 경로를 계산한다.

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# 데이터 경로
# ============================================================

#raw 폴더의 원본 데이터는 수정하지 않고, 정제 결과는 processed 폴더에 별도로 저장한다.

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"


# ============================================================
# 결과물 경로
# ============================================================

OUTPUT_DIR = BASE_DIR / "outputs"

FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR = OUTPUT_DIR / "tables"

MODEL_DIR = OUTPUT_DIR / "models"


# ============================================================
# 파일 경로
# ============================================================

RAW_DATA_PATH = RAW_DIR / "adult.csv"

PROCESSED_DATA_PATH = (
    PROCESSED_DIR / "adult_cleaned.csv"
)

REPORT_PATH = (
    BASE_DIR / "report.md"
)


# ============================================================
# EDA 결과
# ============================================================

#EDA_SUMMARY_PATH:
# 전체 표본 수, 컬럼 수, 고소득 비율, 대학 학위 비율 등 핵심 요약 저장

# MISSING_VALUES_PATH:
# 컬럼별 결측값 개수와 비율 저장

# DUPLICATE_RESULT_PATH:
# 중복 제거 전후 행 수와 제거된 행 수 저장

# DESCRIPTIVE_STATS_PATH:
# 수치형 변수의 평균, 표준편차, 분위수 등 기술통계 저장

EDA_SUMMARY_PATH = (
    TABLE_DIR / "eda_summary.json"
)

MISSING_VALUES_PATH = (
    TABLE_DIR / "missing_values.csv"
)

DUPLICATE_RESULT_PATH = (
    TABLE_DIR / "duplicate_result.csv"
)

DESCRIPTIVE_STATS_PATH = (
    TABLE_DIR / "descriptive_stats.csv"
)

# Pandas/Polars 성능 비교 결과
BENCHMARK_RESULT_PATH = (
    TABLE_DIR / "data_engine_benchmark.json"
)

# ============================================================
# Adult Census Income 컬럼
# ============================================================

# 원본 adult.csv는 첫 번째 행에 15개 컬럼명이 포함되어 있다.
CSV_HAS_HEADER = True

# CSV 헤더가 프로젝트의 Adult 데이터 계약과 일치하는지 검증할 때 사용한다.
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]


# ============================================================
# 공통 설정
# ============================================================

#데이터 분할, PSM, 머신러닝 모델의 결과를 재현하기 위한 공통 난수 시드이다.
RANDOM_STATE = 42


# ============================================================
# 분석 기준
# ============================================================

# education 값이 아래 학위 목록에 포함되면 college_degree를 1로 생성한다.
# frozenset을 사용해 실행 중 학위 기준이 변경되지 않도록 한다.

COLLEGE_DEGREES = frozenset(
    {
        "Bachelors",
        "Masters",
        "Prof-school",
        "Doctorate",
    }
)


# ============================================================
# 필요한 디렉터리 생성
# ============================================================

#분석 결과를 저장하기 전에 필요한 데이터·출력 폴더를 생성한다.
#이미 존재하는 폴더는 유지하며, main.py 시작 시 한 번 호출한다.

def ensure_directories() -> None:

    for directory in [
        RAW_DIR,
        PROCESSED_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        MODEL_DIR
    ]:

        directory.mkdir(parents=True, exist_ok=True)
