from pathlib import Path


# ============================================================
# 프로젝트 기본 경로
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# 데이터 경로
# ============================================================

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

RAW_DATA_PATH = (
    RAW_DIR / "adult.csv"
)

PROCESSED_DATA_PATH = (
    PROCESSED_DIR / "adult_clean.csv"
)

REPORT_PATH = (
    BASE_DIR / "report.md"
)


# ============================================================
# EDA 결과
# ============================================================

EDA_SUMMARY_PATH = (
    TABLE_DIR / "eda_summary.csv"
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


# ============================================================
# Adult Census Income 컬럼
# ============================================================

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

RANDOM_STATE = 42


# ============================================================
# 분석 기준
# ============================================================

COLLEGE_DEGREES = [
    "Bachelors",
    "Masters",
    "Prof-school",
    "Doctorate"
]


# ============================================================
# 필요한 디렉터리 생성
# ============================================================

def ensure_directories() -> None:

    for directory in [
        RAW_DIR,
        PROCESSED_DIR,
        FIGURE_DIR,
        TABLE_DIR,
        MODEL_DIR
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )