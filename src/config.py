from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "adult.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "adult_cleaned.csv"
FIGURE_DIR = ROOT_DIR / "outputs" / "figures"
TABLE_DIR = ROOT_DIR / "outputs" / "tables"
MODEL_DIR = ROOT_DIR / "outputs" / "models"
REPORT_PATH = ROOT_DIR / "report.md"

RANDOM_STATE = 42

# 프로젝트에서 '대학 학위 보유'로 정의하는 교육 수준.
COLLEGE_DEGREES = ["Bachelors", "Masters", "Prof-school", "Doctorate"]


def ensure_directories() -> None:
    for directory in [
        RAW_DATA_PATH.parent,
        PROCESSED_DATA_PATH.parent,
        FIGURE_DIR,
        TABLE_DIR,
        MODEL_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

