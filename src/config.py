"""프로젝트 공통 경로·설정 상수."""

from pathlib import Path


# ============================================================
# 프로젝트 기본 경로
# ============================================================

#src/config.py 기준 프로젝트 루트

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

# ============================================================
# 데이터 파일 경로
# ============================================================

RAW_DATA_PATH = RAW_DIR / "adult.csv"
PROCESSED_DATA_PATH = (PROCESSED_DIR / "adult_cleaned.csv")

CSV_HAS_HEADER = True

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
    "income",
]

RAW_TARGET_COLUMN = "income"
TARGET_COLUMN = "high_income"

# ============================================================
# 사용자 연관성 분석 변수
# ============================================================

# 홈페이지에서 관심 변수 또는 통제 변수로 선택할 수 있는 변수와
# 연관성 분석 시 사용할 변수 유형을 명시한다.
# fnlwgt:Census 표본 가중치이므로 사용자 분석 변수에서 제외한다.
# education-num:education의 숫자 표현으로 정보가 중복되므로 제외한다.
# income / high_income:결과변수이므로 설명변수로 사용할 수 없다.

ANALYSIS_VARIABLE_TYPES = {
    "age": "continuous",
    "workclass": "categorical",
    "education": "categorical",
    "marital-status": "categorical",
    "occupation": "categorical",
    "relationship": "categorical",
    "race": "categorical",
    "sex": "binary",
    "capital-gain": "continuous",
    "capital-loss": "continuous",
    "hours-per-week": "continuous",
    "native-country": "categorical",
}

ANALYSIS_VARIABLES = tuple(ANALYSIS_VARIABLE_TYPES.keys())

# ============================================================
# 고소득 예측 변수
# ============================================================

# 사용자 입력을 받아 고소득 확률을 예측할 때 사용하는 피처.
# income / high_income:정답 변수이므로 제외한다.
# education-num:education과 중복되므로 제외한다.
# fnlwgt:Census 표본 가중치이며 개인의 실질적 특성이 아니므로 제외한다.

PREDICTION_FEATURE_COLUMNS = [
    "age",
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

# ============================================================
# 공통 설정
# ============================================================

#데이터 분할, PSM, 머신러닝 모델의 결과를 재현하기 위한 공통 난수 시드이다.
RANDOM_STATE = 42

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
        MODEL_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
