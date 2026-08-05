"""모델 채택을 위한 실험 스크립트 (프로덕션 코드 아님).

src/modeling.py에 들어가는 모델·하이퍼파라미터는 이론적 추정이 아니라
이 스크립트의 실제 실행 결과로 결정한다. 결과는 docs/MODEL_SELECTION_LOG.md에
기록하고, 여기 코드는 그 기록을 재현하기 위한 학습 기록용으로 남긴다.

src/data.py가 현재 깨져있어(KNOWN_ISSUES.md) import할 수 없으므로,
scripts/make_local_test_sample.py와 동일하게 정제 로직을 여기서 직접 재현한다.

전처리는 모델마다 다르다: HistGradientBoosting은 src/modeling.py와 동일하게
categorical_features="from_dtype" 네이티브 범주형 처리를 쓰고(원-핫 인코딩 없음),
Logistic Regression·Random Forest는 원-핫 인코딩(+ LR만 StandardScaler)을 그대로 쓴다
(두 모델 다 카테고리 dtype을 분기 기준으로 못 쓰기 때문). 그래서 세 모델을 같은
X_train/X_test가 아니라 모델별로 인코딩된 데이터로 각각 학습·평가한다.

실행:
    python scripts/experiments/model_selection_experiment.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "adult.csv"
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 20

COLLEGE_DEGREES = ["Bachelors", "Masters", "Prof-school", "Doctorate"]
EXCLUDED_COLUMNS = ["income", "high_income", "education-num", "fnlwgt"]
TARGET_COLUMN = "high_income"


def load_and_clean() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH, na_values="?", skipinitialspace=True)
    cleaned = raw.copy()
    cleaned.columns = cleaned.columns.astype(str).str.strip()

    object_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].astype("string").str.strip().str.rstrip(".")
        cleaned[column] = cleaned[column].replace({"?": pd.NA, "": pd.NA})
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    income_map = {"<=50K": 0, ">50K": 1}
    cleaned["high_income"] = cleaned["income"].map(income_map).astype("Int64")
    cleaned = cleaned.dropna(subset=["high_income"]).reset_index(drop=True)
    cleaned["college_degree"] = cleaned["education"].isin(COLLEGE_DEGREES).astype(int)
    return cleaned.dropna().reset_index(drop=True)


def build_preprocessing_ohe(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Logistic Regression·Random Forest용 전처리. 둘 다 카테고리 dtype을 직접
    분기 기준으로 못 쓰므로 원-핫 인코딩이 필요하다 (LR은 추가로 스케일링도 필요).
    """
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def build_preprocessing_native(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """HistGradientBoosting용 전처리. src/modeling.py와 동일하게 원-핫 인코딩 없이
    categorical_features="from_dtype"로 category dtype 컬럼을 그대로 분기 기준에 쓴다.
    결측치도 모델이 자체 처리하므로 범주형 imputer가 필요 없다. ColumnTransformer가
    category dtype을 그대로 넘기도록 set_output(transform="pandas")를 지정해야 한다.
    """
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                "passthrough",
                categorical_columns,
            ),
        ]
    )
    preprocessing.set_output(transform="pandas")
    return preprocessing


def candidate_search_spaces(
    ohe_preprocessing: ColumnTransformer, native_preprocessing: ColumnTransformer
) -> dict[str, tuple[Pipeline, dict]]:
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("preprocessing", ohe_preprocessing),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3_000, class_weight="balanced", random_state=RANDOM_STATE
                        ),
                    ),
                ]
            ),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("preprocessing", ohe_preprocessing),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": randint(100, 600),
                "model__max_depth": [None, 5, 10, 15, 20, 30],
                "model__min_samples_leaf": randint(1, 10),
            },
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    ("preprocessing", native_preprocessing),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            categorical_features="from_dtype",
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__learning_rate": loguniform(1e-2, 3e-1),
                "model__max_depth": [None, 3, 5, 8, 12],
                "model__max_iter": randint(100, 400),
                "model__l2_regularization": uniform(0, 1),
            },
        ),
    }


def main() -> None:
    df = load_and_clean()
    print(f"데이터 로딩 완료: {df.shape}")

    feature_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int)

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    for column in categorical_columns:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)

    # object dtype 버전(LR·RF의 OneHotEncoder용)과 category dtype 버전
    # (HGB의 categorical_features="from_dtype"용)을 둘 다 준비한다.
    X_native = X.copy()
    for column in categorical_columns:
        X_native[column] = X_native[column].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # 동일한 split이 되도록 인덱스만 그대로 가져와 category dtype 버전에 적용한다.
    X_train_native = X_native.loc[X_train.index]
    X_test_native = X_native.loc[X_test.index]

    train_data_by_model = {
        "logistic_regression": (X_train, X_test),
        "random_forest": (X_train, X_test),
        "hist_gradient_boosting": (X_train_native, X_test_native),
    }

    ohe_preprocessing = build_preprocessing_ohe(numeric_columns, categorical_columns)
    native_preprocessing = build_preprocessing_native(numeric_columns, categorical_columns)
    spaces = candidate_search_spaces(ohe_preprocessing, native_preprocessing)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_searches = {}

    for name, (pipeline, param_distributions) in spaces.items():
        model_X_train, _ = train_data_by_model[name]
        print(f"\n=== {name} 탐색 시작 (n_iter={N_ITER}, cv={CV_FOLDS}) ===")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=N_ITER,
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        start = time.perf_counter()
        search.fit(model_X_train, y_train)
        search_seconds = time.perf_counter() - start

        results.append(
            {
                "model": name,
                "cv_roc_auc_mean": search.best_score_,
                "search_seconds": search_seconds,
                "best_params": search.best_params_,
            }
        )
        fitted_searches[name] = search
        print(f"{name}: cv_roc_auc={search.best_score_:.4f}, 탐색 시간={search_seconds:.1f}초")
        print(f"best_params={search.best_params_}")

    results_df = pd.DataFrame(results).sort_values("cv_roc_auc_mean", ascending=False)
    winner_name = results_df.iloc[0]["model"]
    winner_search = fitted_searches[winner_name]
    _, winner_X_test = train_data_by_model[winner_name]

    predict_start = time.perf_counter()
    prediction = winner_search.predict(winner_X_test)
    probability = winner_search.predict_proba(winner_X_test)[:, 1]
    predict_seconds = time.perf_counter() - predict_start

    final_metrics = {
        "model_name": winner_name,
        "best_params": winner_search.best_params_,
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "predict_seconds_per_1000_rows": predict_seconds / len(X_test) * 1000,
    }

    print("\n" + "=" * 60)
    print("[모델 비교 결과 (train, CV ROC-AUC 기준)]")
    print("=" * 60)
    print(results_df[["model", "cv_roc_auc_mean", "search_seconds"]].to_string(index=False))
    print(f"\n채택된 모델: {winner_name}")
    print(f"best_params: {winner_search.best_params_}")
    print("\n[held-out 테스트셋 최종 성능]")
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
