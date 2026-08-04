"""고소득 여부 예측용 sklearn Pipeline."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_DIR, RANDOM_STATE, TABLE_DIR


def train_income_model(df: pd.DataFrame) -> dict:
    # income은 정답 원문, education-num은 education과 중복, fnlwgt는 표본 가중치이므로 제외한다.
    excluded = ["income", "high_income", "education-num", "fnlwgt"]
    feature_columns = [column for column in df.columns if column not in excluded]
    X = df[feature_columns].copy()
    y = df["high_income"].astype(int)

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    # pandas StringDtype의 pd.NA를 sklearn이 안정적으로 처리할 수 있게 변환한다.
    for column in categorical_columns:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan)
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction)),
        "recall": float(recall_score(y_test, prediction)),
        "f1": float(f1_score(y_test, prediction)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }

    joblib.dump(pipeline, MODEL_DIR / "income_pipeline.joblib")
    (TABLE_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[ML] 테스트 성능")
    print(pd.Series(metrics).to_string())
    return metrics
