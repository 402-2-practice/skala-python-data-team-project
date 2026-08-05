"""src.modeling.train_income_model()의 산출물 계약을 합성 데이터로 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODEL_DIR, TABLE_DIR, ensure_directories
from src.modeling import train_income_model


def test_train_income_model_produces_expected_metrics_and_artifacts():
    ensure_directories()
    rng = np.random.default_rng(0)
    n = 120
    age = rng.integers(20, 65, size=n)
    sex = rng.choice(["Male", "Female"], size=n)
    high_income = (age > 40).astype(int)  # 예측 가능한 신호를 넣어 학습이 되게 한다.

    df = pd.DataFrame({"age": age, "sex": sex, "high_income": high_income})

    metrics = train_income_model(df)

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    assert (MODEL_DIR / "income_pipeline.joblib").exists()
    assert (TABLE_DIR / "model_metrics.json").exists()


def test_train_income_model_excludes_leakage_columns():
    ensure_directories()
    rng = np.random.default_rng(1)
    n = 200
    age = rng.integers(20, 65, size=n)
    # age가 high_income과 완전히 분리되지 않도록 노이즈를 섞는다. age만으로는 100%
    # 정확도가 나올 수 없게 만들어야, income 컬럼이 새서 100%가 나오는 경우를
    # 이 테스트가 실제로 구분해낼 수 있다.
    prob_high_income = np.clip((age - 20) / 45, 0.05, 0.95)
    high_income = (rng.random(n) < prob_high_income).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": rng.choice(["Male", "Female"], size=n),
            "high_income": high_income,
            # income/education-num/fnlwgt는 예측 입력에서 제외되어야 한다.
            "income": np.where(high_income == 1, ">50K", "<=50K"),
            "education-num": rng.integers(1, 16, size=n),
            "fnlwgt": rng.integers(10_000, 300_000, size=n),
        }
    )

    # income 컬럼이 제외되지 않으면 정답이 그대로 새어 들어가 정확도가 1.0에 가깝게 나온다.
    metrics = train_income_model(df)

    assert metrics["accuracy"] < 1.0
