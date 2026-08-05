"""src.modeling.train_income_model()의 산출물 계약을 합성 데이터로 검증한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.modeling as modeling
from src.modeling import predict_income, train_income_model


def test_train_income_model_produces_expected_metrics_and_artifacts():
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

    assert (modeling.MODEL_DIR / "income_pipeline.joblib").exists()
    assert (modeling.TABLE_DIR / "model_metrics.json").exists()


def test_train_income_model_excludes_leakage_columns():
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


def test_reloaded_model_matches_original_predictions():
    """joblib으로 저장한 파이프라인을 다시 불러왔을 때 원본과 동일한 확률을 내는지 확인한다.

    저장 자체는 성공해도 재로딩 후 예측이 미묘하게 달라지는 경우(예: 범주형 dtype이
    직렬화 과정에서 깨지는 것)를 이 비교가 없으면 못 잡는다.
    """
    rng = np.random.default_rng(3)
    n = 150
    age = rng.integers(20, 65, size=n)
    sex = rng.choice(["Male", "Female"], size=n)
    high_income = (age > 40).astype(int)
    df = pd.DataFrame({"age": age, "sex": sex, "high_income": high_income})

    # 같은 df·RANDOM_STATE로 학습하므로 evaluate_income_model()의 결과와
    # train_income_model()이 디스크에 저장하는 파이프라인은 동일해야 한다.
    original_evaluation = modeling.evaluate_income_model(df)
    train_income_model(df)

    X, _, _, _ = modeling._split_features(df)
    original_proba = original_evaluation.pipeline.predict_proba(X)[:, 1]
    reloaded_proba = predict_income(df)["probability"].to_numpy()

    assert np.allclose(original_proba, reloaded_proba)
