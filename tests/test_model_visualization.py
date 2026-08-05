"""src.model_visualization.create_model_visualizations()가 요구 산출물을 생성하는지 검증한다."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import src.model_visualization as model_visualization
from src.model_visualization import create_model_visualizations


def test_create_model_visualizations_writes_expected_figures():
    (model_visualization.TABLE_DIR / "model_metrics.json").write_text(
        json.dumps(
            {"accuracy": 0.83, "precision": 0.62, "recall": 0.85, "f1": 0.72, "roc_auc": 0.93}
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(0)
    n = 100
    y_test = rng.integers(0, 2, size=n)
    predictions = pd.DataFrame(
        {
            "row_id": range(n),
            "y_test": y_test,
            "y_pred": rng.integers(0, 2, size=n),
            "y_proba": np.clip(y_test * 0.6 + rng.random(n) * 0.4, 0, 1),
        }
    )
    predictions.to_csv(model_visualization.TABLE_DIR / "model_predictions.csv", index=False)

    create_model_visualizations()

    for filename in [
        "model_performance_metrics.png",
        "model_roc_curve.png",
        "model_confusion_matrix.png",
    ]:
        path = model_visualization.FIGURE_DIR / filename
        assert path.exists() and path.stat().st_size > 0


def test_create_model_visualizations_skips_gracefully_without_model_outputs():
    # model 단계를 아직 안 돌렸으면 예외 없이 경고만 출력하고 넘어가야 한다.
    create_model_visualizations()
