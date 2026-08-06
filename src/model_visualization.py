"""예측 모델 진단 시각화 — Accuracy/Precision/Recall/F1, ROC curve, Confusion matrix.

담당: 이서현 (시각화·보고서)

modeling.py의 train_income_model()이 저장한 산출물(model_metrics.json,
model_predictions.csv)에 의존하므로, main.py에서 model 단계 이후에 호출해야 한다.
modeling.py는 roc_curve.csv/confusion_matrix.csv처럼 이미 계산된 곡선/행렬을 저장하지
않고, 테스트셋 행 단위 예측(model_predictions.csv: row_id, y_test, y_pred, y_proba)만
남긴다 — ROC curve와 confusion matrix는 여기서 sklearn.metrics로 직접 계산한다.
주제 중심(학위·소득·PSM) 차트는 src/visualization.py를 참고.
"""

from __future__ import annotations

import json
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve as sk_roc_curve

from src.config import FIGURE_DIR, TABLE_DIR

INCOME_CLASS_LABELS = ["<=50K", ">50K"]


def _require_keys(data: dict, keys: list[str], chart_name: str) -> None:
    """model_metrics.json에 차트가 필요로 하는 키가 다 있는지 확인한다."""
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{chart_name}에 필요한 model_metrics.json 키가 없습니다: {missing}")


def _require_columns(df: pd.DataFrame, columns: list[str], chart_name: str) -> None:
    """model_predictions.csv에 차트가 필요로 하는 컬럼이 다 있는지 확인한다."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{chart_name}에 필요한 model_predictions.csv 컬럼이 없습니다: {missing}")


# ============================================================
# [그룹 비교] 예측 모델 성능 지표 막대그래프 (Seaborn)
# - modeling.py의 train_income_model()이 저장한 model_metrics.json 사용
# - x축: 지표(accuracy/precision/recall/f1만 — ROC-AUC는 임계값 전체를 요약한 별도 성격의 지표라 제외)
# - y축: 점수(0~1)
# - model 단계를 먼저 실행해야 파일이 존재 — 없으면 경고만 출력하고 건너뜀
# ============================================================
def plot_model_metrics() -> None:
    model_metrics_path = TABLE_DIR / "model_metrics.json"
    if not model_metrics_path.exists():
        print("[시각화 경고] model_metrics.json이 없습니다. model 단계를 먼저 실행하세요.")
        return

    model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    metric_names = ["accuracy", "precision", "recall", "f1"]
    _require_keys(model_metrics, metric_names, "plot_model_metrics")
    metrics_table = pd.DataFrame(
        {"metric": metric_names, "score": [model_metrics[name] for name in metric_names]}
    )
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=metrics_table, x="metric", y="score", hue="metric", legend=False)
    ax.set(title="Income prediction model performance", xlabel="Metric", ylabel="Score")
    ax.set_ylim(0, 1.08)
    for patch, value in zip(ax.patches, metrics_table["score"]):
        bar = cast(Rectangle, patch)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}",
            ha="center",
        )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_performance_metrics.png", dpi=160)
    plt.close()


# ============================================================
# [그룹 비교] ROC curve (Seaborn)
# - model_predictions.csv(테스트셋 행 단위 y_test/y_proba)에서 sklearn.metrics.roc_curve로
#   fpr/tpr을 직접 계산해 선 그래프로 표현, 대각선은 무작위 분류 기준선
# - 제목에 AUC 값을 함께 표시해 막대그래프에서 뺀 ROC-AUC를 별도로 강조
# ============================================================
def plot_roc_curve() -> None:
    predictions_path = TABLE_DIR / "model_predictions.csv"
    model_metrics_path = TABLE_DIR / "model_metrics.json"
    if not (predictions_path.exists() and model_metrics_path.exists()):
        print(
            "[시각화 경고] model_predictions.csv 또는 model_metrics.json이 없습니다. "
            "model 단계를 먼저 실행하세요."
        )
        return

    model_metrics = json.loads(model_metrics_path.read_text(encoding="utf-8"))
    _require_keys(model_metrics, ["roc_auc"], "plot_roc_curve")
    predictions = pd.read_csv(predictions_path)
    _require_columns(predictions, ["y_test", "y_proba"], "plot_roc_curve")
    fpr, tpr, _ = sk_roc_curve(predictions["y_test"], predictions["y_proba"])
    roc_points = pd.DataFrame({"fpr": fpr, "tpr": tpr})

    plt.figure(figsize=(6, 6))
    ax = sns.lineplot(data=roc_points, x="fpr", y="tpr", color="#4C72B0", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set(
        title=f"ROC curve (AUC = {model_metrics['roc_auc']:.3f})",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_roc_curve.png", dpi=160)
    plt.close()


# ============================================================
# [그룹 비교] Confusion matrix (Seaborn)
# - model_predictions.csv(테스트셋 y_test/y_pred)에서 sklearn.metrics.confusion_matrix로
#   클래스 순서를 [0, 1](<=50K, >50K)로 고정한 2x2 행렬을 히트맵으로 표현
# - 각 셀에 건수와 실제 클래스 기준 행 비율을 함께 표시해 클래스 불균형 상황에서도 해석 가능
# - class_weight="balanced"로 인한 recall 우선(오탐 증가) 경향을 셀 단위로 직접 확인
# ============================================================
def plot_confusion_matrix() -> None:
    predictions_path = TABLE_DIR / "model_predictions.csv"
    if not predictions_path.exists():
        print("[시각화 경고] model_predictions.csv가 없습니다. model 단계를 먼저 실행하세요.")
        return

    predictions = pd.read_csv(predictions_path)
    _require_columns(predictions, ["y_test", "y_pred"], "plot_confusion_matrix")
    confusion = sk_confusion_matrix(
        predictions["y_test"],
        predictions["y_pred"],
        labels=[0, 1],
    )
    confusion_table = pd.DataFrame(
        confusion, index=INCOME_CLASS_LABELS, columns=INCOME_CLASS_LABELS
    )
    row_percent = confusion_table.div(confusion_table.sum(axis=1), axis=0).mul(100)
    annotations = confusion_table.copy().astype(str)
    for actual_label in INCOME_CLASS_LABELS:
        for predicted_label in INCOME_CLASS_LABELS:
            count = confusion_table.loc[actual_label, predicted_label]
            percent = row_percent.loc[actual_label, predicted_label]
            annotations.loc[actual_label, predicted_label] = f"{count:,}\n({percent:.1f}%)"

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        confusion_table,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
    )
    ax.set(title="Confusion matrix (test set)", xlabel="Predicted", ylabel="Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_confusion_matrix.png", dpi=160)
    plt.close()


def create_model_visualizations() -> None:
    sns.set_theme(style="whitegrid")

    plot_model_metrics()
    plot_roc_curve()
    plot_confusion_matrix()
