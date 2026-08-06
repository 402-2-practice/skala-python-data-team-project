"""src.report.generate_report()가 JSON·상관계수 산출물을 report.md로 정확히 합치는지 검증한다."""

from __future__ import annotations

import json

import pandas as pd

import src.report as report

_EDA = {
    "dataset": {"rows": 100, "columns": 16},
    "college_degree_distribution": {"college_degree_rate": 0.25},
    "target_distribution": {"high_income_rate": 0.24},
}
_BENCHMARK = {
    "pandas_median_seconds": 0.01,
    "polars_median_seconds": 0.005,
}
_WELCH = {
    "no_degree_mean": 0.15,
    "degree_mean": 0.45,
    "mean_difference": 0.30,
    "t_statistic": 10.0,
    "p_value": 0.0001,
    "significant_at_0_05": True,
}
_PSM = {
    "matched_pairs": 40,
    "matched_no_degree_rate": 0.20,
    "matched_degree_rate": 0.48,
    "matched_rate_difference": 0.28,
    "p_value": 0.0002,
    "max_smd_before": 0.25,
    "max_smd_after": 0.05,
}
_SENSITIVITY = {
    "matched_no_degree_rate": 0.30,
    "matched_degree_rate": 0.47,
    "matched_rate_difference": 0.17,
    "p_value": 0.001,
    "max_smd_before": 0.9,
    "max_smd_after": 0.06,
}
_MODEL = {
    "accuracy": 0.81,
    "precision": 0.57,
    "recall": 0.86,
    "f1": 0.69,
    "roc_auc": 0.91,
}
_CORRELATION_COLUMNS = ["high_income", "college_degree", "education-num", "age"]
_CORRELATIONS = pd.DataFrame(
    [
        [1.00, 0.35, 0.33, 0.20],
        [0.35, 1.00, 0.60, 0.05],
        [0.33, 0.60, 1.00, 0.10],
        [0.20, 0.05, 0.10, 1.00],
    ],
    index=_CORRELATION_COLUMNS,
    columns=_CORRELATION_COLUMNS,
)


def _write_fixture_json(filename: str, payload: dict) -> None:
    (report.TABLE_DIR / filename).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_generate_report_combines_all_stage_outputs():
    _write_fixture_json("eda_summary.json", _EDA)
    _write_fixture_json("data_engine_benchmark.json", _BENCHMARK)
    _write_fixture_json("welch_ttest.json", _WELCH)
    _write_fixture_json("psm_result.json", _PSM)
    _write_fixture_json("psm_sensitivity_result.json", _SENSITIVITY)
    _write_fixture_json("model_metrics.json", _MODEL)
    _CORRELATIONS.to_csv(report.TABLE_DIR / "correlations.csv")

    report.generate_report()

    assert report.REPORT_PATH.exists()
    content = report.REPORT_PATH.read_text(encoding="utf-8")
    assert "대학 학위" in content
    assert "0.81" in content  # accuracy
    assert "40" in content  # matched_pairs
    # 인과관계를 확정적으로 증명했다고 표현하지 않는다는 해석 원칙이 실제로 report에 들어가는지 확인한다.
    assert "확정적 증명이 아니" in content
    # high_income과의 상관계수 표가 절댓값 큰 순으로 정렬되어 들어가는지 확인한다.
    assert "| college_degree | 0.350 |" in content
