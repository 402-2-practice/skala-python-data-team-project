"""src.report.generate_report()가 다섯 개 JSON 산출물을 report.md로 정확히 합치는지 검증한다."""

from __future__ import annotations

import json

from src.config import REPORT_PATH, TABLE_DIR, ensure_directories
from src.report import generate_report

_EDA = {
    "rows": 100,
    "columns": 16,
    "college_degree_pct": 25.0,
    "high_income_pct": 24.0,
    "pandas_seconds": 0.01,
    "polars_seconds": 0.005,
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


def _write_fixture_json(filename: str, payload: dict) -> None:
    (TABLE_DIR / filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_generate_report_combines_all_stage_outputs():
    ensure_directories()
    _write_fixture_json("eda_summary.json", _EDA)
    _write_fixture_json("welch_ttest.json", _WELCH)
    _write_fixture_json("psm_result.json", _PSM)
    _write_fixture_json("psm_sensitivity_result.json", _SENSITIVITY)
    _write_fixture_json("model_metrics.json", _MODEL)

    generate_report()

    assert REPORT_PATH.exists()
    content = REPORT_PATH.read_text(encoding="utf-8")
    assert "대학 학위" in content
    assert "0.81" in content  # accuracy
    assert "40" in content  # matched_pairs
    # 인과관계를 확정적으로 증명했다고 표현하지 않는다는 해석 원칙이 실제로 report에 들어가는지 확인한다.
    assert "확정적 증명이 아니" in content
