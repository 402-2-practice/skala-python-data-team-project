"""테스트가 outputs/tables·models·figures와 report.md 등 실제 프로젝트 산출물을
합성 데이터로 덮어쓰지 않도록, 각 모듈이 이미 바인딩한 경로 상수를 매 테스트마다
tmp_path 하위 디렉터리로 바꿔치기한다.

각 모듈이 `from src.config import TABLE_DIR`처럼 값을 직접 import해서 쓰기 때문에,
src.config 쪽을 패치해도 이미 바인딩된 이름에는 반영되지 않는다. 그래서 모듈별로
개별 패치한다. 서로 다른 테스트 파일이 welch_ttest.json 같은 같은 파일명을 공유해도
매 테스트 격리 디렉터리가 새로 생기므로 파일 내용이 섞이지 않는다.
"""

from __future__ import annotations

import pytest

import src.model_visualization as model_visualization
import src.modeling as modeling
import src.report as report
import src.statistics as statistics
import src.visualization as visualization


@pytest.fixture(autouse=True)
def isolate_output_paths(tmp_path, monkeypatch):
    table_dir = tmp_path / "tables"
    model_dir = tmp_path / "models"
    figure_dir = tmp_path / "figures"
    for directory in (table_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for module in (modeling, model_visualization, report, statistics, visualization):
        if hasattr(module, "TABLE_DIR"):
            monkeypatch.setattr(module, "TABLE_DIR", table_dir)

    if hasattr(modeling, "MODEL_DIR"):
        monkeypatch.setattr(modeling, "MODEL_DIR", model_dir)
    if hasattr(visualization, "FIGURE_DIR"):
        monkeypatch.setattr(visualization, "FIGURE_DIR", figure_dir)
    if hasattr(model_visualization, "FIGURE_DIR"):
        monkeypatch.setattr(model_visualization, "FIGURE_DIR", figure_dir)
    if hasattr(report, "REPORT_PATH"):
        monkeypatch.setattr(report, "REPORT_PATH", tmp_path / "report.md")
