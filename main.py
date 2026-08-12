"""프로젝트 전체 파이프라인 실행 진입점.

담당: 고동민 (테스트·통합·문서 QA)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import RAW_DATA_PATH, ensure_directories
from src.data import load_and_clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Adult 대학 학위 가치 분석")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument(
        "--stage",
        choices=["all", "eda", "statistics", "visualization", "model", "report"],
        default="all",
    )
    args = parser.parse_args()
    ensure_directories()

    if not args.data.exists():
        raise FileNotFoundError(
            f"데이터 파일이 없습니다: {args.data}\n"
            "adult.csv를 data/raw/adult.csv에 복사하거나 --data 경로를 지정하세요."
        )

    df = load_and_clean(args.data)

    if args.stage in ["all", "eda"]:
        from src.eda import run_eda

        run_eda(data_path=args.data,)
    if args.stage in ["all", "statistics", "visualization",]:
        from src.statistics import run_statistics

        run_statistics(df)
    if args.stage in ["all", "visualization"]:
        from src.visualization import create_visualizations

        create_visualizations(df)
    if args.stage in ["all", "model"]:
        from src.model_visualization import create_model_visualizations
        from src.modeling import train_income_model

        train_income_model(df)
        create_model_visualizations()


if __name__ == "__main__":
    main()
