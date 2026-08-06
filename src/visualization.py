"""Seaborn 정적 시각화와 Plotly 인터랙티브 시각화 — 주제 중심(학위·소득·PSM) 차트.

담당: 이서현 (시각화·보고서)

예측 모델 진단 차트(성능 지표/ROC curve/confusion matrix)는 model 단계 산출물에 의존하므로
src/model_visualization.py로 분리했다. main.py에서 model 단계 이후에 호출해야 한다.
"""

from __future__ import annotations

import json
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.patches import Rectangle

from src.config import FIGURE_DIR, TABLE_DIR


def _require_columns(df: pd.DataFrame, columns: list[str], chart_name: str) -> None:
    """차트가 필요로 하는 컬럼이 df에 다 있는지 확인한다.

    없으면 pandas가 던지는 원시 KeyError 대신, 어떤 차트가 어떤 컬럼을 원했는지
    바로 알 수 있는 메시지로 실패시킨다.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{chart_name}에 필요한 컬럼이 없습니다: {missing}")


# ============================================================
# [그룹 비교] 학력별 고소득률 인터랙티브 막대그래프 (Plotly)
# - x축: education, y축: high_income_rate(%)
# - hover: 표본 크기(size), education-num
# ============================================================
def plot_education_income_rate(df: pd.DataFrame) -> None:
    _require_columns(
        df, ["education", "education-num", "high_income"], "plot_education_income_rate"
    )
    education_rate = (
        df.groupby(["education", "education-num"], observed=True)["high_income"]
        .agg(["mean", "size"])
        .reset_index()
        .sort_values("education-num")
    )
    # round(2)로 반올림 
    education_rate["high_income_rate"] = (education_rate["mean"] * 100).round(2)
    fig = px.bar(
        education_rate,
        x="education",
        y="high_income_rate",
        hover_data=["size", "education-num"],
        title="Interactive high-income rate by education",
        labels={"education": "Education", "high_income_rate": "High-income rate (%)"},
    )
    fig.write_html(FIGURE_DIR / "education_income_rate.html", include_plotlyjs="cdn")


# ============================================================
# [분포] age 히스토그램 (Seaborn)
# - x축: age, hue: income, kde 곡선 포함
# - 고소득/저소득 집단의 연령 분포 비교
# - 저소득층은 20대 초중반에 몰려있고, 고소득층은 30~50대에 넓게 분포하는지 확인
# ============================================================
def plot_age_distribution(df: pd.DataFrame) -> None:
    _require_columns(df, ["age", "income"], "plot_age_distribution")
    plt.figure(figsize=(8, 5))
    # binwidth를 지정하지 않으면 Seaborn이 자동으로 고른 bin 경계가 정수 나이와
    # 어긋나서 특정 나이만 유독 튀어 보이는 톱니 패턴이 생긴다. age는 정수이므로
    # binwidth=1로 고정해 나이 한 살 단위로 정확히 집계한다.
    # common_norm=False로 두 집단을 각각 독립적으로 정규화한다 — 기본값(True)은 두
    # 집단을 합쳐서 정규화하기 때문에, 표본 수가 훨씬 많은 <=50K 집단이 표본 크기
    # 차이만으로 더 높게 그려져 "분포 모양" 비교가 왜곡된다.
    ax = sns.histplot(
        data=df,
        x="age",
        hue="income",
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        binwidth=1,
    )
    ax.set(title="Age distribution by income group", xlabel="Age", ylabel="Density")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "age_distribution_by_income.png", dpi=160)
    plt.close()


# ============================================================
# [그룹 비교] 자본 이득 유무(이진 지표) → 고소득률 막대그래프 (Seaborn)
# - x축: capital-gain 유무(0 vs >0), y축: high_income_rate(%)
# - capital-gain은 대부분(약 91~92%)이 0이라 금액 자체보다 "있다/없다"가 더 강한 신호
# - capital-gain=0 집단 vs >0 집단의 고소득률이 약 3배 차이 (정확한 수치는 데이터 정제 로직에 따라 변동)
# - 결과변수(high_income)와 직접 연결되는 비교라 설계 문서의 핵심 결과변수와 일치
# ============================================================
def plot_capital_gain_indicator(df: pd.DataFrame) -> None:
    _require_columns(df, ["capital-gain", "high_income"], "plot_capital_gain_indicator")
    capital_gain_indicator = (
        df.assign(has_capital_gain=(df["capital-gain"] > 0).map({False: "No gain", True: "Has gain"}))
        .groupby("has_capital_gain", observed=True)["high_income"]
        .mean()
        .mul(100)
        .rename("high_income_rate")
        .reset_index()
    )
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=capital_gain_indicator,
        x="has_capital_gain",
        y="high_income_rate",
        hue="has_capital_gain",
        legend=False,
    )
    ax.set(
        title="High-income rate by capital gain presence",
        xlabel="Capital gain",
        ylabel="High-income rate (%)",
    )
    for patch in ax.patches:
        bar = cast(Rectangle, patch)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center",
        )
    ax.margins(y=0.15)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "capital_gain_indicator_income_rate.png", dpi=160)
    plt.close()


# ============================================================
# [그룹 비교] sex/race별 고소득 비율 막대 (Plotly)
# - x축: race, y축: high_income_rate, color: sex
# - 인종/성별 조합별 고소득(>50K) 비율 비교
# - 집단 간 소득 격차 존재 여부 확인
# ============================================================
def plot_race_sex_income_rate(df: pd.DataFrame) -> None:
    _require_columns(df, ["race", "sex", "high_income"], "plot_race_sex_income_rate")
    # round(2)로 반올림 
    sex_race_rate = (
        df.groupby(["race", "sex"], observed=True)["high_income"]
        .mean()
        .mul(100)
        .round(2)
        .rename("high_income_rate")
        .reset_index()
    )
    fig = px.bar(
        sex_race_rate,
        x="race",
        y="high_income_rate",
        color="sex",
        barmode="group",
        title="Interactive high-income rate by race and sex",
        labels={"race": "Race", "high_income_rate": "High-income rate (%)", "sex": "Sex"},
    )
    fig.write_html(FIGURE_DIR / "race_sex_income_rate.html", include_plotlyjs="cdn")


# ============================================================
# [그룹 비교] PSM 공변량 균형 Love Plot (Seaborn/Matplotlib)
# - statistics.py의 propensity_score_matching()이 저장한 psm_balance.csv 사용
# - x축: SMD(표준화 평균차이), y축: 공변량(매칭 전 SMD 상위 15개, "_nan" 결측 더미 제외)
# - 매칭 전(빨강) vs 매칭 후(파랑) 점을 나란히 찍어 균형 개선 여부를 확인
# - 0.1 기준선(점선) 안쪽으로 들어오면 해당 공변량은 "균형이 맞다"고 판단하는 PSM 표준 진단 차트
# - statistics 단계를 먼저 실행해야 파일이 존재 — 없으면 경고만 출력하고 건너뜀
# ============================================================
def plot_psm_balance() -> None:
    psm_balance_path = TABLE_DIR / "psm_balance.csv"
    if not psm_balance_path.exists():
        print("[시각화 경고] psm_balance.csv가 없습니다. statistics 단계를 먼저 실행하세요.")
        return

    balance: pd.DataFrame = pd.read_csv(psm_balance_path)
    non_missing_dummy = ~balance["covariate"].str.endswith("_nan")
    balance = balance.copy()

    balance["max_smd"] = (
        balance[
            [
                "smd_before",
                "smd_after",
            ]
        ]
        .abs()
        .max(axis=1)
    )

    top_balance = (
        balance.loc[non_missing_dummy]
        .sort_values(
            by="max_smd",
            ascending=False,
        )
        .head(15)
        .sort_values(
            by="max_smd",
            ascending=True,
        )
        .reset_index(drop=True)
    )
    y_pos = range(len(top_balance))
    plt.figure(figsize=(8, 8))
    plt.hlines(
        y=y_pos,
        xmin=0,
        xmax=top_balance[["smd_before", "smd_after"]].to_numpy().max(axis=1),
        color="lightgray",
        linewidth=1,
    )
    plt.scatter(top_balance["smd_before"], y_pos, color="#C44E52", label="Before matching", zorder=3)
    plt.scatter(top_balance["smd_after"], y_pos, color="#4C72B0", label="After matching", zorder=3)
    plt.axvline(0.1, color="black", linestyle="--", linewidth=1, label="SMD = 0.1 threshold")
    plt.yticks(list(y_pos), top_balance["covariate"].tolist())
    plt.xlabel("Standardized mean difference (SMD)")
    plt.ylabel("Covariate")
    plt.title("Covariate balance before vs after PSM matching")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "psm_balance_love_plot.png", dpi=160)
    plt.close()


# ============================================================
# [그룹 비교] 3단계 분석 효과 크기 비교 (Plotly)
# - welch_ttest.json(단순 비교) + psm_result.json(주 PSM) + psm_sensitivity_result.json(민감도 분석) 사용
# - x축: 분석 단계(1.단순 비교 / 2.주 PSM / 3.민감도 분석), y축: 고소득률 차이(학위 있음 - 없음, %p 단위)
# - "작은 p-value는 효과 크기를 의미하지 않는다"는 설계 문서 해석 원칙을 시각적으로 뒷받침
# - hover로 p-value, 매칭 표본 수(matched_pairs) 확인 — 단계별 표본 규모 차이도 함께 보고
# - statistics 단계를 먼저 실행해야 세 파일이 모두 존재 — 하나라도 없으면 경고만 출력하고 건너뜀
# ============================================================
def plot_effect_comparison() -> None:
    welch_path = TABLE_DIR / "welch_ttest.json"
    psm_path = TABLE_DIR / "psm_result.json"
    sensitivity_path = TABLE_DIR / "psm_sensitivity_result.json"
    if not (welch_path.exists() and psm_path.exists() and sensitivity_path.exists()):
        print("[시각화 경고] welch_ttest/psm_result/psm_sensitivity_result.json 중 일부가 없습니다. statistics 단계를 먼저 실행하세요.")
        return

    welch = json.loads(welch_path.read_text(encoding="utf-8"))
    psm = json.loads(psm_path.read_text(encoding="utf-8"))
    sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))

    # welch/psm 결과의 effect는 0.32 같은 비율(fraction)이라, 그대로 쓰면 "0.32%"인지
    # "32%p"인지 헷갈린다. 100을 곱해 %p(백분위 포인트) 단위로 명시한다.
    # effect_pp는 round(2)로 hover 소수점을 줄인다. p_value는 statistics.py가
    # p_value == 0(1e-300 미만이라 float64 표현 범위를 벗어나 0으로 언더플로된 경우)을
    # "< 1e-300" 문자열로 바꿔둔 p_value_display를 그대로 쓴다 — raw p_value(float)를
    # 그대로 hover에 포맷하면 0.0인 값은 어떤 소수점/유효숫자 포맷을 적용해도 그대로
    # "0"으로 찍혀 유의성 정보가 사라진다.
    effect_summary = pd.DataFrame(
        [
            {
                "stage": "1. Naive comparison",
                "effect_pp": round(welch["mean_difference"] * 100, 2),
                "p_value": welch["p_value_display"],
                "sample_note": "all rows (unmatched)",
            },
            {
                "stage": "2. Main PSM",
                "effect_pp": round(psm["matched_rate_difference"] * 100, 2),
                "p_value": psm["p_value_display"],
                "sample_note": f"{psm['matched_pairs']} matched pairs",
            },
            {
                "stage": "3. Sensitivity PSM",
                "effect_pp": round(sensitivity["matched_rate_difference"] * 100, 2),
                "p_value": sensitivity["p_value_display"],
                "sample_note": f"{sensitivity['matched_pairs']} matched pairs",
            },
        ]
    )
    fig = px.bar(
        effect_summary,
        x="stage",
        y="effect_pp",
        hover_data=["p_value", "sample_note"],
        title="Interactive high-income rate difference by analysis stage",
        labels={"stage": "Analysis stage", "effect_pp": "High-income rate difference (%p, degree - no degree)"},
    )
    fig.write_html(FIGURE_DIR / "effect_size_comparison.html", include_plotlyjs="cdn")


# ============================================================
# [분포] 수치형 변수 상관관계 히트맵 (Seaborn)
# - statistics.py가 저장한 correlations.csv 사용 (수치형 컬럼 전체 상관계수 행렬)
# - 대각선 위 삼각형은 아래 삼각형과 대칭이라 마스킹해 중복 정보를 없앰
# - 상관계수는 -1~1의 방향성 있는 값이라 0을 중심으로 하는 발산형 컬러맵 사용
# - statistics 단계를 먼저 실행해야 파일이 존재 — 없으면 경고만 출력하고 건너뜀
# ============================================================
def plot_correlation_heatmap() -> None:
    correlations_path = TABLE_DIR / "correlations.csv"
    if not correlations_path.exists():
        print("[시각화 경고] correlations.csv가 없습니다. statistics 단계를 먼저 실행하세요.")
        return

    correlations = pd.read_csv(correlations_path, index_col=0)
    mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
    plt.figure(figsize=(9, 8))
    ax = sns.heatmap(
        correlations,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation coefficient"},
    )
    ax.set_title("Correlation heatmap of numeric features")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def create_visualizations(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plot_education_income_rate(df)
    plot_age_distribution(df)
    plot_capital_gain_indicator(df)
    plot_race_sex_income_rate(df)
    plot_psm_balance()
    plot_effect_comparison()
    plot_correlation_heatmap()
