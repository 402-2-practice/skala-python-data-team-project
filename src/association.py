"""
Adult Census Income 범용 연관성 분석 모듈.

Purpose:
    사용자가 선택한 관심 변수(exposure)와 통제 변수(controls)를 바탕으로
    고소득 여부(target)와의 조정 전·후 연관성을 분석한다.

Main Tasks:
    1. 관심 변수를 binary / continuous / categorical로 자동 판별한다.
    2. 해당 분석에 필요한 변수만 선택해 결측치를 처리한다.
    3. 변수 유형에 맞는 조정 전 요약 통계를 계산한다.
    4. 이진 결과변수에 대해 Logistic Regression을 수행한다.
    5. coefficient, odds ratio, p-value, 95% CI를 반환한다.

Caution:
    이 모듈에서 계산하는 결과는 관찰 데이터에 기반한 연관성이다.
    통제변수를 포함하더라도 측정되지 않은 교란요인이 존재할 수 있으므로
    결과를 확정적인 인과효과로 해석하지 않는다.

    modeling.py의 머신러닝 모델과 목적이 다르다.
    - association.py: 변수와 고소득 여부의 관계 설명
    - modeling.py: 새로운 입력의 고소득 여부 예측
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, pointbiserialr
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from src.data import prepare_analysis_data


# ============================================================
# 분석 타입 정의
# ============================================================

VariableType = Literal[
    "binary",
    "continuous",
    "categorical",
]


class AssociationError(ValueError):
    """연관성 분석 입력 또는 모델 적합 과정에서 발생하는 오류."""


@dataclass(frozen=True)
class AnalysisRequest:
    """사용자가 요청한 연관성 분석 조건.

    Args:
        exposure:
            고소득 여부와의 관계를 확인할 관심 변수.

        target:
            분석할 결과변수.
            현재 프로젝트에서는 기본적으로 high_income을 사용한다.

        controls:
            exposure와 target의 관계를 볼 때 함께 조정할 통제 변수.
    """

    exposure: str
    target: str = "high_income"
    controls: tuple[str, ...] = ()


# ============================================================
# 공통 입력 검증
# ============================================================

def _validate_request(
    df: pd.DataFrame,
    request: AnalysisRequest,
) -> None:
    """분석 요청 자체가 논리적으로 유효한지 확인한다."""

    if not isinstance(df, pd.DataFrame):
        raise AssociationError(
            "연관성 분석 입력은 pandas.DataFrame이어야 합니다."
        )

    if df.empty:
        raise AssociationError(
            "연관성 분석용 데이터프레임이 비어 있습니다."
        )

    if request.exposure == request.target:
        raise AssociationError(
            "관심 변수(exposure)와 결과 변수(target)는 "
            "같은 변수일 수 없습니다."
        )

    if request.exposure in request.controls:
        raise AssociationError(
            f"관심 변수 '{request.exposure}'가 "
            "통제 변수에도 포함되어 있습니다."
        )

    if request.target in request.controls:
        raise AssociationError(
            f"결과 변수 '{request.target}'는 "
            "통제 변수로 사용할 수 없습니다."
        )

    if len(set(request.controls)) != len(request.controls):
        raise AssociationError(
            "통제 변수에 중복된 변수가 있습니다."
        )

    required_columns = [
        request.target,
        request.exposure,
        *request.controls,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise AssociationError(
            "분석에 필요한 변수가 없습니다: "
            f"{missing_columns}"
        )


# ============================================================
# 변수 유형 판별
# ============================================================

def infer_variable_type(
    series: pd.Series,
) -> VariableType:
    """관심 변수를 binary / continuous / categorical로 분류한다.

    판별 기준:
        - 고유값 2개: binary
        - 고유값이 3개 이상이고 숫자형: continuous
        - 그 외: categorical

    Notes:
        숫자로 코딩된 범주형 변수는 향후 config.py의 변수 메타데이터를
        이용해 명시적으로 구분하는 방식으로 발전시키는 것이 더 안전하다.
    """

    non_null = series.dropna()

    if non_null.empty:
        raise AssociationError(
            f"변수 '{series.name}'에 분석 가능한 값이 없습니다."
        )

    unique_count = non_null.nunique()

    if unique_count < 2:
        raise AssociationError(
            f"변수 '{series.name}'에 서로 다른 값이 "
            "2개 이상 필요합니다."
        )

    if unique_count == 2:
        return "binary"

    if pd.api.types.is_numeric_dtype(non_null):
        return "continuous"

    return "categorical"


# ============================================================
# Target 검증
# ============================================================

def _validate_binary_target(
    series: pd.Series,
) -> None:
    """현재 Logistic Regression용 target이 정확히 0/1인지 검증한다.

    잘못된 값을 int로 변환한 뒤 검사하지 않고,
    원래 값을 그대로 확인해 데이터 오류가 숨겨지는 것을 막는다.
    """

    values = set(
        series.dropna().unique()
    )

    if values != {0, 1}:
        raise AssociationError(
            f"결과 변수 '{series.name}'는 "
            "0과 1을 모두 포함하는 이진 변수여야 합니다. "
            f"현재 값: {values}"
        )


# ============================================================
# 범주 순서 결정
# ============================================================

def _ordered_levels(
    series: pd.Series,
) -> list:
    """범주형 변수의 기준 범주를 재현 가능하게 결정한다.

    값의 문자열 표현을 기준으로 정렬하며,
    첫 번째 값을 Logistic Regression의 기준 범주로 사용한다.
    """

    values = (
        series
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(
        values,
        key=lambda value: str(value),
    )


# ============================================================
# 조정 전 분석
# ============================================================

def _unadjusted_summary(
    df: pd.DataFrame,
    request: AnalysisRequest,
    exposure_type: VariableType,
) -> dict:
    """통제변수를 적용하기 전 exposure와 target의 관계를 요약한다."""

    exposure = request.exposure
    target = request.target

    # --------------------------------------------------------
    # 이진형 관심 변수
    # --------------------------------------------------------

    if exposure_type == "binary":

        levels = _ordered_levels(
            df[exposure]
        )

        reference = levels[0]
        comparison = levels[1]

        grouped = (
            df
            .groupby(
                exposure,
                observed=True,
            )[target]
            .agg(
                n="size",
                target_rate="mean",
            )
            .reset_index()
        )

        rates = {
            row[exposure]: float(
                row["target_rate"]
            )
            for _, row in grouped.iterrows()
        }

        return {
            "method": "binary_group_comparison",
            "reference_level": reference,
            "comparison_level": comparison,
            "reference_rate": rates[reference],
            "comparison_rate": rates[comparison],
            "rate_difference": (
                rates[comparison]
                - rates[reference]
            ),
            "groups": grouped.to_dict(
                orient="records"
            ),
        }

    # --------------------------------------------------------
    # 연속형 관심 변수
    # --------------------------------------------------------

    if exposure_type == "continuous":

        exposure_values = pd.to_numeric(
            df[exposure],
            errors="raise",
        ).astype(float)

        target_values = (
            df[target]
            .astype(int)
        )

        correlation, p_value = (
            pointbiserialr(
                target_values,
                exposure_values,
            )
        )

        return {
            "method": "point_biserial_correlation",
            "correlation": float(correlation),
            "p_value": float(p_value),
            "exposure_mean": float(
                exposure_values.mean()
            ),
            "exposure_std": float(
                exposure_values.std(ddof=1)
            ),
        }

    # --------------------------------------------------------
    # 다범주형 관심 변수
    # --------------------------------------------------------

    grouped = (
        df
        .groupby(
            exposure,
            observed=True,
        )[target]
        .agg(
            n="size",
            target_rate="mean",
        )
        .reset_index()
        .sort_values(
            "target_rate",
            ascending=False,
        )
    )

    contingency = pd.crosstab(
        df[exposure],
        df[target],
    )

    chi2, p_value, dof, _ = (
        chi2_contingency(
            contingency
        )
    )

    return {
        "method": "categorical_group_comparison",
        "chi2_statistic": float(chi2),
        "chi2_p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "groups": grouped.to_dict(
            orient="records"
        ),
    }


# ============================================================
# Logistic Regression 입력 행렬 생성
# ============================================================

def _build_design_matrix(
    df: pd.DataFrame,
    request: AnalysisRequest,
    exposure_type: VariableType,
) -> tuple[
    pd.DataFrame,
    list[str],
    dict,
]:
    """exposure와 controls를 Logistic Regression 입력 X로 변환한다.

    처리 원칙:
        - continuous: 숫자 그대로 사용
        - binary: 기준 범주를 0, 비교 범주를 1로 변환
        - categorical: one-hot encoding 후 첫 범주를 기준 범주로 제거
        - 범주형 controls도 동일하게 one-hot encoding
    """

    exposure = request.exposure
    controls = list(request.controls)

    predictors = [
        exposure,
        *controls,
    ]

    X = df[predictors].copy()

    categorical_columns: list[str] = []

    exposure_metadata: dict = {
        "type": exposure_type,
    }

    # --------------------------------------------------------
    # 관심 변수 처리
    # --------------------------------------------------------

    if exposure_type == "binary":

        levels = _ordered_levels(
            X[exposure]
        )

        reference = levels[0]
        comparison = levels[1]

        X[exposure] = (
            X[exposure]
            .map(
                {
                    reference: 0.0,
                    comparison: 1.0,
                }
            )
            .astype(float)
        )

        exposure_metadata.update(
            {
                "reference_level": reference,
                "comparison_level": comparison,
            }
        )

    elif exposure_type == "continuous":

        X[exposure] = pd.to_numeric(
            X[exposure],
            errors="raise",
        ).astype(float)

        exposure_metadata[
            "interpretation_unit"
        ] = 1

    else:

        levels = _ordered_levels(
            X[exposure]
        )

        X[exposure] = pd.Categorical(
            X[exposure],
            categories=levels,
        )

        categorical_columns.append(
            exposure
        )

        exposure_metadata[
            "reference_level"
        ] = levels[0]

        exposure_metadata[
            "levels"
        ] = levels

    # --------------------------------------------------------
    # 통제 변수 처리
    # --------------------------------------------------------

    for column in controls:

        if pd.api.types.is_numeric_dtype(
            X[column]
        ):
            X[column] = pd.to_numeric(
                X[column],
                errors="raise",
            ).astype(float)

        else:
            levels = _ordered_levels(
                X[column]
            )

            X[column] = pd.Categorical(
                X[column],
                categories=levels,
            )

            categorical_columns.append(
                column
            )

    # 범주형 변수는 기준 범주 하나를 제외하고 dummy 변수로 변환한다.
    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=True,
            dtype=float,
        )

    # --------------------------------------------------------
    # exposure에 해당하는 회귀계수 이름 확인
    # --------------------------------------------------------

    if exposure_type in {
        "binary",
        "continuous",
    }:

        exposure_terms = [
            exposure
        ]

    else:

        prefix = f"{exposure}_"

        exposure_terms = [
            column
            for column in X.columns
            if column.startswith(prefix)
        ]

    # 값이 하나뿐인 통제변수는 회귀에 아무 정보도 주지 않으므로 제거한다.
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:

        X = X.drop(
            columns=constant_columns
        )

        exposure_terms = [
            term
            for term in exposure_terms
            if term in X.columns
        ]

    if not exposure_terms:
        raise AssociationError(
            f"관심 변수 '{exposure}'의 회귀계수를 "
            "생성할 수 없습니다."
        )

    return (
        X.astype(float),
        exposure_terms,
        exposure_metadata,
    )


# ============================================================
# 조정 후 Logistic Regression
# ============================================================

def _fit_logistic_association(
    df: pd.DataFrame,
    request: AnalysisRequest,
    exposure_type: VariableType,
) -> dict:
    """통제변수를 포함한 이항 Logistic Regression을 수행한다."""

    X, exposure_terms, exposure_metadata = (
        _build_design_matrix(
            df,
            request,
            exposure_type,
        )
    )

    y = (
        df[request.target]
        .astype(int)
    )

    # 절편 추가
    X = sm.add_constant(
        X,
        has_constant="add",
    )

    # 정확한 선형 종속이 있으면 Logit 추정이 불안정하므로
    # statsmodels 내부 오류가 발생하기 전에 명확한 메시지를 제공한다.
    matrix_rank = np.linalg.matrix_rank(
        X.to_numpy(
            dtype=float
        )
    )

    if matrix_rank < X.shape[1]:
        raise AssociationError(
            "설명변수 사이에 완전한 선형 종속성이 있습니다. "
            "서로 중복되거나 동일 정보를 가진 통제변수를 "
            "제거한 뒤 다시 분석하세요."
        )

    try:
        model = sm.Logit(
            y,
            X,
        )

        fitted = model.fit(
            disp=False,
            maxiter=200,
        )

    except PerfectSeparationError as exc:
        raise AssociationError(
            "일부 변수 조합이 고소득 여부를 완전히 분리하여 "
            "Logistic Regression을 추정할 수 없습니다."
        ) from exc

    except np.linalg.LinAlgError as exc:
        raise AssociationError(
            "Logistic Regression 행렬 계산에 실패했습니다. "
            "중복되거나 지나치게 유사한 통제변수를 확인하세요."
        ) from exc

    except (TypeError, ValueError) as exc:
        raise AssociationError(
            f"Logistic Regression 적합에 실패했습니다: {exc}"
        ) from exc

    converged = bool(
        fitted.mle_retvals.get(
            "converged",
            True,
        )
    )

    if not converged:
        raise AssociationError(
            "Logistic Regression이 수렴하지 않았습니다. "
            "변수 조합이나 희소한 범주를 확인하세요."
        )

    confidence_interval = (
        fitted.conf_int(
            alpha=0.05
        )
    )

    # --------------------------------------------------------
    # 전체 회귀계수 결과
    # --------------------------------------------------------

    result_rows = []

    for term in fitted.params.index:

        if term == "const":
            continue

        coefficient = float(
            fitted.params[term]
        )

        ci_low = float(
            confidence_interval.loc[
                term,
                0,
            ]
        )

        ci_high = float(
            confidence_interval.loc[
                term,
                1,
            ]
        )

        result_rows.append(
            {
                "term": term,
                "role": (
                    "exposure"
                    if term in exposure_terms
                    else "control"
                ),
                "coefficient": coefficient,
                "odds_ratio": float(
                    np.exp(coefficient)
                ),
                "p_value": float(
                    fitted.pvalues[term]
                ),
                "ci_95_low": float(
                    np.exp(ci_low)
                ),
                "ci_95_high": float(
                    np.exp(ci_high)
                ),
            }
        )

    exposure_results = [
        row
        for row in result_rows
        if row["role"] == "exposure"
    ]

    return {
        "method": "binary_logistic_regression",
        "converged": converged,
        "pseudo_r_squared": float(
            fitted.prsquared
        ),
        "aic": float(
            fitted.aic
        ),
        "bic": float(
            fitted.bic
        ),
        "exposure_metadata": (
            exposure_metadata
        ),
        "exposure_effects": (
            exposure_results
        ),
        "all_terms": result_rows,
    }


# ============================================================
# 최종 연관성 분석 진입점
# ============================================================

def analyze_association(
    df: pd.DataFrame,
    request: AnalysisRequest,
) -> dict:
    """사용자의 분석 요청을 받아 조정 전·후 연관성 분석을 한 번에 수행한다.

    실행 흐름:
        1. 입력 요청 검증
        2. 필요한 열만 선택하고 결측 제거
        3. target의 0/1 구조 검증
        4. exposure 유형 자동 판별
        5. 조정 전 분석
        6. controls를 포함한 Logistic Regression
        7. 웹/API에서 사용하기 쉬운 dict로 반환

    Returns:
        분석 요청, 표본 수, 변수 유형,
        조정 전 결과와 Logistic Regression 결과를 담은 dict.
    """

    _validate_request(
        df,
        request,
    )

    required_columns = [
        request.target,
        request.exposure,
        *request.controls,
    ]

    # 5번에서 data.py에 추가한 함수.
    # 전체 데이터가 아니라 이번 분석에서 실제 사용할 변수에 대해서만
    # 결측 행을 제거한다.
    analysis_df = prepare_analysis_data(
        df,
        required_columns,
    )

    if analysis_df.empty:
        raise AssociationError(
            "결측값을 제외한 뒤 분석 가능한 표본이 없습니다."
        )

    _validate_binary_target(
        analysis_df[
            request.target
        ]
    )

    exposure_type = (
        infer_variable_type(
            analysis_df[
                request.exposure
            ]
        )
    )

    unadjusted = (
        _unadjusted_summary(
            analysis_df,
            request,
            exposure_type,
        )
    )

    adjusted = (
        _fit_logistic_association(
            analysis_df,
            request,
            exposure_type,
        )
    )

    return {
        "request": {
            "target": request.target,
            "exposure": request.exposure,
            "controls": list(
                request.controls
            ),
        },

        "analysis": {
            "exposure_type": exposure_type,
            "sample_size": int(
                len(analysis_df)
            ),
            "unadjusted": unadjusted,
            "adjusted": adjusted,
        },

        "interpretation_note": (
            "통제변수를 포함한 결과는 관측된 변수들을 조정한 "
            "조건부 연관성을 의미하며 확정적인 인과효과를 의미하지 않는다."
        ),
    }