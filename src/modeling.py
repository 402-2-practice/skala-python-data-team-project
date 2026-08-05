"""고소득 여부 예측용 sklearn Pipeline.

담당: 원광식 (ML 모델링)

모델과 하이퍼파라미터는 이론적 선택이 아니라
scripts/experiments/model_selection_experiment.py의 실제 실행 결과로
채택했다. 근거와 비교 대상(Logistic Regression, Random Forest)은
docs/MODEL_SELECTION_LOG.md에 기록되어 있다.

HistGradientBoosting 전처리는 원-핫 인코딩 대신 네이티브 범주형 처리
(categorical_features="from_dtype")를 쓴다. BEST_MODEL_PARAMS는 원래 원-핫 인코딩
기준으로 탐색됐지만, model_selection_experiment.py를 네이티브 범주형 기준으로 재실행해
같은 하이퍼파라미터가 그대로 유효함을 확인했다 (2026-08-05 재검증,
docs/MODEL_SELECTION_LOG.md "재검증" 절 참고).
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_DIR, RANDOM_STATE, TABLE_DIR

logger = logging.getLogger(__name__)

TARGET_COLUMN = "high_income"
# income은 정답 원문, education-num은 education과 중복, fnlwgt는 표본 가중치이므로 제외한다.
EXCLUDED_COLUMNS = ["income", "high_income", "education-num", "fnlwgt"]
# 성별·인종처럼 특정 집단만 예측을 놓치고 있는지 확인하기 위해 지정한 민감 변수.
FAIRNESS_GROUP_COLUMNS = ["sex", "race"]
# test_size=0.2에서 stratify가 실패하지 않으려면 클래스별로 최소 몇 개는 있어야 하므로, 그 하한선을 미리 정해둔다.
MIN_SAMPLES_PER_CLASS = 10
# 양성 표본이 이보다 적으면 한두 명만 틀려도 recall이 크게 흔들리므로, 그 집단의 진단은 참고용으로만 본다.
MIN_RELIABLE_GROUP_POSITIVES = 30
# 신뢰 가능한(표본 충분한) 집단 중 recall이 이보다 낮으면 그 집단만 유독 놓치고 있다고 보고 경고한다.
FAIRNESS_MIN_GROUP_RECALL = 0.60
# 신뢰 가능한 집단 간 recall 최대-최소 격차가 이보다 크면 형평성 문제로 보고 경고한다.
FAIRNESS_MAX_RECALL_GAP = 0.10

# HistGradientBoosting을 채택한 이유: docs/MODEL_SELECTION_LOG.md 실험에서 교차검증 ROC-AUC가
# 가장 높았고(0.9258) 탐색 시간도 더 짧았기 때문이다 (Logistic Regression·Random Forest 대비).
BEST_MODEL_PARAMS = {
    "learning_rate": 0.14447746112718687,
    "max_depth": 5,
    "max_iter": 154,
    "l2_regularization": 0.45606998421703593,
}


class ModelingError(RuntimeError):
    """데이터 인입 시점부터 학습·저장·추론까지, 이 모듈이 던지는 모든 오류의 기반 클래스."""


@dataclass
class ModelEvaluation:
    """학습·평가 결과를 담는 컨테이너. 디스크 저장 없이 순수하게 값만 갖고 있어서
    pytest에서 파일 I/O 없이 바로 검증할 수 있다 (tests/ 담당자용).
    """

    pipeline: Pipeline
    metrics: dict
    fairness: pd.DataFrame
    feature_importance: pd.DataFrame
    model_card: dict = field(default_factory=dict)


def _validate_input(df: pd.DataFrame) -> None:
    """main.py가 넘겨주는 df가 기대한 데이터 계약을 만족하는지 확인한다.

    src/data.py가 앞으로 바뀌어도 이 함수가 계약 위반을 여기서 바로
    잡아내야, 문제가 sklearn 내부 에러로 애매하게 터지는 걸 막을 수 있다.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ModelingError("train_income_model()은 pandas.DataFrame을 받아야 합니다.")
    if df.empty:
        raise ModelingError("모델 학습용 데이터프레임이 비어 있습니다.")
    if TARGET_COLUMN not in df.columns:
        raise ModelingError(
            f"타깃 컬럼 '{TARGET_COLUMN}'이 없습니다. "
            "src.data의 정제 함수(clean_data 등)를 거친 데이터인지 확인하세요."
        )

    _check_excluded_column_casing(df.columns)

    feature_columns = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    if not feature_columns:
        raise ModelingError(
            f"제외 컬럼({EXCLUDED_COLUMNS})을 뺀 뒤 남는 피처가 없습니다. 입력 df의 컬럼 구성을 확인하세요."
        )

    try:
        target = df[TARGET_COLUMN].astype(int)
    except (TypeError, ValueError) as exc:
        raise ModelingError(
            f"타깃 컬럼 '{TARGET_COLUMN}'을 정수로 변환할 수 없습니다 (예: 결측치나 0/1이 아닌 값 포함). "
            f"원본 예외: {exc}"
        ) from exc

    class_counts = target.value_counts()
    if class_counts.shape[0] < 2:
        raise ModelingError(
            f"타깃 '{TARGET_COLUMN}'에 클래스가 {class_counts.shape[0]}개뿐이라 분류 모델을 학습할 수 없습니다."
        )
    if class_counts.min() < MIN_SAMPLES_PER_CLASS:
        raise ModelingError(
            f"타깃 클래스 중 표본이 너무 적습니다 (최소 클래스 {class_counts.min()}개, "
            f"기준 {MIN_SAMPLES_PER_CLASS}개). train_test_split(stratify=...)이 실패할 수 있습니다."
        )


def _check_excluded_column_casing(columns: pd.Index) -> None:
    """스키마 변경으로 컬럼명 대소문자가 바뀌어 EXCLUDED_COLUMNS가 못 걸러내는 상황을 막는다.

    예: 업스트림에서 "income"이 "Income"으로 바뀌면 EXCLUDED_COLUMNS(소문자 고정)는
    이를 제외하지 못하고, 정답 원문이 그대로 피처로 들어가는 유출이 생긴다. 대소문자만
    다른 채로 남아 있는 컬럼을 발견하면 자동으로 걸러내지 않고 명시적으로 실패시켜서,
    "몰래 통과"가 아니라 스키마를 고치도록 강제한다.
    """
    excluded_lower = {name.lower() for name in EXCLUDED_COLUMNS}
    for column in columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if column.lower() in excluded_lower:
            raise ModelingError(
                f"컬럼 '{column}'이 제외 대상({EXCLUDED_COLUMNS})과 대소문자만 다른 이름으로 존재합니다. "
                "스키마 변경(대소문자 변경)으로 정답/유출 컬럼이 피처에 섞여 들어갈 수 있으니, "
                "EXCLUDED_COLUMNS를 갱신하거나 입력 데이터의 컬럼명을 표준화하세요."
            )


def _coerce_feature_dtypes(X: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    """범주형 컬럼을 pandas category dtype으로 맞춘다.

    HistGradientBoostingClassifier(categorical_features="from_dtype")는 입력 DataFrame에서
    dtype이 category인 컬럼을 범주형으로 자동 인식한다. pandas의 pd.NA는 sklearn이 못 알아듣고
    예외를 던지므로, 결측치는 먼저 np.nan으로 바꾼 뒤 category로 변환한다 (category dtype은
    NaN을 별도 레벨 없이 결측으로 허용하고, HGB는 결측을 자체적으로 분기 처리한다).
    """
    X = X.copy()
    for column in categorical_columns:
        X[column] = X[column].astype(object).where(X[column].notna(), np.nan).astype("category")
    return X


def _split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """df를 (피처 X, 타깃 y)로 나누고, 전처리 단계에서 쓸 수 있게 컬럼을
    수치형/범주형으로 분류해서 함께 반환한다.
    """
    feature_columns = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int)

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    X = _coerce_feature_dtypes(X, categorical_columns)

    return X, y, numeric_columns, categorical_columns


def _build_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    """전처리(수치형 결측치 처리) + 채택 모델(HistGradientBoosting)을 하나의
    sklearn Pipeline으로 묶는다.

    범주형 컬럼은 OneHotEncoder로 더미 변수화하지 않는다. HistGradientBoosting은
    categorical_features="from_dtype"로 pandas category dtype 컬럼을 직접 분기 기준으로
    쓸 수 있어서, 더미 변수 폭발(메모리)과 그로 인한 학습/추론 속도 저하를 피할 수 있다.
    결측치도 모델이 자체적으로 분기 처리하므로 범주형 imputer도 필요 없다.
    ColumnTransformer는 set_output(transform="pandas")로 category dtype을 그대로
    보존해서 넘겨야 모델이 이를 인식한다.

    예전엔 Logistic Regression도 후보였어서 수치형 컬럼에 StandardScaler를 같이 썼는데,
    지금은 HistGradientBoosting만 남았다. 이 모델은 트리 기반이라 분기 기준이 값의 순서만
    보고 스케일에는 영향받지 않으므로, 스케일링이 무의미해져 제거했다.
    """
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                "passthrough",
                categorical_columns,
            ),
        ]
    )
    preprocessing.set_output(transform="pandas")
    model = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        **BEST_MODEL_PARAMS,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def _fairness_by_group(
    df_test: pd.DataFrame, y_test: pd.Series, prediction: np.ndarray, group_columns: list[str]
) -> pd.DataFrame:
    """민감 변수 집단별 Recall/False Negative Rate을 비교한다.

    모델이 특정 집단에서 실제 고소득자를 유독 많이 놓치는지 확인하기 위한 진단용 지표다.
    `reliable`이 False인 행은 실제 양성 표본이 너무 적어(<MIN_RELIABLE_GROUP_POSITIVES)
    recall 추정이 불안정하다는 뜻이므로 참고용으로만 봐야 한다.
    """
    rows = []
    for column in group_columns:
        if column not in df_test.columns:
            continue
        for group_value, idx in df_test.groupby(column, observed=True).groups.items():
            group_y = y_test.loc[idx]
            group_pred = pd.Series(prediction, index=y_test.index).loc[idx]
            positives = int((group_y == 1).sum())
            if positives == 0:
                continue
            recall = recall_score(group_y, group_pred, zero_division=0)
            rows.append(
                {
                    "group_column": column,
                    "group_value": group_value,
                    "n": int(len(idx)),
                    "n_actual_positive": positives,
                    "recall": float(recall),
                    "false_negative_rate": float(1 - recall),
                    "reliable": positives >= MIN_RELIABLE_GROUP_POSITIVES,
                }
            )
    return pd.DataFrame(rows)


def _assess_fairness(fairness: pd.DataFrame) -> dict:
    """집단별 recall 격차/최저치가 기준을 넘는지 판정하고, 위반 시 경고 로그를 남긴다.

    표본이 부족해 신뢰할 수 없는(reliable=False) 집단은 판정에서 제외한다 — 그런 집단은
    recall 자체가 추정 오차가 커서, 기준 미달로 잡히더라도 모델 결함인지 표본 부족인지
    구분할 수 없기 때문이다.
    """
    if fairness.empty:
        return {"status": "skipped", "reason": "그룹별 표본 없음"}

    reliable = fairness[fairness["reliable"]]
    if reliable.empty:
        return {"status": "skipped", "reason": "신뢰 가능한(표본 충분한) 집단 없음"}

    violations = []
    for column, group in reliable.groupby("group_column"):
        min_recall_row = group.loc[group["recall"].idxmin()]
        max_recall_row = group.loc[group["recall"].idxmax()]
        min_recall = float(min_recall_row["recall"])
        max_recall = float(max_recall_row["recall"])
        recall_gap = max_recall - min_recall

        if min_recall < FAIRNESS_MIN_GROUP_RECALL:
            violations.append(
                {
                    "group_column": column,
                    "group_value": min_recall_row["group_value"],
                    "type": "min_recall",
                    "value": min_recall,
                    "threshold": FAIRNESS_MIN_GROUP_RECALL,
                }
            )
        if recall_gap > FAIRNESS_MAX_RECALL_GAP:
            # 격차 수치만 남기면 model_card만 보고는 어느 집단을 고쳐야 할지 알 수 없으므로,
            # 격차의 양 끝(recall이 가장 낮은/높은 집단)을 함께 남긴다.
            violations.append(
                {
                    "group_column": column,
                    "type": "recall_gap",
                    "value": recall_gap,
                    "threshold": FAIRNESS_MAX_RECALL_GAP,
                    "lowest_group_value": min_recall_row["group_value"],
                    "lowest_group_recall": min_recall,
                    "highest_group_value": max_recall_row["group_value"],
                    "highest_group_recall": max_recall,
                }
            )

    status = "fail" if violations else "pass"
    if violations:
        for violation in violations:
            if violation["type"] == "min_recall":
                logger.warning(
                    "[공정성 경고] %s=%s 집단 recall(%.3f)이 기준(%.3f) 미달입니다.",
                    violation["group_column"],
                    violation["group_value"],
                    violation["value"],
                    violation["threshold"],
                )
            else:
                logger.warning(
                    "[공정성 경고] %s 집단 간 recall 격차(%.3f)가 기준(%.3f)을 초과했습니다 "
                    "(최저 %s=%.3f vs 최고 %s=%.3f).",
                    violation["group_column"],
                    violation["value"],
                    violation["threshold"],
                    violation["lowest_group_value"],
                    violation["lowest_group_recall"],
                    violation["highest_group_value"],
                    violation["highest_group_recall"],
                )

    return {
        "status": status,
        "min_recall_threshold": FAIRNESS_MIN_GROUP_RECALL,
        "max_recall_gap_threshold": FAIRNESS_MAX_RECALL_GAP,
        "violations": violations,
    }


def _feature_importance(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 10
) -> pd.DataFrame:
    """테스트셋 기준 permutation importance를 계산한다.

    HistGradientBoosting은 RandomForest와 달리 `.feature_importances_`를
    제공하지 않아서 permutation importance로 대체한다. 원본 컬럼(예: education,
    college_degree) 단위로 나오기 때문에 원-핫 인코딩된 더미 변수 단위로
    쪼개지는 것보다 해석하기 쉽다 — "학력이 소득 예측에 얼마나 기여하는가"라는
    이 프로젝트의 핵심 질문과 바로 연결된다.
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return importance.reset_index(drop=True)


def _build_model_card(
    metrics: dict,
    train_rows: int,
    test_rows: int,
    feature_columns: list[str],
    fairness_assessment: dict,
) -> dict:
    """재현성을 위한 메타데이터. 같은 코드를 나중에 다시 돌렸을 때
    "그때 그 결과"가 어떤 환경·데이터 규모에서 나온 건지 추적하기 위함이다.

    fairness_assessment는 단순 진단표(model_fairness_by_group.csv)와 달리, 판정 결과
    (pass/fail)와 위반 내역을 담아 CI나 배포 게이트가 사람이 읽지 않고도 확인할 수 있게 한다.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": metrics["model_name"],
        "hyperparameters": BEST_MODEL_PARAMS,
        "categorical_encoding": "native_from_dtype",
        "random_state": RANDOM_STATE,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_columns": feature_columns,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "selection_reference": "docs/MODEL_SELECTION_LOG.md",
        "fairness_check": fairness_assessment,
    }


def evaluate_income_model(df: pd.DataFrame) -> ModelEvaluation:
    """학습·평가만 수행하고 디스크에는 아무것도 쓰지 않는다 (테스트용 진입점).

    train_income_model()이 이 함수를 감싸서 저장까지 하는 구조라, tests/에서는
    파일 시스템 없이 이 함수만 호출해서 결과를 검증할 수 있다.
    """
    # 1. 데이터 계약 검증 (타깃 컬럼, 클래스 수/균형, 제외 컬럼 대소문자 등)
    _validate_input(df)

    # 2. 피처(X)/타깃(y) 분리 + 수치형/범주형 컬럼 구분
    X, y, numeric_columns, categorical_columns = _split_features(df)

    # 3. 학습 80% / 테스트(held-out) 20% 분리 — 테스트셋은 최종 평가에만 쓰고
    #    이후 어떤 학습·튜닝 과정에도 노출시키지 않는다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 4. 전처리+모델 파이프라인 구성 후 학습 데이터로 학습
    #    (수치형 median imputer는 이 시점에 X_train에만 fit되므로 테스트셋 유출 없음)
    pipeline = _build_pipeline(numeric_columns, categorical_columns)
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        raise ModelingError(f"모델 학습에 실패했습니다: {exc}") from exc

    # 5. 테스트셋으로 1회 예측 후 평가 지표 계산
    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": "hist_gradient_boosting",
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }

    # 6. 민감 변수(성별·인종)별 Recall/False Negative Rate 진단 + 기준 위반 판정
    fairness = _fairness_by_group(X_test, y_test, prediction, FAIRNESS_GROUP_COLUMNS)
    fairness_assessment = _assess_fairness(fairness)

    # 7. 피처 중요도 (permutation importance) — 학력 관련 변수가 실제로
    #    예측에 얼마나 기여하는지, 팀의 인과추론 분석과 대조해볼 근거 자료
    feature_importance = _feature_importance(pipeline, X_test, y_test)

    model_card = _build_model_card(
        metrics, len(X_train), len(X_test), list(X.columns), fairness_assessment
    )

    return ModelEvaluation(
        pipeline=pipeline,
        metrics=metrics,
        fairness=fairness,
        feature_importance=feature_importance,
        model_card=model_card,
    )


def _save_outputs(evaluation: ModelEvaluation) -> None:
    """학습된 파이프라인(joblib)과 평가 지표·진단 결과(json/csv)를 디스크에 남긴다.

    model_metrics.json은 src/report.py가 report.md를 만들 때 그대로 읽으므로
    파일명·경로·키 이름을 바꾸면 안 된다.
    """
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(evaluation.pipeline, MODEL_DIR / "income_pipeline.joblib")
        (TABLE_DIR / "model_metrics.json").write_text(
            json.dumps(evaluation.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (TABLE_DIR / "model_card.json").write_text(
            json.dumps(evaluation.model_card, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not evaluation.fairness.empty:
            evaluation.fairness.to_csv(TABLE_DIR / "model_fairness_by_group.csv", index=False)
        evaluation.feature_importance.to_csv(
            TABLE_DIR / "model_feature_importance.csv", index=False
        )
    except (OSError, TypeError) as exc:
        raise ModelingError(f"모델/지표 저장에 실패했습니다: {exc}") from exc


def train_income_model(df: pd.DataFrame) -> dict:
    """main.py 진입점. 반환값(dict)의 키(accuracy/precision/recall/f1/roc_auc)는
    src/report.py가 model_metrics.json에서 그대로 읽으므로 이름을 바꾸지 않는다.
    """
    evaluation = evaluate_income_model(df)
    _save_outputs(evaluation)

    print("\n[ML] 채택 모델: hist_gradient_boosting (근거: docs/MODEL_SELECTION_LOG.md)")
    print("\n[ML] 테스트 성능")
    print(pd.Series(evaluation.metrics).to_string())
    if not evaluation.fairness.empty:
        print("\n[ML] 집단별 Recall / False Negative Rate (reliable=False는 표본 부족으로 참고용)")
        print(evaluation.fairness.to_string(index=False))
    fairness_status = evaluation.model_card.get("fairness_check", {}).get("status")
    print(f"\n[ML] 공정성 판정: {fairness_status}")
    print("\n[ML] 피처 중요도 (permutation importance, 상위 5개)")
    print(evaluation.feature_importance.head(5).to_string(index=False))

    return evaluation.metrics


def predict_income(df: pd.DataFrame) -> pd.DataFrame:
    """저장된 income_pipeline.joblib으로 새 데이터에 대한 예측값/확률을 반환한다.

    train_income_model()과 별도 프로세스(배포 서버, 배치 작업 등)에서 호출되는
    표준 추론 진입점이다. 학습 때와 동일하게 EXCLUDED_COLUMNS를 제외한 나머지
    컬럼을 피처로 쓰고, 범주형 컬럼은 같은 방식(category dtype)으로 맞춘다.
    입력 df에 타깃(high_income)이나 income 컬럼이 있어도 없어도 동작한다.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ModelingError("predict_income()은 pandas.DataFrame을 받아야 합니다.")
    if df.empty:
        raise ModelingError("추론할 데이터프레임이 비어 있습니다.")

    model_path = MODEL_DIR / "income_pipeline.joblib"
    if not model_path.exists():
        raise ModelingError(
            f"학습된 파이프라인을 찾을 수 없습니다: {model_path}. "
            "train_income_model()을 먼저 실행해 모델을 저장하세요."
        )

    feature_columns = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    if not feature_columns:
        raise ModelingError(
            f"제외 컬럼({EXCLUDED_COLUMNS})을 뺀 뒤 남는 피처가 없습니다. 입력 df의 컬럼 구성을 확인하세요."
        )

    X = df[feature_columns].copy()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    X = _coerce_feature_dtypes(X, categorical_columns)

    try:
        pipeline: Pipeline = joblib.load(model_path)
    except Exception as exc:
        raise ModelingError(f"파이프라인을 불러오는 데 실패했습니다: {exc}") from exc

    try:
        prediction = pipeline.predict(X)
        probability = pipeline.predict_proba(X)[:, 1]
    except Exception as exc:
        raise ModelingError(
            f"추론에 실패했습니다 (입력 컬럼이 학습 시점과 다를 수 있습니다): {exc}"
        ) from exc

    return pd.DataFrame(
        {"prediction": prediction.astype(int), "probability": probability}, index=df.index
    )
