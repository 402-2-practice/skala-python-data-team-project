# 모델 채택 기록

> `src/modeling.py`에 들어간 모델·하이퍼파라미터는 이론적으로 고른 게 아니라
> `scripts/experiments/model_selection_experiment.py`를 실제로 실행한 결과로 정했다.
> 재현하려면 그 스크립트를 그대로 실행하면 된다.

## 실행 조건

- 데이터: `data/raw/adult.csv` 전체 (정제 후 30,139행)
- 학습/테스트 분리: 80/20, stratify, `random_state=42`
- 후보 3개, 각각 `RandomizedSearchCV`(`n_iter=20`, `cv=5`, `StratifiedKFold`, `scoring="roc_auc"`)로 하이퍼파라미터 탐색
- 모델 선택 기준: 학습셋 내부 5-fold 교차검증 ROC-AUC 평균 (테스트셋은 최종 1회 평가에만 사용, 모델 선택에는 관여하지 않음)

## 탐색 공간

| 모델 | 탐색한 하이퍼파라미터 |
|---|---|
| Logistic Regression | `C` (log-uniform 1e-3~1e2) |
| Random Forest | `n_estimators`(100~600), `max_depth`(None/5/10/15/20/30), `min_samples_leaf`(1~10) |
| HistGradientBoosting | `learning_rate`(log-uniform 1e-2~3e-1), `max_depth`(None/3/5/8/12), `max_iter`(100~400), `l2_regularization`(0~1) |

## 재검증 (2026-08-05, 네이티브 범주형 처리)

src/modeling.py가 HistGradientBoosting 전처리를 OneHotEncoder에서 네이티브 범주형
처리(`categorical_features="from_dtype"`)로 바꾸면서, 아래 원본 실험(OneHotEncoder
기준)이 채택한 하이퍼파라미터가 그대로 유효한지 재실행해서 확인했다.
Logistic Regression·Random Forest는 카테고리 dtype을 분기 기준으로 못 쓰므로
원-핫 인코딩을 그대로 쓰고, HistGradientBoosting만 네이티브 범주형으로 바꿔 비교했다.

| 모델 | CV ROC-AUC | 탐색 시간 |
|---|---|---|
| **HistGradientBoosting (네이티브 범주형)** | **0.9254** | 7.6초 |
| Random Forest (원-핫) | 0.9134 | 40.3초 |
| Logistic Regression (원-핫) | 0.9053 | 6.0초 |

채택된 하이퍼파라미터는 원본 실험과 **동일**했다:

```python
{
    "learning_rate": 0.14447746112718687,
    "max_depth": 5,
    "max_iter": 154,
    "l2_regularization": 0.45606998421703593,
}
```

Held-out 테스트셋: Accuracy 0.8341 / Precision 0.6214 / Recall 0.8541 / F1 0.7194 /
ROC-AUC 0.9255 (예측 시간 1,000행당 0.005초) — 원본 실험(ROC-AUC 0.9246)과 오차
범위 내로 동일하거나 소폭 개선. **결론: 네이티브 범주형 처리로 바꿔도 BEST_MODEL_PARAMS를
다시 튜닝할 필요는 없다.** 탐색 시간도 원-핫 대비 짧아졌다(36.2초 → 7.6초) —
더미 변수가 없어 피처 수가 줄어든 결과로 보인다.

## 결과 (2026-08-04 실행, OneHotEncoder 기준 — 원본 실험)

| 모델 | CV ROC-AUC | 탐색 시간 |
|---|---|---|
| **HistGradientBoosting** | **0.9258** | 36.2초 |
| Random Forest | 0.9134 | 50.0초 |
| Logistic Regression | 0.9053 | 7.6초 |

**채택: HistGradientBoosting** — 정확도(CV ROC-AUC)가 가장 높으면서 탐색 시간도 Random Forest보다 짧았다. 정확도와 효율성 두 기준 모두에서 우위였기 때문에 이견 없이 채택.

채택된 하이퍼파라미터:

```python
{
    "learning_rate": 0.14447746112718687,
    "max_depth": 5,
    "max_iter": 154,
    "l2_regularization": 0.45606998421703593,
}
```

## Held-out 테스트셋 최종 성능

| 지표 | 값 |
|---|---|
| Accuracy | 0.8339 |
| Precision | 0.6198 |
| Recall | 0.8614 |
| F1 | 0.7209 |
| ROC-AUC | 0.9246 |
| 예측 시간 (1,000행당) | 0.007초 |

Random Forest(50.0초 탐색) 대비 HistGradientBoosting(36.2초)이 탐색도 더 빠르고, 예측도 1,000행당 0.007초로 실시간 서비스에도 부담 없는 수준. Logistic Regression은 압도적으로 빠르지만(7.6초) ROC-AUC가 가장 낮아 최종 후보에서 제외.

## 남겨진 개선 여지 (다음 라운드에 시도해볼만한 것)

- `n_iter`를 20 → 50 이상으로 늘려서 더 넓게 탐색 (현재는 시간 예산상 20으로 제한)
- HistGradientBoosting의 `early_stopping`, `max_leaf_nodes` 파라미터는 이번 탐색 공간에 포함 안 함
- class_weight="balanced" 고정 — threshold 조정(예: `predict_proba` 기반 커스텀 임계값)으로 precision/recall 트레이드오프를 조정하는 실험은 아직 안 함
