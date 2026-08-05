# ML 모델링 파트 요약

> `src/modeling.py` 담당(원광식). 코드를 안 읽어도 결과를 파악할 수 있게 정리한 문서.
> 모델 채택 과정과 실험 근거는 [`docs/MODEL_SELECTION_LOG.md`](./MODEL_SELECTION_LOG.md) 참고.

## 최종 모델

**HistGradientBoosting** (Logistic Regression, Random Forest와 실제 하이퍼파라미터 탐색으로 비교 후 채택)

| 지표 | 값 |
|---|---|
| Accuracy | 0.834 |
| Precision | 0.620 |
| Recall | 0.861 |
| F1 | 0.721 |
| ROC-AUC | 0.925 |

테스트셋 6,028행 (전체 30,139행 중 20%, 학습에 전혀 쓰이지 않은 held-out).

## 피처 중요도 (Permutation Importance)

> **정민규(통계·PSM) 참고용** — PSM 스토리를 정할 때 이 결과와 대조해보면 좋을 것 같습니다.

| 순위 | 피처 | importance |
|---|---|---|
| 1 | marital-status | 0.084 |
| 2 | capital-gain | 0.062 |
| 3 | age | 0.039 |
| 4 | capital-loss | 0.015 |
| 5 | occupation | 0.015 |
| **6** | **college_degree** | **0.015** |
| 7 | hours-per-week | 0.012 |
| **8** | **education** | **0.009** |
| 9 | relationship | 0.005 |
| 10 | workclass | 0.004 |
| 11 | sex | 0.003 |
| 12 | native-country | 0.002 |
| 13 | race | 0.0004 |

**학력 관련 변수(`college_degree`, `education`)가 13개 중 6·8위로, 상위 변수가 아닙니다.** marital-status(혼인상태)가 압도적 1위예요. 예측 기여도가 높다는 것과 인과관계가 있다는 것은 다른 이야기라 이 결과 자체가 PSM 결론을 좌우하진 않지만, report.md 스토리텔링에 참고할 만합니다.

## 집단별 진단 (Recall / False Negative Rate)

민감 변수(성별·인종)별로 모델이 실제 고소득자를 놓치는 비율에 차이가 있는지 확인한 결과입니다. `reliable=False`는 양성 표본이 30개 미만이라 추정이 불안정한 집단입니다 — 참고만 하고 결론의 근거로는 쓰지 않는 게 좋습니다.

| 집단 | n | 양성 표본 | Recall | reliable |
|---|---|---|---|---|
| sex=Female | 1,956 | 231 | 0.749 | True |
| sex=Male | 4,072 | 1,270 | 0.882 | True |
| race=White | 5,150 | 1,368 | 0.865 | True |
| race=Black | 594 | 82 | 0.841 | True |
| race=Asian-Pac-Islander | 174 | 41 | 0.854 | True |
| race=Amer-Indian-Eskimo | 63 | 6 | 0.667 | **False** |
| race=Other | 47 | 4 | 0.500 | **False** |

## 산출물 위치

- `outputs/models/income_pipeline.joblib` — 학습된 파이프라인
- `outputs/tables/model_metrics.json` — 위 성능 지표
- `outputs/tables/model_feature_importance.csv` — 전체 피처 중요도
- `outputs/tables/model_fairness_by_group.csv` — 전체 집단별 진단
- `outputs/tables/model_card.json` — 재현용 메타데이터(하이퍼파라미터, 학습 환경, 실행 시각)
- `outputs/tables/model_predictions.csv` — 테스트셋 개별 예측값(`row_id`,`y_test`,`y_pred`,`y_proba`). 시각화 파트(`model_visualization.py`)의 컨퓨전매트릭스·ROC 커브 입력

## 알려진 이슈 / 남은 작업

- `src/data.py` 결함 3건 미해결 — [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) 참고, 고쳐지면 `main.py`(고동민 담당) 연결 후 재검증 필요
- 하이퍼파라미터 탐색은 `n_iter=20`으로 제한 — 더 넓게 탐색해볼 여지 있음 (`MODEL_SELECTION_LOG.md` 참고)
