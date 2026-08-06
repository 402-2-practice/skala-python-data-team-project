# 데이터 파이프라인 결함 목록 (해결됨)

> 작성: 원광식(ML 모델링) 담당, `feat/modeling` 브랜치에서 `python main.py` 실행 중 발견.
> `src/data.py`, `src/config.py`는 윤찬웅 담당 파일이라 직접 수정하지 않고 여기에 정리만 함.
> `main.py` 연결은 고동민(테스트·통합·문서 QA) 담당이라 최종 반영은 그쪽에서.
>
> **2026-08-06 업데이트**: 아래 3건 모두 `feat/data`(PR #12), `feat/wire-model-visualization`(PR #11)에서
> 해결되어 `main`에 merge됨. `python main.py` 전체 파이프라인이 처음부터 끝까지 에러 없이 실행되고,
> pandas/polars 행 수도 32,561행으로 일치함을 재확인했다. 과거 기록 보존을 위해 원문은 남겨둔다.

## 1. `src/data.py` import 경로 오류 (즉시 크래시)

**현상**: `python main.py` 실행 시 바로 아래 에러 발생.

```
ModuleNotFoundError: No module named 'config'
```

**원인**: `src/data.py` 10번째 줄

```python
from config import (
    ADULT_COLUMNS,
    COLLEGE_DEGREES,
    PROCESSED_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
```

`config.py`는 `src/config.py`에 있으므로 `from src.config import (...)`이어야 함.

**제안**: `from config` → `from src.config`로 변경.

**상태**: 해결됨 — 현재 `src/data.py`는 `from src.config import (...)`로 수정되어 있음.

---

## 2. `main.py`가 기대하는 `load_and_clean` 함수가 사라짐

**현상**: 1번을 고쳐도 아래 에러로 이어서 크래시.

```
ImportError: cannot import name 'load_and_clean' from 'src.data'
```

**원인**: `main.py`는 아래처럼 호출하는데

```python
from src.data import load_and_clean
...
df, load_comparison = load_and_clean(args.data)
```

새 `src/data.py`에는 `load_and_clean`이 없고, 이름과 반환 형태가 다른 `run_data_pipeline(path) -> (cleaned_df, comparison)`으로 바뀜 (`comparison`에 `best_tool`, `pandas_columns`, `polars_columns`가 추가됨. `pandas_rows`, `polars_rows`, `pandas_seconds`, `polars_seconds`는 기존과 동일하게 유지되어 있어서 `eda.py`, `report.py`와는 호환 가능해 보임).

**제안**: 아래 둘 중 하나로 확정 필요
- (a) `run_data_pipeline`을 유지하고 `main.py`에서 그에 맞게 호출부만 수정 (main.py는 고동민 담당이라 윤찬웅·고동민 협의 후 진행)
- (b) 함수명을 다시 `load_and_clean`으로 되돌려서 기존 계약 유지

**상태**: 해결됨 — (b)로 확정. `src/data.py`에 `load_and_clean(path)`이 다시 정의되어 있고
`main.py`도 그대로 `from src.data import load_and_clean`으로 호출함.

---

## 3. `load_data_polars`가 헤더 있는 파일을 `has_header=False`로 읽음 (조용히 잘못된 결과)

**현상**: 직접 재현해서 확인함.

```python
import polars as pl
df = pl.read_csv("data/raw/adult.csv", has_header=False, new_columns=ADULT_COLUMNS, null_values=" ?")
print(df.shape)   # (32562, 15)  <- pandas는 (32561, 15)
```

- 실제 `data/raw/adult.csv`는 첫 줄이 헤더(`age,workclass,fnlwgt,...`)인 파일인데, `has_header=False`로 읽어서 헤더 줄이 데이터 첫 행으로 들어감
- 그 여파로 `pandas_rows`(32,561) vs `polars_rows`(32,562) 행 수가 어긋남
- 모든 컬럼 dtype이 숫자가 아닌 문자열(String)로 깨짐 (헤더 문자열이 섞여 들어가서 타입 추론이 실패)

**영향**: `TEAM_WORKFLOW.md`의 완료 정의 중 "Pandas와 Polars 결과 행 수가 일치한다"를 위반함. 지금은 크래시가 안 나서 눈에 안 띄지만, EDA 요약에 잘못된 행 수/타입이 그대로 들어감.

**제안**: `load_data_polars`에서 `has_header=False`, `new_columns=ADULT_COLUMNS` 제거하고 기본 옵션(헤더 자동 인식)으로 읽기.

**상태**: 해결됨 — `src/config.py`의 `CSV_HAS_HEADER = True`를 읽어 pandas/polars 둘 다 헤더를 인식하도록 수정됨.
현재 `outputs/tables/data_engine_benchmark.json`에서 `pandas_rows == polars_rows == 32561`,
`results_match: true`로 재확인함.

---

## 확인된 것 (참고용 — 문제 없음)

- `clean_data()`가 만드는 `high_income`, `college_degree` 컬럼은 그대로 유지되어 있어서 `src/modeling.py`, `src/statistics.py`는 위 3가지가 고쳐지면 별도 수정 없이 바로 연결 가능함.
