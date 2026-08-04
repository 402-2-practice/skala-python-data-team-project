# 데이터 파이프라인 결함 목록

> 작성: 원광식(ML 모델링) 담당, `feat/modeling` 브랜치에서 `python main.py` 실행 중 발견.
> `src/data.py`, `src/config.py`는 윤찬웅 담당 파일이라 직접 수정하지 않고 여기에 정리만 함.
> `main.py` 연결은 고동민(테스트·통합·문서 QA) 담당이라 최종 반영은 그쪽에서.
> 각 항목에 코멘트/체크 남겨주세요.

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

**상태**: 미해결

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

**상태**: 미해결 (함수명 확정 필요 — A 코멘트 요청)

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

**상태**: 미해결

---

## 확인된 것 (참고용 — 문제 없음)

- `clean_data()`가 만드는 `high_income`, `college_degree` 컬럼은 그대로 유지되어 있어서 `src/modeling.py`, `src/statistics.py`는 위 3가지가 고쳐지면 별도 수정 없이 바로 연결 가능함.
