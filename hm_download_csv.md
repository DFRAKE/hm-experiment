# 명세서: H&M 데이터셋 CSV 3개만 받기

> Claude Code 작업 지시용 · 2026-11-24 · 김승진
> 다음 단계 명세서: `hm_list_vs_swipe_experiment.md`

---

## 목표

Kaggle 대회 **H&M Personalized Fashion Recommendations**에서 **CSV 3개만** 받아 `~/hm-experiment/data/raw/`에 둔다.
**이미지 폴더는 받지 않는다** (전체 zip은 약 30GB).

| 파일 | 예상 크기 | 필수 |
| --- | --- | --- |
| `transactions_train.csv` | 약 3.5GB | ✅ |
| `articles.csv` | 수십 MB | ✅ |
| `customers.csv` | 수백 MB | ✅ |

---

## 지켜야 할 것

- 전체 대회 다운로드(`competition_download`에 `path` 없이 호출, 또는 `kaggle competitions download -c ...`에 `-f` 없이 호출)는 **절대 하지 않는다.**
- `~/.kaggle/kaggle.json` 내용이나 API 키를 **출력하거나 로그에 남기지 않는다.**
- 인증 정보가 없으면 **멈추고 사용자에게 아래 "인증 안내"를 보여 준다.** 직접 계정을 만들거나 키를 추측하지 않는다.
- 403 에러(대회 규칙 미동의)가 나면 멈추고 사용자에게 알린다.
- 기존 `data/raw/`에 같은 파일이 있고 크기가 정상이면 다시 받지 않는다.

---

## 작업 순서

### 1. 폴더 준비

```bash
mkdir -p ~/hm-experiment/data/raw
cd ~/hm-experiment
```

`~/hm-experiment/.gitignore`가 없으면 만들고, 아래 줄이 없으면 추가한다.

```
data/
.kaggle/
kaggle.json
*.zip
```

### 2. 디스크 공간 확인

`df -h ~` 로 여유 공간이 **10GB 이상**인지 확인한다. 부족하면 멈추고 알린다.

### 3. 인증 확인

다음 중 하나라도 있으면 통과:

- `~/.kaggle/kaggle.json` 파일 (권한 600으로 맞춘다: `chmod 600 ~/.kaggle/kaggle.json`)
- 환경변수 `KAGGLE_USERNAME`, `KAGGLE_KEY`

없으면 멈추고 "인증 안내"를 출력한다.

### 4. 패키지 설치

```bash
python3 -m pip install --upgrade kagglehub
```

가상환경이 있으면 그 안에 설치한다. 없으면 `python3 -m venv .venv && source .venv/bin/activate` 후 설치해도 된다.

### 5. 다운로드 스크립트

`~/hm-experiment/scripts/download_hm.py` 로 저장하고 실행한다.

```python
import os, shutil, zipfile, sys
import kagglehub

COMP = "h-and-m-personalized-fashion-recommendations"
OUT = os.path.expanduser("~/hm-experiment/data/raw")
FILES = ["articles.csv", "customers.csv", "transactions_train.csv"]
MIN_BYTES = {"articles.csv": 10_000_000, "customers.csv": 100_000_000, "transactions_train.csv": 3_000_000_000}

os.makedirs(OUT, exist_ok=True)
for f in FILES:
    dst = os.path.join(OUT, f)
    if os.path.exists(dst) and os.path.getsize(dst) >= MIN_BYTES[f]:
        print(f"skip {f} (already exists)")
        continue
    p = kagglehub.competition_download(COMP, path=f)   # 반드시 path 지정
    if str(p).endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            z.extract(f, OUT)
    else:
        shutil.copy(p, dst)
    print(f"done {f}")

for f in FILES:
    size = os.path.getsize(os.path.join(OUT, f))
    ok = size >= MIN_BYTES[f]
    print(f"{'OK ' if ok else 'BAD'} {f} {size/1e6:,.1f} MB")
    if not ok:
        sys.exit(1)
```

- 다운로드가 끊기면 같은 스크립트를 다시 실행한다 (이미 받은 파일은 건너뜀).
- kagglehub 캐시(`~/.cache/kagglehub/`)에 남은 사본은 복사가 끝난 뒤 **사용자에게 물어보고** 지운다.

### 6. 내용 확인

`transactions_train.csv`는 크므로 **청크로 읽는다.**

```python
import pandas as pd
raw = "~/hm-experiment/data/raw/"
a = pd.read_csv(raw + "articles.csv", dtype={"article_id": str})
c = pd.read_csv(raw + "customers.csv")
rows, dmin, dmax = 0, None, None
for ch in pd.read_csv(raw + "transactions_train.csv", usecols=["t_dat"], chunksize=2_000_000):
    rows += len(ch)
    lo, hi = ch.t_dat.min(), ch.t_dat.max()
    dmin = lo if dmin is None else min(dmin, lo)
    dmax = hi if dmax is None else max(dmax, hi)
print("articles", a.shape, "customers", c.shape)
print("transactions rows", rows, "range", dmin, "~", dmax)
print("articles columns", list(a.columns))
```

**기대값** (공개 자료 기준, 다르면 보고만 하고 멈추지 않는다):

| 항목 | 기대값 |
| --- | --- |
| 구매 기록 행 수 | 31,788,324 |
| 기간 | 2018-09-20 ~ 2020-09-22 |
| 상품 수 (`articles` 행) | 105,542 |
| 고객 수 (`customers` 행) | 1,371,980 |

---

## 인증 안내 (멈출 때 사용자에게 보여 줄 문구)

```
Kaggle 인증 정보가 없습니다.
1) kaggle.com → 오른쪽 위 프로필 → Settings → API → "Create New Token"
2) 받은 kaggle.json 을 ~/.kaggle/ 에 넣어 주세요:
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
3) 대회 페이지 Rules 탭에서 동의했는지 확인해 주세요:
   https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules
끝나면 "다시 해" 라고 말해 주세요.
```

---

## 완료 기준

- [ ] `data/raw/`에 CSV 3개, 크기 기준 통과
- [ ] 6번 확인 결과를 표로 보고 (행 수, 기간, 칼럼 목록)
- [ ] 이미지 폴더나 전체 zip을 받지 않았음
- [ ] `.gitignore`에 `data/` 포함
- [ ] 비밀 정보가 화면이나 파일에 출력되지 않았음

## 보고 형식

```
✅ 다운로드 완료
| 파일 | 크기 |
| 구매 기록 | 행 수 / 기간 |
| 상품 | 행 수 / 칼럼 |
| 고객 | 행 수 |
다음 단계: hm_list_vs_swipe_experiment.md 3장(데이터 준비)
```
