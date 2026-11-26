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
        print(f"skip {f} (already exists)", flush=True)
        continue
    print(f"downloading {f} ...", flush=True)
    p = kagglehub.competition_download(COMP, path=f)   # 반드시 path 지정
    if str(p).endswith(".zip"):
        with zipfile.ZipFile(p) as z:
            z.extract(f, OUT)
    else:
        shutil.copy(p, dst)
    print(f"done {f}", flush=True)

for f in FILES:
    size = os.path.getsize(os.path.join(OUT, f))
    ok = size >= MIN_BYTES[f]
    print(f"{'OK ' if ok else 'BAD'} {f} {size/1e6:,.1f} MB", flush=True)
    if not ok:
        sys.exit(1)
