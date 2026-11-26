"""상품 풀(article_pool.parquet)에 있는 상품의 이미지만 한 장씩 받는다.

- 대회 전체 다운로드는 하지 않는다 (항상 path 지정).
- 저장: data/raw/images/<앞3자리>/<article_id>.jpg  (대회 폴더 구조 그대로)
- 이미 받은 파일(크기 > 0)은 건너뛴다 -> 끊겨도 다시 실행하면 이어서 받음
- 대회에 이미지가 없는 상품은 data/raw/images_missing.txt 에 기록

실행: python scripts/download_images.py [--limit 20] [--workers 8]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMP = "h-and-m-personalized-fashion-recommendations"
OUT = ROOT / "data" / "raw"
MISSING = OUT / "images_missing.txt"


def rel_path(aid: str) -> str:
    return f"images/{aid[:3]}/{aid}.jpg"


CANARY = "0108775015"   # 대회에 이미지가 있는 것으로 확인된 상품. 이것도 404면 요청 제한으로 본다.


def _get(aid: str, out: Path) -> str:
    import kagglehub
    try:
        kagglehub.competition_download(COMP, path=rel_path(aid), output_dir=str(out))
        return "ok"
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "404" in msg:
            return "404"
        if "403" in msg:
            return "forbidden"
        return "error"


def canary_ok() -> bool:
    """Kaggle은 요청 제한 때 404를 돌려준다. 있는 이미지를 받아 보고 되는지 확인."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        return _get(CANARY, Path(d)) == "ok"


def fetch(aid: str, delay: float) -> tuple[str, str]:
    dst = OUT / rel_path(aid)
    if dst.exists() and dst.stat().st_size > 0:
        return aid, "skip"
    if dst.exists():
        dst.unlink()
    time.sleep(delay)
    st = _get(aid, OUT)
    if st == "404":
        # 진짜 없는 이미지인지, 요청 제한인지 카나리로 구분
        return aid, ("missing" if canary_ok() else "throttled")
    return aid, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.5, help="요청마다 쉬는 시간(초)")
    args = ap.parse_args()
    logging.getLogger("kagglehub").setLevel(logging.WARNING)
    if not canary_ok():
        print("카나리 이미지도 받지 못함: 요청 제한 중이거나 인증 문제. 나중에 다시 실행하세요.", flush=True)
        sys.exit(3)

    ids = pd.read_parquet(ROOT / "data" / "interim" / "article_pool.parquet")["article_id"].astype(str).tolist()
    known_missing = set(MISSING.read_text().split()) if MISSING.exists() else set()
    todo = [a for a in ids if a not in known_missing]
    if args.limit:
        todo = todo[:args.limit]

    from tqdm import tqdm
    counts = {"ok": 0, "skip": 0, "missing": 0, "forbidden": 0, "throttled": 0, "error": 0}
    errors = []
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex, open(MISSING, "a") as miss:
        futs = [ex.submit(fetch, a, args.delay) for a in todo]
        for f in tqdm(as_completed(futs), total=len(futs), unit="img", mininterval=5):
            aid, st = f.result()
            counts[st] += 1
            n = sum(counts.values())
            if n % 1000 == 0:
                el = time.time() - t0
                print(f"[progress] {n}/{len(todo)} {counts} {n/el:.1f}/s, ETA {(len(todo)-n)/max(n/el,1e-9)/60:.0f} min",
                      flush=True)
            if st == "missing":
                miss.write(aid + "\n"); miss.flush()
            elif st in ("error", "forbidden"):
                errors.append(aid)
            if counts["forbidden"] >= 3:
                print("403 반복: 대회 규칙 동의 또는 권한 문제. 중단합니다.", flush=True)
                ex.shutdown(cancel_futures=True)
                sys.exit(2)
            if counts["throttled"] >= 3:
                print("요청 제한(404) 감지. 중단합니다. 시간을 두고 다시 실행하면 이어서 받습니다.", flush=True)
                ex.shutdown(cancel_futures=True)
                sys.exit(4)
    dt = time.time() - t0
    size = sum(p.stat().st_size for p in (OUT / "images").rglob("*.jpg")) if (OUT / "images").exists() else 0
    print(f"done {len(todo)} in {dt:.0f}s ({len(todo)/max(dt,1e-9):.1f}/s) | {counts} | images on disk {size/1e9:.2f} GB",
          flush=True)
    if errors:
        print(f"errors (다시 실행하면 재시도): {len(errors)}  e.g. {errors[:5]}", flush=True)


if __name__ == "__main__":
    main()
