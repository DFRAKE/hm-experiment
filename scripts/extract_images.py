"""Kaggle에서 받은 대회 전체 zip에서 상품 풀의 이미지만 꺼낸다 (전체 압축 해제 안 함).

- 저장: data/raw/images/<앞3자리>/<article_id>.jpg
- 이미 있는 파일은 건너뛴다
- zip에 없는 상품(대회에 이미지가 없는 상품)은 data/raw/images_missing.txt 에 기록
- zip 파일은 지우지 않는다 (확인 후 사용자가 결정)

실행: python scripts/extract_images.py ~/Downloads/h-and-m-personalized-fashion-recommendations.zip
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("사용법: python scripts/extract_images.py <zip 경로>")
    zpath = Path(sys.argv[1]).expanduser()
    ids = pd.read_parquet(ROOT / "data" / "interim" / "article_pool.parquet")["article_id"].astype(str).tolist()

    with zipfile.ZipFile(zpath) as z:
        names = {n for n in z.namelist() if n.endswith(".jpg")}
        prefix = next((n[: n.index("images/")] for n in names if "images/" in n), "")
        print(f"zip 안 jpg {len(names):,}장 (경로 접두어 '{prefix}')", flush=True)
        ok = skip = 0
        missing = []
        for a in tqdm(ids, unit="img", mininterval=5):
            rel = f"images/{a[:3]}/{a}.jpg"
            dst = OUT / rel
            if dst.exists() and dst.stat().st_size > 0:
                skip += 1
                continue
            src = prefix + rel
            if src not in names:
                missing.append(a)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".jpg.tmp")
            with z.open(src) as f, open(tmp, "wb") as g:
                g.write(f.read())
            tmp.rename(dst)
            ok += 1

    (OUT / "images_missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    size = sum(p.stat().st_size for p in (OUT / "images").rglob("*.jpg"))
    print(f"꺼냄 {ok:,} / 이미 있음 {skip:,} / zip에 없음 {len(missing):,} (풀 {len(ids):,}) | "
          f"images 폴더 {size/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
