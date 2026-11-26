"""명세 수정 v3: 상품 사진 임베딩 (FashionCLIP, patrickjohncyh/fashion-clip, 512차원, 정규화).

- 대상: 상품 풀 중 사진이 있는 상품. 사진은 data/raw/images/<앞3자리>/<id>.jpg 파일, 없으면 대회 공식 zip
  (data/raw/h-and-m-personalized-fashion-recommendations.zip, --zip 으로 바꿈) 안에서 바로 읽는다 (풀지 않음, 디스크 절약)
- zip 이 있으면 zip 에 없는 풀 상품을 data/raw/images_missing.txt 에 적는다 (공식 전체 사진 기준)
- 풀의 모든 상품이 "사진 있음" 또는 images_missing.txt 에 있어야 실행한다
  (사진을 다 받기 전에 돌려서 풀이 잘못 줄어드는 것을 막음). --allow-partial 은 점검용이며 --out 이 필수.
- 배치마다 img_emb_parts/batch_XXXXX.npy 로 저장 -> 끊겨도 다시 실행하면 이어서 함
- 열리지 않는(깨진) 사진은 사진 없음으로 처리하고 data/interim/images_unreadable.txt 에 남긴다

출력: data/interim/article_img_emb.npy (N, 512), article_img_ids.json
실행: python -m src.embed_image [--restart] [--allow-partial --out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

ZIP_NAME = "h-and-m-personalized-fashion-recommendations.zip"


def image_path(cfg: dict, aid: str) -> Path:
    return cfg["paths"]["raw"] / "images" / aid[:3] / f"{aid}.jpg"


def pick_device(want: str) -> str:
    import torch
    if want and want != "auto":
        return want
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(ecfg: dict, device: str):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(ecfg["model"]).to(device).eval()
    proc = CLIPProcessor.from_pretrained(ecfg["model"])
    return model, proc


def image_features(model, proc, images, device: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        x = proc(images=images, return_tensors="pt").to(device)
        out = model.get_image_features(**x)
        e = out if torch.is_tensor(out) else out.pooler_output     # transformers 버전에 따라 반환형이 다름
        e = e / e.norm(dim=-1, keepdim=True)
    return e.float().cpu().numpy()


def main(argv=None) -> None:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="사진을 다 받기 전 점검용 (--out 필수)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--zip", type=Path, default=None, help=f"대회 공식 zip (기본 data/raw/{ZIP_NAME})")
    args = ap.parse_args(argv)
    cfg = load_config()
    ecfg = cfg["embed_image"]
    raw = cfg["paths"]["raw"]
    out = args.out or cfg["paths"]["interim"]
    if args.allow_partial and args.out is None:
        sys.exit("--allow-partial 은 결과가 실제 실험에 섞이지 않도록 --out 을 따로 지정해야 합니다.")

    pool_ids = pd.read_parquet(cfg["paths"]["interim"] / "article_pool.parquet")["article_id"].astype(str)
    pool_ids = sorted(pool_ids)
    missing_file = raw / "images_missing.txt"
    on_disk = {a for a in pool_ids if image_path(cfg, a).exists() and image_path(cfg, a).stat().st_size > 0}
    zpath = args.zip or raw / ZIP_NAME
    zf, in_zip = None, {}
    if zpath.exists():
        import zipfile
        zf = zipfile.ZipFile(zpath)
        for name in zf.namelist():
            if name.endswith(".jpg") and "/images/" in "/" + name:
                in_zip[Path(name).stem] = name
        miss = sorted(a for a in pool_ids if a not in in_zip and a not in on_disk)
        tmp = missing_file.with_suffix(".tmp")
        tmp.write_text("\n".join(miss) + ("\n" if miss else ""))
        os.replace(tmp, missing_file)
        print(f"zip {zpath.name}: 사진 {len(in_zip):,}장 | 풀 중 zip에 없음 {len(miss):,}", flush=True)
    known_missing = set(missing_file.read_text().split()) if missing_file.exists() else set()
    have_set = on_disk | (set(in_zip) & set(pool_ids))
    have = [a for a in pool_ids if a in have_set]
    unresolved = [a for a in pool_ids if a not in known_missing and a not in have_set]
    print(f"pool {len(pool_ids):,} | 사진 있음 {len(have):,} | 대회에 사진 없음 {len(known_missing):,} "
          f"| 아직 모름 {len(unresolved):,}", flush=True)
    if unresolved and not args.allow_partial:
        sys.exit("아직 받지 않은 사진이 있습니다. scripts/extract_images.py 를 먼저 끝내세요.")

    ids = have
    bs = ecfg["batch_size"]
    n_batches = (len(ids) + bs - 1) // bs
    fp = hashlib.sha256(json.dumps({"ids": ids, "cfg": ecfg}, sort_keys=True).encode()).hexdigest()
    parts = out / "img_emb_parts"
    meta_path = parts / "meta.json"
    if args.restart and parts.exists():
        shutil.rmtree(parts)
    parts.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] != fp:
            sys.exit("상품 목록이나 설정이 저장된 배치와 다릅니다. --restart 로 다시 실행하세요.")
    else:
        meta = {"fingerprint": fp, "model": ecfg["model"], "n_items": len(ids), "batch_size": bs,
                "n_batches": n_batches, "elapsed_sec": 0.0, "partial": bool(unresolved)}
        meta_path.write_text(json.dumps(meta, indent=2))

    def part(b: int) -> Path:
        return parts / f"batch_{b:05d}.npz"

    todo = [b for b in range(n_batches) if not part(b).exists()]
    print(f"batches {n_batches} (size {bs}) | done {n_batches - len(todo)} | todo {len(todo)}", flush=True)
    if todo:
        from tqdm import tqdm
        device = pick_device(ecfg.get("device", "auto"))
        model, proc = load_model(ecfg, device)
        t0 = time.time()
        try:
            for b in tqdm(todo, desc=f"embed_image[{device}]", unit="batch", mininterval=5, file=sys.stdout):
                chunk = ids[b * bs:(b + 1) * bs]
                ims, ok_ids = [], []
                for a in chunk:
                    try:
                        src = image_path(cfg, a) if a in on_disk else io.BytesIO(zf.read(in_zip[a]))
                        with Image.open(src) as im:
                            ims.append(im.convert("RGB"))
                        ok_ids.append(a)
                    except Exception:  # noqa: BLE001  깨진 사진은 사진 없음으로
                        pass
                emb = image_features(model, proc, ims, device) if ims else np.zeros((0, ecfg["dim"]), np.float32)
                tmp = parts / f".tmp_{b:05d}.npz"
                np.savez(tmp, ids=np.array(ok_ids), emb=emb.astype(np.float32))
                os.replace(tmp, part(b))
        finally:
            meta["elapsed_sec"] = round(meta["elapsed_sec"] + time.time() - t0, 1)
            meta_path.write_text(json.dumps(meta, indent=2))

    got_ids, embs = [], []
    for b in range(n_batches):
        z = np.load(part(b))
        got_ids += z["ids"].tolist()
        embs.append(z["emb"])
    emb = np.concatenate(embs)
    assert emb.shape == (len(got_ids), ecfg["dim"]), emb.shape
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1, atol=1e-3), (norms.min(), norms.max())
    unreadable = sorted(set(ids) - set(got_ids))
    np.save(out / "article_img_emb.npy", emb)
    (out / "article_img_ids.json").write_text(json.dumps(got_ids))
    (out / "images_unreadable.txt").write_text("\n".join(unreadable) + ("\n" if unreadable else ""))
    rnd = np.random.default_rng(0).choice(len(emb), size=min(2000, len(emb)), replace=False)
    S = emb[rnd] @ emb[rnd].T
    off = S[np.triu_indices(len(rnd), 1)]
    print(f"saved article_img_emb.npy {emb.shape} | 깨진 사진 {len(unreadable)} | embed time {meta['elapsed_sec']:.0f}s "
          f"| 무작위 두 상품 유사도 평균 {off.mean():.3f} (p10 {np.quantile(off, .1):.3f}, p90 {np.quantile(off, .9):.3f})",
          flush=True)


if __name__ == "__main__":
    main()
