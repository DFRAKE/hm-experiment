"""3.4 상품 임베딩: intfloat/multilingual-e5-small, 384차원, 정규화, CPU, 배치 64.

- 배치마다 data/interim/emb_parts/batch_XXXXX.npy 로 저장한다. 끊겨도 다시 실행하면 저장된 배치는 건너뛴다.
- 모든 배치가 모이면 data/interim/article_emb.npy + article_ids.json 으로 합친다.
- 상품 풀이나 설정이 바뀌면(meta.json 불일치) 이어 하지 않고 멈춘다. --restart 로 처음부터 다시 한다.

실행: python -m src.embed [--restart]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from src.config import load_config


def passage_text(r) -> str:
    def v(x):
        return "" if pd.isna(x) else str(x).strip()
    return (f"passage: {v(r.prod_name)}. {v(r.product_type_name)}. {v(r.colour_group_name)}. "
            f"{v(r.section_name)}. {v(r.detail_desc)}")


def load_texts(cfg: dict) -> tuple[list[str], list[str]]:
    pool = pd.read_parquet(cfg["paths"]["interim"] / "article_pool.parquet")
    pool = pool.sort_values("article_id").reset_index(drop=True)
    return pool["article_id"].tolist(), [passage_text(r) for r in pool.itertuples()]


def fingerprint(ids: list[str], texts: list[str], ecfg: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(ecfg, sort_keys=True).encode())
    for i, t in zip(ids, texts):
        h.update(i.encode()); h.update(b"\0"); h.update(t.encode()); h.update(b"\n")
    return h.hexdigest()


def encode(texts: list[str], cfg: dict) -> np.ndarray:
    """상품 풀과 같은 모델·설정으로 텍스트를 임베딩 (채점용 풀 밖 상품 등 소량)."""
    import torch
    from sentence_transformers import SentenceTransformer
    ecfg = cfg["embed"]
    torch.set_num_threads(ecfg.get("threads") or os.cpu_count())
    model = SentenceTransformer(ecfg["model"], device="cpu")
    return model.encode(texts, batch_size=ecfg["batch_size"], normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart", action="store_true", help="저장된 배치를 지우고 처음부터")
    args = ap.parse_args(argv)
    cfg = load_config()
    ecfg = cfg["embed"]
    interim = cfg["paths"]["interim"]
    parts = interim / "emb_parts"
    meta_path = parts / "meta.json"

    ids, texts = load_texts(cfg)
    fp = fingerprint(ids, texts, ecfg)
    bs = ecfg["batch_size"]
    n_batches = (len(ids) + bs - 1) // bs

    if args.restart and parts.exists():
        shutil.rmtree(parts)
    parts.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] != fp:
            sys.exit("상품 풀이나 임베딩 설정이 저장된 배치와 다릅니다. --restart 로 다시 실행하세요.")
    else:
        meta = {"fingerprint": fp, "model": ecfg["model"], "n_items": len(ids),
                "batch_size": bs, "n_batches": n_batches, "elapsed_sec": 0.0}
        meta_path.write_text(json.dumps(meta, indent=2))

    def part(b):
        return parts / f"batch_{b:05d}.npy"

    todo = [b for b in range(n_batches) if not part(b).exists()]
    print(f"items {len(ids):,} | batches {n_batches} (size {bs}) | done {n_batches - len(todo)} | todo {len(todo)}",
          flush=True)

    if todo:
        import torch
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm

        torch.set_num_threads(ecfg.get("threads") or os.cpu_count())
        model = SentenceTransformer(ecfg["model"], device="cpu")
        dim = model.get_sentence_embedding_dimension()
        assert dim == ecfg["dim"], f"차원 {dim} != {ecfg['dim']}"
        t0 = time.time()
        try:
            for b in tqdm(todo, desc="embed", unit="batch", mininterval=5, file=sys.stdout):
                chunk = texts[b * bs:(b + 1) * bs]
                emb = model.encode(chunk, batch_size=bs, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
                tmp = parts / f".tmp_{b:05d}.npy"
                np.save(tmp, emb)
                os.replace(tmp, part(b))       # 다 쓴 배치만 보이게 (중간에 끊겨도 깨진 파일 없음)
        finally:
            meta["elapsed_sec"] = round(meta["elapsed_sec"] + time.time() - t0, 1)
            meta_path.write_text(json.dumps(meta, indent=2))

    emb = np.concatenate([np.load(part(b)) for b in range(n_batches)])
    assert emb.shape == (len(ids), ecfg["dim"]), emb.shape
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1, atol=1e-3), (norms.min(), norms.max())
    np.save(interim / "article_emb.npy", emb)
    (interim / "article_ids.json").write_text(json.dumps(ids))
    print(f"saved article_emb.npy {emb.shape} {emb.dtype} | norm {norms.min():.4f}~{norms.max():.4f} "
          f"| embed time {meta['elapsed_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
