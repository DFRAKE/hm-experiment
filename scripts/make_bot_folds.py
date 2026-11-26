"""조건에 맞는 고객 전체(bot_candidates, 7,780명)를 서로 겹치지 않는 K개 묶음으로 나눈다 (명세 수정 v3 4-5).

각 묶음은 층(구매 수 × 다양성)마다 고르게 나눠 담고, 묶음 안에서 원 설계와 같은 방식(prepare.sample_and_assign)으로
A/B를 층화 배정한다. 메인 1,000명 표(bots.parquet)는 건드리지 않는다.
실행: .venv/bin/python scripts/make_bot_folds.py [--k 7]
출력: data/interim/bot_folds/bots_f{1..K}.parquet, balance_f{k}.md, folds_summary.md
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config  # noqa: E402
from src.prepare import balance_markdown, balance_table, sample_and_assign  # noqa: E402

FOLD_SEED = 20201001


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=7)
    args = ap.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    interim = cfg["paths"]["interim"]
    cand = pd.read_parquet(interim / "bot_candidates.parquet")
    main_ids = set(pd.read_parquet(interim / "bots.parquet").index)
    rng = np.random.default_rng(FOLD_SEED)
    fold = pd.Series(0, index=cand.index)
    offset = 0
    for s in sorted(cand["stratum"].unique()):          # 층마다 섞어서 돌아가며 담기 → 묶음마다 층 비율이 같음
        ids = rng.permutation(np.sort(cand.index[cand["stratum"] == s].to_numpy()))
        fold.loc[ids] = (np.arange(len(ids)) + offset) % args.k + 1
        offset += len(ids)
    out = interim / "bot_folds"
    out.mkdir(exist_ok=True)
    thr = cfg["bots"]["smd_threshold"]
    L = ["# 고객 묶음 (bot_candidates 전체를 겹치지 않게 나눔)", "",
         f"후보 {len(cand):,}명 → {args.k}개 묶음. 묶음 안 A/B 층화 배정 (원 설계와 같은 방법).", "",
         "| 묶음 | 봇 수 | A | B | 메인 1,000명과 겹침 | SMD 최대 |", "|---|---:|---:|---:|---:|---:|"]
    for k in range(1, args.k + 1):
        c = cand[fold == k]
        fc = copy.deepcopy(cfg)
        fc["bots"]["n_bots"], fc["bots"]["seed"] = len(c), FOLD_SEED + k
        bots, _ = sample_and_assign(c, fc)
        assert len(bots) == len(c) and set(bots.index) == set(c.index)
        bots.to_parquet(out / f"bots_f{k}.parquet")
        bt = balance_table(bots, thr)
        (out / f"balance_f{k}.md").write_text(balance_markdown(bt, bots, thr), encoding="utf-8")
        smd = bt["abs_SMD"].max()
        n = bots["group"].value_counts()
        L.append(f"| f{k} | {len(bots):,} | {n.get('A', 0)} | {n.get('B', 0)} | {len(main_ids & set(bots.index))} | {smd:.3f} |")
    (out / "folds_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
