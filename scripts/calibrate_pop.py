"""분석 3-1: 봇에 '유행 따르기'를 넣으면 실제 고객 구매를 더 잘 설명하나 (검증 주만, src.calibrate_bots 와 같은 방식).

찜 판단 점수 = 취향 유사도 + pop_coef × 상품 인기도(log1p 판매 수 표준화). pop_weight(β) = 인기도 1표준편차를
취향 유사도 몇 표준편차(봇마다)로 칠지. fixed = 모든 봇 같은 β, personal = 학습 기간에 인기 상품을 많이 산 고객일수록 큰 β.

    AUC = 검증 주에 실제로 산 상품의 점수가 무작위 상품보다 높을 확률
    통과율/오탐율/J = 전체 규칙(취향+가격+이미 있어요)을 실제 산 상품 / 무작위 상품이 통과하는 비율

출력: data/results/analysis/calibrate_pop.{md,csv}
실행: .venv/bin/python scripts/calibrate_pop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibrate_bots import Calib  # noqa: E402
from src.config import load_config  # noqa: E402
from src.items import load_pool  # noqa: E402

OUT = ROOT / "data" / "results" / "analysis"
BETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def evaluate(c: Calib, beta: float, mode: str) -> dict:
    auc, tpr, fpr, ttpr, tfpr = [], [], [], [], []
    for b in c.bots(pop_weight=beta, pop_mode=mode):
        sr = np.sort(b.score(c.pool, c.rand))
        sb = b.score(c.pool, c.buy[b.bot_id])
        auc.append(((np.searchsorted(sr, sb, "left") + np.searchsorted(sr, sb, "right")) / 2 / len(sr)).mean())
        ttpr.append((sb >= b.like_th).mean())
        tfpr.append((sr >= b.like_th).mean())
        tpr.append(b.rule_mask(c.pool, c.buy[b.bot_id])[2].mean())
        fpr.append(b.rule_mask(c.pool, c.rand)[2].mean())
    return {"pop_weight": beta, "pop_mode": mode, "auc": np.mean(auc), "taste_tpr": np.mean(ttpr),
            "taste_fpr": np.mean(tfpr), "tpr": np.mean(tpr), "fpr": np.mean(fpr)}


def main() -> None:
    cfg = load_config()
    c = Calib(cfg, load_pool(cfg))
    rows = [evaluate(c, 0.0, "fixed")]
    for mode in ("fixed", "personal"):
        rows += [evaluate(c, b, mode) for b in BETAS if b]
    t = pd.DataFrame(rows)
    t["taste_j"] = t.taste_tpr - t.taste_fpr
    t["j"] = t.tpr - t.fpr
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "calibrate_pop.csv", index=False)
    best = t.loc[t.auc.idxmax()]
    L = ["# 분석 3-1: 유행 따르는 봇이 실제 구매를 더 잘 설명하나 (검증 주)", "",
         f"봇 {len(c.buy):,}명, 검증 주 구매 중 상품 풀 안 {c.n_valid_in_pool:,}건. 메인 봇 = 유행 따르기 0.", "",
         "| 방식 | 유행 가중치 β | AUC | 취향 통과율 | 취향 오탐율 | 취향 J | 전체 규칙 통과율 | 전체 오탐율 | 전체 J |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples():
        L.append(f"| {'메인 (끔)' if r.pop_weight == 0 else r.pop_mode} | {r.pop_weight:g} | {r.auc:.3f} | {r.taste_tpr:.1%} "
                 f"| {r.taste_fpr:.1%} | {r.taste_j:+.3f} | {r.tpr:.1%} | {r.fpr:.1%} | {r.j:+.3f} |")
    L += ["", f"AUC 최대: {best.pop_mode} β={best.pop_weight:g} (AUC {best.auc:.3f}, 메인 {t.auc.iat[0]:.3f})."]
    (OUT / "calibrate_pop.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
