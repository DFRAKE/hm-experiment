"""봇 가정 보정 (명세 수정 v3 4-2). **검증 주(valid)만** 쓴다. 정답 주(test)는 읽지 않는다.

질문: 학습 기간으로 만든 봇 규칙이, 같은 고객이 검증 주에 실제로 산 상품을 얼마나 '찜'으로 판정하나?
그리고 그게 아무 상품(무작위)을 찜으로 판정하는 비율보다 얼마나 높은가?

    통과율(TPR)  = 검증 주에 실제로 산 상품 중 규칙을 통과하는 비율 (높을수록 규칙이 실제 구매를 설명)
    오탐율(FPR)  = 무작위 풀 상품 중 규칙을 통과하는 비율 (높을수록 아무거나 찜함)
    J = TPR - FPR (Youden J, 0이면 무작위와 구분 못 함, 음수면 규칙이 실제 구매와 반대로 감)
    AUC = 실제 구매 상품의 취향 유사도가 무작위 상품보다 높을 확률 (기준선과 무관, 0.5 = 구분 못 함)

규칙의 부분마다 따로 본다 (각 부분은 서로 다른 값만 쓴다):
    1. 취향: 반감기(taste_half_life_days) -> AUC,  찜 기준선(like_quantile) -> 취향 부분 TPR/FPR/J
    2. 가격 상한: 기준 가격(price_basis: all|recent) × 배수(price_mult) -> 가격 부분 TPR/FPR/J
    3. 이미 있어요: type(최근 산 종류, 원 설계, 참고용) / near_dup(최근 산 같은 종류 옷과 거의 같은 옷, v3)
       near_dup 은 최근 일수(recent_days) × '거의 같음' 기준(dup_quantile)
    4. 전체 규칙: 현재 값 vs 제안 값

제안 규칙 (config 는 자동으로 바꾸지 않는다):
    - 반감기: 평균 AUC 최대. 단 현재 값과 차이가 0.005 미만이면 현재 값 유지
    - 찜 기준선: 그 반감기에서 J 최대. 무작위 상품 찜률 5~20% (like_quantile 0.80~0.95) 안에서만
      (봇이 아무거나 다 찜하면 '넘긴 이유' 신호가 의미 없어짐). J 차이 0.01 미만이면 현재 값 유지
    - 가격·이미 있어요 (2026-11-25 1차 보정 뒤 바꾼 규칙, 명세 수정 v3 4-2 참고):
      실제로 산 옷을 10% 넘게 거절하지 않으면서(TPR ≥ 0.90) 무작위 옷을 가장 많이 거절하는(FPR 최소) 설정.
      구매 기록에는 '비싸서 안 산 것'이 남지 않아 J 로는 이 두 규칙을 고를 수 없다 (1차 보정에서 J 최대 = 규칙 끄기).
      그래서 "실제 행동을 크게 어기지 않는 범위에서 이유 신호를 최대한 남긴다"를 기준으로 삼는다.
      이미 있어요는 near_dup 후보 중에서만 고른다.
평균은 봇마다 낸 값의 평균 (봇 1명 = 가중치 1). 검증 주 구매 중 상품 풀(사진 있는 상품)에 있는 것만 쓴다.

실행: python -m src.calibrate_bots [--out DIR]     출력: bot_calibration.csv, bot_calibration.md
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.bots import build_profiles, load_bot_table, load_train_tx
from src.config import experiment_tag, load_config, results_dir
from src.items import ItemPool, load_pool

SPLIT = "valid"                   # 이 모듈은 검증 주만 쓴다
GRID = {
    "taste_half_life_days": [3, 7, 14, 30, 60, float("inf")],
    "like_quantile": [0.70, 0.80, 0.85, 0.90, 0.95],
    "price_basis": ["all", "recent"],
    "price_mult": [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, float("inf")],
    "type_recent_days": [0, 3, 7, 14, 28],          # already_have=type (원 설계, 참고용)
    "dup_recent_days": [14, 28, 70],                # already_have=near_dup (70 = 학습 기간 전체)
    "dup_quantile": [0.90, 0.95, 0.99, 0.999],
}
Q_RANGE = (0.80, 0.95)
MIN_TPR = 0.90
TOL_AUC, TOL_J = 0.005, 0.01
KEYS = ("taste_half_life_days", "like_quantile", "price_basis", "price_mult", "already_have", "recent_days",
        "dup_quantile")
N_RANDOM = 2000
RANDOM_SEED = 20200909            # 찜 기준선 계산용 표본(seed·봇별)과 다른 무작위 상품


def load_valid_purchases(cfg: dict, bot_ids) -> pd.DataFrame:
    lo, hi = (pd.Timestamp(d) for d in cfg["periods"][SPLIT])
    w = pd.read_parquet(cfg["paths"]["interim"] / "tx_2020q3.parquet", columns=["t_dat", "customer_id", "article_id"],
                        filters=[("t_dat", ">=", lo), ("t_dat", "<=", hi)])
    return w[w["customer_id"].isin(set(bot_ids))].drop_duplicates(["customer_id", "article_id"])


def with_rule(cfg: dict, **kw) -> dict:
    c = copy.deepcopy(cfg)
    c["bot_rule"].update(kw)
    return c


class Calib:
    def __init__(self, cfg: dict, pool: ItemPool, seed: int = 0):
        self.cfg, self.pool, self.seed = cfg, pool, seed
        self.tx, self.table = load_train_tx(cfg), load_bot_table(cfg)
        v = load_valid_purchases(cfg, self.table.index)
        self.n_valid = len(v)
        v = v[v["article_id"].isin(pool.index)]
        self.n_valid_in_pool = len(v)
        self.buy = {b: pool.rows(g["article_id"]) for b, g in v.groupby("customer_id")}
        self.rand = np.random.default_rng(RANDOM_SEED).choice(len(pool), size=N_RANDOM, replace=False)

    def bots(self, **kw):
        return [b for b in build_profiles(self.tx, self.pool, self.table, with_rule(self.cfg, **kw), self.seed)
                if b.bot_id in self.buy]

    def taste(self, h: float, q: float) -> dict:
        auc, tpr, fpr = [], [], []
        for b in self.bots(taste_half_life_days=h, like_quantile=q):
            sr = np.sort(self.pool.bot_emb[self.rand] @ b.taste)
            sb = self.pool.bot_emb[self.buy[b.bot_id]] @ b.taste
            auc.append(((np.searchsorted(sr, sb, "left") + np.searchsorted(sr, sb, "right")) / 2 / len(sr)).mean())
            tpr.append((sb >= b.like_th).mean())
            fpr.append((sr >= b.like_th).mean())
        return {"auc": np.mean(auc), "tpr": np.mean(tpr), "fpr": np.mean(fpr)}

    def price(self, basis: str, m: float) -> dict:
        tpr, fpr = [], []
        for b in self.bots(price_basis=basis, price_mult=m):
            tpr.append((self.pool.price[self.buy[b.bot_id]] <= b.price_th).mean())
            fpr.append((self.pool.price[self.rand] <= b.price_th).mean())
        return {"tpr": np.mean(tpr), "fpr": np.mean(fpr)}

    def already(self, mode: str, d: int, q: float | None) -> dict:
        tpr, fpr, n_types = [], [], []
        kw = {"already_have": mode, "recent_days": d} | ({"dup_quantile": q} if q is not None else {})
        for b in self.bots(**kw):
            tpr.append(1 - b.dup_mask(self.pool, self.buy[b.bot_id]).mean())
            fpr.append(1 - b.dup_mask(self.pool, self.rand).mean())
            n_types.append(len(b.recent_types))
        return {"tpr": np.mean(tpr), "fpr": np.mean(fpr), "recent_types_mean": np.mean(n_types)}

    def full(self, **kw) -> dict:
        tpr, fpr = [], []
        for b in self.bots(**kw):
            tpr.append(b.rule_mask(self.pool, self.buy[b.bot_id])[2].mean())
            fpr.append(b.rule_mask(self.pool, self.rand)[2].mean())
        return {"tpr": np.mean(tpr), "fpr": np.mean(fpr)}


def _pick_constrained(t: pd.DataFrame) -> pd.Series:
    """TPR ≥ MIN_TPR 중 FPR 최소 (같으면 TPR 높은 쪽)."""
    ok = t[t["tpr"] >= MIN_TPR]
    return ok.sort_values(["fpr", "tpr"], ascending=[True, False]).iloc[0]


def run(cfg: dict) -> tuple[pd.DataFrame, dict]:
    pool = load_pool(cfg)
    c = Calib(cfg, pool)
    rc = cfg["bot_rule"]
    cur = {k: rc.get(k) for k in KEYS}
    cur["price_basis"] = cur["price_basis"] or "all"
    cur["already_have"] = cur["already_have"] or "type"
    rows = []
    for h in GRID["taste_half_life_days"]:
        for q in GRID["like_quantile"]:
            rows.append({"part": "taste", "taste_half_life_days": h, "like_quantile": q, **c.taste(h, q)})
    for basis in GRID["price_basis"]:
        for m in GRID["price_mult"]:
            rows.append({"part": "price", "price_basis": basis, "price_mult": m, **c.price(basis, m)})
    for d in GRID["type_recent_days"]:
        rows.append({"part": "already", "already_have": "type", "recent_days": d, **c.already("type", d, None)})
    for d in GRID["dup_recent_days"]:
        for q in GRID["dup_quantile"]:
            rows.append({"part": "already", "already_have": "near_dup", "recent_days": d, "dup_quantile": q,
                         **c.already("near_dup", d, q)})
    t = pd.DataFrame(rows)
    t["j"] = t["tpr"] - t["fpr"]

    taste = t[t["part"] == "taste"]
    auc = taste.groupby("taste_half_life_days")["auc"].mean()
    h = auc.idxmax()
    if cur["taste_half_life_days"] in auc.index and auc.max() - auc[cur["taste_half_life_days"]] < TOL_AUC:
        h = cur["taste_half_life_days"]
    qs = taste[(taste["taste_half_life_days"] == h) & taste["like_quantile"].between(*Q_RANGE)].set_index("like_quantile")
    q = qs["j"].idxmax()
    if cur["like_quantile"] in qs.index and qs["j"].max() - qs.at[cur["like_quantile"], "j"] < TOL_J:
        q = cur["like_quantile"]
    pr = _pick_constrained(t[t["part"] == "price"])
    al = _pick_constrained(t[(t["part"] == "already") & (t["already_have"] == "near_dup")])
    prop = {"taste_half_life_days": float(h), "like_quantile": float(q),
            "price_basis": pr["price_basis"], "price_mult": float(pr["price_mult"]),
            "already_have": "near_dup", "recent_days": int(al["recent_days"]), "dup_quantile": float(al["dup_quantile"])}
    full_cfg = lambda d: {k: v for k, v in d.items() if v is not None}  # noqa: E731
    full = pd.DataFrame([{"part": "full", "setting": "current", **cur, **c.full(**full_cfg(cur))},
                         {"part": "full", "setting": "proposed", **prop, **c.full(**prop)}])
    full["j"] = full["tpr"] - full["fpr"]
    t = pd.concat([t, full], ignore_index=True)
    meta = {"experiment": experiment_tag(cfg), "split": SPLIT, "bots_with_valid_in_pool": len(c.buy),
            "valid_purchases": c.n_valid, "valid_purchases_in_pool": c.n_valid_in_pool,
            "current": cur, "proposed": prop, "q_range": Q_RANGE, "min_tpr": MIN_TPR}
    return t, meta


def markdown(t: pd.DataFrame, meta: dict) -> str:
    f = lambda x: "∞" if x == float("inf") else f"{x:g}"  # noqa: E731
    L = [f"# 봇 가정 보정 ({meta['experiment']}, 검증 주만)", "",
         f"검증 주 구매 {meta['valid_purchases']:,}건 중 상품 풀 안 {meta['valid_purchases_in_pool']:,}건, "
         f"봇 {meta['bots_with_valid_in_pool']:,}명. 통과율 = 실제 산 상품이 규칙을 통과한 비율, "
         "오탐율 = 무작위 상품이 통과한 비율, J = 통과율 − 오탐율.", "",
         "## 1. 취향 (반감기 × 찜 기준선)", "", "| 반감기(일) | 기준선 | AUC | 통과율 | 오탐율 | J |", "|---:|---:|---:|---:|---:|---:|"]
    for r in t[t["part"] == "taste"].itertuples():
        L.append(f"| {f(r.taste_half_life_days)} | {r.like_quantile:.2f} | {r.auc:.3f} | {r.tpr:.1%} | {r.fpr:.1%} | {r.j:+.3f} |")
    L += ["", "## 2. 가격 상한 (기준 가격 × 배수)", "", "기준 가격: all = 학습 기간 전체 중앙값, recent = 최근 14일 중앙값", "",
          "| 기준 | 배수 | 통과율 | 오탐율 | J |", "|---|---:|---:|---:|---:|"]
    for r in t[t["part"] == "price"].itertuples():
        L.append(f"| {r.price_basis} | {f(r.price_mult)} | {r.tpr:.1%} | {r.fpr:.1%} | {r.j:+.3f} |")
    L += ["", "## 3. 이미 있어요", "", "type = 최근 N일에 산 종류면 넘김(원 설계), near_dup = 최근 N일에 산 같은 종류 옷과 "
          "'거의 같은' 옷만 넘김. 기준 = 같은 종류 무작위 두 상품 유사도의 분위수", "",
          "| 방식 | 최근 일수 | 거의 같음 기준 | 통과율 | 오탐율 | J |", "|---|---:|---:|---:|---:|---:|"]
    for r in t[t["part"] == "already"].itertuples():
        dq = "" if pd.isna(r.dup_quantile) else f"{r.dup_quantile:g}"
        L.append(f"| {r.already_have} | {int(r.recent_days)} | {dq} | {r.tpr:.1%} | {r.fpr:.1%} | {r.j:+.3f} |")
    L += ["", f"가격·이미 있어요 제안 = 통과율 ≥ {MIN_TPR:.0%} 중 오탐율 최소.", "",
          "## 4. 전체 규칙", "",
          "| 설정 | 반감기 | 기준선 | 가격 | 이미 있어요 | 통과율 | 오탐율 | J |", "|---|---:|---:|---|---|---:|---:|---:|"]
    for r in t[t["part"] == "full"].itertuples():
        al = f"{r.already_have} {int(r.recent_days)}일" + ("" if pd.isna(r.dup_quantile) or r.already_have == "type"
                                                         else f" q{r.dup_quantile:g}")
        L.append(f"| {'현재' if r.setting == 'current' else '제안'} | {f(r.taste_half_life_days)} | {r.like_quantile:.2f} "
                 f"| {r.price_basis} ×{f(r.price_mult)} | {al} | {r.tpr:.1%} | {r.fpr:.1%} | {r.j:+.3f} |")
    L += ["", "제안 값은 config.yaml 에 자동으로 들어가지 않는다. 확인 후 직접 바꾼다."]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    t, meta = run(cfg)
    out = args.out or results_dir(cfg, SPLIT)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "bot_calibration.csv", index=False)
    (out / "bot_calibration_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    md = markdown(t, meta)
    (out / "bot_calibration.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
