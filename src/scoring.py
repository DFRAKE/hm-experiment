"""6장 채점. 정답 주(test) 데이터는 이 모듈에서만 읽는다.

    python -m src.scoring coverage                  # 3.4 커버리지 (정답 주 구매 중 상품 풀 밖 비율)
    python -m src.scoring calibrate                 # 6.1 근접 정답 기준값을 검증 주로 조정
    python -m src.scoring score --split valid       # 검증 주로 채점 (몇 번이든)
    python -m src.scoring score --split test --final   # 정답 주 최종 채점 (마지막에 한 번만)
    python -m src.scoring score --split valid --out DIR  # 결과를 다른 폴더에 (민감도 분석·점검용)

지표 (명세 수정 v3): 메인은 취향 회복도(src.metrics_taste, 다음 주 구매를 안 씀),
다음 주 구매 근접 적중은 현실 점검. 비교는 list vs swipe, list_v2 vs swipe_v2 (세션에 있는 것만),
pop(P) 세션이 있으면 인기 기준선 대비 배율(lift)도 낸다.

검증·정답 주 구매에는 상품 풀 밖 상품도 있다. 근접 정답 판정을 위해 그 상품들만 같은 모델로 임베딩해
data/interim/scoring_emb_cache.npz 에 둔다 (채점에서만 쓰고 봇·추천기는 보지 않음).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import experiment_tag, load_config, results_dir, session_meta, sessions_dir
from src.items import ItemPool, load_pool
from src.metrics_taste import taste_metrics

COMPARISONS = [("list", "swipe"), ("list_v2", "swipe_v2")]   # (A쪽 방식, B쪽 방식)
BASELINE = "pop"
BIN = 10


# ---------------------------------------------------------------- 정답 데이터
def load_week(cfg: dict, split: str) -> pd.DataFrame:
    lo, hi = (pd.Timestamp(d) for d in cfg["periods"][split])
    return pd.read_parquet(cfg["paths"]["interim"] / "tx_2020q3.parquet",
                           columns=["t_dat", "customer_id", "article_id"],
                           filters=[("t_dat", ">=", lo), ("t_dat", "<=", hi)])


def load_answer_week(cfg: dict) -> pd.DataFrame:
    return load_week(cfg, "test")


@dataclass
class Truth:
    """봇 한 명의 정답: 상품 풀 각 상품의 '정답 구매와 최대 유사도'와 근접 정답 여부."""
    answer_ids: set[str]
    maxsim: np.ndarray        # (풀 크기,)
    pair_match: np.ndarray    # (풀 크기,) 종류·색이 정답 구매 중 하나와 모두 같음

    def near(self, tau: float) -> np.ndarray:
        return (self.maxsim >= tau) | self.pair_match


def extra_embeddings(cfg: dict, ids: list[str], articles: pd.DataFrame) -> dict[str, np.ndarray]:
    """상품 풀 밖 상품 임베딩 (캐시)."""
    from src.embed import encode, passage_text
    path = cfg["paths"]["interim"] / "scoring_emb_cache.npz"
    cache = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cache = dict(zip(z["ids"].tolist(), z["emb"]))
    todo = [a for a in ids if a not in cache]
    if todo:
        meta = articles.set_index("article_id").loc[todo].reset_index()
        emb = encode([passage_text(r) for r in meta.itertuples()], cfg)
        cache.update(zip(todo, emb))
        keys = sorted(cache)
        np.savez(path, ids=np.array(keys), emb=np.stack([cache[k] for k in keys]))
    return {a: cache[a] for a in ids}


def build_truths(cfg: dict, split: str, pool: ItemPool, bot_ids) -> dict[str, Truth]:
    from src.prepare import load_articles
    week = load_week(cfg, split)
    week = week[week["customer_id"].isin(set(bot_ids))]
    articles = load_articles(cfg)
    arts = sorted(week["article_id"].unique())
    outside = [a for a in arts if a not in pool.index]
    extra = extra_embeddings(cfg, outside, articles) if outside else {}
    # 현실 점검(다음 주 구매)은 실험 버전과 상관없이 항상 설명 문장 좌표(e5)로 잰다
    emb = {a: (pool.text_emb[pool.index[a]] if a in pool.index else extra[a]) for a in arts}

    meta = articles.set_index("article_id")
    pool_pair = (pool.df["product_type_name"].astype(str) + "|" + pool.df["colour_group_name"].astype(str)).to_numpy()
    out = {}
    for b, g in week.groupby("customer_id"):
        ids = sorted(set(g["article_id"]))
        E = np.stack([emb[a] for a in ids])
        pairs = {f"{meta.at[a, 'product_type_name']}|{meta.at[a, 'colour_group_name']}" for a in ids}
        out[b] = Truth(set(ids), (pool.text_emb @ E.T).max(axis=1), np.isin(pool_pair, list(pairs)))
    missing = set(bot_ids) - set(out)
    if missing:
        raise ValueError(f"{split} 주에 구매가 없는 봇 {len(missing)}명 (3.3 조건 위반)")
    return out


# ---------------------------------------------------------------- 6.1 기준값 조정 (검증 주)
def calibrate(cfg: dict, grid=None) -> pd.DataFrame:
    pool = load_pool(cfg)
    bots = pd.read_parquet(cfg["paths"]["interim"] / "bots.parquet").index
    truths = build_truths(cfg, "valid", pool, bots)
    grid = np.round(np.arange(0.86, 0.99, 0.005), 3) if grid is None else grid
    pair = np.array([t.pair_match.mean() for t in truths.values()])
    rows = []
    for tau in grid:
        frac = np.array([t.near(tau).mean() for t in truths.values()])
        sim_only = np.array([(t.maxsim >= tau).mean() for t in truths.values()])
        rows.append({"tau": tau, "near_frac_mean": frac.mean(), "near_frac_median": np.median(frac),
                     "sim_only_mean": sim_only.mean(), "pair_only_mean": pair.mean(),
                     "bots_with_no_near": float((frac == 0).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 세션별 지표
def load_sessions(cfg: dict, seeds) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = sessions_dir(cfg)
    want = json.loads(json.dumps(session_meta(cfg)))
    for s in seeds:
        mp = d / f"meta_seed{s}.json"
        if not mp.exists():
            print(f"[경고] {mp.name} 없음: 세션을 만든 설정을 확인할 수 없음 (명세 수정 v3 이전 세션)", flush=True)
        elif json.loads(mp.read_text()) != want:
            raise ValueError(f"{mp} 의 설정이 지금 설정과 다름. 세션을 돌린 설정(HM_CONFIG)으로 채점하세요.")
    logs = pd.concat([pd.read_parquet(d / f"log_seed{s}.parquet") for s in seeds], ignore_index=True)
    finals = pd.concat([pd.read_parquet(d / f"final_seed{s}.parquet") for s in seeds], ignore_index=True)
    return logs, finals


def session_metrics(logs: pd.DataFrame, finals: pd.DataFrame, truths: dict[str, Truth],
                    pool: ItemPool, tau: float, exposure: int) -> pd.DataFrame:
    logs = logs.sort_values(["seed", "group", "bot_id", "set_no", "pos"])
    fin_by = {k: g["article_id"].tolist() for k, g in finals.sort_values("rank").groupby(["seed", "group", "bot_id"])}
    rows = []
    for (seed, mode, b), g in logs.groupby(["seed", "group", "bot_id"], sort=False):
        t = truths[b]
        near = t.near(tau)
        shown = pool.rows(g["article_id"])
        final = fin_by[(seed, mode, b)]
        frows = pool.rows(final)
        like = (g["action"] == "like").to_numpy()
        r = {"seed": seed, "group": mode, "bot_id": b, "ab_group": g["ab_group"].iat[0],
             "assigned": bool(g["assigned"].iat[0]),
             "near_hit20": float(near[frows].any()),
             "exact_hit20": float(bool(t.answer_ids & set(final))),
             "near_recall20": float(near[frows].sum() / near.sum()) if near.sum() else np.nan,
             "first_like": float(np.argmax(like) + 1) if like.any() else np.nan}
        ms = t.maxsim[shown]
        for e in range(BIN, exposure + 1, BIN):
            r[f"curve_{e}"] = float(ms[e - BIN:e].mean())
        for lo in range(0, exposure, 20):
            r[f"like_{lo + 1}_{lo + 20}"] = float(like[lo:lo + 20].mean())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6.3 통계
def _boot_means(x: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    x = x[~np.isnan(x)]
    return x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)


def ab_diff(xl: np.ndarray, xs: np.ndarray, rng, n_boot) -> tuple[float, float, float]:
    d = _boot_means(xs, rng, n_boot) - _boot_means(xl, rng, n_boot)
    return float(np.nanmean(xs) - np.nanmean(xl)), *np.percentile(d, [2.5, 97.5]).tolist()


# (칼럼, 이름, 종류, 높을수록 좋은가). 종류: taste = 취향 회복도(메인), reality = 다음 주 구매(현실 점검)
METRICS = [
    ("pref_all", "추천 상품 선호율 (60장, 변덕 없이 찜할 비율)", "taste", True),
    ("final_like20", "최종 20개 중 찜할 개수", "taste", True),
    ("final_like20_new", "최종 20개 중 찜할 새 상품 수", "taste", True),
    ("final_taste20", "최종 20개 중 취향 기준선 넘는 개수", "taste", True),
    ("repeat_attr", "비선호 속성 반복 노출률", "taste", False),
    ("repeat_price", "  ㄴ 비싸요 뒤 비싼 상품 비율", "taste", False),
    ("repeat_type", "  ㄴ 이미 있어요 뒤 같은 종류 비율", "taste", False),
    ("first_pref", "첫 '찜할' 상품까지 노출 수", "taste", False),
    ("first_like", "첫 찜까지 노출 수 (변덕 포함)", "taste", False),
    ("near_hit20", "근접 적중률@20 (다음 주 구매)", "reality", True),
    ("exact_hit20", "정확 적중률@20 (다음 주 구매)", "reality", True),
    ("near_recall20", "근접 재현율@20 (다음 주 구매)", "reality", True),
]


def metric_list(m: pd.DataFrame) -> list[tuple[str, str, str, bool]]:
    pref = [c for c in m.columns if c.startswith("pref_") and c != "pref_all"]
    curve = [c for c in m.columns if c.startswith("curve_")]
    like = [c for c in m.columns if c.startswith("like_")]
    return ([x for x in METRICS if x[0] in m.columns]
            + [(c, f"선호율 곡선 ~{c.split('_')[1]}장", "taste_curve", True) for c in pref]
            + [(c, f"다음 주 구매 유사도 곡선 ~{c.split('_')[1]}장", "reality_curve", True) for c in curve]
            + [(c, f"찜률 {c[5:].replace('_', '~')}번째 (변덕 포함)", "observed", True) for c in like])


def summarize(m: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """비교마다(A쪽 vs B쪽): 배정된 봇끼리 A/B 차이 + 같은 봇 전원 paired 차이 + 인기 기준선(P) 대비 배율."""
    sc = cfg["scoring"]
    rng = np.random.default_rng(sc["bootstrap_seed"])
    n_boot = sc["n_boot"]
    metrics = metric_list(m)
    cols = [c for c, *_ in metrics]
    seeds = sorted(m["seed"].unique())
    modes = set(m["group"])

    a = m[m["assigned"]]
    per_bot = a.groupby(["group", "bot_id"])[cols].mean()   # seed 평균
    paired = m.groupby(["group", "bot_id"])[cols].mean()
    pop = paired.loc[BASELINE].mean() if BASELINE in modes else None
    rows = []
    for ma, mb in COMPARISONS:
        if not {ma, mb} <= modes:
            continue
        for col, name, kind, higher in metrics:
            xa, xb = per_bot.loc[ma, col].to_numpy(), per_bot.loc[mb, col].to_numpy()
            diff, lo, hi = ab_diff(xa, xb, rng, n_boot)
            seed_d = {}
            for s in seeds:
                q = a[a["seed"] == s].groupby("group")[col].mean()
                seed_d[f"seed{s}_diff"] = q[mb] - q[ma]
            pd_ = (paired.loc[mb, col] - paired.loc[ma, col]).to_numpy()
            pboot = _boot_means(pd_, rng, n_boot)
            pa, pb = float(np.nanmean(paired.loc[ma, col])), float(np.nanmean(paired.loc[mb, col]))
            r = {"comparison": f"{ma}_vs_{mb}", "metric": col, "name": name, "kind": kind, "higher_better": higher,
                 "a_mode": ma, "b_mode": mb, "a": np.nanmean(xa), "b": np.nanmean(xb),
                 "diff": diff, "ci_low": lo, "ci_high": hi,
                 "seed_diff_mean": np.mean(list(seed_d.values())),
                 "seed_diff_min": min(seed_d.values()), "seed_diff_max": max(seed_d.values()), **seed_d,
                 "paired_a": pa, "paired_b": pb, "paired_diff": float(np.nanmean(pd_)),
                 "paired_ci_low": float(np.percentile(pboot, 2.5)),
                 "paired_ci_high": float(np.percentile(pboot, 97.5)),
                 "n_a": int((~np.isnan(xa)).sum()), "n_b": int((~np.isnan(xb)).sum())}
            if pop is not None:
                p = float(pop[col])
                r.update({"pop_paired": p,
                          "lift_a": pa / p if p else np.nan, "lift_b": pb / p if p else np.nan})
            rows.append(r)
    summary = pd.DataFrame(rows)

    curve = []
    for col, _, kind, _ in metrics:
        if kind not in ("taste_curve", "reality_curve"):
            continue
        for mode in sorted(modes):
            x = per_bot.loc[mode, col].to_numpy()
            bm = _boot_means(x, rng, n_boot)
            curve.append({"metric": "pref" if kind == "taste_curve" else "near_sim", "group": mode,
                          "exposed": int(col.split("_")[1]), "mean": float(np.nanmean(x)),
                          "ci_low": float(np.percentile(bm, 2.5)), "ci_high": float(np.percentile(bm, 97.5))})
    return summary, pd.DataFrame(curve)


def bot_profiles(cfg: dict, pool: ItemPool, seeds) -> dict:
    """세션 때와 똑같이 봇 프로필을 다시 만든다 (학습 기간 데이터만 씀)."""
    from src.bots import build_profiles, load_bot_table, load_train_tx
    tx, table = load_train_tx(cfg), load_bot_table(cfg)
    return {s: {b.bot_id: b for b in build_profiles(tx, pool, table, cfg, s)} for s in seeds}


def reason_table(logs: pd.DataFrame) -> pd.DataFrame:
    """방식별·세트별 넘김 이유 비율 (배정된 봇)."""
    sk = logs[logs["assigned"] & (logs["action"] == "skip")]
    t = sk.pivot_table(index=["group", "set_no"], columns="reason", values="pos", aggfunc="size", fill_value=0)
    return t.div(t.sum(axis=1), axis=0)


def score(cfg: dict, split: str, seeds, out=None) -> dict:
    pool = load_pool(cfg)
    logs, finals = load_sessions(cfg, seeds)
    tau = cfg["scoring"]["near_threshold"]
    exposure = cfg["session"]["exposure"]
    mt, check = taste_metrics(logs, finals, bot_profiles(cfg, pool, seeds), pool, exposure)
    truths = build_truths(cfg, split, pool, logs["bot_id"].unique())
    m = session_metrics(logs, finals, truths, pool, tau, exposure)
    m = m.merge(mt, on=["seed", "group", "bot_id"], how="left", validate="one_to_one")
    assert m["pref_all"].notna().all()
    summary, curve = summarize(m, cfg)
    out = results_dir(cfg, split) if out is None else out
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "session_metrics.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    curve.to_csv(out / "learning_curve.csv", index=False)
    reasons = reason_table(logs)
    reasons.to_csv(out / "skip_reasons.csv")
    near_size = np.array([t.near(tau).sum() for t in truths.values()])
    meta = {"experiment": experiment_tag(cfg), "split": split, "seeds": [int(s) for s in seeds], "near_threshold": tau,
            "near_set_size_mean": float(near_size.mean()), "near_set_frac_mean": float(near_size.mean() / len(pool)),
            "n_sessions": len(m), "modes": sorted(m["group"].unique()), **check}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return {"summary": summary, "curve": curve, "reasons": reasons, "meta": meta}


# ---------------------------------------------------------------- 3.4 커버리지
def pool_coverage(answer_tx: pd.DataFrame, pool_ids: set[str], bot_ids=None) -> pd.DataFrame:
    """전체 / 봇 기준으로 정답 주 구매 중 상품 풀 밖 비율."""
    rows = []
    scopes = [("전체", answer_tx)]
    if bot_ids is not None:
        scopes.append(("봇 1,000명", answer_tx[answer_tx["customer_id"].isin(set(bot_ids))]))
    for name, t in scopes:
        out = ~t["article_id"].isin(pool_ids)
        arts = t["article_id"].drop_duplicates()
        per_cust = out.groupby(t["customer_id"]).all()   # 정답 주 구매가 전부 풀 밖인 고객
        rows.append({
            "scope": name,
            "customers": int(t["customer_id"].nunique()),
            "purchases": len(t),
            "purchases_outside_pool": int(out.sum()),
            "purchase_outside_rate": float(out.mean()),
            "articles": len(arts),
            "articles_outside_pool": int((~arts.isin(pool_ids)).sum()),
            "article_outside_rate": float((~arts.isin(pool_ids)).mean()),
            "customers_all_outside": int(per_cust.sum()),
            "customers_all_outside_rate": float(per_cust.mean()),
        })
    return pd.DataFrame(rows)


def coverage_markdown(t: pd.DataFrame) -> str:
    lines = ["## 상품 풀 커버리지 (정답 주 구매 중 상품 풀 밖 비율)", "",
             "| 기준 | 구매 건수 | 풀 밖 구매 | 상품 종류 | 풀 밖 상품 | 정답 주 구매가 전부 풀 밖인 고객 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in t.itertuples():
        lines.append(f"| {r.scope} | {r.purchases:,} | {r.purchases_outside_pool:,} ({r.purchase_outside_rate:.1%}) "
                     f"| {r.articles:,} | {r.articles_outside_pool:,} ({r.article_outside_rate:.1%}) "
                     f"| {r.customers_all_outside:,} / {r.customers:,} ({r.customers_all_outside_rate:.1%}) |")
    lines += ["", "풀 밖 구매는 두 그룹 모두 정확 적중이 불가능하다. 근접 정답 판정에는 영향이 적다."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI
def main(argv=None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["coverage", "calibrate", "score"])
    ap.add_argument("--split", choices=["valid", "test"], default="valid")
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["session"]["seeds"])
    ap.add_argument("--final", action="store_true", help="정답 주 최종 채점 확인 (test에 필수)")
    ap.add_argument("--out", type=Path, default=None, help="결과 폴더 (기본: 실험 버전별 results 폴더)")
    args = ap.parse_args(argv)
    interim, results = cfg["paths"]["interim"], cfg["paths"]["results"]

    if args.task == "coverage":
        pool_ids = set(pd.read_parquet(interim / "article_pool.parquet", columns=["article_id"])["article_id"])
        bots = pd.read_parquet(interim / "bots.parquet", columns=["group"])
        t = pool_coverage(load_answer_week(cfg), pool_ids, bots.index)
        results.mkdir(parents=True, exist_ok=True)
        t.to_csv(results / "coverage.csv", index=False)
        (results / "coverage.md").write_text(coverage_markdown(t), encoding="utf-8")
        print(coverage_markdown(t))
        return

    if args.task == "calibrate":
        t = calibrate(cfg)
        (results / "valid").mkdir(parents=True, exist_ok=True)
        t.to_csv(results / "valid" / "threshold_calibration.csv", index=False)
        print(t.round(4).to_string(index=False))
        return

    if args.split == "test":
        marker = results / "test" / "FINAL_DONE"
        if not args.final:
            raise SystemExit("정답 주 채점은 마지막에 한 번만 합니다. 확인했으면 --final 을 붙이세요.")
        if marker.exists():
            raise SystemExit(f"정답 주 채점을 이미 했습니다 ({marker.read_text().strip()}). "
                             "다시 하려면 이 파일을 직접 지우고 이유를 report.md에 남기세요.")
    if args.split == "test" and args.out is not None:
        raise SystemExit("정답 주 채점은 기본 결과 폴더에만 씁니다 (--out 불가).")
    r = score(cfg, args.split, args.seeds, out=args.out)
    if args.split == "test":
        marker.write_text(pd.Timestamp.now().isoformat(timespec="seconds"))
    s = r["summary"]
    print(json.dumps(r["meta"], ensure_ascii=False))
    show_cols = ["metric", "a", "b", "diff", "ci_low", "ci_high", "seed_diff_min", "seed_diff_max",
                 "paired_diff", "paired_ci_low", "paired_ci_high"] + [c for c in ("lift_a", "lift_b") if c in s]
    for comp, g in s.groupby("comparison", sort=False):
        for kind, title in (("taste", "취향 회복도 (메인)"), ("taste_curve", "선호율 곡선"),
                            ("reality", "다음 주 구매 (현실 점검)")):
            k = g[g["kind"] == kind]
            if len(k):
                print(f"\n[{comp}] {title}  (a={k['a_mode'].iat[0]}, b={k['b_mode'].iat[0]})")
                print(k[show_cols].round(4).to_string(index=False))
    print("\nskip reasons (assigned):\n", r["reasons"].round(3).to_string())


if __name__ == "__main__":
    main()
