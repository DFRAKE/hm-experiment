"""3장 데이터 준비 (임베딩 제외).

3.2 기간 자르기  -> data/interim/tx_2020q3.parquet
3.3 봇 후보 / 묶음 / 1,000명 뽑기 / A·B 배정 / 균형 확인
3.4 상품 풀 (학습 기간 판매 상품만; 태그, 대표 가격, 인기도) -> data/interim/article_pool.parquet

실행: python -m src.prepare [--force-cut]
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

from src.config import load_config

TX_DTYPES = {"t_dat": str, "customer_id": str, "article_id": str,
             "price": "float64", "sales_channel_id": "int8"}
ARTICLE_COLS = ["article_id", "prod_name", "product_type_name", "product_group_name",
                "colour_group_name", "index_group_name", "section_name",
                "garment_group_name", "detail_desc"]
TAG_COLS = ["product_type_name", "colour_group_name", "index_group_name", "garment_group_name"]
GROUPS = ("A", "B")  # A = 목록형, B = 스와이프형


# ---------------------------------------------------------------- 3.2 기간 자르기
def cut_transactions(cfg: dict, force: bool = False) -> pd.DataFrame:
    out = cfg["paths"]["interim"] / "tx_2020q3.parquet"
    if out.exists() and not force:
        print(f"skip cut ({out.name} already exists)")
        return pd.read_parquet(out)
    start = cfg["periods"]["cut_start"]
    parts = []
    reader = pd.read_csv(cfg["paths"]["raw"] / "transactions_train.csv",
                         dtype=TX_DTYPES, chunksize=cfg["read"]["chunksize"])
    for ch in reader:
        parts.append(ch[ch["t_dat"] >= start])   # 'YYYY-MM-DD' 문자열 비교
    tx = pd.concat(parts, ignore_index=True)
    tx["t_dat"] = pd.to_datetime(tx["t_dat"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(out, index=False)
    print(f"cut {len(tx):,} rows -> {out}")
    return tx


def split_periods(tx: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "valid", "test"):
        lo, hi = cfg["periods"][name]
        out[name] = tx[tx["t_dat"].between(pd.Timestamp(lo), pd.Timestamp(hi))]
    return out


def load_articles(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(cfg["paths"]["raw"] / "articles.csv", dtype={"article_id": str},
                       usecols=ARTICLE_COLS)


def load_ages(cfg: dict) -> pd.Series:
    c = pd.read_csv(cfg["paths"]["raw"] / "customers.csv", usecols=["customer_id", "age"],
                    dtype={"customer_id": str})
    return c.set_index("customer_id")["age"]


def _check_train_only(tx: pd.DataFrame, train_end: str, who: str) -> None:
    if len(tx) and tx["t_dat"].max() > pd.Timestamp(train_end):
        raise ValueError(f"{who}: 학습 기간 이후 거래가 들어옴 (max={tx['t_dat'].max().date()})")


# ---------------------------------------------------------------- 3.3 봇 후보
def bot_features(train_tx: pd.DataFrame, articles: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """학습 기간 거래만으로 봇 특징을 만든다 (누수 방지 검사 포함)."""
    _check_train_only(train_tx, train_end, "bot_features")
    t = train_tx.merge(articles[["article_id", "index_group_name", "product_type_name"]],
                       on="article_id", how="left")
    g = t.groupby("customer_id")
    feat = pd.DataFrame({
        "n_train": g.size(),
        "diversity": g["product_type_name"].nunique(),
        "price_median": g["price"].median(),
    })
    # 주 카테고리: 최빈값. 동률이면 더 최근에 산 카테고리, 그래도 같으면 이름순.
    cc = (t.groupby(["customer_id", "index_group_name"])
          .agg(n=("t_dat", "size"), last=("t_dat", "max")).reset_index()
          .sort_values(["customer_id", "n", "last", "index_group_name"],
                       ascending=[True, False, False, True]))
    feat["main_cat"] = cc.drop_duplicates("customer_id").set_index("customer_id")["index_group_name"]
    return feat


def select_candidates(periods: dict, cfg: dict) -> tuple[pd.Index, list[dict]]:
    b = cfg["bots"]
    n_tr = periods["train"]["customer_id"].value_counts()
    n_va = periods["valid"]["customer_id"].value_counts()
    n_te = periods["test"]["customer_id"].value_counts()
    everyone = pd.Index(pd.concat([p["customer_id"] for p in periods.values()]).unique())

    def cnt(s, ids):
        return s.reindex(ids, fill_value=0)

    funnel = [{"step": "0. 2020-07-01~09-22 구매 고객", "n": len(everyone)}]
    ids = everyone[cnt(n_tr, everyone).values >= b["min_train_purchases"]]
    funnel.append({"step": f"1. 학습 기간 구매 ≥ {b['min_train_purchases']}", "n": len(ids)})
    ids = ids[cnt(n_va, ids).values >= b["min_valid_purchases"]]
    funnel.append({"step": f"2. + 검증 주 구매 ≥ {b['min_valid_purchases']}", "n": len(ids)})
    ids = ids[cnt(n_te, ids).values >= b["min_test_purchases"]]
    funnel.append({"step": f"3. + 정답 주 구매 ≥ {b['min_test_purchases']}", "n": len(ids)})
    ids = ids[cnt(n_tr, ids).values <= b["max_train_purchases"]]
    funnel.append({"step": f"4. + 학습 기간 구매 ≤ {b['max_train_purchases']}", "n": len(ids)})
    return ids.sort_values(), funnel


def count_labels(bins: list[int]) -> list[tuple[str, int, int | None]]:
    edges = list(bins) + [None]
    return [(f"{lo}-{hi - 1}" if hi else f"{lo}+", lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]


def make_strata(cand: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """카테고리 × 구매 수 구간 × 다양성으로 묶고, 작은 묶음은 같은 카테고리의 인접 구매 수 구간과 합친다."""
    b = cfg["bots"]
    labels = count_labels(b["count_bins"])
    cand = cand.copy()
    cut_edges = list(b["count_bins"]) + [np.inf]
    cand["count_bin"] = pd.cut(cand["n_train"], cut_edges, right=False,
                               labels=[l[0] for l in labels]).astype(str)
    med = cand["diversity"].median()
    cand["div_level"] = np.where(cand["diversity"] <= med, "low", "high")
    cand["stratum"] = ""
    merges = []
    for (cat, div), sub in cand.groupby(["main_cat", "div_level"]):
        groups = [[i] for i in range(len(labels))]
        size = lambda grp: int(sub["count_bin"].isin([labels[i][0] for i in grp]).sum())
        while len(groups) > 1:
            sizes = [size(g) for g in groups]
            small = [i for i, s in enumerate(sizes) if s < b["min_stratum_size"]]
            if not small:
                break
            i = min(small, key=lambda k: (sizes[k], k))
            nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
            j = min(nbrs, key=lambda k: (sizes[k], k))
            lo_i, hi_i = sorted((i, j))
            merged = groups[lo_i] + groups[hi_i]
            merges.append({"category": cat, "diversity": div,
                           "merged": " + ".join(f"{labels[k][0]}({size([k])})" for k in merged),
                           "reason": f"{labels[groups[i][0]][0]} 구간 {sizes[i]}명 < {b['min_stratum_size']}"})
            groups[lo_i] = merged
            del groups[hi_i]
        for grp in groups:
            lo = labels[grp[0]][1]
            hi = labels[grp[-1]][2]
            cl = f"{lo}-{hi - 1}" if hi else f"{lo}+"
            mask = (cand["main_cat"] == cat) & (cand["div_level"] == div) & \
                   cand["count_bin"].isin([labels[k][0] for k in grp])
            cand.loc[mask, "stratum"] = f"{cat} | {cl} | {div}"
        if len(groups) == 1 and size(groups[0]) < b["min_stratum_size"]:
            warnings.warn(f"{cat}/{div}: 구매 수 구간을 모두 합쳐도 {size(groups[0])}명 < "
                          f"{b['min_stratum_size']} (그대로 둠)")
    return cand, merges


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """비례 배분 (최대 잔여법)."""
    quota = sizes / sizes.sum() * n
    alloc = np.floor(quota).astype(int)
    rest = int(n - alloc.sum())
    order = (quota - alloc).sort_values(ascending=False, kind="stable").index[:rest]
    alloc[order] += 1
    return alloc


def sample_and_assign(cand: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = cfg["bots"]
    rng = np.random.default_rng(b["seed"])
    sizes = cand["stratum"].value_counts().sort_index()
    alloc = allocate(sizes, b["n_bots"])
    picked = []
    for s in sizes.index:
        ids = np.sort(cand.index[cand["stratum"] == s].to_numpy())
        picked.append(rng.choice(ids, size=int(alloc[s]), replace=False))
    bots = cand.loc[np.concatenate(picked)].copy()

    # 층화 무작위 배정: 묶음 안에서 섞고 A/B 번갈아. 홀수 묶음마다 시작 그룹을 바꿔 전체 500/500 유지.
    bots["group"] = ""
    start = 0
    for s in sizes.index:
        ids = bots.index[bots["stratum"] == s].to_numpy()
        perm = rng.permutation(np.sort(ids))
        for k, cid in enumerate(perm):
            bots.loc[cid, "group"] = GROUPS[(start + k) % 2]
        if len(perm) % 2:
            start = 1 - start
    table = pd.DataFrame({"candidates": sizes, "sampled": alloc})
    gc = bots.groupby(["stratum", "group"]).size().unstack(fill_value=0)
    table = table.join(gc.reindex(columns=list(GROUPS), fill_value=0)).fillna(0).astype(int)
    return bots, table


def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    sd = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def smd_binary(p1: float, p2: float) -> float:
    sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return float((p1 - p2) / sd) if sd > 0 else 0.0


def balance_table(bots: pd.DataFrame, threshold: float) -> pd.DataFrame:
    A, B = (bots[bots["group"] == g] for g in GROUPS)
    rows = []
    for col, name in [("n_train", "학습 기간 구매 수 (평균)"), ("diversity", "취향 다양성 (평균)"),
                      ("price_median", "가격 중앙값 (평균)"), ("age", "나이 (평균, 결측 제외)")]:
        rows.append({"item": name, "A": A[col].mean(), "B": B[col].mean(),
                     "SMD": smd_continuous(A[col], B[col])})
    for cat in sorted(bots["main_cat"].unique()):
        pa, pb = (A["main_cat"] == cat).mean(), (B["main_cat"] == cat).mean()
        rows.append({"item": f"주 카테고리 비율: {cat}", "A": pa, "B": pb, "SMD": smd_binary(pa, pb)})
    t = pd.DataFrame(rows)
    t["abs_SMD"] = t["SMD"].abs()
    t["ok"] = t["abs_SMD"] < threshold
    return t


def balance_markdown(t: pd.DataFrame, bots: pd.DataFrame, threshold: float) -> str:
    def fmt(item, v):
        if "비율" in item:
            return f"{v:.1%}"
        if "가격" in item:
            return f"{v:.4f}"
        return f"{v:.2f}"
    nA, nB = (bots["group"] == "A").sum(), (bots["group"] == "B").sum()
    lines = ["## 그룹 균형 확인", "",
             f"그룹 A(목록형) {nA}명 / 그룹 B(스와이프형) {nB}명. 기준: |SMD| < {threshold}", "",
             "| 항목 | 그룹 A | 그룹 B | SMD | 통과 |", "| --- | ---: | ---: | ---: | :---: |"]
    for r in t.itertuples():
        lines.append(f"| {r.item} | {fmt(r.item, r.A)} | {fmt(r.item, r.B)} | {r.SMD:+.3f} | "
                     f"{'✅' if r.ok else '⚠️'} |")
    miss = bots.groupby("group")["age"].apply(lambda s: s.isna().sum())
    lines += ["", f"나이 결측: A {miss.get('A', 0)}명, B {miss.get('B', 0)}명. "
                  "가격은 H&M 원본 단위(실제 통화 아님)로 상대 비교에만 쓴다."]
    bad = t[~t["ok"]]
    if len(bad):
        lines += ["", "⚠️ **경고**: 아래 항목은 SMD ≥ 기준이라 결과 분석 때 보정 변수로 함께 보고한다 "
                      "(seed를 바꿔 다시 배정하지 않음).", ""]
        lines += [f"- {r.item} (SMD {r.SMD:+.3f})" for r in bad.itertuples()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 3.4 상품 풀
def pool_stats(train_tx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """대표 가격(학습 기간 중앙값)과 인기도(학습 기간 마지막 7일 판매 수). 학습 기간 거래만 사용."""
    _check_train_only(train_tx, cfg["periods"]["train"][1], "pool_stats")
    lo, hi = cfg["pool"]["popularity_window"]
    w = train_tx[train_tx["t_dat"].between(pd.Timestamp(lo), pd.Timestamp(hi))]
    return pd.DataFrame({"price": train_tx.groupby("article_id")["price"].median(),
                         "popularity": w["article_id"].value_counts()})


def build_pool(train_tx: pd.DataFrame, articles: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """학습 기간에 1건 이상 팔린 상품만 (검증·정답 주에만 팔린 상품은 존재 자체가 미래 정보라 제외).

    학습 기간 거래만 받으므로 미래 상품이 들어올 경로가 없다.
    """
    stats = pool_stats(train_tx, cfg)
    pool = articles[articles["article_id"].isin(stats.index)].copy()
    pool["price"] = pool["article_id"].map(stats["price"])
    pool["popularity"] = pool["article_id"].map(stats["popularity"]).fillna(0).astype(int)
    return pool.sort_values("article_id").reset_index(drop=True)


# ---------------------------------------------------------------- main
def build_all(tx: pd.DataFrame, articles: pd.DataFrame, ages: pd.Series, cfg: dict) -> dict:
    """파일 입출력 없이 3.3~3.4 계산 전체를 수행한다 (테스트에서 가짜 데이터로도 호출)."""
    periods = split_periods(tx, cfg)
    ids, funnel = select_candidates(periods, cfg)
    train_end = cfg["periods"]["train"][1]
    train_c = periods["train"][periods["train"]["customer_id"].isin(ids)]
    cand = bot_features(train_c, articles, train_end).loc[ids]
    cand["age"] = ages.reindex(cand.index).to_numpy()
    cand.index.name = "customer_id"
    cand, merges = make_strata(cand, cfg)
    bots, strata = sample_and_assign(cand, cfg)

    thr = cfg["bots"]["smd_threshold"]
    bal = balance_table(bots, thr)
    if (~bal["ok"]).any():
        warnings.warn("균형 확인: SMD ≥ %.2f 항목 있음 -> %s" % (thr, list(bal.loc[~bal["ok"], "item"])))

    pool = build_pool(periods["train"], articles, cfg)
    return {"periods": periods, "funnel": funnel, "merges": merges, "cand": cand,
            "bots": bots, "strata": strata, "bal": bal, "pool": pool}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-cut", action="store_true", help="tx_2020q3.parquet 다시 만들기")
    args = ap.parse_args(argv)
    cfg = load_config()
    interim, results = cfg["paths"]["interim"], cfg["paths"]["results"]
    interim.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    tx = cut_transactions(cfg, force=args.force_cut)
    r = build_all(tx, load_articles(cfg), load_ages(cfg), cfg)
    periods, funnel, merges = r["periods"], r["funnel"], r["merges"]
    cand, bots, strata, bal, pool = r["cand"], r["bots"], r["strata"], r["bal"], r["pool"]
    thr = cfg["bots"]["smd_threshold"]

    cand.to_parquet(interim / "bot_candidates.parquet")
    bots.to_parquet(interim / "bots.parquet")
    strata.rename_axis("stratum").to_csv(interim / "strata.csv")
    bal.to_csv(results / "balance.csv", index=False)
    (results / "balance.md").write_text(balance_markdown(bal, bots, thr), encoding="utf-8")
    pool.to_parquet(interim / "article_pool.parquet", index=False)

    tx_path = interim / "tx_2020q3.parquet"
    summary = {
        "tx_rows": len(tx), "tx_file_mb": round(tx_path.stat().st_size / 1e6, 1),
        "tx_range": [str(tx["t_dat"].min().date()), str(tx["t_dat"].max().date())],
        "period_rows": {k: len(v) for k, v in periods.items()},
        "funnel": funnel, "diversity_median": float(cand["diversity"].median()),
        "merges": merges, "n_strata": int(len(strata)),
        "groups": bots["group"].value_counts().sort_index().to_dict(),
        "pool": {"total": len(pool),
                 "popularity_gt0": int((pool["popularity"] > 0).sum()),
                 "by_index_group": pool["index_group_name"].value_counts().to_dict()},
    }
    (interim / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
