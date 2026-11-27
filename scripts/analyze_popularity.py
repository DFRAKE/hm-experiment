"""분석 1: 다음 주 구매 맞히기에서 목록형이 이기는 이유 (인기 편향). **검증 주만** 쓴다 (정답 주는 이미 한 번 채점함).

질문
  (a) 고객이 다음 주에 실제로 산 옷은 인기 상품에 얼마나 몰려 있나?
  (b) 방식별 최종 추천 20개는 얼마나 인기 상품 위주인가?
  (c) 인기 상품은 '근접 정답'(다음 주 구매와 비슷한 옷)이 되기 쉬운가? (흔한 검정 티셔츠 효과)
  (d) 학습 기간에 인기 상품을 주로 산 고객 / 아닌 고객으로 나누면 차이가 어떻게 갈리나?
  (e) 최종 20개의 인기도가 높을수록 근접 적중이 오르나?

인기도 = 상품 풀의 popularity (학습 기간 마지막 7일 판매 수, 목록형 정렬 기준과 같음). 백분위는 풀 안 순위.
출력: data/results/analysis/popularity.{md,csv}, data/results/figures/fig5_popularity.png

실행: .venv/bin/python scripts/analyze_popularity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bots import load_bot_table, load_train_tx  # noqa: E402
from src.config import load_config, results_dir, sessions_dir  # noqa: E402
from src.items import load_pool  # noqa: E402
from src.scoring import build_truths, load_week  # noqa: E402

OUT = ROOT / "data" / "results" / "analysis"
FIG = ROOT / "data" / "results" / "figures"
SPLIT = "valid"
TOP = 0.05                    # '인기 상품' = 풀에서 인기도 상위 5%
N_BOOT, SEED = 1000, 0
BINS = [(0, 0), (1, 1), (2, 3), (4, 9), (10, 30), (31, 99), (100, 10**9)]
BIN_LABELS = ["0", "1", "2~3", "4~9", "10~30", "31~99", "100+"]
COMPS = [("list", "swipe", "원래 엔진"), ("list_v2", "swipe_v2", "같은 엔진 v2")]
MODE_KO = {"list": "목록 (인기순)", "swipe": "스와이프", "list_v2": "목록 v2", "swipe_v2": "스와이프 v2",
           "pop": "인기 기준선"}


def boot_diff(xa: np.ndarray, xb: np.ndarray, rng) -> tuple[float, float, float]:
    xa, xb = xa[~np.isnan(xa)], xb[~np.isnan(xb)]
    ba = xa[rng.integers(0, len(xa), (N_BOOT, len(xa)))].mean(1)
    bb = xb[rng.integers(0, len(xb), (N_BOOT, len(xb)))].mean(1)
    lo, hi = np.percentile(bb - ba, [2.5, 97.5])
    return float(xb.mean() - xa.mean()), float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    return "스와이프" if lo > 0 else ("목록" if hi < 0 else "비김")


def main() -> None:
    cfg = load_config()
    pool = load_pool(cfg)
    pop = pool.df["popularity"].to_numpy(dtype=float)
    pct = pd.Series(pop).rank(pct=True).to_numpy()          # 풀 안 인기 백분위 (동률 평균)
    top = pct > 1 - TOP
    table = load_bot_table(cfg)
    bots = table.index
    rng = np.random.default_rng(SEED)
    L = ["# 분석 1: 다음 주 구매 맞히기에서 목록형이 이기는 이유 (검증 주)", "",
         f"인기도 = 학습 기간 마지막 7일 판매 수 (목록형 정렬 기준). '인기 상품' = 상품 풀 {len(pool):,}개 중 상위 {TOP:.0%}"
         f" ({top.sum():,}개, 7일 판매 {pop[top].min():.0f}건 이상).", ""]
    rows = []

    # (a) 다음 주 실제 구매
    week = load_week(cfg, SPLIT)
    week = week[week["customer_id"].isin(set(bots))].drop_duplicates(["customer_id", "article_id"])
    inpool = week["article_id"].isin(pool.index)
    r = pool.rows(week.loc[inpool, "article_id"])
    buy_top, buy_pct = top[r].mean(), pct[r].mean()
    L += ["## (a) 다음 주에 실제로 산 옷", "",
          f"- 봇 {len(bots):,}명의 검증 주 구매 {len(week):,}건 중 상품 풀 안 {inpool.mean():.1%}.",
          f"- 그중 **인기 상위 {TOP:.0%} 상품이 {buy_top:.1%}** (아무렇게나 고르면 {TOP:.0%}, {buy_top / TOP:.1f}배). "
          f"평균 인기 백분위 {buy_pct:.2f} (아무렇게나 0.50).", ""]
    rows += [{"part": "a", "what": "next_week_top_share", "value": buy_top},
             {"part": "a", "what": "next_week_mean_pct", "value": buy_pct}]

    # (b) 방식별 최종 20개 인기도 (배정된 봇)
    sd = sessions_dir(cfg)
    seeds = cfg["session"]["seeds"]
    fin = pd.concat([pd.read_parquet(sd / f"final_seed{s}.parquet") for s in seeds], ignore_index=True)
    fr = pool.rows(fin["article_id"])
    fin["pct"], fin["top"] = pct[fr], top[fr]
    fb = fin[fin["assigned"]].groupby("group")[["pct", "top"]].mean()
    L += ["## (b) 방식별 최종 추천 20개의 인기도 (배정된 봇, seed 5개)", "",
          "| 방식 | 평균 인기 백분위 | 인기 상위 5% 비율 |", "|---|---:|---:|"]
    for g in ["list", "swipe", "list_v2", "swipe_v2"]:
        L.append(f"| {MODE_KO[g]} | {fb.at[g, 'pct']:.2f} | {fb.at[g, 'top']:.1%} |")
        rows += [{"part": "b", "what": f"final_pct_{g}", "value": fb.at[g, "pct"]},
                 {"part": "b", "what": f"final_top_{g}", "value": fb.at[g, "top"]}]
    allfin = fin.groupby("group")[["pct", "top"]].mean()        # 봇 전원 (pop 기준선 포함)
    if "pop" in allfin.index:
        L.append(f"| {MODE_KO['pop']} (참고, 봇 전원) | {allfin.at['pop', 'pct']:.2f} | {allfin.at['pop', 'top']:.1%} |")
    L.append("")

    # (c) 인기 상품이 근접 정답이 되기 쉬운가
    tau = cfg["scoring"]["near_threshold"]
    truths = build_truths(cfg, SPLIT, pool, bots)
    near_top, near_rest, by_dec = [], [], np.zeros(len(BINS))
    dec = np.digitize(pop, [b[1] + 1 for b in BINS[:-1]])
    cnt = np.bincount(dec, minlength=len(BINS))
    for t in truths.values():
        nr = t.near(tau)
        near_top.append(nr[top].mean())
        near_rest.append(nr[~top].mean())
        by_dec += np.bincount(dec, weights=nr, minlength=len(BINS)) / cnt
    by_dec /= len(truths)
    nt, nrest = float(np.mean(near_top)), float(np.mean(near_rest))
    L += ["## (c) 인기 상품은 근접 정답이 되기 쉬운가", "",
          f"고객 한 명 기준, 아무 상품 하나가 그 고객의 '근접 정답'(다음 주 구매와 설명 유사도 ≥ {tau} 또는 종류·색 같음)일 확률:",
          "", f"- 인기 상위 {TOP:.0%} 상품: **{nt:.2%}**", f"- 나머지 상품: {nrest:.2%}  → **{nt / nrest:.1f}배**", "",
          "| 7일 판매 수 | " + " | ".join(BIN_LABELS) + " |", "|---|" + "---:|" * len(BINS),
          "| 상품 수 | " + " | ".join(f"{c:,}" for c in cnt) + " |",
          "| 근접 정답일 확률 | " + " | ".join(f"{v:.2%}" for v in by_dec) + " |", ""]
    rows += [{"part": "c", "what": "near_rate_top", "value": nt}, {"part": "c", "what": "near_rate_rest", "value": nrest}]
    rows += [{"part": "c", "what": f"near_rate_bin_{BIN_LABELS[i]}", "value": v} for i, v in enumerate(by_dec)]

    # (c2) 근접 적중을 판정 방식별로 나누기: 설명 유사도만 / 종류·색 같음만 / 정확히 같은 상품
    fin_by = {k: pool.rows(g.sort_values("rank")["article_id"]) for k, g in fin.groupby(["seed", "group", "bot_id"])}
    rec = []
    for (s, g, b), fr_ in fin_by.items():
        t = truths[b]
        rec.append({"group": g, "assigned": (g in ("list", "list_v2", "pop")) == (table.at[b, "group"] == "A"),
                    "near": t.near(tau)[fr_].any(), "sim_only": (t.maxsim[fr_] >= tau).any(),
                    "pair_only": t.pair_match[fr_].any(),
                    "exact": bool(t.answer_ids & {pool.ids[i] for i in fr_})})
    rec = pd.DataFrame(rec)
    rc = rec[rec["assigned"]].groupby("group")[["near", "sim_only", "pair_only", "exact"]].mean()
    L += ["## (c2) 근접 적중은 어느 판정에서 나오나 (배정된 봇)", "",
          "근접 적중 = 최종 20개 중 하나라도 (설명 유사도 ≥ 기준) 또는 (종류·색이 다음 주 구매와 같음).", "",
          "| 방식 | 근접 적중 (둘 중 하나) | 설명 유사도 기준만 적용 | 종류·색 기준만 적용 | 정확히 같은 상품 |", "|---|---:|---:|---:|---:|"]
    for g in ["list", "swipe", "list_v2", "swipe_v2"]:
        L.append(f"| {MODE_KO[g]} | {rc.at[g, 'near']:.1%} | {rc.at[g, 'sim_only']:.1%} | {rc.at[g, 'pair_only']:.1%} "
                 f"| {rc.at[g, 'exact']:.1%} |")
        rows += [{"part": "c2", "what": f"{g}_{k}", "value": rc.at[g, k]} for k in rc.columns]
    L.append("")

    # (d) 고객의 인기 추종 성향으로 나누기
    tx = load_train_tx(cfg)
    tx = tx[tx["customer_id"].isin(set(bots)) & tx["article_id"].isin(pool.index)]
    tend = pd.Series(pct[pool.rows(tx["article_id"])], index=tx["customer_id"].to_numpy()).groupby(level=0).mean()
    tert = pd.qcut(tend, 3, labels=["낮음", "중간", "높음"])
    m = pd.read_parquet(results_dir(cfg, SPLIT) / "session_metrics.parquet")
    a = m[m["assigned"]].groupby(["group", "bot_id"])[["near_hit20", "pref_all", "final_like20"]].mean().reset_index()
    a["tert"] = a["bot_id"].map(tert)
    L += ["## (d) 학습 기간에 인기 상품을 얼마나 샀나로 고객을 3등분", "",
          "성향 = 학습 기간에 산 옷의 평균 인기 백분위. 차이 = 스와이프 − 목록 (95% 구간, 0을 포함하면 비김).", "",
          "| 비교 | 성향 | 고객 수 (목록/스와이프) | 목록 근접 적중 | 스와이프 근접 적중 | 차이 (%p) | 취향 선호율 차이 (%p) |",
          "|---|---|---:|---:|---:|---|---|"]
    for ma, mb, cname in COMPS:
        for tl in ["낮음", "중간", "높음"]:
            xa = a[(a.group == ma) & (a.tert == tl)]
            xb = a[(a.group == mb) & (a.tert == tl)]
            d, lo, hi = boot_diff(xa["near_hit20"].to_numpy(), xb["near_hit20"].to_numpy(), rng)
            dp, plo, phi = boot_diff(xa["pref_all"].to_numpy(), xb["pref_all"].to_numpy(), rng)
            L.append(f"| {cname} | {tl} | {len(xa)}/{len(xb)} | {xa['near_hit20'].mean():.1%} | {xb['near_hit20'].mean():.1%} "
                     f"| {d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}] {verdict(lo, hi)} "
                     f"| {dp * 100:+.1f} [{plo * 100:+.1f}, {phi * 100:+.1f}] {verdict(plo, phi)} |")
            rows.append({"part": "d", "what": f"{ma}_vs_{mb}", "tert": tl, "near_a": xa["near_hit20"].mean(),
                         "near_b": xb["near_hit20"].mean(), "near_diff": d, "near_lo": lo, "near_hi": hi,
                         "pref_diff": dp, "pref_lo": plo, "pref_hi": phi})
    L.append("")

    # (e) 최종 20개 인기도와 근접 적중 (세션 단위, 방식 안에서)
    fp = fin.groupby(["seed", "group", "bot_id"])["pct"].mean().rename("final_pct").reset_index()
    mm = m.merge(fp, on=["seed", "group", "bot_id"])
    L += ["## (e) 최종 20개가 인기 상품 위주일수록 근접 적중이 오르나 (같은 방식 안에서, 봇 전원 × seed)", "",
          "| 방식 | 최종 20개 인기 백분위 하위 1/3 | 중간 | 상위 1/3 |", "|---|---:|---:|---:|"]
    for g in ["list", "swipe", "list_v2", "swipe_v2"]:
        x = mm[mm.group == g]
        q = pd.qcut(x["final_pct"].rank(method="first"), 3, labels=False)
        v = x.groupby(q)["near_hit20"].mean()
        L.append(f"| {MODE_KO[g]} | {v[0]:.1%} | {v[1]:.1%} | {v[2]:.1%} |")
        rows += [{"part": "e", "what": g, "tert": str(i), "value": v[i]} for i in range(3)]
    L.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "popularity.csv", index=False)
    (OUT / "popularity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    fig(by_dec, fb, buy_top, pd.DataFrame([r for r in rows if r["part"] == "d"]))
    print("\n".join(L))


def fig(by_dec, fb, buy_top, d):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_figures as mf          # 같은 색·글꼴
    plt = mf.plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ax.bar(range(len(BINS)), by_dec * 100, color=mf.INK2, width=0.7)
    ax.set_xticks(range(len(BINS)), BIN_LABELS, fontsize=8.5)
    ax.set_xlabel("상품의 학습 기간 마지막 7일 판매 수")
    ax.set_ylabel("근접 정답일 확률 (%)")
    ax.set_title("인기 상품일수록 '다음 주 구매와 비슷한 옷'이 되기 쉽다", loc="left")
    ax = axes[1]
    gs = ["list", "swipe", "list_v2", "swipe_v2"]
    cols = [mf.LIST, mf.SWIPE, mf.LIST, mf.SWIPE]
    ax.bar(range(4), fb.loc[gs, "top"] * 100, color=cols, width=0.6)
    for i, g in enumerate(gs):
        ax.text(i, fb.at[g, "top"] * 100, f"{fb.at[g, 'top']:.0%}", ha="center", va="bottom", fontsize=9, color=mf.INK)
    ax.axhline(buy_top * 100, color=mf.INK, lw=1, ls="--")
    ax.text(1, buy_top * 100 + 1.5, f"실제 다음 주 구매 {buy_top:.0%}", ha="center", va="bottom", fontsize=9, color=mf.INK,
            bbox=dict(boxstyle="round,pad=0.2", fc=mf.SURF, ec="none"))
    ax.axhline(TOP * 100, color=mf.INK2, lw=1, ls=":")
    ax.set_xticks(range(4), ["목록\n(인기순)", "스와이프", "목록 v2", "스와이프 v2"])
    ax.set_ylabel("최종 20개 중 인기 상위 5% 비율 (%)")
    ax.set_title("최종 추천의 인기 상품 비율", loc="left")
    ax = axes[2]
    tl = ["낮음", "중간", "높음"]
    for k, (comp, color, name) in enumerate([("list_vs_swipe", mf.V1, "원래 엔진"), ("list_v2_vs_swipe_v2", mf.V2, "같은 엔진 v2")]):
        x = d[d.what == comp].set_index("tert").loc[tl]
        xs = np.arange(3) + (k - 0.5) * 0.18
        ax.errorbar(xs, x.near_diff * 100, yerr=[(x.near_diff - x.near_lo) * 100, (x.near_hi - x.near_diff) * 100],
                    fmt="o", color=color, ms=7, capsize=3, lw=1.5, label=name)
    mf.zero(ax)
    ax.set_xticks(range(3), [f"인기 추종 {t}" for t in tl])
    ax.set_ylabel("근접 적중 차이 (%p, + = 스와이프 우세)")
    ax.set_title("고객 성향별 근접 적중 차이", loc="left")
    ax.legend(loc="lower left")
    fig.suptitle("⑤ 다음 주 구매 맞히기에서 목록형이 이기는 이유: 인기 편향 (검증 주)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=mf.INK)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig5_popularity.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
