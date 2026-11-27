"""분석 2: 취향 찾기와 다음 주 구매 맞히기를 둘 다 잡는 설정이 있나 (이미 돌린 조건, 검증 주).

봇은 그대로(메인 고객·메인 봇 규칙)이고 **추천기 설정만 바꾼 조건**만 쓴다. 봇 설정이 바뀌면 '좋아할 옷 비율'의 눈금 자체가
바뀌어 같은 그림에 놓을 수 없기 때문이다.
  가로 = 60장 중 봇이 좋아할 옷 비율 (취향, 메인 지표), 세로 = 최종 20개 근접 적중률 (다음 주 구매, 현실 점검)
출력: data/results/analysis/tradeoff.{md,csv}, data/results/figures/fig6_tradeoff.png

실행: .venv/bin/python scripts/analyze_tradeoff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_figures as mf  # noqa: E402

RES = ROOT / "data" / "results"
OUT = RES / "analysis"
REC_FACTORS = ["wpop", "negw", "explore", "variation", "insert", "explore_x_wpop"]
REC_PRESETS = ["tag-0", "tag-0p6"]
MODE_KO = {"list": "목록 (인기순)", "swipe": "스와이프", "list_v2": "목록 v2", "swipe_v2": "스와이프 v2"}


def load() -> pd.DataFrame:
    t = pd.read_csv(RES / "all_runs_valid.csv")
    keep = ((t.kind == "main") | t.factor.isin(REC_FACTORS) | t.run.isin(REC_PRESETS)) & (t.kind != "fold")
    t = t[keep & t.metric.isin(["pref_all", "near_hit20"])].drop_duplicates(["run", "comparison", "metric"])
    rows = []
    for (run, comp), g in t.groupby(["run", "comparison"]):
        g = g.set_index("metric")
        a, b = comp.split("_vs_")
        info = g.iloc[0]
        for mode, col in ((a, "a"), (b, "b")):
            rows.append({"run": run, "factor": info.factor, "value": info.value, "label": info.label, "mode": mode,
                         "pref": g.at["pref_all", col], "near": g.at["near_hit20", col]})
    return pd.DataFrame(rows).drop_duplicates(["run", "mode"])


def pareto(d: pd.DataFrame) -> pd.Series:
    """취향·근접 적중 둘 다에서 자기보다 나은 점이 없는 점."""
    p, n = d.pref.to_numpy(), d.near.to_numpy()
    dom = ((p[None, :] >= p[:, None]) & (n[None, :] >= n[:, None]) & ((p[None, :] > p[:, None]) | (n[None, :] > n[:, None])))
    return pd.Series(~dom.any(1), index=d.index)


def main() -> None:
    d = load()
    d["pareto"] = pareto(d)
    main_ = d[d.run == "main"].set_index("mode")
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "tradeoff.csv", index=False)

    L = ["# 분석 2: 취향 찾기와 다음 주 구매 맞히기를 둘 다 잡는 설정이 있나 (검증 주)", "",
         f"봇·고객은 메인 그대로, 추천기 설정만 바꾼 조건 {d.run.nunique()}개 × 방식 4개 = 점 {len(d)}개.",
         "취향 = 60장 중 봇이 좋아할 옷 비율, 구매 = 최종 20개 근접 적중률.", "",
         "## 메인 조건", "", "| 방식 | 취향 | 구매 |", "|---|---:|---:|"]
    for m in MODE_KO:
        L.append(f"| {MODE_KO[m]} | {main_.at[m, 'pref']:.1%} | {main_.at[m, 'near']:.1%} |")
    sw = d[d["mode"] == "swipe_v2"]
    best_near = sw.sort_values("near", ascending=False).head(5)
    L += ["", "## 스와이프 v2에서 구매 적중이 가장 높은 설정 5개", "", "| 조건 | 취향 | 구매 |", "|---|---:|---:|"]
    for r in best_near.itertuples():
        L.append(f"| {r.run} | {r.pref:.1%} | {r.near:.1%} |")
    both = sw[(sw.pref > main_.at["list_v2", "pref"]) & (sw.near >= main_.at["list_v2", "near"])]
    L += ["", f"**메인 목록 v2(취향 {main_.at['list_v2', 'pref']:.1%}, 구매 {main_.at['list_v2', 'near']:.1%})보다 "
          f"두 지표 모두 같거나 나은 스와이프 v2 설정: {len(both)}개**"]
    for r in both.itertuples():
        L.append(f"- {r.run}: 취향 {r.pref:.1%}, 구매 {r.near:.1%}")
    both1 = d[(d["mode"].str.startswith("swipe")) & (d.near >= main_.at["list", "near"])]
    L += [f"- 원래 목록(인기순, 구매 {main_.at['list', 'near']:.1%})만큼 구매를 맞히는 스와이프 설정: {len(both1)}개", ""]
    # wpop 따라 변하는 모습
    wp = d[(d.factor == "wpop") | (d.run == "main")].copy()
    wp["w"] = np.where(wp.run == "main", 0.1, pd.to_numeric(wp.value, errors="coerce"))
    L += ["## 인기도 가중치(w_pop)를 올리면 (v2 둘 다)", "",
          "| w_pop | 목록 v2 취향 | 목록 v2 구매 | 스와이프 v2 취향 | 스와이프 v2 구매 |", "|---:|---:|---:|---:|---:|"]
    for w, g in wp[wp["mode"].isin(["list_v2", "swipe_v2"])].groupby("w"):
        g = g.set_index("mode")
        L.append(f"| {w:g} | {g.at['list_v2', 'pref']:.1%} | {g.at['list_v2', 'near']:.1%} "
                 f"| {g.at['swipe_v2', 'pref']:.1%} | {g.at['swipe_v2', 'near']:.1%} |")
    # 같은 w_pop에서 v2 두 방식 차이와 판정 (부트스트랩 95% 구간)
    t = pd.read_csv(RES / "all_runs_valid.csv")
    t = t[(t.kind != "fold") & (t.comparison == "list_v2_vs_swipe_v2")
          & ((t.factor == "wpop") | (t.run == "main") | t.run.str.startswith("explore-0__wpop"))]
    t = t.drop_duplicates(["run", "metric"]).set_index(["run", "metric"])
    ko = {"swipe": "스와이프", "list": "목록", "tie": "비김"}
    L += ["", "## 같은 w_pop에서 스와이프 v2 − 목록 v2 (95% 구간, 0을 포함하면 비김)", "",
          "| 조건 | 취향 차이 (%p) | 최종 20개 중 좋아할 옷 (개) | 구매 근접 적중 차이 (%p) |", "|---|---|---|---|"]
    runs = ["wpop-0", "wpop-0p05", "main", "wpop-0p2", "wpop-0p3", "wpop-0p5", "wpop-0p75", "wpop-1", "wpop-2",
            "explore-0__wpop-0", "explore-0__wpop-0p3", "explore-0__wpop-1"]
    for run in runs:
        c = []
        for m, k in (("pref_all", 100), ("final_like20", 1), ("near_hit20", 100)):
            r = t.loc[(run, m)]
            c.append(f"{r['diff'] * k:+.1f} [{r.ci_low * k:+.1f}, {r.ci_high * k:+.1f}] {ko[r.verdict]}")
        L.append(f"| {'main (w_pop 0.1)' if run == 'main' else run} | " + " | ".join(c) + " |")
    pf = d[d.pareto].sort_values("pref").drop_duplicates(["mode", "pref", "near"])
    pf.loc[pf["mode"] == "list", "run"] = "(모든 조건 같음: 원래 목록은 추천기 설정을 안 씀)"
    L += ["", "## 둘 다에서 더 나은 점이 없는 설정 (파레토 경계)", "", "| 조건 | 방식 | 취향 | 구매 |", "|---|---|---:|---:|"]
    for r in pf.itertuples():
        L.append(f"| {r.run} | {MODE_KO[r.mode]} | {r.pref:.1%} | {r.near:.1%} |")
    (OUT / "tradeoff.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    fig(d, wp)


def fig(d, wp):
    plt = mf.plt
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    color = {"list": mf.LIST, "list_v2": mf.LIST, "swipe": mf.SWIPE, "swipe_v2": mf.SWIPE}
    marker = {"list": "s", "swipe": "s", "list_v2": "o", "swipe_v2": "o"}
    for m in MODE_KO:
        x = d[d["mode"] == m]
        ax.scatter(x.pref * 100, x.near * 100, s=22, color=color[m], marker=marker[m], alpha=0.35, lw=0)
    for m in ("list_v2", "swipe_v2"):
        x = wp[wp["mode"] == m].sort_values("w")
        ax.plot(x.pref * 100, x.near * 100, color=color[m], lw=1.5, marker="o", ms=4)
        for r in x.itertuples():
            if r.w in (0, 0.1, 0.3, 1, 2):
                ax.annotate(f"{r.w:g}", (r.pref * 100, r.near * 100), xytext=(4, 3), textcoords="offset points",
                            fontsize=8, color=mf.INK2)
    mm = d[d.run == "main"].set_index("mode")
    for m in MODE_KO:
        ax.scatter(mm.at[m, "pref"] * 100, mm.at[m, "near"] * 100, s=90, color=color[m], marker=marker[m],
                   edgecolor=mf.SURF, lw=2, zorder=5)
        ax.annotate(MODE_KO[m] + " (메인)", (mm.at[m, "pref"] * 100, mm.at[m, "near"] * 100), xytext=(8, -4),
                    textcoords="offset points", fontsize=9, color=mf.INK, fontweight="bold")
    ax.set_xlabel("취향: 60장 중 봇이 좋아할 옷 비율 (%)  →  오른쪽이 좋음")
    ax.set_ylabel("구매: 최종 20개 근접 적중률 (%)  →  위가 좋음")
    ax.set_title("⑥ 취향과 다음 주 구매는 맞바꾸는 관계 (추천기 설정만 바꾼 조건, 검증 주)\n"
                 "□ 원래 엔진, ○ v2 엔진, 선 = 인기도 가중치 w_pop (숫자)", loc="left")
    fig.savefig(mf.OUT / "fig6_tradeoff.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
