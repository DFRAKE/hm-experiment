"""분석 3-2, 4: 유행 따르는 봇으로 돌린 결과 + seed 20개 결과 (검증 주). all_runs_valid.csv 에서 뽑아 정리만 한다.

출력: data/results/analysis/popbots.md, data/results/figures/fig7_popbots.png
실행: .venv/bin/python scripts/analyze_popbots.py   (scripts/sweep.py 로 popw-*, seeds-20 을 돌린 뒤)
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
RUNS = [("main", "메인 봇 (유행 안 따름)"), ("popw-0p5", "유행 β=0.5"), ("popw-1", "유행 β=1"), ("popw-2", "유행 β=2"),
        ("popw-1-personal", "유행 β=1, 사람마다 다름"), ("popw-1__wpop-0", "유행 β=1 × v2 인기도 가중치 0"),
        ("popw-1__wpop-0p3", "유행 β=1 × v2 인기도 가중치 0.3"), ("popw-1__wpop-1", "유행 β=1 × v2 인기도 가중치 1")]
COMP = [("list_vs_swipe", "원래 엔진"), ("list_v2_vs_swipe_v2", "같은 엔진 v2")]
METRICS = [("pref_all", "60장 중 좋아할 옷 비율", 100, "%p"), ("final_like20", "최종 20개 중 좋아할 옷", 1, "개"),
           ("near_hit20", "다음 주 구매 근접 적중", 100, "%p")]
KO = {"swipe": "스와이프", "list": "목록", "tie": "비김"}


def cell(r, k):
    return f"{r.a * k:.1f} → {r.b * k:.1f} ({r['diff'] * k:+.1f} [{r.ci_low * k:+.1f}, {r.ci_high * k:+.1f}] {KO[r.verdict]})"


def main() -> None:
    t = pd.read_csv(RES / "all_runs_valid.csv").drop_duplicates(["run", "comparison", "metric"])
    t = t.set_index(["run", "comparison", "metric"])
    cal = pd.read_csv(OUT / "calibrate_pop.csv")
    auc = {0.0: cal.auc.iat[0], **{r.pop_weight: r.auc for r in cal[cal.pop_mode == "fixed"].itertuples()}}
    L = ["# 분석 3-2: 유행 따르는 봇으로 다시 돌리기 (검증 주)", "",
         "봇의 찜 판단 점수 = 사진 취향 유사도 + β × 상품 인기도. 메인 봇은 β = 0.",
         f"실제 구매 설명력(AUC, scripts/calibrate_pop.py): β=0 {auc[0.0]:.3f}, β=0.5 {auc[0.5]:.3f}, β=1 {auc[1.0]:.3f}, "
         f"β=2 {auc[2.0]:.3f}.", "", "칸 = 목록 → 스와이프 (차이 [95% 구간] 판정).", ""]
    for comp, cname in COMP:
        L += [f"## {cname}", "", "| 조건 | " + " | ".join(m[1] for m in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for run, name in RUNS:
            if comp == "list_vs_swipe" and "__wpop" in run:
                continue                      # 원래 엔진은 v2 인기도 가중치를 쓰지 않음 (popw-1 과 같음)
            L.append(f"| {name} | " + " | ".join(cell(t.loc[(run, comp, m)], k) for m, _, k, _ in METRICS) + " |")
        L.append("")
    # 다른 고객 7묶음
    L += ["## 유행 β=1 봇을 다른 고객 7묶음으로 반복", "", "| 묶음 | 원래 엔진 취향 차이 | 같은 엔진 v2 취향 차이 | v2 최종 20개 차이 | v2 구매 근접 적중 차이 |",
          "|---|---|---|---|---|"]
    cnt = {"swipe": 0, "tie": 0, "list": 0}
    for k in range(1, 8):
        run = f"popw-1__f{k}"
        a = t.loc[(run, "list_vs_swipe", "pref_all")]
        b = t.loc[(run, "list_v2_vs_swipe_v2", "pref_all")]
        c = t.loc[(run, "list_v2_vs_swipe_v2", "final_like20")]
        d = t.loc[(run, "list_v2_vs_swipe_v2", "near_hit20")]
        cnt[b.verdict] += 1
        f = lambda r, s: f"{r['diff'] * s:+.1f} {KO[r.verdict]}"  # noqa: E731
        L.append(f"| f{k} | {f(a, 100)} | {f(b, 100)} | {f(c, 1)} | {f(d, 100)} |")
    L += ["", f"같은 엔진 v2 취향: 스와이프 {cnt['swipe']} / 비김 {cnt['tie']} / 목록 {cnt['list']} (7묶음).", ""]
    # seed 20
    L += ["# 분석 4: seed 5개 → 20개 (메인 조건)", "", "| 비교 | 지표 | seed 5개 | seed 20개 |", "|---|---|---|---|"]
    for comp, cname in COMP:
        for m, name, k, _ in METRICS:
            L.append(f"| {cname} | {name} | {cell(t.loc[('main', comp, m)], k)} | {cell(t.loc[('seeds-20', comp, m)], k)} |")
    L += ["", "seed를 4배로 늘려도 숫자와 판정이 그대로다. 구간 폭도 거의 안 줄었다: 오차는 seed(봇의 변덕·추천기 무작위)보다 "
          "'어떤 고객 1,000명을 뽑았나'에서 주로 온다. 그래서 seed보다 고객 묶음 반복(7묶음)이 더 중요한 점검이다.", ""]
    (OUT / "popbots.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    fig(t)


def fig(t):
    plt = mf.plt
    runs = [("main", "메인 봇\n(유행 0)"), ("popw-0p5", "유행\n0.5"), ("popw-1", "유행\n1"), ("popw-2", "유행\n2"),
            ("popw-1-personal", "유행 1\n사람마다")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (m, title, k) in zip(axes, [("pref_all", "취향: 좋아할 옷 비율 차이 (%p)", 100),
                                        ("near_hit20", "구매: 근접 적중 차이 (%p)", 100)]):
        for j, (comp, cname) in enumerate(COMP):
            color = mf.V1 if j == 0 else mf.V2
            xs = np.arange(len(runs)) + (j - 0.5) * 0.22
            r = [t.loc[(run, comp, m)] for run, _ in runs]
            y = np.array([x["diff"] for x in r]) * k
            lo = np.array([x.ci_low for x in r]) * k
            hi = np.array([x.ci_high for x in r]) * k
            ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt="o", color=color, ms=7, capsize=3, lw=1.5, label=cname)
        mf.zero(ax)
        ax.set_xticks(range(len(runs)), [n for _, n in runs])
        ax.set_title(title + "  + = 스와이프 우세", loc="left")
    axes[0].legend(loc="center right")
    fig.suptitle("⑦ 봇이 유행을 따르면: 원래 스와이프 엔진은 크게 지고, 같은 엔진 v2는 근소하게 이긴다 (검증 주)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=mf.INK)
    fig.savefig(mf.OUT / "fig7_popbots.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
