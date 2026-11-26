"""보고서 그래프 (검증 주). 실행: .venv/bin/python scripts/make_figures.py  → data/results/figures/*.png

색: 스와이프 = 파랑, 목록 = 주황 (모든 그림에서 같음). 비교 종류(원래 엔진 / 같은 엔진)는 보라 / 청록.
차이 표(열지도)는 파랑 = 스와이프 우세, 빨강 = 목록 우세, 회색 = 0.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
OUT = RES / "figures"
MAIN_DIR = RES / "bot-image_rec-text" / "valid"

SWIPE, LIST = "#2a78d6", "#eb6834"
V1, V2 = "#6250d6", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COMP = {"list_vs_swipe": ("원래 엔진 (목록 인기순 vs 스와이프)", V1),
        "list_v2_vs_swipe_v2": ("같은 엔진 (목록 v2 vs 스와이프 v2)", V2)}

plt.rcParams.update({
    "font.family": "AppleGothic", "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def zero(ax):
    ax.axhline(0, color=INK2, lw=1, zorder=1)


# 1. 노출 10장마다 좋아할 옷 비율 -----------------------------------------------
def fig_learning_curve():
    lc = pd.read_csv(MAIN_DIR / "learning_curve.csv")
    lc = lc[lc.metric == "pref"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (title, pair) in zip(axes, [("원래 엔진", ("list", "swipe")), ("같은 엔진 (v2)", ("list_v2", "swipe_v2"))]):
        for g, color, name in [(pair[0], LIST, "목록"), (pair[1], SWIPE, "스와이프")]:
            d = lc[lc.group == g].sort_values("exposed")
            ax.fill_between(d.exposed, d.ci_low * 100, d.ci_high * 100, color=color, alpha=0.15, lw=0)
            ax.plot(d.exposed, d["mean"] * 100, color=color, lw=2, marker="o", ms=5, label=name)
            ax.annotate(name, (d.exposed.iloc[-1], d["mean"].iloc[-1] * 100), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(10, 61, 10), [f"{e - 9}~{e}" for e in range(10, 61, 10)])
        ax.set_xlabel("노출 순번 (10장 구간)")
        ax.set_xlim(7, 66)
    axes[0].set_ylabel("봇이 좋아할 옷 비율 (%)")
    axes[0].legend(loc="lower left", ncol=2)
    fig.suptitle("① 보여 줄수록 스와이프가 취향을 더 잘 찾는다 (메인 고객 1,000명, 색 띠 = 95% 구간)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.savefig(OUT / "fig1_learning_curve.png")
    plt.close(fig)


# 2. 설정 값별 차이 ---------------------------------------------------------------
FACTORS = [("noise", "봇 변덕 확률", False), ("explore", "세트 20장 중 탐색 상품 수", False),
           ("price", "봇 가격 상한 배수 (작을수록 '비싸요' 잦음)", False), ("likeq", "봇 까다로움 (찜 기준선 분위수)", False),
           ("insert", "목록 v2에 끼워 넣는 추천 수 (v2만)", True), ("wpop", "v2 인기도 가중치 (v2만)", True)]


def factor_rows(t, factor, metric):
    """factor 조건 + 메인 값(메인 결과)을 값 순서로."""
    d = t[(t.kind == "factor") & (t.factor == factor) & (t.metric == metric)].copy()
    d["x"] = d["value"].astype(float)
    return d.sort_values("x")


def fig_factors(t):
    fig, axes = plt.subplots(2, len(FACTORS), figsize=(17, 6.4), sharey="row")
    for j, (f, label, v2only) in enumerate(FACTORS):
        for i, (m, mname) in enumerate([("pref_all", "취향: 좋아할 옷 비율 차이 (%p)"),
                                        ("near_hit20", "다음 주 구매 근접 적중 차이 (%p)")]):
            ax = axes[i, j]
            d = factor_rows(t, f, m)
            xs_all = sorted(d.x.unique())
            pos = {x: k for k, x in enumerate(xs_all)}       # 값 간격과 상관없이 같은 간격으로
            for comp, (cname, color) in COMP.items():
                if v2only and comp == "list_vs_swipe":
                    continue
                c = d[d.comparison == comp].drop_duplicates("x")
                x = c.x.map(pos)
                ax.fill_between(x, c.ci_low * 100, c.ci_high * 100, color=color, alpha=0.15, lw=0)
                ax.plot(x, c["diff"] * 100, color=color, lw=2, marker="o", ms=5, label=cname)
                mv = c[c.is_main_value]
                ax.scatter(mv.x.map(pos), mv["diff"] * 100, s=90, facecolor="none", edgecolor=INK, lw=1.2, zorder=5)
            zero(ax)
            ax.set_xticks(range(len(xs_all)), ["끔" if not np.isfinite(x) else f"{x:g}" for x in xs_all], fontsize=8)
            if i == 0:
                ax.set_title(label, loc="left", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(mname)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.0))
    fig.suptitle("② 설정을 하나씩 바꿨을 때 차이 (스와이프 - 목록). 0 위 = 스와이프 우세, 동그라미 = 메인 값",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig2_factors.png")
    plt.close(fig)


# 3. 다른 고객 묶음 반복 ------------------------------------------------------------
FOLD_LABEL = {"main": "메인", "noise-0p3": "변덕 0.3", "explore-6": "탐색 6장", "likeq-0p7": "까다로움 0.7",
              "price-2": "가격 상한 2", "rule-off": "이유 규칙 끔", "rule-orig": "원 설계 규칙",
              "rec-image": "B안 (순환 주의)"}


def fig_folds(t):
    fold = t[t.kind == "fold"]
    same = t[~t.kind.isin(["fold", "reference"])]
    conds = list(FOLD_LABEL)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for j, (comp, (cname, _)) in enumerate(COMP.items()):
        for i, (m, mname) in enumerate([("pref_all", "취향: 좋아할 옷 비율 차이 (%p)"),
                                        ("near_hit20", "다음 주 구매 근접 적중 차이 (%p)")]):
            ax = axes[i, j]
            for k, c in enumerate(conds):
                y = len(conds) - 1 - k
                d = fold[(fold.base_cond == c) & (fold.comparison == comp) & (fold.metric == m)]
                col = [SWIPE if v == "swipe" else LIST if v == "list" else INK2 for v in d.verdict]
                ax.scatter(d["diff"] * 100, np.full(len(d), y) + np.linspace(-0.12, 0.12, len(d)),
                           c=col, s=28, alpha=0.85, edgecolor=SURF, lw=0.6, zorder=3)
                r = same[(same.run == c) & (same.comparison == comp) & (same.metric == m)].head(1)
                if not r.empty:
                    ax.plot([r.ci_low.iloc[0] * 100, r.ci_high.iloc[0] * 100], [y - 0.32] * 2, color=INK, lw=1.5)
                    ax.scatter(r["diff"] * 100, [y - 0.32], marker="D", s=30, color=INK, zorder=4)
            ax.axvline(0, color=INK2, lw=1)
            ax.set_yticks(range(len(conds)), [FOLD_LABEL[c] for c in reversed(conds)])
            ax.grid(axis="y", visible=False)
            ax.set_xlabel(mname)
            if i == 0:
                ax.set_title(cname, loc="left")
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", ls="", color=SWIPE, label="묶음: 스와이프 우세"),
           Line2D([], [], marker="o", ls="", color=INK2, label="묶음: 비김"),
           Line2D([], [], marker="o", ls="", color=LIST, label="묶음: 목록 우세"),
           Line2D([], [], marker="D", color=INK, label="메인 고객 1,000명 (선 = 95% 구간)")]
    fig.legend(handles=leg, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("③ 서로 겹치지 않는 다른 고객 7묶음(약 1,110명씩)으로 반복. 점 하나 = 한 묶음",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT / "fig3_folds.png")
    plt.close(fig)


# 4. 두 설정을 함께 ---------------------------------------------------------------
CMAP = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
GRIDS = [("noise_x_explore", "변덕", "탐색 상품 수"), ("noise_x_price", "변덕", "가격 상한 배수"),
         ("noise_x_likeq", "변덕", "까다로움")]


def fig_grids(t):
    g = t[(t.kind == "grid") & (t.metric == "pref_all")].copy()
    parts = g["value"].str.split(", ", expand=True)
    g["r"] = parts[0].str.split("=").str[1].astype(float)
    g["c"] = parts[1].str.split("=").str[1].astype(float)
    lim = 15
    fig, axes = plt.subplots(2, len(GRIDS), figsize=(13, 7))
    for j, (gname, rname, cname) in enumerate(GRIDS):
        for i, (comp, (title, _)) in enumerate(COMP.items()):
            ax = axes[i, j]
            d = g[(g.factor == gname) & (g.comparison == comp)].drop_duplicates(["r", "c"])
            pv = d.pivot(index="r", columns="c", values="diff") * 100
            vd = d.pivot(index="r", columns="c", values="verdict")
            ax.imshow(pv.values, cmap=CMAP, vmin=-lim, vmax=lim, aspect="auto")
            for a in range(pv.shape[0]):
                for b in range(pv.shape[1]):
                    v = pv.values[a, b]
                    mark = {"swipe": "▲", "list": "▼", "tie": "비김"}[vd.values[a, b]]
                    ax.text(b, a, f"{v:+.1f}\n{mark}", ha="center", va="center", fontsize=9,
                            color="white" if abs(v) > 10 else INK)
            ax.set_xticks(range(pv.shape[1]), [f"{c:g}" for c in pv.columns])
            ax.set_yticks(range(pv.shape[0]), [f"{r:g}" for r in pv.index])
            ax.set_xlabel(cname)
            ax.set_ylabel(rname)
            ax.grid(False)
            ax.set_title(f"{title.split(' (')[0]} · {rname} × {cname}", loc="left", fontsize=10)
    fig.suptitle("④ 두 설정을 함께 바꿨을 때 취향 차이 (%p, 스와이프 - 목록). 파랑 = 스와이프 우세, 빨강 = 목록 우세",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig4_grids.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    t = pd.read_csv(RES / "all_runs_valid.csv", keep_default_na=False, na_values=[""])
    t["is_main_value"] = t["is_main_value"].astype(str) == "True"
    fig_learning_curve()
    fig_factors(t)
    fig_folds(t)
    fig_grids(t)
    print("저장:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
