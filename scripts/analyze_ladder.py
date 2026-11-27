"""재설계 (2026-11-27): 데이터 사다리 결과 정리 (검증 주만).

같은 엔진 v2에서 스와이프가 추천기에 넘기는 데이터를 한 칸씩 늘린다.
    목록 v2   찜만, 2·3세트 개인화 6칸
    S1        찜만, 개인화 20칸            (S1 − 목록 = 화면 구성 효과)
    S2        + 넘김 (모두 '취향 아님' 취급)  (S2 − S1 = 넘김 데이터 효과)
    S3        + 이유 3가지 (= swipe_v2)     (S3 − S2 = 이유 데이터 효과)
    부록       S2 + '비싸요'만 / '이미 있어요'만 구분

사다리 표는 봇 1,000명 전원이 모든 방식을 돈 paired 차이라 단계 차이를 더하면 S3 − 목록과 정확히 같다.
A/B(배정된 봇끼리) 차이는 기존 보고서와 같은 방식의 확인용으로 함께 적는다.

지표: 찜 비율(메인, 변덕 없이 봇 규칙으로 판정) · 첫 찜까지 장수 · 세트별 찜 비율 · 넘긴 카드의 탈락 원인 ·
      체류 대용(찜할 카드가 연속 N장 안 나오면 나감, N = 5·10·15) · 부록: 최종 20개 · 다음 주 구매 근접 적중

출력: data/results/analysis/ladder.{md,csv}, data/results/figures/fig8_ladder.png
실행: .venv/bin/python scripts/analyze_ladder.py   (scripts/sweep.py --only ladder ladder__f1 ... 뒤)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import make_figures as mf  # noqa: E402
from src.config import load_config, results_dir  # noqa: E402
from src.items import load_pool  # noqa: E402
from src.scoring import bot_profiles, load_sessions  # noqa: E402

RES = ROOT / "data" / "results"
OUT = RES / "analysis"
CONFIGS = ROOT / "configs"
LADDER = [("list_v2", "목록 v2"), ("swipe_s1", "S1 찜만"), ("swipe_s2", "S2 +넘김"), ("swipe_v2", "S3 +이유")]
APPX = [("swipe_rp", "S2 +'비싸요'만"), ("swipe_rh", "S2 +'이미 있어요'만")]
NAME = dict(LADDER + APPX)
STEPS = [("list_v2", "swipe_s1", "화면 구성 (S1 − 목록)"), ("swipe_s1", "swipe_s2", "넘김 데이터 (S2 − S1)"),
         ("swipe_s2", "swipe_v2", "이유 데이터 (S3 − S2)")]
APPX_STEPS = [("swipe_s2", "swipe_rp", "'비싸요'만 (− S2)"), ("swipe_s2", "swipe_rh", "'이미 있어요'만 (− S2)"),
              ("swipe_s2", "swipe_v2", "이유 3가지 (S3 − S2)")]
METRICS = [("pref_all", "60장 중 찜 비율", 100, "%p", True), ("first_pref", "첫 찜까지 장수", 1, "장", False),
           ("final_like20", "최종 20개 중 찜할 옷", 1, "개", True), ("near_hit20", "부록: 다음 주 구매 근접 적중", 100, "%p", True)]
DWELL_N = (5, 10, 15)
COLORS = {"list_v2": mf.LIST, "swipe_s1": "#93bdf0", "swipe_s2": "#4a8fe0", "swipe_v2": "#1d5aa8"}
N_BOOT = 1000


# ---------------------------------------------------------------- 통계
def boot_paired(d: np.ndarray, rng) -> tuple[float, float, float]:
    d = d[~np.isnan(d)]
    b = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), *np.percentile(b, [2.5, 97.5]).tolist()


def boot_ab(xa: np.ndarray, xb: np.ndarray, rng) -> tuple[float, float, float]:
    xa, xb = xa[~np.isnan(xa)], xb[~np.isnan(xb)]
    ba = xa[rng.integers(0, len(xa), size=(N_BOOT, len(xa)))].mean(axis=1)
    bb = xb[rng.integers(0, len(xb), size=(N_BOOT, len(xb)))].mean(axis=1)
    return float(xb.mean() - xa.mean()), *np.percentile(bb - ba, [2.5, 97.5]).tolist()


def verdict(d, lo, hi, higher=True) -> str:
    if lo <= 0 <= hi:
        return "비김"
    return "스와이프 쪽" if (d > 0) == higher else "목록 쪽"


def step_word(d, lo, hi, higher=True) -> str:
    if lo <= 0 <= hi:
        return "효과 없음"
    return "올림" if (d > 0) == higher else "내림"


def fmt(d, lo, hi, k, unit):
    return f"{d * k:+.1f}{unit} [{lo * k:+.1f}, {hi * k:+.1f}]"


# ---------------------------------------------------------------- 세션 로그로 계산하는 지표 (체류 대용, 탈락 원인)
def dwell(like: np.ndarray, n: int) -> int:
    """찜할 카드가 연속 n장 안 나오면 그 카드에서 나간다. 나가기 전까지 본 장수 (끝까지 보면 60)."""
    run = 0
    for i, x in enumerate(like):
        run = 0 if x else run + 1
        if run >= n:
            return i + 1
    return len(like)


def log_metrics(cfg: dict) -> pd.DataFrame:
    pool = load_pool(cfg)
    seeds = cfg["session"]["seeds"]
    logs, _ = load_sessions(cfg, seeds)
    prof = bot_profiles(cfg, pool, seeds)
    logs = logs.sort_values(["seed", "group", "bot_id", "set_no", "pos"])
    rows = []
    for (seed, mode, b), g in logs.groupby(["seed", "group", "bot_id"], sort=False):
        bot = prof[seed][b]
        shown = pool.rows(g["article_id"])
        _, taste_ok, like = bot.rule_mask(pool, shown)
        price_ok = pool.price[shown] <= bot.price_th
        r = {"seed": seed, "group": mode, "bot_id": b, "assigned": bool(g["assigned"].iat[0])}
        for n in DWELL_N:
            r[f"dwell_{n}"] = float(dwell(like, n))
            r[f"full_{n}"] = float(dwell(like, n) == len(like))
        set_no = g["set_no"].to_numpy()
        for s in (1, 2, 3):
            m = set_no == s
            r[f"s{s}_like"] = like[m].mean()
            r[f"s{s}_taste"] = (~taste_ok[m]).mean()
            r[f"s{s}_price"] = (taste_ok[m] & ~price_ok[m]).mean()
            r[f"s{s}_dup"] = (taste_ok[m] & price_ok[m] & ~like[m]).mean()
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 정리
def paired_table(per: pd.DataFrame, cols, pairs, rng) -> list[dict]:
    out = []
    for a, b, label in pairs:
        for col in cols:
            d = (per.loc[b, col] - per.loc[a, col]).to_numpy()
            out.append({"a": a, "b": b, "label": label, "metric": col, **dict(zip(("diff", "lo", "hi"), boot_paired(d, rng)))})
    return out


def run_tables(name: str, rng) -> dict:
    cfg = load_config(CONFIGS / f"{name}.yaml")
    rd = results_dir(cfg, "valid")
    m = pd.read_parquet(rd / "session_metrics.parquet")
    summ = pd.read_csv(rd / "summary.csv")
    cols = [c for c, *_ in METRICS] + [f"pref_{e}" for e in range(10, 61, 10)]
    per_all = m.groupby(["group", "bot_id"])[cols].mean()                       # 봇 1,000명 전원 (paired)
    per_ab = m[m["assigned"]].groupby(["group", "bot_id"])[cols].mean()        # 배정된 봇만 (A/B)
    for s in (1, 2, 3):
        per_all[f"set{s}"] = per_all[[f"pref_{20 * s - 10}", f"pref_{20 * s}"]].mean(axis=1)
        per_ab[f"set{s}"] = per_ab[[f"pref_{20 * s - 10}", f"pref_{20 * s}"]].mean(axis=1)
    return {"cfg": cfg, "m": m, "summ": summ, "per_all": per_all, "per_ab": per_ab}


def main() -> None:
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    t = run_tables("ladder", rng)
    per_all, per_ab, summ = t["per_all"], t["per_ab"], t["summ"]
    modes = [m for m, _ in LADDER + APPX]

    # 확인: 사다리 실행의 목록 v2 · 스와이프 v2 는 메인 실행과 같은 세션이어야 한다
    main_s = pd.read_csv(RES / "bot-image_rec-text" / "valid" / "summary.csv")
    key = ["comparison", "metric"]
    chk = summ.merge(main_s, on=key, suffixes=("", "_main"))
    chk = chk[chk.comparison == "list_v2_vs_swipe_v2"]
    same_as_main = bool(np.allclose(chk[["a", "b"]].to_numpy(), chk[["a_main", "b_main"]].to_numpy()))

    csv_rows = []
    L = ["# 재설계: 데이터 사다리 (검증 주, 2026-11-27)", "",
         "같은 엔진 v2, 같은 첫 20장(카테고리 인기순), 같은 봇 1,000명. 스와이프가 추천기에 넘기는 데이터만 한 칸씩 늘림.",
         f"사다리 실행의 목록 v2 · S3 세션이 메인 실행과 같은가: {'예' if same_as_main else '아니오 (확인 필요)'}.", "",
         "## 1. 단계별 값 (봇 1,000명 전원, seed 5개 평균)", "",
         "| 지표 | " + " | ".join(NAME[m] for m in modes) + " |", "|---|" + "---:|" * len(modes)]
    for col, name, k, unit, _ in METRICS:
        L.append(f"| {name} | " + " | ".join(f"{per_all.loc[m, col].mean() * k:.1f}" for m in modes) + " |")
    for s in (1, 2, 3):
        L.append(f"| 찜 비율 {s}세트 ({20 * s - 19}~{20 * s}장) | "
                 + " | ".join(f"{per_all.loc[m, f'set{s}'].mean() * 100:.1f}" for m in modes) + " |")
    for m in modes:
        csv_rows.append({"run": "ladder", "table": "level", "mode": m,
                         **{c: per_all.loc[m, c].mean() for c in [x for x, *_ in METRICS] + ["set1", "set2", "set3"]}})

    L += ["", "## 2. 사다리: 한 칸 올라갈 때의 차이 (같은 봇끼리 paired, [95% 구간])", "",
          "| 단계 | " + " | ".join(n for _, n, *_ in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    step_rows = paired_table(per_all, [c for c, *_ in METRICS] + ["set1", "set2", "set3"], STEPS + APPX_STEPS
                             + [("list_v2", "swipe_v2", "합계 (S3 − 목록)")], rng)
    sr = pd.DataFrame(step_rows).set_index(["label", "metric"])
    for _, _, label in STEPS + [(None, None, "합계 (S3 − 목록)")]:
        cells = []
        for col, _, k, unit, hb in METRICS:
            r = sr.loc[(label, col)]
            cells.append(f"{fmt(r['diff'], r.lo, r.hi, k, unit)} {step_word(r['diff'], r.lo, r.hi, hb)}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "세트별 찜 비율 차이 (%p):", "", "| 단계 | 1세트 | 2세트 | 3세트 |", "|---|---|---|---|"]
    for _, _, label in STEPS + [(None, None, "합계 (S3 − 목록)")]:
        L.append(f"| {label} | " + " | ".join(fmt(sr.loc[(label, f'set{s}'), 'diff'], sr.loc[(label, f'set{s}'), 'lo'],
                                                   sr.loc[(label, f'set{s}'), 'hi'], 100, '') for s in (1, 2, 3)) + " |")
    L += ["", "부록: 이유를 하나씩만 알려 주면 (S2 대비, 찜 비율 %p):", ""]
    for _, _, label in APPX_STEPS:
        r = sr.loc[(label, "pref_all")]
        L.append(f"- {label}: {fmt(r['diff'], r.lo, r.hi, 100, '%p')} {step_word(r['diff'], r.lo, r.hi)}")
    csv_rows += [{"run": "ladder", "table": "step", **r} for r in step_rows]

    # A/B 확인 (기존 보고서 방식)
    L += ["", "## 3. 확인: 배정된 봇끼리 A/B (기존 보고서 방식, 목록 v2 = A그룹, 스와이프 = B그룹)", "",
          "| 비교 | 찜 비율 | 첫 찜까지 | 최종 20개 | 부록: 구매 근접 적중 |", "|---|---|---|---|---|"]
    for m in ("swipe_s1", "swipe_s2", "swipe_v2", "swipe_rp", "swipe_rh"):
        cells = []
        for col, _, k, unit, hb in METRICS:
            d, lo, hi = boot_ab(per_ab.loc["list_v2", col].to_numpy(), per_ab.loc[m, col].to_numpy(), rng)
            cells.append(f"{fmt(d, lo, hi, k, unit)} {verdict(d, lo, hi, hb)}")
            csv_rows.append({"run": "ladder", "table": "ab", "a": "list_v2", "b": m, "metric": col, "diff": d, "lo": lo, "hi": hi})
        L.append(f"| 목록 v2 vs {NAME[m]} | " + " | ".join(cells) + " |")

    # 체류 대용, 탈락 원인
    lm = log_metrics(t["cfg"])          # 봇(고객) ID가 들어 있어 파일로 저장하지 않는다
    pl =lm.groupby(["group", "bot_id"]).mean(numeric_only=True)
    L += ["", "## 4. 체류 대용: 찜할 카드가 연속 N장 안 나오면 나간다", "",
          "| N | " + " | ".join(NAME[m] for m in modes) + " |", "|---|" + "---:|" * len(modes)]
    for n in DWELL_N:
        L.append(f"| {n}장 · 나가기 전 본 장수 | " + " | ".join(f"{pl.loc[m, f'dwell_{n}'].mean():.1f}" for m in modes) + " |")
        L.append(f"| {n}장 · 60장 끝까지 본 비율 | " + " | ".join(f"{pl.loc[m, f'full_{n}'].mean():.0%}" for m in modes) + " |")
    L += ["", "| 단계 (paired) | " + " | ".join(f"N={n} 본 장수" for n in DWELL_N) + " |", "|---|" + "---|" * len(DWELL_N)]
    dw = pd.DataFrame(paired_table(pl, [f"dwell_{n}" for n in DWELL_N], STEPS + [("list_v2", "swipe_v2", "합계 (S3 − 목록)")], rng))
    dw = dw.set_index(["label", "metric"])
    for _, _, label in STEPS + [(None, None, "합계 (S3 − 목록)")]:
        L.append(f"| {label} | " + " | ".join(
            f"{fmt(dw.loc[(label, f'dwell_{n}'), 'diff'], dw.loc[(label, f'dwell_{n}'), 'lo'], dw.loc[(label, f'dwell_{n}'), 'hi'], 1, '장')}"
            for n in DWELL_N) + " |")
    csv_rows += [{"run": "ladder", "table": "dwell_step", **r} for r in dw.reset_index().to_dict("records")]
    for m in modes:
        csv_rows.append({"run": "ladder", "table": "dwell_level", "mode": m,
                         **{c: pl.loc[m, c].mean() for c in pl.columns if c.startswith(("dwell_", "full_"))}})

    L += ["", "## 5. 보여 준 카드는 왜 찜할 카드가 아니었나 (세트별, 보여 준 카드 중 비율)", "",
          "| 방식 | 세트 | 찜할 카드 | 취향 아님 | 취향은 맞는데 비쌈 | 취향·가격 맞는데 이미 있음 |", "|---|---|---:|---:|---:|---:|"]
    for m in modes:
        for s in (1, 2, 3):
            v = [pl.loc[m, f"s{s}_{x}"].mean() for x in ("like", "taste", "price", "dup")]
            L.append(f"| {NAME[m]} | {s} | " + " | ".join(f"{x:.1%}" for x in v) + " |")
            csv_rows.append({"run": "ladder", "table": "cards", "mode": m, "set": s,
                             **dict(zip(("like", "taste", "price", "dup"), v))})

    # 다른 고객 7묶음
    L += ["", "## 6. 다른 고객 7묶음으로 반복 (찜 비율, paired)", "",
          "| 묶음 | " + " | ".join(lbl for _, _, lbl in STEPS) + " | 합계 (S3 − 목록) | A/B 목록 v2 vs S3 |", "|---|---|---|---|---|---|"]
    cnt = {lbl: {"올림": 0, "효과 없음": 0, "내림": 0} for _, _, lbl in STEPS + [(0, 0, "합계")]}
    for k in range(1, 8):
        f = run_tables(f"ladder__f{k}", rng)
        rows = paired_table(f["per_all"], ["pref_all"], STEPS + [("list_v2", "swipe_v2", "합계")], rng)
        cells = []
        for r in rows:
            w = step_word(r["diff"], r["lo"], r["hi"])
            cnt[r["label"]][w] += 1
            cells.append(f"{r['diff'] * 100:+.1f} {w}")
        d, lo, hi = boot_ab(f["per_ab"].loc["list_v2", "pref_all"].to_numpy(), f["per_ab"].loc["swipe_v2", "pref_all"].to_numpy(), rng)
        cells.append(f"{d * 100:+.1f} {verdict(d, lo, hi)}")
        L.append(f"| f{k} | " + " | ".join(cells) + " |")
        csv_rows += [{"run": f"ladder__f{k}", "table": "step", **r} for r in rows]
    L += ["", "7묶음 집계 (올림 / 효과 없음 / 내림): "
          + " · ".join(f"{lbl} {c['올림']}/{c['효과 없음']}/{c['내림']}" for lbl, c in cnt.items()), ""]

    (OUT / "ladder.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(OUT / "ladder.csv", index=False)
    print("\n".join(L))
    fig(per_all, sr, pl)


def fig(per_all, sr, pl):
    plt = mf.plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    xs = np.arange(len(LADDER))
    vals = [per_all.loc[m, "pref_all"].mean() * 100 for m, _ in LADDER]
    ax.bar(xs, vals, width=0.6, color=[COLORS[m] for m, _ in LADDER], edgecolor=mf.SURF, linewidth=2)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", color=mf.INK, fontsize=10)
    for i, (_, _, label) in enumerate(STEPS):
        r = sr.loc[(label, "pref_all")]
        ax.annotate(f"{r['diff'] * 100:+.1f}%p", ((xs[i] + xs[i + 1]) / 2, max(vals[i], vals[i + 1]) + 6),
                    ha="center", color=mf.INK2, fontsize=9)
    ax.set_xticks(xs, [n.replace(" ", "\n", 1) for _, n in LADDER])
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("60장 중 찜할 카드 비율 (%)")
    ax.set_title("찜 비율: 데이터를 한 칸씩 더하면", loc="left")

    ax = axes[1]
    for m, n in LADDER:
        y = [per_all.loc[m, f"set{s}"].mean() * 100 for s in (1, 2, 3)]
        ax.plot([1, 2, 3], y, color=COLORS[m], lw=2, marker="o", ms=6, label=n)
        ax.annotate(n, (3, y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=mf.INK)
    ax.set_xticks([1, 2, 3], ["1세트\n(모두 인기순)", "2세트", "3세트"])
    ax.set_xlim(0.8, 3.7)
    ax.set_ylabel("세트 20장 중 찜할 카드 비율 (%)")
    ax.set_title("세트별: 첫 20장은 같고 그 뒤로 벌어진다", loc="left")

    ax = axes[2]
    w = 0.2
    for j, (m, n) in enumerate(LADDER):
        y = [pl.loc[m, f"dwell_{k}"].mean() for k in DWELL_N]
        ax.bar(np.arange(len(DWELL_N)) + (j - 1.5) * w, y, width=w, color=COLORS[m], edgecolor=mf.SURF, linewidth=1.5, label=n)
    ax.set_xticks(range(len(DWELL_N)), [f"연속 {k}장\n안 맞으면 나감" for k in DWELL_N])
    ax.set_ylim(0, 66)
    ax.set_ylabel("나가기 전까지 본 카드 수 (최대 60)")
    ax.set_title("체류 대용", loc="left")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    fig.suptitle("⑧ 데이터 사다리 (재설계, 검증 주, 봇 1,000명 paired)", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=mf.INK)
    fig.savefig(mf.OUT / "fig8_ladder.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
