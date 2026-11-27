"""experiments.yaml 의 모든 조건(+ 원래 설계)의 검증 주 결과를 한 표로 모은다 (명세 수정 v3 4-4, 4-5).

실행: python scripts/collect_results.py [--quiet]   (scripts/sweep.py 가 중간·끝에 자동 실행)
출력: data/results/all_runs_valid.csv, all_runs_valid.md

판정: 95% 구간이 0을 포함하면 '비김'. 지표마다 좋은 방향(higher_better)을 따라 스와이프 우세 / 목록 우세.
같은 결과가 여러 번 세지지 않게, 비교마다 그 비교에 영향을 주는 설정만으로 조건을 구분한다
(원래 엔진 비교: recommender_v2.* 무시 / 같은 엔진 비교: recommender.* 무시).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from sweep import MAIN, conditions, config_path  # noqa: E402
from src.config import load_config, results_dir  # noqa: E402

RES = ROOT / "data" / "results"
METRICS = ["pref_all", "final_like20", "final_like20_new", "repeat_attr", "first_pref", "near_hit20", "exact_hit20"]
PCT = {"pref_all", "repeat_attr", "near_hit20", "exact_hit20"}
COMPS = {"list_vs_swipe": "원래 엔진: 목록(인기순) vs 스와이프",
         "list_v2_vs_swipe_v2": "같은 엔진: 목록 v2 vs 스와이프 v2"}
IGNORE = {"list_vs_swipe": "recommender_v2.", "list_v2_vs_swipe_v2": "recommender."}
MARK = {"swipe": "▲", "tie": "·", "list": "▼", "na": ""}
ORIGINAL = {"name": "original", "kind": "reference", "factor": "reference",
            "label": "참고: 원래 설계 (설명 문장 봇 + 설명 문장 추천기, 순환 구조, v1만)", "value": None, "keys": [], "set": {}}


def verdict(r) -> str:
    if pd.isna(r["diff"]):
        return "na"
    if r.ci_low <= 0 <= r.ci_high:
        return "tie"
    return "swipe" if (r["diff"] > 0) == bool(r.higher_better) else "list"


def signature(c: dict, comp: str) -> str:
    """그 비교 결과를 좌우하는 설정만 (예: 원래 엔진 비교에는 v2 인기도 가중치가 영향 없음)."""
    return repr(sorted((k, repr(v)) for k, v in c["set"].items()
                       if not k.startswith(IGNORE[comp]) and k != "experiment.variant"))


def load() -> tuple[pd.DataFrame, list[str]]:
    rows, missing = [], []
    for c in conditions() + [ORIGINAL]:
        if c["set"].get("session.ladder"):    # 데이터 사다리 실행은 scripts/analyze_ladder.py 에서 따로 정리
            continue
        if c["name"] == "original":
            f = RES / "valid" / "summary.csv"
        else:
            f = results_dir(load_config(config_path(c["name"])), "valid") / "summary.csv"
        if not f.exists():
            missing.append(c["name"])
            continue
        s = pd.read_csv(f)
        if "comparison" not in s:          # 지표 추가 전 원래 설계 결과
            s = s.rename(columns={"list": "a", "swipe": "b"}).assign(comparison="list_vs_swipe")
        if "higher_better" not in s:
            s["higher_better"] = s["metric"] != "first_pref"
        s = s[s["metric"].isin(METRICS)].copy()
        s["run"], s["kind"], s["factor"], s["label"] = c["name"], c["kind"], c["factor"], c["label"]
        s["value"] = "" if c["value"] is None else str(c["value"])
        s["is_main_value"] = bool(c.get("is_main_value", False))
        s["base_cond"], s["fold"] = c.get("base_cond", ""), c.get("fold", 0)
        s["v2_only"] = bool(c["keys"]) and all(k.startswith("recommender_v2.") for k in c["keys"])
        s["sig"] = [signature(c, comp) for comp in s["comparison"]]
        rows.append(s[["run", "kind", "factor", "label", "value", "is_main_value", "base_cond", "fold", "v2_only",
                       "sig", "comparison", "metric", "higher_better", "a", "b", "diff", "ci_low", "ci_high"]])
    t = pd.concat(rows, ignore_index=True)
    t["verdict"] = t.apply(verdict, axis=1)
    return t, list(dict.fromkeys(missing))


def fmt(m: str, v: float) -> str:
    return f"{v * 100:+.1f}" if m in PCT else f"{v:+.2f}"


def cell(x, m: str) -> str:
    if x.verdict == "na":
        return "해당 없음"
    return f"{MARK[x.verdict]} {fmt(m, x['diff'])} [{fmt(m, x.ci_low)}, {fmt(m, x.ci_high)}]"


def counts(v: pd.Series) -> str:
    c = v.value_counts()
    return f"{c.get('swipe', 0)} / {c.get('tie', 0)} / {c.get('list', 0)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    t, missing = load()
    t.to_csv(RES / "all_runs_valid.csv", index=False)
    same = t[~t["kind"].isin(["fold", "reference"])]                     # 메인 고객 1,000명
    uniq = same.drop_duplicates(["comparison", "sig", "metric"])          # 결과가 같은 조건은 한 번만

    L = ["# 전체 실험 조건 결과 (검증 주)", "",
         "diff = 스와이프 쪽 − 목록 쪽, [ ] = 95% 부트스트랩 구간 (배정된 봇끼리 A/B).",
         "▲ 스와이프 우세 · 비김(구간이 0 포함) ▼ 목록 우세. first_pref(첫 찜까지 장수)와 repeat_attr(싫다고 한 걸 또 보여 줌)는 작을수록 좋음.",
         "비율 지표(pref_all, repeat_attr, near_hit20, exact_hit20)는 %p, 나머지는 개수·장수.", ""]
    if missing:
        L += [f"(아직 결과 없음 {len(missing)}개: {', '.join(missing[:12])}{' …' if len(missing) > 12 else ''})", ""]

    # 1. 견고성 요약
    L += ["## 1. 견고성 요약 (메인 고객 1,000명, 조건마다 판정을 셈)", "",
          "칸 = ▲ 스와이프 우세 / · 비김 / ▼ 목록 우세 조건 수. 비교에 영향을 주지 않는 설정만 다른 조건은 한 번만 셈. "
          "'해당 없음'(그 이유를 말한 세션이 없음)은 세지 않음.", "",
          "| 지표 | " + " | ".join(f"{k} (조건 {uniq[uniq.comparison == k]['sig'].nunique()}개)" for k in COMPS) + " |",
          "|---|" + "---:|" * len(COMPS)]
    for m in METRICS:
        L.append(f"| {m} | " + " | ".join(counts(uniq[(uniq.comparison == k) & (uniq.metric == m)]["verdict"])
                                         for k in COMPS) + " |")
    L.append("")

    # 2. 다른 고객 묶음 반복
    fold = t[t["kind"] == "fold"]
    if not fold.empty:
        L += ["## 2. 다른 고객으로 반복 (서로 겹치지 않는 7묶음, 묶음마다 약 1,110명)", "",
              "칸 = 묶음 판정 ▲/·/▼ 수, 묶음 평균 diff (가장 작은 값 ~ 가장 큰 값). ↳ = 원래 고객 1,000명 결과.", ""]
        for comp, title in COMPS.items():
            L += [f"### {title}", "", "| 조건 | 묶음 수 | " + " | ".join(METRICS) + " |", "|---|---:|" + "---|" * len(METRICS)]
            for base, g in fold[fold.comparison == comp].groupby("base_cond", sort=False):
                if g["v2_only"].iloc[0] and comp == "list_vs_swipe":
                    continue
                cells = []
                for m in METRICS:
                    h = g[(g.metric == m) & (g.verdict != "na")]
                    if h.empty:
                        cells.append("해당 없음")
                        continue
                    cells.append(f"{counts(h['verdict'])}, {fmt(m, h['diff'].mean())} "
                                 f"({fmt(m, h['diff'].min())}~{fmt(m, h['diff'].max())})")
                L.append(f"| {base} | {g['fold'].nunique()} | " + " | ".join(cells) + " |")
                ref = same[(same.run == base) & (same.comparison == comp)].drop_duplicates("metric").set_index("metric")
                if not ref.empty:
                    L.append("| ↳ 메인 고객 |  | " + " | ".join(cell(ref.loc[m], m) if m in ref.index else ""
                                                              for m in METRICS) + " |")
            L.append("")

    # 3. 한 가지씩 바꾼 조건
    L += ["## 3. 한 가지씩 바꾼 조건과 묶음 설정 (메인 고객 1,000명)", ""]
    for comp, title in COMPS.items():
        c = same[(same.comparison == comp) & same["kind"].isin([MAIN, "factor", "preset"])]
        if comp == "list_vs_swipe":
            c = c[~c["v2_only"]]
        L += [f"### {title}", "", "| 조건 | 값 | " + " | ".join(METRICS) + " |", "|---|---|" + "---:|" * len(METRICS)]
        for (factor, label), g in c.groupby(["factor", "label"], sort=False):
            if factor != MAIN:
                L.append(f"| **{label}** |" + " |" * (len(METRICS) + 1))
            for (run, val), h in g.groupby(["run", "value"], sort=False):
                h = h.set_index("metric")
                name = run + (" (메인)" if run == MAIN and factor != MAIN else "")
                v = "" if factor in ("preset", MAIN) else val
                L.append(f"| {name} | {v} | " + " | ".join(cell(h.loc[m], m) if m in h.index else "" for m in METRICS) + " |")
        L.append("")

    # 4. 두 가지를 함께 바꾼 조합: 표 모양 (행 = 첫 설정, 열 = 둘째 설정)
    grid = t[t["kind"] == "grid"]
    if not grid.empty:
        L += ["## 4. 두 가지를 함께 바꾼 조합 (메인 고객 1,000명)", "",
              "칸 = 스와이프 − 목록 diff (▲/·/▼ 판정). 지표: pref_all(취향, %p), near_hit20(다음 주 구매, %p).", ""]
        for (gname, label), g in grid.groupby(["factor", "label"], sort=False):
            parts = g["value"].str.split(", ", expand=True)
            g = g.assign(r=parts[0], col=parts[1])
            for comp in COMPS:
                for m in ("pref_all", "near_hit20"):
                    h = g[(g.comparison == comp) & (g.metric == m)]
                    if h.empty or (comp == "list_vs_swipe" and "wpop" in gname):
                        continue
                    num = lambda x: float(x.split("=")[1])                     # noqa: E731
                    cols = sorted(dict.fromkeys(h["col"]), key=num)
                    h = h.assign(rk=h["r"].map(num)).sort_values("rk")
                    L += [f"**{label}** · {comp} · {m}", "", "| | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
                    for r, rr in h.groupby("r", sort=False):
                        rr = rr.set_index("col")
                        L.append(f"| {r} | " + " | ".join(f"{MARK[rr.loc[x].verdict]} {fmt(m, rr.loc[x]['diff'])}"
                                                         if x in rr.index else "" for x in cols) + " |")
                    L.append("")

    # 5. 참고
    ref = t[t["kind"] == "reference"]
    if not ref.empty:
        L += ["## 5. 참고: 원래 설계", "", ORIGINAL["label"], "", "| 지표 | diff |", "|---|---:|"]
        L += [f"| {r.metric} | {cell(r, r.metric)} |" for _, r in ref.iterrows()]
        L.append("")
    (RES / "all_runs_valid.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if not args.quiet:
        print("\n".join(L))


if __name__ == "__main__":
    main()
