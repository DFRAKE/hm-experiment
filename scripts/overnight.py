"""밤샘 탐색 (2026-11-28, 검증 주만): overnight.yaml 의 라운드를 돌리고 데이터 사다리로 정리한다.

실행:
  .venv/bin/python scripts/overnight.py run --round 1 --jobs 6   # 그 라운드 조건만 세션 + 채점
  .venv/bin/python scripts/overnight.py analyze                   # 끝난 조건 모두 정리 -> analysis/overnight_ladder.{md,csv}
  .venv/bin/python scripts/overnight.py list

조건은 config.yaml + 설정 몇 개 + session.ladder: true. 설정 파일은 configs/<이름>.yaml (scripts/sweep.py 와 같은 방식).
정답 주(test)는 돌리지 않는다. 개별 봇 값은 저장하지 않는다(조건별 평균과 paired 차이만).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import sweep  # noqa: E402
import analyze_ladder as al  # noqa: E402

SPEC = ROOT / "overnight.yaml"
OUT = ROOT / "data" / "results" / "analysis"
MODES = ["list_v2", "swipe_s1", "swipe_s2", "swipe_v2", "swipe_rp", "swipe_rh"]
STEPS = [("screen", "list_v2", "swipe_s1"), ("skip", "swipe_s1", "swipe_s2"), ("reason", "swipe_s2", "swipe_v2"),
         ("total", "list_v2", "swipe_v2"), ("price_only", "swipe_s2", "swipe_rp"), ("have_only", "swipe_s2", "swipe_rh")]


def load_spec() -> list[dict]:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))["rounds"]


def conds(rounds=None) -> list[dict]:
    out = []
    for r in load_spec():
        if rounds and r["round"] not in rounds:
            continue
        for name, st in (r.get("conditions") or {}).items():
            out.append({"name": name, "round": r["round"], "set": {**st, "session.ladder": True}, "variant": name})
    return out


def run(rounds, jobs: int) -> None:
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    todo = []
    for c in conds(rounds):
        p = sweep.write_config(base, c)
        st = sweep.status(p)
        print(f"{st:9s} {c['name']}", flush=True)
        if st == "conflict":
            raise SystemExit(f"{c['name']}: 다른 설정의 세션이 있음")
        if st in ("run", "score"):
            todo.append((c, p, st))
    print(f"돌릴 조건 {len(todo)}개, 동시 {jobs}개", flush=True)
    t0, failed = time.time(), []
    with ThreadPoolExecutor(jobs) as ex:
        futs = [ex.submit(sweep.run_one, *t) for t in todo]
        for i, f in enumerate(as_completed(futs), 1):
            name, rc, sec = f.result()
            print(f"[{i}/{len(todo)} | {(time.time() - t0) / 60:.0f}분] {'끝' if rc == 0 else f'실패 rc={rc}'} {name} "
                  f"({sec / 60:.1f}분)", flush=True)
            if rc:
                failed.append(name)
    if failed:
        print(f"실패: {failed}", flush=True)


def one(name: str, rng) -> dict | None:
    cfg_path = sweep.CONFIGS / f"{name}.yaml"
    if not cfg_path.exists() or sweep.status(cfg_path) != "done":
        return None
    t = al.run_tables(name, rng)
    per = t["per_all"]
    m = t["m"]
    bins = sorted(int(c[5:]) for c in m.columns if c.startswith("pref_") and c[5:].isdigit())
    exposure = bins[-1]
    blocks = range(1, exposure // 20 + 1)          # 20장 단위 구간 (세트 크기와 상관없이 같은 잣대)
    pb = m.groupby(["group", "bot_id"])[[f"pref_{e}" for e in bins]].mean()
    for s in blocks:
        per[f"set{s}"] = pb[[f"pref_{20 * s - 10}", f"pref_{20 * s}"]].mean(axis=1)
    r = {"name": name, "exposure": exposure}
    for m in MODES:
        if m not in per.index.get_level_values(0):
            continue
        r[f"{m}"] = per.loc[m, "pref_all"].mean() * 100
        for s in blocks:
            r[f"{m}_set{s}"] = per.loc[m, f"set{s}"].mean() * 100
        r[f"{m}_final"] = per.loc[m, "final_like20"].mean()
        r[f"{m}_near"] = per.loc[m, "near_hit20"].mean() * 100
    for key, a, b in STEPS:
        d = (per.loc[b, "pref_all"] - per.loc[a, "pref_all"]).to_numpy() * 100
        r[key], r[f"{key}_lo"], r[f"{key}_hi"] = al.boot_paired(d, rng)
        df = (per.loc[b, "final_like20"] - per.loc[a, "final_like20"]).to_numpy()
        r[f"{key}_final"] = df.mean()
    for key in ("screen", "skip", "reason"):
        r[f"{key}_share"] = r[key] / r["total"] * 100 if r["total"] else np.nan
    # 세트 2 -> 3 상승폭 (볼수록 좋아지는가)
    for m in ("swipe_s1", "swipe_s2", "swipe_v2", "list_v2"):
        r[f"{m}_slope"] = r[f"{m}_set3"] - r[f"{m}_set2"]
        r[f"{m}_last"] = r[f"{m}_set{blocks[-1]}"]
    return r


def verdict(d, lo, hi) -> str:
    return "▲" if lo > 0 else ("▼" if hi < 0 else "·")


def analyze() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    names = ["ladder"] + [f"ladder__f{k}" for k in range(1, 8)] + [c["name"] for c in conds()]
    rows = []
    for n in names:
        r = one(n, rng)
        if r:
            r["round"] = next((c["round"] for c in conds() if c["name"] == n), 0)
            rows.append(r)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "overnight_ladder.csv", index=False)
    lines = ["# 밤샘 탐색: 조건별 데이터 사다리 (검증 주, 봇 전원 paired)", "",
             "찜 비율 %. 단계 차이 %p [95% 구간]. ▲ 올림 · 효과 없음 ▼ 내림. 몫 = 단계 / 합계.", "",
             "| 조건 | 라운드 | 목록 v2 | S3 | 화면 | 넘김 | 이유 | 합계 | 몫 화면/넘김/이유 | '비싸요'만 | '이미 있어요'만 | S1 2→3세트 | S3 2→3세트 | 최종20 S3−목록 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        def c(k):
            return f"{r[k]:+.1f} [{r[k + '_lo']:+.1f}, {r[k + '_hi']:+.1f}] {verdict(r[k], r[k + '_lo'], r[k + '_hi'])}"
        lines.append(f"| {r['name']} | {r['round']} | {r['list_v2']:.1f} | {r['swipe_v2']:.1f} | {c('screen')} | {c('skip')} | "
                     f"{c('reason')} | {c('total')} | {r['screen_share']:.0f}/{r['skip_share']:.0f}/{r['reason_share']:.0f} | "
                     f"{c('price_only')} | {c('have_only')} | {r['swipe_s1_slope']:+.1f} | {r['swipe_v2_slope']:+.1f} | "
                     f"{r['total_final']:+.1f} |")
    (OUT / "overnight_ladder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze", "list"])
    ap.add_argument("--round", type=int, nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.round, a.jobs)
    elif a.cmd == "analyze":
        analyze()
    else:
        for c in conds(a.round):
            print(c["round"], c["name"], c["set"])


if __name__ == "__main__":
    main()
