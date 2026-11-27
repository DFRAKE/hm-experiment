"""experiments.yaml 의 실험 조건을 모두 돌리고 결과를 모은다 (명세 수정 v3 4-4, 4-5). 검증 주만.

실행:
  .venv/bin/python scripts/sweep.py              # 안 돌린 조건만 세션 + 채점, 끝나면 collect_results
  .venv/bin/python scripts/sweep.py --list       # 조건 목록과 상태만 보기
  .venv/bin/python scripts/sweep.py --only noise-0 likeq-0p9 --jobs 4

조건 종류 (experiments.yaml):
  factors   : 한 가지만 바꿈                      이름 <factor>-<값>
  presets   : 여러 값을 한꺼번에 (이름 직접)
  grids     : 두 가지 이상을 함께 바꾸는 모든 조합   이름 <factor>-<값>__<factor>-<값>
  replicate : 위 조건을 다른 고객 묶음(bot_folds)으로 반복   이름 <조건>__f<k>
조건마다 configs/<이름>.yaml 을 만들고 HM_CONFIG 로 src.simulate, src.scoring score --split valid 를 돌린다.
세션 설정 기록(meta_seed*.json)이 지금 설정과 같고 채점 결과가 세션보다 새로우면 건너뛴다.
로그: data/results/sweep_logs/<이름>.log
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config, results_dir, session_meta, sessions_dir  # noqa: E402

EXPERIMENTS = ROOT / "experiments.yaml"
CONFIGS = ROOT / "configs"
LOGS = ROOT / "data" / "results" / "sweep_logs"
MAIN = "main"
ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def get(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d[k]
    return d


def put(d: dict, dotted: str, v) -> None:
    *head, last = dotted.split(".")
    for k in head:
        d = d.setdefault(k, {})
    d[last] = v


def value_name(v) -> str:
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).replace(".", "p").replace("-", "m")


def conditions(base: dict | None = None) -> list[dict]:
    """[{name, kind, factor, label, value, keys, set, variant, is_main_value, base_cond, fold}] — 메인 먼저.

    kind: main | factor | preset | grid | fold. 메인과 같은 값은 메인(또는 더 짧은 이름의 조건)으로 대신한다."""
    base = base or yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    spec = yaml.safe_load(EXPERIMENTS.read_text(encoding="utf-8"))
    main = {"name": MAIN, "kind": MAIN, "factor": MAIN, "label": "메인 (config.yaml)", "value": None, "keys": [],
            "set": {}, "variant": "", "is_main_value": True}
    out = [main]
    fac = {}
    for f in spec.get("factors", []):
        main_vals = {get(base, k) for k in f["keys"]}
        if len(main_vals) != 1:
            raise ValueError(f"{f['name']}: 메인 설정에서 {f['keys']} 값이 서로 다름 {main_vals}")
        f = {**f, "main": main_vals.pop()}
        fac[f["name"]] = f
        for v in f["values"]:
            is_main = v == f["main"]
            name = MAIN if is_main else f"{f['name']}-{value_name(v)}"
            out.append({"name": name, "kind": "factor", "factor": f["name"], "label": f["label"], "value": v,
                        "keys": f["keys"], "is_main_value": is_main, "variant": "" if is_main else name,
                        "set": {} if is_main else {k: v for k in f["keys"]}})
    for name, p in (spec.get("presets") or {}).items():
        out.append({"name": name, "kind": "preset", "factor": "preset", "label": p["label"], "value": name,
                    "keys": list(p["set"]), "set": dict(p["set"]), "is_main_value": False,
                    "variant": p["set"].get("experiment.variant", name)})
    for g in spec.get("grids") or []:
        names = list(g["values"])
        for combo in itertools.product(*(g["values"][n] for n in names)):
            parts, st, keys = [], {}, []
            for n, v in zip(names, combo):
                keys += fac[n]["keys"]
                if v != fac[n]["main"]:
                    parts.append(f"{n}-{value_name(v)}")
                    st.update({k: v for k in fac[n]["keys"]})
            name = "__".join(parts) or MAIN
            out.append({"name": name, "kind": "grid", "factor": g["name"], "label": g["label"],
                        "value": ", ".join(f"{n}={v}" for n, v in zip(names, combo)), "keys": keys, "set": st,
                        "is_main_value": not parts, "variant": name if parts else ""})
    rep = spec.get("replicate") or {}
    by_name = {}
    for c in out:
        by_name.setdefault(c["name"], c)
    for cname in rep.get("conditions", []):
        c = by_name[cname]
        for k in range(1, int(rep.get("folds", 0)) + 1):
            name = f"{cname}__f{k}"
            out.append({**c, "name": name, "kind": "fold", "base_cond": cname, "fold": k, "is_main_value": False,
                        "variant": name, "set": {**c["set"], "bots.table": f"bot_folds/bots_f{k}.parquet"}})
    return out


# config.yaml 에 없어도 되는 설정 (기본값이 코드에 있음. config.yaml 에 넣으면 기존 세션 기록과 달라져 다시 돌려야 함)
OPTIONAL_KEYS = {"bots.table", "bot_rule.pop_weight", "bot_rule.pop_mode"}


def build_cfg(base: dict, c: dict) -> dict:
    cfg = copy.deepcopy(base)
    for k, v in c["set"].items():
        if k not in OPTIONAL_KEYS:
            get(base, k)                     # 오타 방지: config.yaml 에 없는 설정이면 KeyError
        put(cfg, k, v)
    put(cfg, "experiment.variant", c["variant"])
    return cfg


def write_config(base: dict, c: dict) -> Path:
    if c["name"] == MAIN:
        return ROOT / "config.yaml"
    CONFIGS.mkdir(exist_ok=True)
    p = CONFIGS / f"{c['name']}.yaml"
    head = f"# 자동 생성 (scripts/sweep.py, experiments.yaml): config.yaml + {c['set']}\n"
    text = head + yaml.safe_dump(build_cfg(base, c), allow_unicode=True, sort_keys=False)
    if not p.exists() or p.read_text(encoding="utf-8") != text:
        p.write_text(text, encoding="utf-8")
    return p


def config_path(name: str) -> Path:
    return ROOT / "config.yaml" if name == MAIN else CONFIGS / f"{name}.yaml"


def status(cfg_path: Path) -> str:
    """done | score (세션은 있음, 채점 필요) | run (세션부터) | conflict (같은 폴더에 다른 설정 세션)"""
    cfg = load_config(cfg_path)
    meta = json.loads(json.dumps(session_meta(cfg)))
    sd, seeds = sessions_dir(cfg), cfg["session"]["seeds"]
    logs = [sd / f"log_seed{s}.parquet" for s in seeds]
    if not any(p.exists() for p in logs):
        return "run"
    for s, lg in zip(seeds, logs):
        m = sd / f"meta_seed{s}.json"
        if lg.exists() and (not m.exists() or json.loads(m.read_text()) != meta):
            return "conflict"
    if not all(p.exists() for p in logs):
        return "run"                          # 중간에 멈춘 실행 (같은 설정) → 다시 돌림
    summ = results_dir(cfg, "valid") / "summary.csv"
    if summ.exists() and summ.stat().st_mtime > max(p.stat().st_mtime for p in logs):
        return "done"
    return "score"


MIN_FREE_GB = 5.0                 # 디스크가 이보다 적으면 새 실행을 시작하지 않음 (스왑이 디스크를 채운 사고 뒤 추가)
STOP = threading.Event()


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 2**30


def run_one(c: dict, cfg_path: Path, st: str) -> tuple[str, int, float]:
    if STOP.is_set() or free_gb() < MIN_FREE_GB:
        STOP.set()
        return c["name"], -1, 0.0            # -1 = 디스크 부족으로 시작 안 함
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **ONE_THREAD, "HM_CONFIG": str(cfg_path), "PYTHONUNBUFFERED": "1"}
    # --overwrite: status() 가 같은 설정임을 확인한 뒤(중간에 멈춘 실행)만 여기로 온다
    steps = ([[sys.executable, "-m", "src.simulate", "--overwrite"]] if st == "run" else []) + \
            [[sys.executable, "-m", "src.scoring", "score", "--split", "valid"]]
    t0 = time.time()
    with open(LOGS / f"{c['name']}.log", "w", encoding="utf-8") as f:
        for cmd in steps:
            f.write(f"$ HM_CONFIG={cfg_path.relative_to(ROOT)} {' '.join(cmd[1:])}\n")
            f.flush()
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if rc:
                return c["name"], rc, time.time() - t0
    return c["name"], 0, time.time() - t0


def unique_conditions(base: dict | None = None, only=None) -> list[dict]:
    seen, out = set(), []
    for c in conditions(base):
        if c["name"] in seen or (only and c["name"] not in only):
            continue
        seen.add(c["name"])
        out.append(c)
    return out


def collect() -> None:
    subprocess.call([sys.executable, str(ROOT / "scripts" / "collect_results.py"), "--quiet"], cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4, help="동시에 돌릴 조건 수 (조건 하나 ≈ CPU 1개, 메모리 1.3GB). "
                    "16GB 맥에서 6개는 스왑이 디스크를 채웠음 → 기본 4")
    ap.add_argument("--only", nargs="+", default=None, help="이 이름의 조건만")
    ap.add_argument("--list", action="store_true", help="목록과 상태만 출력")
    args = ap.parse_args()

    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    todo, counts = [], {}
    for c in unique_conditions(base, args.only):
        p = write_config(base, c)
        st = status(p)
        counts[st] = counts.get(st, 0) + 1
        print(f"{st:9s} {c['kind']:6s} {c['name']:28s} {c['label']}"
              + (f" = {c['value']}" if c["kind"] in ("factor", "grid") else ""))
        if st == "conflict":
            raise SystemExit(f"{c['name']}: 결과 폴더에 다른 설정으로 만든 세션이 있습니다. 확인 후 지우거나 이름을 바꾸세요.")
        if st in ("run", "score"):
            todo.append((c, p, st))
    print(f"\n상태: {counts}")
    if args.list:
        return
    order = {"main": 0, "factor": 2, "preset": 3, "grid": 4, "fold": 5}
    todo.sort(key=lambda t: 1 if t[0].get("base_cond") == MAIN else order[t[0]["kind"]])   # 메인의 다른 고객 묶음 먼저
    print(f"돌릴 조건 {len(todo)}개, 동시 {args.jobs}개. 로그: {LOGS.relative_to(ROOT)}/", flush=True)
    failed, skipped, t0 = [], [], time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_one, *t) for t in todo]      # 목록 순서대로 시작 (앞에 둔 조건이 먼저 끝남)
        for i, fut in enumerate(as_completed(futs), 1):
            name, rc, sec = fut.result()
            if rc == -1:
                skipped.append(name)
                continue
            print(f"[{i}/{len(todo)} | 경과 {(time.time() - t0) / 3600:.1f}시간] {'끝' if rc == 0 else '실패'} "
                  f"{name} ({sec / 60:.1f}분)", flush=True)
            if rc:
                failed.append(name)
            if i % 10 == 0:                 # 중간 결과도 자주 모아 둔다
                collect()
    collect()
    print(f"전체 {(time.time() - t0) / 3600:.1f}시간. 결과: data/results/all_runs_valid.md")
    if skipped:
        print(f"디스크 남은 공간이 {MIN_FREE_GB:.0f}GB 미만이라 시작하지 않은 조건 {len(skipped)}개: {skipped}")
    if failed:
        raise SystemExit(f"실패: {failed} (로그 확인)")


if __name__ == "__main__":
    main()
