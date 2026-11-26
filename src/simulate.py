"""세션 실행: 봇마다 목록형·스와이프형 세션을 돌리고 로그와 최종 추천을 저장한다.

- 봇 1,000명 전원을 모든 방식으로 실행한다 (6.4 paired 부록용). `assigned` 칼럼이 True인 행이 메인 A/B 결과다.
  원 설계: list(A) vs swipe(B) / 수정 설계 v2: list_v2(A') vs swipe_v2(B') / pop(P): 인기 기준선
- 정답 주 데이터는 읽지 않는다. 채점은 src.scoring 에서 한다. --split 은 채점할 주를 기록만 한다.

실행: python -m src.simulate --seeds 0 1 2 3 4 [--limit 50]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.bots import Bot, build_profiles, load_bot_table, load_train_tx
import json

from src.config import experiment_tag, load_config, session_meta, sessions_dir
from src.items import ItemPool, load_pool
from src.modes.common import N_SETS, SET_SIZE, Session
from src.modes.list_mode import run_list_session
from src.modes.swipe_mode import run_swipe_session
from src.recommenders.embedding_tag import EmbeddingTagRecommender
from src.recommenders.list_plus import ListPlusRecommender, PopularityBaseline
from src.recommenders.popularity import PopularityRecommender

MODES = ("list", "swipe", "list_v2", "swipe_v2", "pop")    # 순서 고정 (난수 seed에 쓰임)
MODE_GROUP = {"list": "A", "list_v2": "A", "pop": "A", "swipe": "B", "swipe_v2": "B"}


def run_bot(bot: Bot, k: int, main_cat: str, mode: str, pool: ItemPool, cfg: dict, seed: int) -> Session:
    m = MODES.index(mode)
    react_rng = np.random.default_rng([seed, k, m, 0])     # 봇 반응(변덕)용
    rec_rng = np.random.default_rng([seed, k, m, 1])       # 추천기용
    if mode == "list":
        return run_list_session(bot, pool, PopularityRecommender(pool, main_cat), react_rng, seed)
    if mode == "swipe":
        rec = EmbeddingTagRecommender(pool, rec_rng, cfg["recommender"])
        return run_swipe_session(bot, pool, rec, react_rng, seed)
    if mode == "list_v2":
        rec = ListPlusRecommender(pool, main_cat, rec_rng, cfg["recommender_v2"])
        return run_list_session(bot, pool, rec, react_rng, seed, mode=mode)
    if mode == "swipe_v2":
        rec = EmbeddingTagRecommender(pool, rec_rng, cfg["recommender_v2"], category=main_cat)
        return run_swipe_session(bot, pool, rec, react_rng, seed, mode=mode)
    if mode == "pop":
        return run_list_session(bot, pool, PopularityBaseline(pool, main_cat), react_rng, seed, mode=mode)
    raise ValueError(mode)


def run_seed(seed: int, cfg: dict, pool: ItemPool, train_tx: pd.DataFrame, table: pd.DataFrame,
             modes=MODES, progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    bots = build_profiles(train_tx, pool, table, cfg, seed)
    logs, finals = [], []
    it = enumerate(bots)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, total=len(bots), desc=f"seed {seed}", unit="bot", mininterval=5)
    for k, bot in it:
        for mode in modes:
            s = run_bot(bot, k, table.at[bot.bot_id, "main_cat"], mode, pool, cfg, seed)
            assigned = MODE_GROUP[mode] == bot.group
            for r in s.log:
                r["ab_group"], r["assigned"] = bot.group, assigned
            logs.extend(s.log)
            finals.extend({"bot_id": bot.bot_id, "group": mode, "seed": seed, "ab_group": bot.group,
                           "assigned": assigned, "rank": i, "article_id": a} for i, a in enumerate(s.final))
    return pd.DataFrame(logs), pd.DataFrame(finals)


def main(argv=None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["session"]["seeds"])
    ap.add_argument("--split", choices=["valid", "test"], default="valid",
                    help="채점할 주 (세션 자체는 같음, 기록만 함)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N명만 (빠른 점검용)")
    ap.add_argument("--overwrite", action="store_true", help="다른 설정으로 돌린 기존 세션을 덮어씀")
    args = ap.parse_args(argv)
    assert cfg["session"]["exposure"] == SET_SIZE * N_SETS, "노출 예산은 20장 × 3세트 = 60으로 고정"

    pool = load_pool(cfg)
    train_tx = load_train_tx(cfg)
    table = load_bot_table(cfg)
    if args.limit:
        table = table.iloc[:args.limit]
    out = sessions_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    meta = session_meta(cfg)
    suffix = f"_limit{args.limit}" if args.limit else ""
    for seed in args.seeds:          # 다른 설정으로 만든 세션을 모르고 덮어쓰지 않게
        old_meta = out / f"meta_seed{seed}{suffix}.json"
        old_log = out / f"log_seed{seed}{suffix}.parquet"
        if old_log.exists() and not args.overwrite and (
                not old_meta.exists() or json.loads(old_meta.read_text()) != json.loads(json.dumps(meta))):
            raise SystemExit(f"{old_log} 는 다른 설정(또는 설정 기록 없음)으로 만든 세션입니다. "
                             "experiment.variant 로 폴더를 나누거나 --overwrite 를 붙이세요.")
    for seed in args.seeds:
        t0 = time.time()
        log, final = run_seed(seed, cfg, pool, train_tx, table, progress=True)
        log.to_parquet(out / f"log_seed{seed}{suffix}.parquet", index=False)
        final.to_parquet(out / f"final_seed{seed}{suffix}.parquet", index=False)
        (out / f"meta_seed{seed}{suffix}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        a = log[log["assigned"]]
        rates = a.groupby("group")["action"].apply(lambda x: x.eq("like").mean()).round(3).to_dict()
        print(f"[{experiment_tag(cfg)}] seed {seed}: {len(table)} bots × {len(MODES)} modes, {len(log):,} exposures, "
              f"{time.time() - t0:.0f}s | like rate (assigned) {rates}", flush=True)


if __name__ == "__main__":
    main()
