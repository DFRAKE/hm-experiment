"""그룹 A: 목록형 (일반 쇼핑몰). 주 카테고리를 누르고 인기도순 고정 목록 60개를 본다."""
from __future__ import annotations

import numpy as np

from src.bots import Bot
from src.items import ItemPool
from src.modes.common import FINAL_K, N_SETS, SET_SIZE, Session, show_set
from src.recommenders.base import Recommender


def run_list_session(bot: Bot, pool: ItemPool, rec: Recommender, rng: np.random.Generator,
                     seed: int, mode: str = "list", set_size: int = SET_SIZE, n_sets: int = N_SETS) -> Session:
    s = Session(bot.bot_id, mode)
    for set_no in range(1, n_sets + 1):
        items = rec.first_set(set_size) if set_no == 1 else rec.next_set(set_size, set(s.shown))
        # 목록형 화면에는 넘긴 이유 입력이 없다 -> reason은 로그에만 남기고 추천기에는 None
        show_set(s, bot, pool, rec, items, set_no, pass_reason=False, rng=rng, seed=seed)
    s.final = rec.final_top(FINAL_K, exclude=set())
    return s
