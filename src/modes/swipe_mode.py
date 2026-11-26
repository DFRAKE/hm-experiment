"""그룹 B: 스와이프형 (대추마켓). 1세트 테마 스프레드 20장, 2·3세트는 추천기가 다시 고른다."""
from __future__ import annotations

import numpy as np

from src.bots import Bot
from src.items import ItemPool
from src.modes.common import FINAL_K, N_SETS, SET_SIZE, Session, show_set
from src.recommenders.base import Recommender


def run_swipe_session(bot: Bot, pool: ItemPool, rec: Recommender, rng: np.random.Generator,
                      seed: int, mode: str = "swipe") -> Session:
    s = Session(bot.bot_id, mode)
    for set_no in range(1, N_SETS + 1):
        items = rec.first_set(SET_SIZE) if set_no == 1 else rec.next_set(SET_SIZE, set(s.shown))
        show_set(s, bot, pool, rec, items, set_no, pass_reason=True, rng=rng, seed=seed)
    # 최종 추천: 세션에서 넘긴 상품은 빼고, 찜한 상품은 다시 추천될 수 있다
    # (목록형 최종 추천도 찜한 상품을 포함하므로 찜 상품 포함 여부는 두 그룹이 같다)
    skipped = {r["article_id"] for r in s.log if r["action"] == "skip"}
    s.final = rec.final_top(FINAL_K, exclude=skipped)
    return s
