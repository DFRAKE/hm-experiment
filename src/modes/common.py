"""두 화면 공통: 상품을 보여 주고 반응을 기록한다."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.bots import Bot
from src.items import ItemPool
from src.recommenders.base import Recommender

SET_SIZE = 20
N_SETS = 3            # 노출 예산 60 = 20 × 3
FINAL_K = 20


@dataclass
class Session:
    bot_id: str
    mode: str                                   # list | swipe
    log: list[dict] = field(default_factory=list)
    final: list[str] = field(default_factory=list)

    @property
    def shown(self) -> list[str]:
        return [r["article_id"] for r in self.log]


def show_set(session: Session, bot: Bot, pool: ItemPool, rec: Recommender, items: list[str],
             set_no: int, pass_reason: bool, rng: np.random.Generator, seed: int) -> None:
    seen = set(session.shown)
    for pos, aid in enumerate(items):
        if aid in seen:
            raise RuntimeError(f"같은 상품 중복 노출: {aid}")
        seen.add(aid)
        r = bot.react(pool, pool.index[aid], rng)
        session.log.append({"bot_id": bot.bot_id, "group": session.mode, "seed": seed, "set_no": set_no,
                            "pos": pos, "article_id": aid, "action": r.action, "reason": r.reason,
                            "sim_to_taste": round(r.sim, 6)})
        rec.update(aid, r.action, r.reason if pass_reason else None)
