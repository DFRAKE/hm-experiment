"""v2 목록형 A' (찜 기반 추천 목록)와 인기 기준선 P.

A': 같은 공통 엔진(EmbeddingTagRecommender, 카테고리 고정)을 쓰되 목록 화면이 줄 수 있는 신호만 넘긴다.
    - 찜만 전달. 넘김은 신호가 아님 (목록에서 안 누른 상품은 싫다는 뜻이 아님) -> 이유도 없음
    - 1페이지: 카테고리 인기순 20개
    - 2·3페이지: 인기순 다음 상품 14개 + 엔진 추천 6개를 3칸마다 끼워 넣음
    - 최종 추천: 엔진 상위 20개
P: 주 카테고리 인기순. 아무것도 배우지 않는다 (노출은 원 설계 목록형과 같음).
"""
from __future__ import annotations

import numpy as np

from src.items import ItemPool
from src.recommenders.embedding_tag import EmbeddingTagRecommender
from src.recommenders.popularity import PopularityRecommender


class ListPlusRecommender:
    def __init__(self, pool: ItemPool, category: str, rng: np.random.Generator, params: dict):
        self.base = PopularityRecommender(pool, category)
        self.engine = EmbeddingTagRecommender(pool, rng, params, category=category)
        self.n_insert = params.get("list_insert", 6)

    def first_set(self, k: int) -> list[str]:
        return self.base.first_set(k)

    def update(self, article_id: str, action: str, reason: str | None) -> None:
        if action == "like":                    # 목록형이 아는 건 찜뿐
            self.engine.update(article_id, "like", None)
        self.base.update(article_id, action, None)

    def next_set(self, k: int, exclude: set[str]) -> list[str]:
        n_pop = k - self.n_insert
        pop = self.base.next_set(n_pop, exclude)
        score, ok = self.engine._candidates(exclude | set(pop))
        order = np.flatnonzero(ok)[np.argsort(-score[ok], kind="stable")]
        picks = [self.engine.pool.ids[r] for r in order[:k - len(pop)]]
        # 3칸마다 추천 상품 끼워 넣기 (3, 6, 9, ... 번째 자리)
        out, pi, ri = [], iter(pop), iter(picks)
        for pos in range(1, k + 1):
            src = ri if (pos % 3 == 0) else pi
            nxt = next(src, None)
            if nxt is None:
                nxt = next(pi if src is ri else ri, None)
            if nxt is not None:
                out.append(nxt)
        # 이미 본 상품(끼워 넣은 추천 포함)은 다음 페이지에서 exclude로 빠진다
        return out

    def final_top(self, k: int, exclude: set[str]) -> list[str]:
        return self.engine.final_top(k, exclude)


class PopularityBaseline(PopularityRecommender):
    """P: 찜도 무시하고 주 카테고리 인기순 상위 k개를 최종 추천."""

    def final_top(self, k: int, exclude: set[str]) -> list[str]:
        return [a for a in self.ranked if a not in exclude][:k]
