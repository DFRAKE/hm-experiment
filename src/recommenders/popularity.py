"""그룹 A(목록형)용: 카테고리 하나의 인기도순 고정 목록."""
from __future__ import annotations

import numpy as np

from src.items import ItemPool


def popularity_order(pool: ItemPool, mask: np.ndarray | None = None) -> list[str]:
    """인기도 내림차순, 동률이면 article_id 오름차순."""
    rows = np.arange(len(pool)) if mask is None else np.flatnonzero(mask)
    pop = pool.df["popularity"].to_numpy()[rows]
    order = np.lexsort((np.array(pool.ids)[rows], -pop))
    return [pool.ids[r] for r in rows[order]]


class PopularityRecommender:
    """목록형: 고정 목록을 위에서부터 보여 준다. 반응은 찜만 기억하고 목록은 바꾸지 않는다."""

    def __init__(self, pool: ItemPool, category: str):
        self.category = category
        self.ranked = popularity_order(pool, pool.df["index_group_name"].to_numpy() == category)
        self._cursor = 0
        self.liked: list[str] = []

    def _take(self, k: int, exclude: set[str]) -> list[str]:
        out = []
        while len(out) < k and self._cursor < len(self.ranked):
            a = self.ranked[self._cursor]
            self._cursor += 1
            if a not in exclude:
                out.append(a)
        return out

    def first_set(self, k: int) -> list[str]:
        return self._take(k, set())

    def update(self, article_id: str, action: str, reason: str | None) -> None:
        if action == "like":            # 넘긴 이유는 목록형 화면에 없으므로 받아도 쓰지 않는다
            self.liked.append(article_id)

    def next_set(self, k: int, exclude: set[str]) -> list[str]:
        return self._take(k, exclude)

    def final_top(self, k: int, exclude: set[str]) -> list[str]:
        """찜한 상품을 앞에, 나머지는 같은 카테고리 인기도순으로 채운 k개."""
        out = [a for a in self.liked if a not in exclude][:k]
        seen = set(out)
        for a in self.ranked:
            if len(out) >= k:
                break
            if a not in seen and a not in exclude:
                out.append(a)
                seen.add(a)
        return out
