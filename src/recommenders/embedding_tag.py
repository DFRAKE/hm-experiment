"""그룹 B(스와이프형) 기준 추천기 `EmbeddingTagRecommender` (5장).

- 사용자 벡터 = 정규화(찜 임베딩 평균 − 0.3 × not_my_taste 임베딩 평균). 찜이 없으면 유사도 항은 0.
- 태그 점수: 모든 태그 값 50점 시작, 찜 +10, not_my_taste −5 (0~100으로 자른 뒤 /100).
  expensive → 가격 상한을 그 상품 가격으로 낮춤 (상한 이상은 제외)
  already_have → 그 종류를 다음 출력(다음 세트 또는 최종 추천)에서 제외
- 점수 = 0.7 × 코사인 유사도 + 0.3 × 태그 점수 평균
- 세트 구성: 취향 14 (점수 상위) / 변형 4 (상위 상품과 같은 종류, 다른 색) / 탐색 2 (아직 안 본 카테고리 인기 상품)
"""
from __future__ import annotations

import numpy as np

from src.items import ItemPool
from src.recommenders.popularity import popularity_order

TAG_COLS = ("product_type_name", "colour_group_name", "index_group_name", "garment_group_name")


class EmbeddingTagRecommender:
    def __init__(self, pool: ItemPool, rng: np.random.Generator, params: dict, category: str | None = None):
        """category가 주어지면 (v2: 관심 카테고리를 알고 시작) 그 카테고리 안에서만 추천한다."""
        self.pool, self.rng, self.p = pool, rng, params
        df = pool.df
        self.category = category
        self.codes = {}          # 태그 칼럼 -> 상품별 코드
        self.tag_score = {}      # 태그 칼럼 -> 값별 점수
        for c in TAG_COLS:
            codes, uniq = df[c].factorize(use_na_sentinel=False)
            self.codes[c] = codes
            self.tag_score[c] = np.full(len(uniq), float(params["tag_start"]))
        self.type_code = self.codes["product_type_name"]
        self.colour_code = self.codes["colour_group_name"]
        self.pop = df["popularity"].to_numpy()
        self.log_pop = np.log1p(self.pop) / max(np.log1p(self.pop).max(), 1e-9)   # 0~1
        self.index_group = df["index_group_name"].to_numpy(dtype=object)
        self.garment_group = df["garment_group_name"].to_numpy(dtype=object)
        self.cat_mask = (self.index_group == category) if category else np.ones(len(pool), dtype=bool)
        self.like_sum = np.zeros(pool.emb.shape[1])
        self.n_like = 0
        self.neg_sum = np.zeros(pool.emb.shape[1])
        self.n_neg = 0
        self.price_cap = np.inf
        self.block_types: set[int] = set()       # already_have: 다음 출력에서만 제외
        self.shown_rows: set[int] = set()

    # ------------------------------------------------------------ 인터페이스
    def first_set(self, k: int) -> list[str]:
        """카테고리를 알면 그 카테고리 인기순 k장 (v2, 목록형 1페이지와 같은 시작점).
        모르면 테마 스프레드: index_group_name별 인기 상품을 돌아가며 k장 (원 설계)."""
        if self.category:
            out = popularity_order(self.pool, self.cat_mask)[:k]
            self.shown_rows.update(self.pool.rows(out))
            return out
        cats = sorted(set(self.index_group))
        queues = {c: popularity_order(self.pool, self.index_group == c) for c in cats}
        out, i = [], 0
        while len(out) < k:
            q = queues[cats[i % len(cats)]]
            if q:
                out.append(q.pop(0))
            i += 1
        self.shown_rows.update(self.pool.rows(out))
        return out

    def update(self, article_id: str, action: str, reason: str | None) -> None:
        r = self.pool.index[article_id]
        self.shown_rows.add(r)
        if action == "like":
            self.like_sum += self.pool.emb[r]
            self.n_like += 1
            self._bump_tags(r, self.p["tag_like"])
        elif reason == "not_my_taste":
            self.neg_sum += self.pool.emb[r]
            self.n_neg += 1
            self._bump_tags(r, self.p["tag_not_my_taste"])
        elif reason == "expensive":
            self.price_cap = min(self.price_cap, float(self.pool.price[r]))
        elif reason == "already_have":
            self.block_types.add(int(self.type_code[r]))

    def next_set(self, k: int, exclude: set[str]) -> list[str]:
        n_var, n_exp = self.p["n_variation"], self.p["n_explore"]
        n_taste = k - n_var - n_exp
        score, ok = self._candidates(exclude)
        order = np.flatnonzero(ok)[np.argsort(-score[ok], kind="stable")]
        taste = list(order[:n_taste])
        chosen = set(taste)

        variation = []
        for top in taste:
            if len(variation) >= n_var:
                break
            same = (self.type_code[order] == self.type_code[top]) & (self.colour_code[order] != self.colour_code[top])
            for r in order[same]:
                if r not in chosen:
                    variation.append(r)
                    chosen.add(r)
                    break

        explore = self._explore(ok, chosen, n_exp)
        chosen.update(explore)
        rows = taste + variation + explore
        for r in order:                       # 모자라면 점수 순으로 채움
            if len(rows) >= k:
                break
            if r not in chosen:
                rows.append(r)
                chosen.add(r)
        rows = [rows[i] for i in self.rng.permutation(len(rows))]   # 세트 안 순서는 섞음
        self.block_types = set()
        return [self.pool.ids[r] for r in rows]

    def final_top(self, k: int, exclude: set[str]) -> list[str]:
        score, ok = self._candidates(exclude)
        order = np.flatnonzero(ok)[np.argsort(-score[ok], kind="stable")]
        self.block_types = set()
        return [self.pool.ids[r] for r in order[:k]]

    # ------------------------------------------------------------ 내부
    def _bump_tags(self, row: int, delta: float) -> None:
        for c in TAG_COLS:
            self.tag_score[c][self.codes[c][row]] += delta

    def user_vector(self) -> np.ndarray | None:
        if self.n_like == 0:
            return None
        u = self.like_sum / self.n_like
        if self.n_neg:
            u = u - self.p["neg_weight"] * (self.neg_sum / self.n_neg)
        n = np.linalg.norm(u)
        return u / n if n > 0 else None

    def scores(self) -> np.ndarray:
        u = self.user_vector()
        cos = self.pool.emb @ u.astype(np.float32) if u is not None else np.zeros(len(self.pool))
        tag = np.mean([np.clip(self.tag_score[c][self.codes[c]], 0, 100) / 100 for c in TAG_COLS], axis=0)
        s = self.p["w_cos"] * cos + self.p["w_tag"] * tag
        w_pop = self.p.get("w_pop", 0.0)                           # 명세 기준 구현은 0 (인기도 미반영)
        if w_pop:
            s = s + w_pop * self.log_pop
        return s + 1e-9 * self.pop / max(self.pop.max(), 1)       # 동점이면 인기순

    def _candidates(self, exclude: set[str]) -> tuple[np.ndarray, np.ndarray]:
        ok = self.cat_mask.copy()
        if exclude:
            ok[self.pool.rows(exclude)] = False
        relaxed = ok.copy()
        ok &= self.pool.price < self.price_cap
        if self.block_types:
            ok &= ~np.isin(self.type_code, list(self.block_types))
        if not ok.any():                      # 조건이 너무 좁으면 가격·종류 제한을 푼다
            ok = relaxed
        return self.scores(), ok

    def _explore(self, ok: np.ndarray, chosen: set[int], n: int) -> list[int]:
        """아직 안 본 index_group_name의 인기 상품. 다 봤으면 안 본 garment_group_name으로."""
        out = []
        shown = np.fromiter(self.shown_rows, dtype=np.int64) if self.shown_rows else np.array([], dtype=np.int64)
        for groups in (self.index_group, self.garment_group):
            unseen = sorted(set(groups[ok]) - set(groups[shown]))
            for g in self.rng.permutation(len(unseen)):
                if len(out) >= n:
                    return out
                cand = np.flatnonzero(ok & (groups == unseen[g]))
                cand = [r for r in cand[np.argsort(-self.pop[cand], kind="stable")] if r not in chosen]
                if cand:
                    out.append(cand[0])
                    chosen = chosen | {cand[0]}
        return out
