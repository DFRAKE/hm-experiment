"""4장 봇 설계: 학습 기간 데이터만으로 봇 프로필을 만들고, 상품을 봤을 때 반응한다.

실행: python -m src.bots      (seed = config bots.seed 로 프로필을 만들어 저장 + 점검 출력)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import load_config, sessions_dir
from src.items import ItemPool, load_pool

ACTIONS = ("like", "skip")
REASONS = ("expensive", "already_have", "not_my_taste")


@dataclass(frozen=True)
class Reaction:
    action: str            # like | skip
    reason: str | None     # skip일 때만: expensive | already_have | not_my_taste
    sim: float             # cos(item.bot_emb, taste)  봇 좌표(설정에 따라 사진 또는 설명 문장)


@dataclass
class Bot:
    bot_id: str
    group: str             # A(목록형) | B(스와이프형)
    taste: np.ndarray      # (384,) 정규화
    like_th: float
    price_th: float
    recent_types: frozenset[str]
    noise: float
    # 명세 수정 v3 (이미 있어요 = 거의 같은 옷): None 이면 원 설계처럼 종류만 같아도 이미 있어요
    dup_th: float | None = None
    recent_emb: dict = field(default_factory=dict)   # 종류 -> 최근 산 상품의 봇 좌표 (k, d)
    # 유행 따르기 (2026-11-27 추가, 기본 끔): 찜 판단 점수 = 취향 유사도 + pop_coef × 상품 인기도(표준화)
    pop_coef: float = 0.0
    pop_z: np.ndarray | None = None                  # (풀 크기,) 모든 봇이 같은 배열을 공유

    def score(self, pool: ItemPool, rows) -> np.ndarray | float:
        """찜 판단 점수. 유행 따르기를 끄면(pop_coef 0) 취향 유사도 그대로."""
        sim = pool.bot_emb[rows] @ self.taste
        if self.pop_coef and self.pop_z is not None:
            return sim + self.pop_coef * self.pop_z[rows]
        return sim

    def _is_dup(self, v: np.ndarray, type_: str) -> bool:
        if type_ not in self.recent_types:
            return False
        if self.dup_th is None:
            return True
        E = self.recent_emb.get(type_)
        return E is not None and float((E @ v).max()) >= self.dup_th

    def rule(self, pool: ItemPool, row: int) -> Reaction:
        """변덕 없이 규칙만 적용 (4.2). 봇은 봇 좌표(pool.bot_emb)로만 판단한다."""
        sim = float(pool.bot_emb[row] @ self.taste)
        if float(self.score(pool, row)) >= self.like_th:
            if pool.price[row] > self.price_th:
                return Reaction("skip", "expensive", sim)
            if self._is_dup(pool.bot_emb[row], pool.type_[row]):
                return Reaction("skip", "already_have", sim)
            return Reaction("like", None, sim)
        return Reaction("skip", "not_my_taste", sim)

    def dup_mask(self, pool: ItemPool, rows: np.ndarray) -> np.ndarray:
        """rows 각각이 '이미 있어요' 조건(취향과 무관)에 걸리는지."""
        return np.fromiter((self._is_dup(pool.bot_emb[r], pool.type_[r]) for r in rows), dtype=bool, count=len(rows))

    def rule_mask(self, pool: ItemPool, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """rule 을 여러 상품에 한꺼번에. 반환: (sim, 취향 기준 통과, 변덕 없이 찜)."""
        rows = np.asarray(rows)
        sim = pool.bot_emb[rows] @ self.taste
        taste_ok = self.score(pool, rows) >= self.like_th
        like = taste_ok & (pool.price[rows] <= self.price_th)
        idx = np.flatnonzero(like)
        like[idx] &= ~self.dup_mask(pool, rows[idx])
        return sim, taste_ok, like

    def react(self, pool: ItemPool, row: int, rng: np.random.Generator) -> Reaction:
        """4.2 규칙 + 변덕. 재현성을 위해 반응 1번마다 난수를 정확히 1개 쓴다."""
        r = self.rule(pool, row)
        if rng.random() < self.noise:
            if r.action == "like":
                return Reaction("skip", "not_my_taste", r.sim)
            return Reaction("like", None, r.sim)
        return r


# ---------------------------------------------------------------- 데이터 읽기 (학습 기간만)
def load_train_tx(cfg: dict) -> pd.DataFrame:
    """학습 기간 거래만 읽는다. 검증·정답 주는 메모리에 올리지 않는다."""
    lo, hi = (pd.Timestamp(d) for d in cfg["periods"]["train"])
    return pd.read_parquet(cfg["paths"]["interim"] / "tx_2020q3.parquet",
                           filters=[("t_dat", ">=", lo), ("t_dat", "<=", hi)])


def load_bot_table(cfg: dict) -> pd.DataFrame:
    """봇 1,000명 표. bots.table 로 다른 고객 묶음(scripts/make_bot_folds.py)을 고를 수 있다 (기본 bots.parquet)."""
    return pd.read_parquet(cfg["paths"]["interim"] / bot_table_name(cfg)).sort_index()


def bot_table_name(cfg: dict) -> str:
    return (cfg.get("bots") or {}).get("table") or "bots.parquet"


# ---------------------------------------------------------------- 프로필 만들기
def _article_types(cfg: dict) -> pd.Series:
    """사진이 없어 풀에서 빠진 상품까지 포함한 article_id -> product_type_name (원본 상품 정보, 날짜 없음)."""
    from src.prepare import load_articles
    return load_articles(cfg).set_index("article_id")["product_type_name"]


DUP_SEED = 20200701
DUP_PAIRS = 20000


def dup_threshold(pool: ItemPool, quantile: float) -> float:
    """'거의 같은 옷' 기준: 같은 종류 상품 무작위 두 개의 봇 좌표 유사도 중 상위 (1-quantile) 경계값 (seed 고정)."""
    types = pool.type_.astype(str)
    uniq, inv = np.unique(types, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    start = np.searchsorted(inv[order], np.arange(len(uniq)))
    count = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(DUP_SEED)
    i = rng.integers(0, len(pool), DUP_PAIRS)
    g = inv[i]
    j = order[start[g] + rng.integers(0, count[g])]
    keep = i != j
    sims = np.einsum("nd,nd->n", pool.bot_emb[i[keep]], pool.bot_emb[j[keep]])
    return float(np.quantile(sims, quantile))


def build_profiles(train_tx: pd.DataFrame, pool: ItemPool, bots: pd.DataFrame,
                   cfg: dict, seed: int) -> list[Bot]:
    """4.1. train_tx는 학습 기간 거래여야 한다 (넘으면 ValueError)."""
    train_end = pd.Timestamp(cfg["periods"]["train"][1])
    if len(train_tx) and train_tx["t_dat"].max() > train_end:
        raise ValueError(f"build_profiles: 학습 기간 이후 거래가 들어옴 (max={train_tx['t_dat'].max().date()})")
    rc = cfg["bot_rule"]
    bot_ids = list(bots.index)
    pos = {b: k for k, b in enumerate(bot_ids)}
    t = train_tx[train_tx["customer_id"].isin(pos)]
    missing = set(t["article_id"]) - set(pool.index)
    if missing and not pool.filtered_by_image:
        raise ValueError(f"학습 기간 구매 상품 {len(missing)}개가 상품 풀에 없음")
    # 사진 기준 풀에서는 사진이 없는 상품 구매를 취향 재료에서 뺀다 (가격·최근 종류는 전체 구매 기준 유지)
    tt = t[t["article_id"].isin(pool.index)] if missing else t

    rows = pool.rows(tt["article_id"])
    who = tt["customer_id"].map(pos).to_numpy()
    age = (train_end - tt["t_dat"]).dt.days.to_numpy()
    w = 0.5 ** (age / rc["taste_half_life_days"])

    taste = np.zeros((len(bot_ids), pool.bot_emb.shape[1]), dtype=np.float64)
    np.add.at(taste, who, pool.bot_emb[rows] * w[:, None])
    norm = np.linalg.norm(taste, axis=1, keepdims=True)
    if (norm == 0).any():
        empty = [bot_ids[i] for i in np.flatnonzero(norm[:, 0] == 0)][:5]
        raise ValueError(f"취향을 만들 구매가 없는 봇이 있음 (사진 있는 구매 0건 등): {empty}")
    taste = (taste / norm).astype(np.float32)

    # 가격 상한 기준: all = 학습 기간 전체 구매 가격 중앙값 (원 설계)
    #                recent = 마지막 price_recent_days 일 구매 가격 중앙값 (계절 따라 가격대가 바뀜, v3). 없으면 전체
    price_med = t.groupby("customer_id")["price"].median()
    basis = rc.get("price_basis", "all")
    if basis == "recent":
        p_from = train_end - pd.Timedelta(days=rc["price_recent_days"] - 1)
        price_med = t.loc[t["t_dat"] >= p_from].groupby("customer_id")["price"].median().combine_first(price_med)
    elif basis != "all":
        raise ValueError(f"price_basis 는 all 또는 recent: {basis}")
    recent_from = train_end - pd.Timedelta(days=rc["recent_days"] - 1)
    rt = t.loc[t["t_dat"] >= recent_from]
    if missing:   # 사진이 없어 풀에서 빠진 상품도 "이미 있어요" 판단에는 넣는다 (종류는 원본 상품 정보에서)
        types = rt["article_id"].map(_article_types(cfg)).to_numpy()
    else:
        types = pool.type_[pool.rows(rt["article_id"])]
    recent = rt.assign(type_=types).dropna(subset=["type_"]).groupby("customer_id")["type_"].agg(frozenset)

    # 이미 있어요: type = 최근 산 종류면 넘김 (원 설계) / near_dup = 최근 산 같은 종류 옷과 사진(봇 좌표)이 거의 같을 때만 (v3)
    mode = rc.get("already_have", "type")
    dup_th, recent_emb = None, {}
    if mode == "near_dup":
        dup_th = dup_threshold(pool, rc["dup_quantile"])
        rp = rt[rt["article_id"].isin(pool.index)]       # 사진(봇 좌표)이 있는 최근 구매만 비교 가능
        rrows = pool.rows(rp["article_id"])
        for (c, ty), idx in pd.Series(np.arange(len(rp)), index=[rp["customer_id"], pool.type_[rrows]]).groupby(level=[0, 1]):
            recent_emb.setdefault(c, {})[ty] = pool.bot_emb[rrows[idx.to_numpy()]]
    elif mode != "type":
        raise ValueError(f"already_have 는 type 또는 near_dup: {mode}")

    # 유행 따르기 (기본 끔). pop_weight = 인기도 1표준편차가 취향 유사도 몇 표준편차(봇마다)만큼인지
    #   fixed    : 모든 봇 같은 값
    #   personal : 학습 기간에 인기 상품을 많이 산 고객일수록 크게 (봇 순위 백분위 × 2 × pop_weight, 평균 ≈ pop_weight)
    beta = float(rc.get("pop_weight", 0.0) or 0.0)
    pop_z, beta_k = None, np.zeros(len(bot_ids))
    if beta:
        pop_z = popularity_z(pool)
        pmode = rc.get("pop_mode", "fixed")
        if pmode == "fixed":
            beta_k[:] = beta
        elif pmode == "personal":
            beta_k = 2 * beta * pop_tendency(tt, pool, bot_ids).rank(pct=True).to_numpy()
        else:
            raise ValueError(f"pop_mode 는 fixed 또는 personal: {pmode}")

    out = []
    for k, b in enumerate(bot_ids):
        rng = np.random.default_rng([seed, k])          # 봇마다 독립, seed·순서로 재현
        sample = rng.choice(len(pool), size=rc["like_sample"], replace=False)
        s_sim = pool.bot_emb[sample] @ taste[k]
        coef = float(beta_k[k] * s_sim.std()) if beta else 0.0
        s = s_sim + coef * pop_z[sample] if coef else s_sim
        like_th = float(np.quantile(s, rc["like_quantile"]))
        out.append(Bot(bot_id=b, group=bots.at[b, "group"], taste=taste[k], like_th=like_th,
                       price_th=float(price_med[b] * rc["price_mult"]),
                       recent_types=recent.get(b, frozenset()), noise=rc["noise"],
                       dup_th=dup_th, recent_emb=recent_emb.get(b, {}), pop_coef=coef, pop_z=pop_z))
    return out


def popularity_z(pool: ItemPool) -> np.ndarray:
    """상품 인기도(학습 기간 마지막 7일 판매 수, 목록형 정렬 기준)를 log1p 후 풀 안에서 표준화."""
    x = np.log1p(pool.df["popularity"].to_numpy(dtype=float))
    return ((x - x.mean()) / x.std()).astype(np.float32)


def pop_tendency(train_tx: pd.DataFrame, pool: ItemPool, bot_ids) -> pd.Series:
    """고객의 인기 추종 성향 = 학습 기간에 산 옷(풀 안)의 평균 인기 백분위. 구매가 없으면 0.5."""
    pct = pd.Series(pool.df["popularity"].to_numpy(dtype=float)).rank(pct=True).to_numpy()
    t = train_tx[train_tx["article_id"].isin(pool.index)]
    s = pd.Series(pct[pool.rows(t["article_id"])], index=t["customer_id"].to_numpy()).groupby(level=0).mean()
    return s.reindex(bot_ids).fillna(0.5)


# ---------------------------------------------------------------- 저장 / 읽기
def save_profiles(bots: list[Bot], cfg: dict, seed: int) -> None:
    """점검용 저장. 실험 버전별 폴더에 둔다 (사진 봇이 설명 문장 봇 프로필을 덮어쓰지 않게)."""
    d = sessions_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"bot_taste_seed{seed}.npy", np.stack([b.taste for b in bots]))
    pd.DataFrame({
        "bot_id": [b.bot_id for b in bots], "group": [b.group for b in bots],
        "like_th": [b.like_th for b in bots], "price_th": [b.price_th for b in bots],
        "recent_types": [json.dumps(sorted(b.recent_types)) for b in bots],
        "noise": [b.noise for b in bots], "dup_th": [b.dup_th for b in bots],
    }).to_parquet(d / f"bot_profiles_seed{seed}.parquet", index=False)


# ---------------------------------------------------------------- 점검
def reaction_check(bots: list[Bot], pool: ItemPool, n_items: int, seed: int) -> pd.DataFrame:
    """봇마다 무작위 상품 n_items개를 보여 줬을 때 반응 분포 (노출 순서·추천 없음, 규칙 점검용)."""
    rng = np.random.default_rng(seed)
    rec = []
    for b in bots:
        for row in rng.choice(len(pool), size=n_items, replace=False):
            r0 = b.rule(pool, row)
            r = b.react(pool, row, rng)
            rec.append((b.group, r0.action, r0.reason, r.action, r.reason))
    return pd.DataFrame(rec, columns=["group", "rule_action", "rule_reason", "action", "reason"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    seed = cfg["bots"]["seed"] if args.seed is None else args.seed
    pool = load_pool(cfg)
    bots = build_profiles(load_train_tx(cfg), pool, load_bot_table(cfg), cfg, seed)
    save_profiles(bots, cfg, seed)

    prof = pd.DataFrame({"like_th": [b.like_th for b in bots], "price_th": [b.price_th for b in bots],
                         "n_recent_types": [len(b.recent_types) for b in bots]})
    print(f"profiles: {len(bots)} bots, seed {seed}")
    print(prof.describe(percentiles=[0.1, 0.5, 0.9]).round(4).to_string())
    chk = reaction_check(bots, pool, n_items=60, seed=seed)
    print("\n[규칙만] ", chk["rule_action"].value_counts(normalize=True).round(3).to_dict(),
          chk["rule_reason"].value_counts(normalize=True, dropna=False).round(3).to_dict())
    print("[변덕 포함]", chk["action"].value_counts(normalize=True).round(3).to_dict(),
          chk["reason"].value_counts(normalize=True, dropna=False).round(3).to_dict())


if __name__ == "__main__":
    main()
