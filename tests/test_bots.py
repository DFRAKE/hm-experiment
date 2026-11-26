"""4장 봇 테스트.

- test_no_leakage_profiles: 봇 프로필 생성 함수에 넘어가는 거래의 최대 날짜 ≤ 2020-09-08 (7장 test_no_leakage)
- test_react_rules: 4.2 반응 규칙 (명세 외 추가)
- test_profiles_reproducible: 같은 seed면 같은 프로필 (7장 test_reproducible의 4장 부분)
"""
import numpy as np
import pandas as pd
import pytest

from src import bots as B
from src.config import load_config
from src.items import ItemPool

CFG = load_config()
TRAIN_END = pd.Timestamp(CFG["periods"]["train"][1])
from src.items import pool_ready  # noqa: E402
needs_data = pytest.mark.skipif(not pool_ready(CFG), reason="prepare, embed(+ 사진 설정이면 embed_image) 먼저 실행")


def _tiny_pool():
    emb = np.eye(4, dtype=np.float32)
    ids = [f"{i:010d}" for i in range(4)]
    df = pd.DataFrame({"article_id": ids, "product_type_name": ["T-shirt", "T-shirt", "Dress", "Dress"],
                       "price": [0.01, 0.10, 0.01, 0.01]})
    return ItemPool(ids=ids, index={a: i for i, a in enumerate(ids)}, emb=emb,
                    price=df["price"].to_numpy(), type_=df["product_type_name"].to_numpy(dtype=object), df=df)


def _bot(noise=0.0):
    taste = np.array([1, 1, 1, 0], dtype=np.float32) / np.sqrt(3)     # 0,1,2번 상품과 유사도 0.577
    return B.Bot("b", "A", taste, like_th=0.5, price_th=0.05, recent_types=frozenset({"Dress"}), noise=noise)


def test_react_rules():
    pool, bot = _tiny_pool(), _bot()
    assert bot.rule(pool, 0) == B.Reaction("like", None, pytest.approx(0.57735, abs=1e-4))
    assert bot.rule(pool, 1).reason == "expensive"
    assert bot.rule(pool, 2).reason == "already_have"
    assert bot.rule(pool, 3).reason == "not_my_taste"
    # 변덕 100%: like -> skip(not_my_taste), skip -> like(이유 없음)
    flip = _bot(noise=1.0)
    rng = np.random.default_rng(0)
    assert flip.react(pool, 0, rng) == B.Reaction("skip", "not_my_taste", pytest.approx(0.57735, abs=1e-4))
    for row in (1, 2, 3):
        r = flip.react(pool, row, rng)
        assert (r.action, r.reason) == ("like", None)


def test_noise_rate():
    pool, bot = _tiny_pool(), _bot(noise=0.1)
    rng = np.random.default_rng(1)
    flips = np.mean([bot.react(pool, 0, rng).action == "skip" for _ in range(20000)])
    assert abs(flips - 0.1) < 0.01


@needs_data
def test_no_leakage_profiles():
    from src.items import load_pool
    pool = load_pool(CFG)
    train = B.load_train_tx(CFG)
    assert train["t_dat"].max() <= TRAIN_END            # 로더가 학습 기간만 읽음
    table = B.load_bot_table(CFG).iloc[:50]
    seen = []
    orig = B.build_profiles

    def spy(tx, *a, **k):
        seen.append(tx["t_dat"].max())
        return orig(tx, *a, **k)
    spy(train, pool, table, CFG, seed=0)
    assert seen and seen[0] <= TRAIN_END

    # 학습 기간 이후 거래가 섞이면 거부
    late = train.iloc[:1].assign(t_dat=pd.Timestamp("2020-09-16"))
    with pytest.raises(ValueError):
        B.build_profiles(pd.concat([train, late]), pool, table, CFG, seed=0)


@needs_data
def test_profiles_reproducible():
    from src.items import load_pool
    pool = load_pool(CFG)
    train = B.load_train_tx(CFG)
    table = B.load_bot_table(CFG).iloc[:30]
    a = B.build_profiles(train, pool, table, CFG, seed=0)
    b = B.build_profiles(train, pool, table, CFG, seed=0)
    c = B.build_profiles(train, pool, table, CFG, seed=1)
    for x, y in zip(a, b):
        assert np.array_equal(x.taste, y.taste) and x.like_th == y.like_th
        assert x.price_th == y.price_th and x.recent_types == y.recent_types
    assert any(x.like_th != z.like_th for x, z in zip(a, c))   # seed가 다르면 like_th 표본이 달라짐
    assert all(abs(float(np.linalg.norm(x.taste)) - 1) < 1e-5 for x in a)


# ---------------------------------------------------------------- 명세 수정 v3: 가격 기준·이미 있어요(거의 같은 옷)
def test_near_dup_rule():
    pool = _tiny_pool()
    taste = np.array([1, 0, 1, 1], dtype=np.float32) / np.sqrt(3)       # 0·2·3번과 유사도 0.577
    bot = B.Bot("b", "A", taste, like_th=0.5, price_th=0.05, recent_types=frozenset({"Dress"}), noise=0.0,
                dup_th=0.9, recent_emb={"Dress": pool.bot_emb[[2]]})
    assert bot.rule(pool, 2).reason == "already_have"     # 최근 산 Dress 와 같은 옷
    assert bot.rule(pool, 3).action == "like"             # 같은 Dress 라도 다른 옷이면 찜 (원 설계에서는 already_have)
    rows = np.arange(4)
    assert bot.rule_mask(pool, rows)[2].tolist() == [bot.rule(pool, r).action == "like" for r in rows]
    assert bot.dup_mask(pool, rows).tolist() == [False, False, True, False]


def test_dup_threshold_deterministic():
    pool = _tiny_pool()
    a, b = B.dup_threshold(pool, 0.99), B.dup_threshold(pool, 0.99)
    assert a == b and -1 <= a <= 1


def test_price_basis_and_near_dup_profiles():
    import copy
    pool = _tiny_pool()
    tx = pd.DataFrame({"customer_id": ["c"] * 3, "article_id": [pool.ids[0], pool.ids[0], pool.ids[2]],
                       "t_dat": [pd.Timestamp("2020-07-05"), pd.Timestamp("2020-07-06"), TRAIN_END],
                       "price": [0.01, 0.01, 0.04]})
    table = pd.DataFrame({"group": ["A"]}, index=pd.Index(["c"], name="customer_id"))
    cfg = copy.deepcopy(CFG)
    cfg["bot_rule"].update(like_sample=4, price_mult=1.0, recent_days=14, dup_quantile=0.5,
                           price_basis="all", already_have="type")
    old = B.build_profiles(tx, pool, table, cfg, 0)[0]
    assert old.price_th == pytest.approx(0.01) and old.dup_th is None and old.recent_types == {"Dress"}
    cfg["bot_rule"].update(price_basis="recent", price_recent_days=14, already_have="near_dup")
    new = B.build_profiles(tx, pool, table, cfg, 0)[0]
    assert new.price_th == pytest.approx(0.04)            # 최근 14일 구매(0.04)만 기준
    assert new.dup_th is not None and np.array_equal(new.recent_emb["Dress"], pool.bot_emb[[2]])
    assert new.rule(pool, 2).reason == "already_have"     # 작은 풀은 같은 종류 쌍 유사도가 모두 0이라 dup_th = 0
