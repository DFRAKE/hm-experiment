"""명세 수정 v3 취향 회복도 지표 테스트 (작은 가짜 풀로 정의 확인)."""
import numpy as np
import pandas as pd
import pytest

from src import bots as B
from src.items import ItemPool
from src.metrics_taste import repeat_rates, rule_arrays, taste_metrics


def _pool():
    # 상품 0~3: 봇 좌표에서 0·1·2는 취향 방향, 3은 반대. 2는 비쌈, 1은 최근 산 종류(T)
    ids = [f"{i:010d}" for i in range(4)]
    bot = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], np.float32)
    df = pd.DataFrame({"article_id": ids, "product_type_name": ["S", "T", "S", "S"], "price": [.01, .01, .09, .01]})
    return ItemPool(ids=ids, index={a: i for i, a in enumerate(ids)}, emb=np.eye(4, 2, dtype=np.float32),
                    price=df["price"].to_numpy(), type_=df["product_type_name"].to_numpy(dtype=object), df=df,
                    bot_emb=bot, text_emb=np.eye(4, 2, dtype=np.float32))


def _bot():
    return B.Bot("b", "B", taste=np.array([1, 0], np.float32), like_th=0.5, price_th=0.05,
                 recent_types=frozenset({"T"}), noise=0.1)


def test_rule_arrays_match_bot_rule():
    pool, bot = _pool(), _bot()
    rows = np.arange(4)
    _, taste_ok, like = rule_arrays(bot, pool, rows)
    assert like.tolist() == [bot.rule(pool, r).action == "like" for r in rows] == [True, False, False, False]
    assert taste_ok.tolist() == [True, True, True, False]


def test_repeat_rates():
    # 노출: 비싸요 -> 싼 것 -> 비싼 것 -> 이미 있어요 -> 이미 있는 것 -> 새 것
    reasons = np.array(["expensive", None, None, "already_have", None, None], dtype=object)
    pricey = np.array([1, 0, 1, 0, 0, 0], bool)
    owned = np.array([0, 0, 0, 1, 1, 0], bool)
    r = repeat_rates(reasons, pricey, owned)
    assert r["repeat_price"] == pytest.approx(1 / 5)      # 비싸요 뒤 5장 중 비싼 것 1장
    assert r["repeat_type"] == pytest.approx(1 / 2)       # 이미 있어요 뒤 2장 중 이미 있는 것 1장
    assert r["repeat_attr"] == pytest.approx(2 / 5)
    assert np.isnan(repeat_rates(np.array([None, "not_my_taste"], object), pricey[:2], owned[:2])["repeat_attr"])


def test_taste_metrics_session():
    pool, bot = _pool(), _bot()
    order = [3, 2, 0, 1]
    sims = pool.bot_emb[order] @ bot.taste
    log = pd.DataFrame({"bot_id": "b", "group": "swipe", "seed": 0, "set_no": 1, "pos": range(4),
                        "article_id": [pool.ids[i] for i in order],
                        "action": ["like", "skip", "like", "skip"],          # 0번 노출은 변덕 찜
                        "reason": [None, "expensive", None, "already_have"], "sim_to_taste": sims})
    fin = pd.DataFrame({"bot_id": "b", "group": "swipe", "seed": 0, "rank": [0, 1], "article_id": [pool.ids[0], pool.ids[2]]})
    m, chk = taste_metrics(log, fin, {0: {"b": bot}}, pool, exposure=4)
    r = m.iloc[0]
    assert r["pref_all"] == 0.25 and r["first_pref"] == 3
    assert r["final_like20"] == 1 and r["final_like20_new"] == 0 and r["final_taste20"] == 2
    assert chk["rule_vs_observed_agree"] == 0.75
    with pytest.raises(ValueError):                                      # 프로필이 다르면 멈춤
        taste_metrics(log.assign(sim_to_taste=sims + 0.1), fin, {0: {"b": bot}}, pool, exposure=4)
