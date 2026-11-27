"""5장 세션 테스트 (7장): 노출 예산, 중복 없음, 목록 고정, 목록형 reason 숨김, 재현성."""
import numpy as np
import pandas as pd
import pytest

from src.bots import build_profiles, load_bot_table, load_train_tx
from src.config import load_config
from src.items import load_pool
from src.modes.list_mode import run_list_session
from src.recommenders.base import Recommender
from src.recommenders.embedding_tag import EmbeddingTagRecommender
from src.recommenders.popularity import PopularityRecommender, popularity_order
from src.recommenders.list_plus import ListPlusRecommender
from src.simulate import BASE_MODES as MODES, LADDER_MODES, run_bot, run_seed

CFG = load_config()
EXPOSURE = CFG["session"]["exposure"]
from src.items import pool_ready  # noqa: E402
pytestmark = pytest.mark.skipif(not pool_ready(CFG), reason="prepare, embed(+ 사진 설정이면 embed_image) 먼저 실행")
N_BOTS = 40


@pytest.fixture(scope="module")
def ctx():
    pool = load_pool(CFG)
    train = load_train_tx(CFG)
    table = load_bot_table(CFG)
    # 두 그룹·여러 카테고리가 섞이도록 고르게 40명
    table = table.iloc[np.linspace(0, len(table) - 1, N_BOTS).astype(int)]
    bots = build_profiles(train, pool, table, CFG, seed=0)
    log, final = run_seed(0, CFG, pool, train, table)
    return {"pool": pool, "train": train, "table": table, "bots": bots, "log": log, "final": final}


def test_recommenders_follow_interface(ctx):
    pool = ctx["pool"]
    assert isinstance(PopularityRecommender(pool, "Ladieswear"), Recommender)
    assert isinstance(EmbeddingTagRecommender(pool, np.random.default_rng(0), CFG["recommender"]), Recommender)


def test_equal_exposure(ctx):
    n = ctx["log"].groupby(["group", "bot_id"]).size()
    assert set(n.index.get_level_values("group")) == set(MODES)
    assert (n == EXPOSURE).all(), n[n != EXPOSURE]
    assert len(n) == len(MODES) * N_BOTS
    f = ctx["final"].groupby(["group", "bot_id"]).size()
    assert (f == CFG["session"]["final_k"]).all()


def test_no_duplicate_exposure(ctx):
    log = ctx["log"]
    assert not log.duplicated(["group", "bot_id", "article_id"]).any()
    assert not ctx["final"].duplicated(["group", "bot_id", "article_id"]).any()


def test_list_static(ctx):
    """목록형 노출 순서 = 그 카테고리 인기도순 상위 60개 그대로. 봇 반응과 무관."""
    pool, log, table = ctx["pool"], ctx["log"], ctx["table"]
    cat_mask = pool.df["index_group_name"].to_numpy()
    for b, g in log[log["group"] == "list"].groupby("bot_id"):
        g = g.sort_values(["set_no", "pos"])
        cat = table.at[b, "main_cat"]
        assert g["article_id"].tolist() == popularity_order(pool, cat_mask == cat)[:EXPOSURE]
    # 같은 카테고리 봇 둘: 반응이 달라도 목록은 같다
    by_cat = table.groupby("main_cat").head(2).groupby("main_cat").filter(lambda d: len(d) == 2)
    for _, d in by_cat.groupby("main_cat"):
        a, b = d.index
        la = log[(log.group == "list") & (log.bot_id == a)].sort_values(["set_no", "pos"])
        lb = log[(log.group == "list") & (log.bot_id == b)].sort_values(["set_no", "pos"])
        assert la["article_id"].tolist() == lb["article_id"].tolist()


def test_reason_hidden_in_list(ctx):
    pool, table = ctx["pool"], ctx["table"]

    class Spy(PopularityRecommender):
        def __init__(self, *a):
            super().__init__(*a)
            self.reasons = []

        def update(self, article_id, action, reason):
            self.reasons.append(reason)
            super().update(article_id, action, reason)

    n_with_reason = 0
    for bot in ctx["bots"][:10]:
        rec = Spy(pool, table.at[bot.bot_id, "main_cat"])
        s = run_list_session(bot, pool, rec, np.random.default_rng(0), seed=0)
        assert len(rec.reasons) == EXPOSURE
        assert all(r is None for r in rec.reasons)               # 추천기에는 전달 안 됨
        n_with_reason += sum(r["reason"] is not None for r in s.log)
    assert n_with_reason > 0                                      # 봇은 이유를 계산해서 로그에는 남김


def test_reason_passed_in_swipe(ctx):
    """대조: 스와이프형은 reason을 추천기에 전달한다."""
    log = ctx["log"]
    sw = log[log["group"] == "swipe"]
    assert sw["reason"].isin(["not_my_taste", "expensive", "already_have"]).any()


def test_reproducible(ctx):
    pool, train, table = ctx["pool"], ctx["train"], ctx["table"].iloc[:10]
    a_log, a_fin = run_seed(0, CFG, pool, train, table)
    b_log, b_fin = run_seed(0, CFG, pool, train, table)
    pd.testing.assert_frame_equal(a_log, b_log)
    pd.testing.assert_frame_equal(a_fin, b_fin)
    c_log, _ = run_seed(1, CFG, pool, train, table)
    assert not a_log["action"].reset_index(drop=True).equals(c_log["action"].reset_index(drop=True))


def test_paired_sessions_independent(ctx):
    """같은 봇의 목록형 세션 결과는 스와이프형 실행 여부와 무관 (paired 부록 전제)."""
    pool, table, bot = ctx["pool"], ctx["table"], ctx["bots"][0]
    s1 = run_bot(bot, 0, table.at[bot.bot_id, "main_cat"], "list", pool, CFG, 0)
    run_bot(bot, 0, table.at[bot.bot_id, "main_cat"], "swipe", pool, CFG, 0)
    s2 = run_bot(bot, 0, table.at[bot.bot_id, "main_cat"], "list", pool, CFG, 0)
    assert s1.log == s2.log and s1.final == s2.final


# ---------------------------------------------------------------- 수정 설계 v2 (spec_amendment_v2.md)
def test_v2_same_start(ctx):
    """A'와 B'는 같은 시작점: 1페이지/1세트가 주 카테고리 인기순 20개로 같다."""
    log, table, pool = ctx["log"], ctx["table"], ctx["pool"]
    cat = pool.df["index_group_name"].to_numpy()
    for b in table.index[:10]:
        top = popularity_order(pool, cat == table.at[b, "main_cat"])[:20]
        for mode in ("list_v2", "swipe_v2"):
            g = log[(log.group == mode) & (log.bot_id == b) & (log.set_no == 1)].sort_values("pos")
            assert sorted(g["article_id"]) == sorted(top), mode


def test_v2_category_known(ctx):
    """v2 두 그룹 모두 주 카테고리 안에서만 보여 주고 추천한다."""
    log, final, table, pool = ctx["log"], ctx["final"], ctx["table"], ctx["pool"]
    cat = dict(zip(pool.ids, pool.df["index_group_name"]))
    for df in (log, final):
        v2 = df[df.group.isin(["list_v2", "swipe_v2"])]
        assert (v2["article_id"].map(cat).to_numpy() == table.loc[v2["bot_id"], "main_cat"].to_numpy()).all()


def test_v2_list_gets_likes_only(ctx):
    """A'의 엔진에는 찜만 들어간다 (넘김·이유 없음). B'는 넘김 이유를 받는다."""
    pool, table, bot = ctx["pool"], ctx["table"], ctx["bots"][0]
    rec = ListPlusRecommender(pool, table.at[bot.bot_id, "main_cat"], np.random.default_rng(0), CFG["recommender_v2"])
    calls = []
    orig = rec.engine.update
    rec.engine.update = lambda a, act, why: (calls.append((act, why)), orig(a, act, why))
    s = run_list_session(bot, pool, rec, np.random.default_rng(0), seed=0, mode="list_v2")
    n_like = sum(r["action"] == "like" for r in s.log)
    assert len(calls) == n_like and all(c == ("like", None) for c in calls)
    assert rec.engine.n_neg == 0 and rec.engine.price_cap == np.inf


def test_v2_shared_params():
    """A'와 B'(+ 데이터 사다리 방식)는 같은 엔진 설정을 쓴다 (설정 블록이 하나)."""
    import inspect
    import src.simulate as sim
    src_code = inspect.getsource(sim.run_bot)
    assert src_code.count('cfg["recommender_v2"]') == 3     # list_v2, swipe_v2, 사다리


def test_pop_baseline_ignores_likes(ctx):
    final, table, pool = ctx["final"], ctx["table"], ctx["pool"]
    cat = pool.df["index_group_name"].to_numpy()
    for b, g in final[final.group == "pop"].groupby("bot_id"):
        assert g.sort_values("rank")["article_id"].tolist() == popularity_order(pool, cat == table.at[b, "main_cat"])[:20]


def test_ladder_keeps_base_sessions(ctx):
    """데이터 사다리 실행: 같은 엔진 v2(list_v2, swipe_v2)는 기존 실행과 똑같고, 사다리 방식도 노출 예산·중복 규칙을 지킨다."""
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["session"]["ladder"] = True
    log, final = run_seed(0, cfg, ctx["pool"], ctx["train"], ctx["table"])
    assert set(log["group"]) == set(LADDER_MODES)
    n = log.groupby(["group", "bot_id"]).size()
    assert (n == EXPOSURE).all() and len(n) == len(LADDER_MODES) * N_BOTS
    assert not log.duplicated(["group", "bot_id", "article_id"]).any()
    cols = ["group", "bot_id", "set_no", "pos", "article_id", "action", "reason"]
    for m in ("list_v2", "swipe_v2", "pop"):
        a = ctx["log"][ctx["log"]["group"] == m][cols].reset_index(drop=True)
        b = log[log["group"] == m][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    # 1세트는 모든 같은 엔진 방식이 같은 인기순 20장
    s1 = log[log["set_no"] == 1].groupby(["bot_id", "group"])["article_id"].apply(tuple).unstack()
    assert (s1.nunique(axis=1) == 1).all()
