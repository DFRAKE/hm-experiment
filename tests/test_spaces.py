"""명세 수정 v3 테스트: 봇과 추천기가 서로 다른 좌표(자)를 쓰는지.

- test_rule_uses_bot_space: 봇은 bot_emb로만 찜을 판단한다 (추천기 좌표 emb가 달라도)
- test_recommenders_do_not_read_bot_space: 추천기·세션 코드는 bot_emb를 읽지 않는다 (봇의 정답 자를 못 봄)
- test_bots_do_not_read_rec_space: 봇 코드는 추천기 좌표(pool.emb)를 읽지 않는다
- test_reality_check_uses_text_space: 다음 주 구매 채점은 항상 text_emb
- test_paths_versioned: 원래 설계(text/text)는 원래 폴더, 바뀐 설계는 별도 폴더
- test_image_pool_alignment: (사진 임베딩이 있을 때) 풀 행 순서와 세 좌표가 맞는지
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import bots as B
from src.config import experiment_tag, load_config, results_dir, sessions_dir
from src.items import ItemPool, image_files_ready, load_pool

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()


def _two_space_pool():
    ids = [f"{i:010d}" for i in range(3)]
    rec = np.eye(3, dtype=np.float32)                       # 추천기 좌표
    bot = np.eye(3, dtype=np.float32)[[2, 1, 0]]            # 봇 좌표: 0번과 2번이 뒤바뀜
    df = pd.DataFrame({"article_id": ids, "product_type_name": ["A", "B", "C"], "price": [0.01] * 3})
    return ItemPool(ids=ids, index={a: i for i, a in enumerate(ids)}, emb=rec, price=df["price"].to_numpy(),
                    type_=df["product_type_name"].to_numpy(dtype=object), df=df, bot_emb=bot, text_emb=rec)


def test_rule_uses_bot_space():
    pool = _two_space_pool()
    bot = B.Bot("b", "B", taste=np.array([1, 0, 0], np.float32), like_th=0.5, price_th=1.0,
                recent_types=frozenset(), noise=0.0)
    # 봇 좌표에서 [1,0,0]과 같은 건 2번 상품. 추천기 좌표였다면 0번이 찜됐을 것
    assert bot.rule(pool, 2).action == "like"
    assert bot.rule(pool, 0).action == "skip"


def _src(*globs):
    for g in globs:
        for p in sorted(ROOT.glob(g)):
            yield p, p.read_text(encoding="utf-8")


def test_recommenders_do_not_read_bot_space():
    for p, s in _src("src/recommenders/*.py", "src/modes/*.py", "src/simulate.py"):
        assert "bot_emb" not in s, f"{p} 가 봇 좌표를 읽음"


def test_bots_do_not_read_rec_space():
    s = (ROOT / "src/bots.py").read_text(encoding="utf-8")
    assert not re.search(r"pool\.emb\b", s), "봇이 추천기 좌표(pool.emb)를 읽음"


def test_reality_check_uses_text_space():
    s = (ROOT / "src/scoring.py").read_text(encoding="utf-8")
    assert "pool.text_emb" in s and not re.search(r"pool\.emb\b", s)


def test_paths_versioned():
    orig = {**CFG, "experiment": {"bot_space": "text", "rec_space": "text"}}
    new = {**CFG, "experiment": {"bot_space": "image", "rec_space": "text"}}
    assert sessions_dir(orig) == CFG["paths"]["interim"] / "sessions"
    assert results_dir(orig, "valid") == CFG["paths"]["results"] / "valid"
    assert experiment_tag(new) == "bot-image_rec-text"
    assert sessions_dir(new) == CFG["paths"]["interim"] / "sessions" / "bot-image_rec-text"
    assert results_dir(new, "valid") == CFG["paths"]["results"] / "bot-image_rec-text" / "valid"
    with pytest.raises(ValueError):
        experiment_tag({**CFG, "experiment": {"bot_space": "video"}})


@pytest.mark.skipif(not image_files_ready(CFG), reason="embed_image 먼저 실행")
def test_image_pool_alignment():
    cfg = {**CFG, "experiment": {"bot_space": "image", "rec_space": "text"}}
    pool = load_pool(cfg)
    interim = CFG["paths"]["interim"]
    img_ids = json.loads((interim / "article_img_ids.json").read_text())
    img = np.load(interim / "article_img_emb.npy")
    text_ids = json.loads((interim / "article_ids.json").read_text())
    text = np.load(interim / "article_emb.npy")
    assert set(pool.ids) == set(img_ids) & set(text_ids)
    for a in pool.ids[:: max(1, len(pool) // 50)]:
        r = pool.index[a]
        assert np.array_equal(pool.bot_emb[r], img[img_ids.index(a)])
        assert np.array_equal(pool.emb[r], text[text_ids.index(a)])
        assert pool.df.at[r, "article_id"] == a
    assert pool.bot_emb.shape[1] == CFG["embed_image"]["dim"] and pool.emb.shape[1] == CFG["embed"]["dim"]
