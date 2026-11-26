"""7장 테스트 중 3장(데이터 준비) 단계에서 돌릴 수 있는 것.

- test_no_leakage: 학습 기간 특징(봇 특징, 상품 대표 가격·인기도)에 넘어가는 거래의 최대 날짜 ≤ 2020-09-08
  (4장 봇 프로필 함수 bots.py가 생기면 여기에 같은 검사를 추가한다)
- test_stratified: 모든 묶음에서 두 그룹 인원 차이 ≤ 1
- test_group_balance: 균형 확인 항목 SMD < 0.1 (아니면 경고)
"""
import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from src import prepare
from src.config import load_config

CFG = load_config()
TRAIN_END = pd.Timestamp(CFG["periods"]["train"][1])
INTERIM = CFG["paths"]["interim"]
BOTS = INTERIM / "bots.parquet"
needs_data = pytest.mark.skipif(not BOTS.exists(), reason="python -m src.prepare 먼저 실행")


def _fake_data(n_customers=200, seed=0):
    rng = np.random.default_rng(seed)
    cats = ["Ladieswear", "Divided", "Menswear", "Sport", "Baby/Children"]
    arts = pd.DataFrame({
        "article_id": [f"{i:010d}" for i in range(100)],
        "prod_name": "x", "product_group_name": "g", "colour_group_name": "c",
        "section_name": "s", "garment_group_name": "gg", "detail_desc": "d",
        "index_group_name": [cats[i % 5] for i in range(100)],
        "product_type_name": [f"type{i % 12}" for i in range(100)],
    })
    days = pd.date_range("2020-07-01", "2020-09-22")
    rows = []
    for c in range(n_customers):
        for _ in range(rng.integers(3, 40)):
            rows.append((rng.choice(days), f"cust{c:04d}", arts.article_id[rng.integers(100)],
                         rng.uniform(0.01, 0.05), 2))
    tx = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"])
    ages = pd.Series(rng.integers(18, 70, n_customers).astype(float),
                     index=[f"cust{c:04d}" for c in range(n_customers)])
    return tx, arts, ages


def test_no_leakage(monkeypatch):
    """가짜 데이터로 전체 준비 과정을 돌리며, 학습 기간 함수에 넘어간 거래의 최대 날짜를 기록한다."""
    seen = {}
    for name in ("bot_features", "pool_stats"):
        orig = getattr(prepare, name)

        def spy(tx, *a, _orig=orig, _name=name, **k):
            seen[_name] = tx["t_dat"].max()
            return _orig(tx, *a, **k)
        monkeypatch.setattr(prepare, name, spy)

    tx, arts, ages = _fake_data()
    assert tx["t_dat"].max() > TRAIN_END            # 가짜 데이터에 검증/정답 주 거래가 실제로 있음
    cfg = copy.deepcopy(CFG)
    cfg["bots"]["n_bots"] = 20
    cfg["bots"]["min_stratum_size"] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepare.build_all(tx, arts, ages, cfg)
    assert set(seen) == {"bot_features", "pool_stats"}
    for name, mx in seen.items():
        assert mx <= TRAIN_END, f"{name}에 {mx.date()} 거래가 들어감"

    # 학습 기간 이후 거래를 직접 넣으면 거부해야 한다
    monkeypatch.undo()
    with pytest.raises(ValueError):
        prepare.bot_features(tx, arts, CFG["periods"]["train"][1])
    with pytest.raises(ValueError):
        prepare.pool_stats(tx, CFG)


@needs_data
def test_no_leakage_real_data():
    """실제 데이터: 학습 구간 최대 날짜와, 저장된 봇 특징이 학습 기간 거래만으로 재현되는지."""
    tx = pd.read_parquet(INTERIM / "tx_2020q3.parquet")
    periods = prepare.split_periods(tx, CFG)
    assert periods["train"]["t_dat"].max() <= TRAIN_END
    bots = pd.read_parquet(BOTS)
    train_b = periods["train"][periods["train"]["customer_id"].isin(bots.index)]
    feat = prepare.bot_features(train_b, prepare.load_articles(CFG), CFG["periods"]["train"][1])
    feat = feat.loc[bots.index]
    for col in ("n_train", "diversity", "main_cat"):
        assert (feat[col] == bots[col]).all(), col
    assert np.allclose(feat["price_median"], bots["price_median"])


def test_pool_no_future_items_fake():
    """가짜 데이터: 검증·정답 주에만 팔린 상품은 풀에 들어가지 않는다."""
    tx, arts, _ = _fake_data()
    periods = prepare.split_periods(tx, CFG)
    future_only = set(arts["article_id"]) - set(periods["train"]["article_id"])
    # 미래에만 팔린 상품을 하나 확실히 만든다
    new_id = "9999999999"
    arts = pd.concat([arts, arts.iloc[[0]].assign(article_id=new_id)], ignore_index=True)
    late = pd.DataFrame({"t_dat": [pd.Timestamp("2020-09-10"), pd.Timestamp("2020-09-20")],
                         "customer_id": "cust0000", "article_id": new_id, "price": 0.02,
                         "sales_channel_id": 2})
    periods = prepare.split_periods(pd.concat([tx, late], ignore_index=True), CFG)
    pool = prepare.build_pool(periods["train"], arts, CFG)
    assert new_id not in set(pool["article_id"])
    assert not (future_only | {new_id}) & set(pool["article_id"])


@needs_data
def test_pool_no_future_items():
    """실제 상품 풀의 모든 상품이 학습 기간(≤ 2020-09-08)에 1건 이상 팔렸다."""
    pool = pd.read_parquet(INTERIM / "article_pool.parquet")
    tx = pd.read_parquet(INTERIM / "tx_2020q3.parquet", columns=["t_dat", "article_id"])
    train_ids = set(prepare.split_periods(tx, CFG)["train"]["article_id"])
    assert pool["article_id"].is_unique
    missing = set(pool["article_id"]) - train_ids
    assert not missing, f"학습 기간에 안 팔린 상품 {len(missing)}개: {sorted(missing)[:5]}"
    assert pool["price"].notna().all()          # 대표 가격 = 학습 기간 중앙값이 모두 있어야 함
    assert len(pool) == len(train_ids)          # 학습 기간 판매 상품은 빠짐없이 들어감


@needs_data
def test_stratified():
    bots = pd.read_parquet(BOTS)
    assert len(bots) == CFG["bots"]["n_bots"]
    assert bots.index.is_unique
    gc = bots.groupby(["stratum", "group"]).size().unstack(fill_value=0)
    diff = (gc["A"] - gc["B"]).abs()
    assert (diff <= 1).all(), diff[diff > 1].to_dict()
    assert abs((bots["group"] == "A").sum() - (bots["group"] == "B").sum()) <= 1

    # 비례 배분: 묶음별 뽑은 수가 후보 비율 × 1,000과 1명 미만 차이
    cand = pd.read_parquet(INTERIM / "bot_candidates.parquet")
    quota = cand["stratum"].value_counts() / len(cand) * len(bots)
    sampled = bots["stratum"].value_counts().reindex(quota.index, fill_value=0)
    assert ((sampled - quota).abs() < 1).all()


@needs_data
def test_group_balance():
    bots = pd.read_parquet(BOTS)
    thr = CFG["bots"]["smd_threshold"]
    t = prepare.balance_table(bots, thr)
    expected = {"학습 기간 구매 수 (평균)", "취향 다양성 (평균)", "가격 중앙값 (평균)",
                "나이 (평균, 결측 제외)"} | {f"주 카테고리 비율: {c}" for c in bots["main_cat"].unique()}
    assert set(t["item"]) == expected
    bad = t[t["abs_SMD"] >= thr]
    for r in bad.itertuples():
        warnings.warn(f"SMD {r.SMD:+.3f} ≥ {thr}: {r.item} (보정 변수로 보고)")
