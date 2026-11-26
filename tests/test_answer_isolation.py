"""7장 test_answer_used_only_in_scoring: 정답 주 데이터는 scoring 모듈에서만 쓴다.

예외: prepare.py는 3.3 봇 후보 조건 3(정답 주 구매 1건 이상)을 확인하려고 정답 주를 읽는다.
이때 저장되는 결과(bots.parquet)에는 정답 주에서 나온 칼럼이 없어야 한다.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from src.config import load_config

CFG = load_config()
SRC = Path(__file__).resolve().parent.parent / "src"
ALLOWED = {"scoring.py", "prepare.py"}
ANSWER_ACCESS = re.compile(r"""periods"?\]?\[["']test["']\]|load_answer_week|load_week\(|from src\.scoring|import scoring""")


def test_answer_used_only_in_scoring():
    offenders = []
    for f in SRC.rglob("*.py"):
        if f.name in ALLOWED:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if ANSWER_ACCESS.search(line):
                offenders.append(f"{f.relative_to(SRC)}:{i}: {line.strip()}")
    assert not offenders, "정답 주 접근:\n" + "\n".join(offenders)


@pytest.mark.skipif(not (CFG["paths"]["interim"] / "bots.parquet").exists(), reason="prepare 먼저")
def test_prepare_outputs_have_no_answer_columns():
    bots = pd.read_parquet(CFG["paths"]["interim"] / "bots.parquet")
    assert set(bots.columns) <= {"n_train", "diversity", "price_median", "main_cat", "age",
                                 "count_bin", "div_level", "stratum", "group"}


@pytest.mark.skipif(not __import__("src.items", fromlist=["pool_ready"]).pool_ready(CFG), reason="embed(+embed_image) 먼저")
def test_simulation_reads_no_future_rows(monkeypatch):
    """세션 실행 중 읽는 거래는 모두 학습 기간 안 (pandas.read_parquet 감시)."""
    import src.bots as bots_mod
    from src.items import load_pool
    from src.simulate import run_seed
    seen = []
    real = pd.read_parquet

    def spy(*a, **k):
        df = real(*a, **k)
        if "t_dat" in df.columns:
            seen.append(df["t_dat"].max())
        return df
    monkeypatch.setattr(pd, "read_parquet", spy)
    pool = load_pool(CFG)
    train = bots_mod.load_train_tx(CFG)
    table = bots_mod.load_bot_table(CFG).iloc[:5]
    run_seed(0, CFG, pool, train, table)
    assert seen and max(seen) <= pd.Timestamp(CFG["periods"]["train"][1])


def test_calibration_reads_valid_only():
    """봇 가정 보정(src/calibrate_bots.py)은 검증 주만 읽는다."""
    from src import calibrate_bots
    s = (SRC / "calibrate_bots.py").read_text(encoding="utf-8")
    assert calibrate_bots.SPLIT == "valid"
    assert not re.search(r"""["']test["']""", s)
