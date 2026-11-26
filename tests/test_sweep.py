"""scripts/sweep.py: experiments.yaml 조건 펼치기."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sweep", ROOT / "scripts" / "sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_value_name():
    assert [sweep.value_name(v) for v in (0.0, 0.1, 1.0, 7, float("inf"), 0.70)] == ["0", "0p1", "1", "7", "inf", "0p7"]


def test_conditions_main_value_reuses_main(tmp_path, monkeypatch):
    base = {"bot_rule": {"noise": 0.1}, "recommender": {"neg_weight": 0.3}, "recommender_v2": {"neg_weight": 0.3},
            "experiment": {"variant": ""}}
    f = tmp_path / "e.yaml"
    f.write_text("factors:\n  - {name: noise, label: n, keys: [bot_rule.noise], values: [0.0, 0.1]}\n"
                 "  - {name: negw, label: w, keys: [recommender.neg_weight, recommender_v2.neg_weight], values: [0.6]}\n"
                 "presets:\n  p: {label: x, set: {bot_rule.noise: 0.5}}\n")
    monkeypatch.setattr(sweep, "EXPERIMENTS", f)
    cs = sweep.conditions(base)
    names = [c["name"] for c in cs]
    assert names == ["main", "noise-0", "main", "negw-0p6", "p"]
    cfg = sweep.build_cfg(base, cs[3])
    assert cfg["recommender"]["neg_weight"] == cfg["recommender_v2"]["neg_weight"] == 0.6
    assert cfg["experiment"]["variant"] == "negw-0p6" and base["recommender"]["neg_weight"] == 0.3
    with pytest.raises(KeyError):          # config.yaml 에 없는 설정 이름 (오타)
        sweep.build_cfg(base, {"name": "x", "set": {"bot_rule.nosie": 0.2}})


def test_repo_experiments_expand():
    cs = sweep.unique_conditions()
    assert cs[0]["name"] == "main" and len({c["name"] for c in cs}) == len(cs)
    same = {}
    for c in sweep.conditions():          # 같은 이름이면 같은 설정이어야 함 (grid 의 한 가지 조합 = factor 조건)
        assert same.setdefault(c["name"], c["set"]) == c["set"], c["name"]


def test_grid_and_fold_names(tmp_path, monkeypatch):
    base = {"bot_rule": {"noise": 0.1, "like_quantile": 0.8}, "experiment": {"variant": ""}, "bots": {}}
    f = tmp_path / "e.yaml"
    f.write_text("factors:\n  - {name: noise, label: n, keys: [bot_rule.noise], values: [0.0, 0.1]}\n"
                 "  - {name: likeq, label: q, keys: [bot_rule.like_quantile], values: [0.7, 0.8]}\n"
                 "grids:\n  - {name: g, label: g, values: {noise: [0.0, 0.1], likeq: [0.7, 0.8]}}\n"
                 "replicate: {folds: 2, conditions: [main, noise-0]}\n")
    monkeypatch.setattr(sweep, "EXPERIMENTS", f)
    names = [c["name"] for c in sweep.unique_conditions(base)]
    assert names == ["main", "noise-0", "likeq-0p7", "noise-0__likeq-0p7",
                     "main__f1", "main__f2", "noise-0__f1", "noise-0__f2"]
    fold = {c["name"]: c for c in sweep.conditions(base)}["noise-0__f2"]
    cfg = sweep.build_cfg(base, fold)
    assert cfg["bots"]["table"] == "bot_folds/bots_f2.parquet" and cfg["bot_rule"]["noise"] == 0.0
    assert cfg["experiment"]["variant"] == "noise-0__f2"
