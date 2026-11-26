import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """path가 없으면 환경변수 HM_CONFIG, 그것도 없으면 config.yaml (민감도 분석 등 설정 바꿔 돌리기용)."""
    path = Path(path or os.environ.get("HM_CONFIG") or ROOT / "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = ROOT / v
    return cfg


# ---------------------------------------------------------------- 실험 버전별 경로 (명세 수정 v3)
def spaces(cfg: dict) -> tuple[str, str]:
    e = cfg.get("experiment") or {}
    bot, rec = e.get("bot_space", "text"), e.get("rec_space", "text")
    for s in (bot, rec):
        if s not in ("text", "image"):
            raise ValueError(f"experiment 좌표는 text 또는 image: {s}")
    return bot, rec


def variant(cfg: dict) -> str:
    """민감도 분석 등 같은 좌표에서 설정만 바꾼 실행의 이름 (결과 폴더를 따로 쓰게). 기본은 빈 문자열."""
    v = str((cfg.get("experiment") or {}).get("variant") or "")
    if v and not v.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"experiment.variant 는 영문·숫자·-·_ 만: {v}")
    return v


def experiment_tag(cfg: dict) -> str:
    bot, rec = spaces(cfg)
    v = variant(cfg)
    return f"bot-{bot}_rec-{rec}" + (f"__{v}" if v else "")


def is_original_design(cfg: dict) -> bool:
    return spaces(cfg) == ("text", "text") and not variant(cfg)


def session_meta(cfg: dict) -> dict:
    """세션 결과를 좌우하는 설정. 세션 파일 옆에 저장하고, 채점 때 같은지 확인한다."""
    meta = {"experiment": experiment_tag(cfg), "bot_rule": cfg["bot_rule"],
            "recommender": cfg.get("recommender"), "recommender_v2": cfg.get("recommender_v2"),
            "session": cfg["session"]}
    table = (cfg.get("bots") or {}).get("table")
    if table and table != "bots.parquet":       # 기본 봇 표면 기록하지 않음 (기존 세션 기록과 같게)
        meta["bots_table"] = table
    return meta


def sessions_dir(cfg: dict) -> Path:
    d = cfg["paths"]["interim"] / "sessions"
    return d if is_original_design(cfg) else d / experiment_tag(cfg)


def results_dir(cfg: dict, split: str) -> Path:
    """정답 주를 한 번만 여는 표시(results/test/FINAL_DONE)는 실험 버전과 상관없이 하나만 둔다."""
    r = cfg["paths"]["results"]
    return r / split if is_original_design(cfg) else r / experiment_tag(cfg) / split
