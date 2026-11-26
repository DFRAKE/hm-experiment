"""상품 풀 + 임베딩 로더 (3.4 결과물, 명세 수정 v3에서 사진 임베딩 추가).

좌표(임베딩)는 세 가지를 들고 다닌다.
- emb      : 추천기가 쓰는 좌표 (experiment.rec_space)
- bot_emb  : 봇이 찜을 판단하는 좌표 (experiment.bot_space)
- text_emb : 설명 문장 좌표 (e5). 다음 주 구매와 비교하는 현실 점검 채점은 항상 이것을 쓴다.

어느 한쪽이라도 image이면, 사진이 없는 상품은 상품 풀에서 뺀다 (카드로 보여 줄 수 없으므로).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config, spaces


@dataclass
class ItemPool:
    ids: list[str]                 # article_id, 임베딩 행 순서
    index: dict[str, int]          # article_id -> 행 번호
    emb: np.ndarray                # 추천기 좌표 (N, d) float32, 정규화
    price: np.ndarray              # (N,) 학습 기간 대표 가격
    type_: np.ndarray              # (N,) product_type_name
    df: pd.DataFrame               # 태그·인기도 등 전체 칼럼 (행 순서 = ids)
    bot_emb: np.ndarray | None = None    # 봇 좌표. None이면 emb와 같음
    text_emb: np.ndarray | None = None   # 설명 문장 좌표. None이면 emb와 같음
    filtered_by_image: bool = False      # 사진 없는 상품을 뺐는지

    def __post_init__(self) -> None:
        if self.bot_emb is None:
            self.bot_emb = self.emb
        if self.text_emb is None:
            self.text_emb = self.emb

    def __len__(self) -> int:
        return len(self.ids)

    def rows(self, article_ids) -> np.ndarray:
        return np.fromiter((self.index[a] for a in article_ids), dtype=np.int64)


def image_files_ready(cfg: dict) -> bool:
    interim = cfg["paths"]["interim"]
    return (interim / "article_img_emb.npy").exists() and (interim / "article_img_ids.json").exists()


def pool_ready(cfg: dict) -> bool:
    """현재 설정으로 load_pool 에 필요한 파일이 다 있는지 (테스트 skip 조건용)."""
    if not (cfg["paths"]["interim"] / "article_emb.npy").exists():
        return False
    return "image" not in spaces(cfg) or image_files_ready(cfg)


def load_pool(cfg: dict | None = None) -> ItemPool:
    cfg = cfg or load_config()
    interim = cfg["paths"]["interim"]
    df = pd.read_parquet(interim / "article_pool.parquet")
    ids = json.loads((interim / "article_ids.json").read_text())
    text = np.load(interim / "article_emb.npy")
    df = df.set_index("article_id").loc[ids].reset_index()
    if len(df) != len(ids) or text.shape[0] != len(ids):
        raise ValueError("상품 풀과 임베딩 개수가 다릅니다. python -m src.embed 를 다시 실행하세요.")

    bot_space, rec_space = spaces(cfg)
    img = None
    if "image" in (bot_space, rec_space):
        if not image_files_ready(cfg):
            raise FileNotFoundError("사진 임베딩이 없습니다. python -m src.embed_image 를 먼저 실행하세요.")
        img_ids = json.loads((interim / "article_img_ids.json").read_text())
        img_all = np.load(interim / "article_img_emb.npy")
        pos = {a: i for i, a in enumerate(img_ids)}
        keep = [i for i, a in enumerate(ids) if a in pos]
        ids = [ids[i] for i in keep]
        text = text[keep]
        df = df.iloc[keep].reset_index(drop=True)
        img = img_all[[pos[a] for a in ids]]

    def pick(space: str) -> np.ndarray:
        return img if space == "image" else text

    return ItemPool(ids=ids, index={a: i for i, a in enumerate(ids)}, emb=pick(rec_space),
                    price=df["price"].to_numpy(dtype=float),
                    type_=df["product_type_name"].to_numpy(dtype=object), df=df,
                    bot_emb=pick(bot_space), text_emb=text, filtered_by_image=img is not None)
