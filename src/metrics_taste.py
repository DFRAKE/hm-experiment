"""명세 수정 v3 4-1: 취향 회복도 지표. 다음 주 구매(검증·정답 주)는 쓰지 않는다.

쇼핑몰이 봇의 숨은 취향을 얼마나 빨리·정확히 찾았는지 잰다. 봇 프로필(학습 기간 데이터로만 만든 취향·찜 기준선·
가격 상한·최근 산 종류)을 seed별로 똑같이 다시 만들어, 보여 준 상품과 최종 추천을 **변덕 없는 규칙**(Bot.rule)으로 판정한다.
(변덕은 봇이 실수로 찜/넘김을 뒤집는 것이라, 쇼핑몰이 잘 골랐는지 잴 때는 뺀다.)

세션 하나(봇 × 방식 × seed)마다:
- pref_10 … pref_60 : 노출 10장 구간마다 봇이 변덕 없이 찜할 상품 비율 (추천 상품 선호율, H-04)
- pref_all          : 60장 전체 선호율
- first_pref        : 처음으로 '변덕 없이 찜할' 상품이 나온 노출 순번 (1부터, 없으면 NaN)
- final_like20      : 최종 추천 20개 중 봇이 변덕 없이 찜할 개수 (0~20)
- final_like20_new  : 그중 세션에서 보여 주지 않은 새 상품만 (이미 찜한 상품을 다시 담은 것 제외)
- final_taste20     : 최종 20개 중 취향 기준선만 넘는 개수 (가격·이미 있어요 무시)
- repeat_attr       : 비선호 속성 반복 노출률. 봇이 '비싸요' 또는 '이미 있어요'로 넘긴 뒤 남은 노출 중
                      같은 이유에 또 걸리는 상품 비율 (봇의 숨은 조건 기준, 취향과 무관)
                        · '비싸요'를 한 번이라도 말한 뒤: 가격 > 봇 가격 상한
                        · '이미 있어요'를 한 번이라도 말한 뒤: 봇의 이미 있어요 조건(Bot.dup_mask)에 걸림
                          (near_dup = 최근 산 옷과 거의 같은 옷, type = 최근 산 종류)
                      이유는 봇이 실제로 말한 것(로그)을 쓴다. 목록형은 이유를 추천기에 못 보내지만 봇은 똑같이 넘긴다.
                      해당 이유를 한 번도 말하지 않은 세션은 NaN
- repeat_price, repeat_type : repeat_attr 를 이유별로 나눈 것
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.bots import Bot
from src.items import ItemPool

BIN = 10


def rule_arrays(bot: Bot, pool: ItemPool, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bot.rule 을 여러 상품에 한꺼번에 적용. 반환: (sim, 취향 기준 통과, 변덕 없이 찜)."""
    return bot.rule_mask(pool, rows)


def repeat_rates(reasons: np.ndarray, too_pricey: np.ndarray, owned: np.ndarray) -> dict[str, float]:
    """노출 순서대로 reasons(봇이 말한 넘김 이유, 찜이면 None), too_pricey(가격 > 상한), owned(이미 있어요 조건)를 받아
    반복 노출률 3개를 계산. 이유를 말한 '다음' 노출부터 센다."""
    price_said = type_said = False
    n = {"price": 0, "type": 0, "attr": 0}
    hit = {"price": 0, "type": 0, "attr": 0}
    for r, p, o in zip(reasons, too_pricey, owned):
        hp, ht = price_said and bool(p), type_said and bool(o)
        if price_said:
            n["price"] += 1
            hit["price"] += hp
        if type_said:
            n["type"] += 1
            hit["type"] += ht
        if price_said or type_said:
            n["attr"] += 1
            hit["attr"] += hp or ht
        if r == "expensive":
            price_said = True
        elif r == "already_have":
            type_said = True
    return {f"repeat_{k}": (hit[k] / n[k] if n[k] else np.nan) for k in n}


def taste_metrics(logs: pd.DataFrame, finals: pd.DataFrame, profiles: dict[int, dict[str, Bot]],
                  pool: ItemPool, exposure: int, sim_tol: float = 1e-4) -> tuple[pd.DataFrame, dict]:
    """profiles[seed][bot_id] = 그 seed로 세션을 돌릴 때 쓴 봇. 반환: (세션별 지표, 점검 값).

    로그에 남은 sim_to_taste 와 다시 계산한 값이 다르면 프로필이 세션 때와 다르다는 뜻이라 멈춘다."""
    logs = logs.sort_values(["seed", "group", "bot_id", "set_no", "pos"])
    fin_by = {k: g["article_id"].tolist() for k, g in finals.sort_values("rank").groupby(["seed", "group", "bot_id"])}
    rows_out, max_dev, agree, n_obs = [], 0.0, 0, 0
    for (seed, mode, b), g in logs.groupby(["seed", "group", "bot_id"], sort=False):
        bot = profiles[seed][b]
        shown = pool.rows(g["article_id"])
        sim, _, like = rule_arrays(bot, pool, shown)
        max_dev = max(max_dev, float(np.abs(sim - g["sim_to_taste"].to_numpy()).max()))
        agree += int((like == (g["action"] == "like").to_numpy()).sum())
        n_obs += len(g)

        final = fin_by[(seed, mode, b)]
        frows = pool.rows(final)
        _, f_taste, f_like = rule_arrays(bot, pool, frows)
        is_new = ~np.isin(final, g["article_id"].to_numpy())
        reasons = g["reason"].where(g["action"] == "skip").to_numpy(dtype=object)
        r = {"seed": seed, "group": mode, "bot_id": b,
             "pref_all": float(like.mean()),
             "first_pref": float(np.argmax(like) + 1) if like.any() else np.nan,
             "final_like20": float(f_like.sum()),
             "final_like20_new": float((f_like & is_new).sum()),
             "final_taste20": float(f_taste.sum()),
             **repeat_rates(reasons, pool.price[shown] > bot.price_th, bot.dup_mask(pool, shown))}
        for e in range(BIN, exposure + 1, BIN):
            r[f"pref_{e}"] = float(like[e - BIN:e].mean())
        rows_out.append(r)
    if max_dev > sim_tol:
        raise ValueError(f"봇 프로필이 세션 때와 다름 (sim_to_taste 최대 차이 {max_dev:.2e}). "
                         "세션을 돌린 설정(config, HM_CONFIG)과 같은 설정으로 채점하세요.")
    check = {"sim_max_dev": max_dev, "rule_vs_observed_agree": agree / n_obs if n_obs else np.nan}
    return pd.DataFrame(rows_out), check
