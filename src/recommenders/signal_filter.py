"""데이터 사다리 (재설계 2026-11-27): 스와이프 화면이 추천기에 넘기는 데이터를 단계별로 줄인다.

추천 엔진과 화면(20장 × 3세트, 매 세트 엔진이 통째로 고름)은 그대로 두고, 봇의 반응 중 어디까지를 엔진에 전달할지만 바꾼다.

    like          S1  찜만 전달. 넘김은 신호가 아님 (목록 v2와 같은 데이터)
    skip          S2  + 넘겼다는 사실. 이유는 모름 -> 모든 넘김을 '취향 아님'으로 처리
    expensive     부록  + '비싸요'만 구분, 나머지 넘김은 '취향 아님'
    already_have  부록  + '이미 있어요'만 구분, 나머지 넘김은 '취향 아님'
    (S3 = 이유 3가지 모두 = 기존 swipe_v2, 이 필터를 쓰지 않음)
"""
from __future__ import annotations

LEVELS = ("like", "skip", "expensive", "already_have")


class SignalFilter:
    def __init__(self, rec, level: str):
        if level not in LEVELS:
            raise ValueError(level)
        self.rec, self.level = rec, level

    def update(self, article_id: str, action: str, reason: str | None) -> None:
        if action == "like":
            self.rec.update(article_id, "like", None)
        elif self.level == "like":
            self.rec.update(article_id, "skip", None)          # 본 상품 기록만 (엔진은 이유 없는 넘김을 무시)
        elif self.level == "skip":
            self.rec.update(article_id, "skip", "not_my_taste")
        else:
            self.rec.update(article_id, "skip", reason if reason == self.level else "not_my_taste")

    def first_set(self, k: int) -> list[str]:
        return self.rec.first_set(k)

    def next_set(self, k: int, exclude: set[str]) -> list[str]:
        return self.rec.next_set(k, exclude)

    def final_top(self, k: int, exclude: set[str]) -> list[str]:
        return self.rec.final_top(k, exclude)
