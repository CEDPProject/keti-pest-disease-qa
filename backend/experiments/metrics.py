import os
import re

import numpy as np
from sklearn.metrics import f1_score

from backend.core.classes import CLASS_KEYS, KEY_TO_SECTION, NORMAL_KEY

SOURCES = ("원본", "NCPMS")
LATIN = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})")


# 파일 이름으로 사진 출처나 정상 사진 부위를 구분한다
def image_source(path: str, key: str) -> str:
    b = os.path.basename(path)
    if key == NORMAL_KEY:
        return "잎" if "잎" in b else ("열매" if "열매" in b else "기타")
    return "NCPMS" if b.startswith("ncpms_") else "원본"


# 8클래스 macro-F1 을 계산한다
def macro_f1(golds: list, preds: list) -> float:
    return f1_score(golds, preds, labels=CLASS_KEYS, average="macro", zero_division=0)


# 질병 사진만 대상으로 출처별 정확도를 계산한다
def source_accuracy(golds: list, preds: list, srcs: list, ndigits: int) -> dict:
    out = {}
    for s in SOURCES:
        idx = [i for i in range(len(golds)) if srcs[i] == s and golds[i] != NORMAL_KEY]
        out[s] = round(sum(preds[i] == golds[i] for i in idx) / len(idx), ndigits) if idx else None
    return out


# 기준 예측 대비 오답→정답, 정답→오답으로 바뀐 위치를 찾는다
def prediction_changes(golds: list, base: list, final: list):
    w2c = [i for i in range(len(golds)) if base[i] != golds[i] and final[i] == golds[i]]
    c2w = [i for i in range(len(golds)) if base[i] == golds[i] and final[i] != golds[i]]
    return w2c, c2w


# 정렬된 마진 목록의 개수·최소·중앙·최대·평균을 요약한다
def margin_stats(xs: list):
    if not xs:
        return None
    return {"n": len(xs), "min": xs[0], "median": xs[len(xs) // 2], "max": xs[-1],
            "mean": round(sum(xs) / len(xs), 4)}


# 점수 목록을 softmax 확률로 바꾼다
def softmax(x):
    x = np.array(x)
    e = np.exp(x - x.max())
    return e / e.sum()


# 문장에서 라틴어 이명(속명 + 종명) 후보를 모은다
def latin_names(text: str) -> set:
    return {f"{m.group(1)} {m.group(2)}" for m in LATIN.finditer(text or "")}


# 답변 속 학명을 정답·오귀속·날조로 판정하는 채점기
class SciNameScorer:
    LEVELS = ("fabrication", "misattribution", "correct")

    # 클래스별 KB 학명과 채점에서 뺄 영명 라틴어를 준비한다
    def __init__(self, kb: dict):
        self.sci = {k: latin_names(kb[sec]["scientific"]) for k, sec in KEY_TO_SECTION.items()}
        self.english = set()
        for sec in KEY_TO_SECTION.values():
            self.english |= latin_names(kb[sec].get("english", ""))

    # 영명을 뺀 학명 후보만 남긴다
    def scientific_only(self, names: set) -> set:
        return {n for n in names if n not in self.english}

    # 학명 후보 집합을 가장 심각한 환각 등급 하나로 판정한다
    def classify(self, names: set, gold_key: str) -> str:
        found = self.scientific_only(names)
        if not found:
            return "none"
        types = set()
        for name in found:
            if name in self.sci.get(gold_key, set()):
                types.add("correct")
            elif any(name in s for s in self.sci.values()):
                types.add("misattribution")
            else:
                types.add("fabrication")
        for level in self.LEVELS:
            if level in types:
                return level
        return "none"

    # 행 목록의 등급별 건수와 환각률을 집계한다
    @staticmethod
    def aggregate(rows: list) -> dict:
        n = len(rows)
        fab = sum(r["halluc"] == "fabrication" for r in rows)
        mis = sum(r["halluc"] == "misattribution" for r in rows)
        cor = sum(r["halluc"] == "correct" for r in rows)
        non = sum(r["halluc"] == "none" for r in rows)
        return {"n": n, "fabrication": fab, "misattribution": mis, "correct": cor, "none": non,
                "halluc_n": fab + mis, "halluc_rate": round((fab + mis) / n, 4)}
