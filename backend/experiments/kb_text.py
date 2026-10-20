import re

from backend.core.classes import CLASS_KEYS, KEY_TO_LABEL, KEY_TO_SECTION, NORMAL_KEY

VISUAL_QUERY = ("병반 반점 딱지 부스럼 융기 돌출 코르크 분화구 검은점 "
                "부패 곰팡이 균사 포자 색깔 표면 모양 크기 테두리")
NORMAL_RULE = "병징에 해당하는 이상 소견(반점·병반·변색·곰팡이 등)이 없으면 정상."
NORMAL_PASSAGE = "병징에 해당하는 이상 소견(반점 병반 변색 곰팡이 무름)이 없는 건강한 상태."
DISEASE_WORDS = list(KEY_TO_LABEL.values()) + ["궤양", "더뎅이", "잿빛", "곰팡이", "수지",
                                               "검은점", "탄저", "멜라노스", "푸른곰팡이"]


# 캡션에서 병명이 들어간 어절을 지워 정답 누설을 막는다
def clean_caption(caption: str) -> str:
    for w in DISEASE_WORDS:
        caption = re.sub(r"\S*" + re.escape(w) + r"\S*", "", caption)
    return re.sub(r"\s+", " ", caption).strip()


# 클래스별 KB 병징 시각 특징 문장을 고른다
def symptom_guides(kb, top_n: int = 3) -> dict:
    guides = {}
    for key in CLASS_KEYS:
        if key == NORMAL_KEY:
            guides[key] = [NORMAL_RULE]
        else:
            lines = kb.retrieve_lines(KEY_TO_SECTION[key], VISUAL_QUERY, "병징", top_n=top_n)
            guides[key] = [line["text"] for line in lines]
    return guides
