"""생성된 QA 답변 텍스트 → 예측 병명(클래스 키) 추출.

baseline(zero/few-shot)과 파인튜닝 모델 모두 동일 규칙으로 라벨을 뽑아
공정 비교(macro-F1)한다.

추출 우선순위 (앞선 규칙이 이기면 즉시 확정):
  1) 굵게 강조된 **병명**      — 파인튜닝 모델의 진단 출력 형식
  2) '진단: X' / '병명: X'     — 콜론이 반드시 있어야 함('진단됩니다' 오탐 방지)
  3) 본문 등장 위치가 가장 앞선 병명 — CANON 배열 순서가 아니라 '텍스트상 위치' 기준
  부정문("~은 아니고", "~는 제외")으로 언급된 병명은 후보에서 제외.

※ 과거 버그: CANON 배열 순서로 매칭해 "더뎅이병보다는 궤양병일 가능성이 높다"에서
   더뎅이병을 반환했다. 위치/강조 기반으로 교체함.
"""

import re

# 정규 클래스명
CANON = [
    "검은점무늬병",
    "잿빛곰팡이병",
    "녹색곰팡이병",
    "검은무늬병",
    "더뎅이병",
    "탄저병",
    "궤양병",
    "수지병",
]
LABEL_TO_KEY = {
    "궤양병": "canker", "더뎅이병": "scab", "잿빛곰팡이병": "gray_mold",
    "수지병": "gummosis", "검은점무늬병": "black_spot", "녹색곰팡이병": "green_mold",
    "탄저병": "anthracnose", "검은무늬병": "black_blotch",
}
# 동의어 → 정규명
SYNONYM = {
    "회색곰팡이병": "잿빛곰팡이병",
    "푸른곰팡이병": "녹색곰팡이병",
    "멜라노스": "검은점무늬병",
    "흑점병": "검은점무늬병",
}
# 부정 표현 (병명 직후 window 안에 있으면 후보 제외)
NEG_PAT = re.compile(r"^(?:은|는|이|가)?\s*(?:아니|아님|아닙|제외|배제)")
NEG_WINDOW = 6


def _canon_of(token: str):
    """토큰(공백제거)이 정규명/동의어면 정규명 반환, 아니면 None."""
    t = token.replace(" ", "").strip()
    for name in CANON:
        if name in t:
            return name
    for syn, canon in SYNONYM.items():
        if syn in t:
            return canon
    return None


def _candidates(text_nospace: str):
    """(위치, 정규명) 목록. 부정문으로 언급된 것은 제외."""
    out = []
    names = [(n, n) for n in CANON] + [(s, c) for s, c in SYNONYM.items()]
    for surface, canon in names:
        start = 0
        while True:
            i = text_nospace.find(surface, start)
            if i < 0:
                break
            tail = text_nospace[i + len(surface): i + len(surface) + NEG_WINDOW]
            if not NEG_PAT.match(tail):
                out.append((i, canon))
            start = i + 1
    return sorted(out)


def extract_label(text: str) -> str:
    """예측 클래스 키 반환. 못 찾으면 'unknown'."""
    if not text:
        return "unknown"
    t = text.replace(" ", "")

    # 1) 굵게 강조된 병명 (**병명**)
    for bold in re.findall(r"\*\*(.+?)\*\*", text):
        canon = _canon_of(bold)
        if canon:
            return LABEL_TO_KEY[canon]

    # 2) '진단: X' / '병명: X' — 콜론 필수
    m = re.search(r"(?:진단|병명|의심\s*병명|의심되는\s*병명)\s*[:：]\s*([^\n,.]+)", text)
    if m:
        canon = _canon_of(m.group(1))
        if canon:
            return LABEL_TO_KEY[canon]

    # 3) 본문에서 등장 위치가 가장 앞선 병명 (부정문 제외)
    cands = _candidates(t)
    if cands:
        return LABEL_TO_KEY[cands[0][1]]

    return "unknown"


if __name__ == "__main__":
    tests = [
        # (입력, 기대)
        ("사진 속 감귤에서 관찰되는 증상으로 볼 때 **궤양병**(으)로 진단됩니다.", "canker"),
        ("**궤양병**(으)로 진단됩니다. 더뎅이병과 감별 필요.", "canker"),      # 강조 우선
        ("진단: 검은무늬병\n증상은...", "black_blotch"),
        ("이 감귤은 검은점무늬병으로 보입니다.", "black_spot"),
        ("회색곰팡이병(잿빛곰팡이병) 가능성", "gray_mold"),
        ("더뎅이병보다는 궤양병일 가능성이 높습니다.", "scab"),               # 위치 기준: 더뎅이병이 먼저
        ("궤양병 또는 더뎅이병 가능성이 있습니다.", "canker"),                 # 위치 기준
        ("잿빛곰팡이병은 아니고 탄저병으로 보입니다.", "anthracnose"),         # 부정문 제외
        ("이 감귤은 **검은반점병**으로 진단됩니다.", "unknown"),               # 없는 병명 → unknown
        ("정상으로 보입니다.", "unknown"),
    ]
    ok = 0
    for txt, exp in tests:
        got = extract_label(txt)
        good = got == exp
        ok += good
        print(f"{'OK  ' if good else 'FAIL'} exp={exp:13s} got={got:13s} | {txt[:42]}")
    print(f"\n{ok}/{len(tests)} 통과")
