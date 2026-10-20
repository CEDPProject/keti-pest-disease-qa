import re

from backend.core.classes import LABEL_TO_KEY

CANON = [
    "검은점무늬병",
    "잿빛곰팡이병",
    "녹색곰팡이병",
    "더뎅이병",
    "탄저병",
    "궤양병",
    "수지병",
    "정상",
]
SYNONYM = {
    "회색곰팡이병": "잿빛곰팡이병",
    "푸른곰팡이병": "녹색곰팡이병",
    "멜라노스": "검은점무늬병",
    "흑점병": "검은점무늬병",
    "검은무늬병": "검은점무늬병",
    "건강": "정상", "건전": "정상",
}
NEG_PAT = re.compile(r"^(?:은|는|이|가)?\s*(?:아니|아님|아닙|제외|배제)")
NEG_WINDOW = 6


# 토큰 안에 들어 있는 정규 병명(동의어 포함)을 찾는다
def _canon_of(token: str):
    t = token.replace(" ", "").strip()
    for name in CANON:
        if name in t:
            return name
    for syn, canon in SYNONYM.items():
        if syn in t:
            return canon
    return None


# 부정문으로 언급된 것을 뺀 (위치, 정규 병명) 목록을 만든다
def _candidates(text_nospace: str):
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


# 생성 답변에서 병명을 찾아 클래스 키로 돌려준다
def extract_label(text: str) -> str:
    if not text:
        return "unknown"
    t = text.replace(" ", "")

    for bold in re.findall(r"\*\*(.+?)\*\*", text):
        canon = _canon_of(bold)
        if canon:
            return LABEL_TO_KEY[canon]

    m = re.search(r"(?:진단|병명|의심\s*병명|의심되는\s*병명)\s*[:：]\s*([^\n,.]+)", text)
    if m:
        canon = _canon_of(m.group(1))
        if canon:
            return LABEL_TO_KEY[canon]

    cands = _candidates(t)
    if cands:
        return LABEL_TO_KEY[cands[0][1]]

    return "unknown"


# 대표 문장으로 병명 추출 규칙을 점검한다
def self_test():
    tests = [
        ("사진 속 감귤에서 관찰되는 증상으로 볼 때 **궤양병**(으)로 진단됩니다.", "canker"),
        ("**궤양병**(으)로 진단됩니다. 더뎅이병과 감별 필요.", "canker"),
        ("진단: 검은무늬병\n증상은...", "black_spot"),
        ("이 감귤은 검은점무늬병으로 보입니다.", "black_spot"),
        ("회색곰팡이병(잿빛곰팡이병) 가능성", "gray_mold"),
        ("더뎅이병보다는 궤양병일 가능성이 높습니다.", "scab"),
        ("궤양병 또는 더뎅이병 가능성이 있습니다.", "canker"),
        ("잿빛곰팡이병은 아니고 탄저병으로 보입니다.", "anthracnose"),
        ("이 감귤은 **검은반점병**으로 진단됩니다.", "unknown"),
        ("이 감귤은 **정상**(건강)으로 보입니다.", "normal"),
        ("뚜렷한 병징이 없어 건강한 상태로 보입니다.", "normal"),
    ]
    ok = 0
    for txt, exp in tests:
        got = extract_label(txt)
        good = got == exp
        ok += good
        print(f"{'OK  ' if good else 'FAIL'} exp={exp:13s} got={got:13s} | {txt[:42]}")
    print(f"\n{ok}/{len(tests)} 통과")


if __name__ == "__main__":
    self_test()
