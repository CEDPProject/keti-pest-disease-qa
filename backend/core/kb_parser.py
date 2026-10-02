import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # core/ 자기참조
from classes import KB_PATH, CLASSES  # noqa: E402


def _bullets(block: str) -> list:
    """'- ' 로 시작하는 줄만 추출."""
    out = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
    return out


def parse_kb() -> dict:
    """{KB섹션명: {scientific, english, symptoms[], ecology[], control[], pesticides[]}}"""
    with open(KB_PATH, encoding="utf-8") as f:
        text = f.read()

    sections = {}
    # '## ' 기준 분리
    for sec in re.split(r"\n(?=## )", text):
        sec = sec.strip()
        if not sec.startswith("## "):
            continue
        name = sec.splitlines()[0].replace("## ", "").strip()

        # 학명/영명 (헤더 직후 bullet)
        sci = re.search(r"학명:\s*(.+)", sec)
        eng = re.search(r"영명:\s*(.+)", sec)

        # ### 하위 섹션 분리
        subs = {}
        for part in re.split(r"\n(?=### )", sec):
            if part.strip().startswith("### "):
                sub_name = part.splitlines()[0].replace("### ", "").strip()
                subs[sub_name] = part

        pesticides = re.findall(r"약제명:\s*(.+)", subs.get("방제약제", ""))

        sections[name] = {
            "scientific": sci.group(1).strip() if sci else "",
            "english": eng.group(1).strip() if eng else "",
            "symptoms": _bullets(subs.get("형태/병징", "")),
            "ecology": _bullets(subs.get("발생생태 및 피해", "")),
            "control": _bullets(subs.get("방제 및 과원관리", "")),
            "pesticides": [p.strip() for p in pesticides],
        }
    return sections


def kb_for_classes() -> dict:
    """사진 8클래스에 해당하는 KB만 반환 {label_ko: knowledge}."""
    kb = parse_kb()
    out = {}
    for label_ko, info in CLASSES.items():
        sec = info["kb_section"]
        if sec not in kb:
            raise KeyError(f"KB 섹션 없음: '{sec}' (클래스 {label_ko})")
        out[label_ko] = kb[sec]
    return out


if __name__ == "__main__":
    kb = kb_for_classes()
    for label, k in kb.items():
        print(f"\n=== {label} ({k['english']}) ===")
        print(f"  학명: {k['scientific']}")
        print(f"  증상 {len(k['symptoms'])} / 생태 {len(k['ecology'])} / 방제 {len(k['control'])} / 약제 {len(k['pesticides'])}")
