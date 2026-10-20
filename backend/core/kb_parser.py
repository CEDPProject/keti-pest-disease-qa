import re

from backend.core.classes import CLASSES, KB_PATH


# 블록에서 '- ' 로 시작하는 항목 줄만 뽑는다
def _bullets(block: str) -> list:
    out = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
    return out


# KB 문서를 병해충별 학명·영명·병징·생태·방제·약제 사전으로 파싱한다
def parse_kb() -> dict:
    with open(KB_PATH, encoding="utf-8") as f:
        text = f.read()

    sections = {}
    for sec in re.split(r"\n(?=## )", text):
        sec = sec.strip()
        if not sec.startswith("## "):
            continue
        name = sec.splitlines()[0].replace("## ", "").strip()

        sci = re.search(r"학명:\s*(.+)", sec)
        eng = re.search(r"영명:\s*(.+)", sec)

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


# 진단 대상 질병 클래스의 KB 항목만 한글 라벨 기준으로 돌려준다
def kb_for_classes() -> dict:
    kb = parse_kb()
    out = {}
    for label_ko, info in CLASSES.items():
        sec = info["kb_section"]
        if sec not in kb:
            raise KeyError(f"KB 섹션 없음: '{sec}' (클래스 {label_ko})")
        out[label_ko] = kb[sec]
    return out


# 클래스별 KB 항목 개수를 출력한다
def main():
    kb = kb_for_classes()
    for label, k in kb.items():
        print(f"\n=== {label} ({k['english']}) ===")
        print(f"  학명: {k['scientific']}")
        print(f"  증상 {len(k['symptoms'])} / 생태 {len(k['ecology'])} / "
              f"방제 {len(k['control'])} / 약제 {len(k['pesticides'])}")


if __name__ == "__main__":
    main()
