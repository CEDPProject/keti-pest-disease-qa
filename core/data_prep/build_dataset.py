"""split.json + KB → 파인튜닝/평가 데이터셋 생성 (의도 다양화 버전).

VLM QA 어시스턴트가 되도록, 이미지 + '다양한 의도의 텍스트 질문' → '의도에 맞는 답변'을 학습.
의도 4종: 진단(full) / 병명(name) / 방제(control) / 증상(symptom) — 모두 KB grounding.

산출물 (vlm_finetune/data/):
  - train.json           : LLaMA-Factory 멀티모달 SFT (의도 4종 균형)
  - test.jsonl           : 평가용 (고정 진단질문, gold=full 답변)
  - target_answers.json  : 클래스×의도 답변 참고본
  - images_aug/          : 증강 이미지

원본 이미지 → 진단(full), 증강 3장 → 방제/증상/병명 각 1개. (test 누수 방지: 증강은 train만)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
import json
import random

from PIL import Image, ImageEnhance

from classes import CLASSES, SPLIT_JSON, DATA_DIR, AUG_IMAGE_DIR, TEST_JSONL
from kb_parser import kb_for_classes


AUG_DIR = AUG_IMAGE_DIR
AUG_MAX_SIDE = 1024
SEED = 42

# ---- 의도별 질문 ----
Q_FULL = [
    "이 감귤 사진을 보고 어떤 병해충인지 진단하고, 관찰되는 증상과 방제 방법을 알려주세요.",
    "이 감귤에 어떤 병이 의심되나요? 증상과 관리 방법도 함께 설명해주세요.",
    "사진 속 감귤의 상태를 진단해주세요. 병명, 증상, 방제법을 알려주세요.",
    "이 감귤 이미지에서 보이는 병해충 증상을 설명하고, 의심 병명과 방제 방법을 알려주세요.",
]
Q_NAME = [
    "이 감귤은 무슨 병인가요?",
    "이 사진의 병명만 알려주세요.",
    "이거 무슨 병이야?",
]
Q_CONTROL = [
    "이 병은 어떻게 방제해야 하나요?",
    "방제 방법과 약제를 알려주세요.",
    "이거 어떻게 관리하고 무슨 약을 써요?",
]
Q_SYMPTOM = [
    "이 감귤에 어떤 증상이 보이나요?",
    "사진에서 관찰되는 증상을 설명해주세요.",
    "어떤 병징이 나타나 있나요?",
]

EVAL_QUESTION = Q_FULL[0]  # 평가는 고정 진단질문


# ---- 의도별 정답 빌더 (모두 병명을 **굵게** 포함 → 라벨 추출 가능) ----
def _ident(k):
    parts = []
    if k["scientific"] and "미확정" not in k["scientific"]:
        parts.append(f"학명 {k['scientific']}")
    if k["english"] and "가칭" not in k["english"]:
        parts.append(f"영명 {k['english']}")
    return f"({' / '.join(parts)})" if parts else ""


def build_full(label, k):
    lines = [f"사진 속 감귤에서 관찰되는 증상으로 볼 때 **{label}**(으)로 진단됩니다.", ""]
    if _ident(k):
        lines += [_ident(k), ""]
    lines.append("## 관찰 증상")
    lines += [f"- {s}" for s in k["symptoms"][:3]]
    if k["ecology"]:
        lines += ["", "## 발생 원인 및 생태"] + [f"- {s}" for s in k["ecology"][:2]]
    lines += ["", "## 방제 및 관리"] + [f"- {s}" for s in k["control"][:3]]
    if k["pesticides"]:
        lines += ["", "## 추천 약제(예시)",
                  f"- {', '.join(k['pesticides'][:3])} 등 (정확한 등록 약제·사용 시기는 농약안전정보시스템 확인)"]
    lines += ["", "## 추가 확인",
              "- 이미지만으로 확정 진단은 어렵습니다. 병반의 위치·크기, 발생 시기, 주변 나무 상태를 함께 확인하세요."]
    return "\n".join(lines)


def build_name(label, k):
    reason = k["symptoms"][0] if k["symptoms"] else ""
    out = [f"이 감귤은 **{label}**(으)로 진단됩니다."]
    if _ident(k):
        out.append(_ident(k))
    if reason:
        out += ["", f"판단 근거: {reason}"]
    out += ["", "정확한 확진과 방제 방법은 증상 부위·발생 시기를 함께 확인해 주세요."]
    return "\n".join(out)


def build_control(label, k):
    lines = [f"**{label}**의 방제 및 관리 방법입니다.", "", "## 방제 및 관리"]
    lines += [f"- {s}" for s in k["control"][:3]]
    if k["pesticides"]:
        lines += ["", "## 추천 약제(예시)",
                  f"- {', '.join(k['pesticides'][:3])} 등 (정확한 등록 약제·사용 시기는 농약안전정보시스템 확인)"]
    lines += ["", "약제는 등록 정보를 확인하고, 발생 초기에 예방적으로 관리하는 것이 효과적입니다."]
    return "\n".join(lines)


def build_symptom(label, k):
    lines = [f"이 감귤에서는 **{label}**의 증상이 관찰됩니다.", "", "## 관찰 증상"]
    lines += [f"- {s}" for s in k["symptoms"][:4]]
    lines += ["", "동일 증상이라도 병 확진에는 발생 부위·시기 확인이 필요합니다."]
    return "\n".join(lines)


INTENTS = [
    ("full", Q_FULL, build_full),
    ("control", Q_CONTROL, build_control),
    ("symptom", Q_SYMPTOM, build_symptom),
    ("name", Q_NAME, build_name),
]


def augment(img, rng):
    im = img
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(rng.uniform(-12, 12), resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    w, h = im.size
    scale = rng.uniform(0.85, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x0, y0, x0 + cw, y0 + ch))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.85, 1.15))
    w, h = im.size
    if max(w, h) > AUG_MAX_SIDE:
        r = AUG_MAX_SIDE / max(w, h)
        im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
    return im


def sample(image_path, question, answer):
    return {"messages": [{"role": "user", "content": "<image>" + question},
                         {"role": "assistant", "content": answer}], "images": [image_path]}


def main():
    os.makedirs(AUG_DIR, exist_ok=True)
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    kb = kb_for_classes()

    # 클래스×의도 답변 참고 저장
    ref = {label: {name: builder(label, kb[label]) for name, _, builder in INTENTS}
           for label in CLASSES}
    with open(os.path.join(DATA_DIR, "target_answers.json"), "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)

    rng = random.Random(SEED)
    train = []
    for i, row in enumerate(split["train"]):
        label, key, src = row["label_ko"], row["key"], row["image_path"]
        kk = kb[label]

        # 원본 → 진단(full)
        train.append(sample(src, rng.choice(Q_FULL), build_full(label, kk)))

        # 증강 3장 → control / symptom / name
        img = Image.open(src).convert("RGB")
        aug_dir = os.path.join(AUG_DIR, key)
        os.makedirs(aug_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        for a, (name, qs, builder) in enumerate(INTENTS[1:], 1):  # control, symptom, name
            aug_img = augment(img, rng)
            ap = os.path.join(aug_dir, f"{base}_aug{a}.jpg")
            aug_img.save(ap, "JPEG", quality=92)
            train.append(sample(os.path.abspath(ap), rng.choice(qs), builder(label, kk)))

    rng.shuffle(train)
    with open(os.path.join(DATA_DIR, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    # 평가셋 (원본 test, 고정 진단질문, gold=full)
    with open(os.path.join(DATA_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
        for row in split["test"]:
            label = row["label_ko"]
            f.write(json.dumps({"image": row["image_path"], "label_ko": label, "key": row["key"],
                                "question": EVAL_QUESTION,
                                "gold_answer": build_full(label, kb[label])},
                               ensure_ascii=False) + "\n")

    # 의도 분포 출력
    from collections import Counter
    cnt = Counter()
    for s in train:
        q = s["messages"][0]["content"]
        for name, qs, _ in INTENTS:
            if any(x in q for x in qs):
                cnt[name] += 1
                break
    print(f"train {len(train)}개 / test {len(split['test'])}개")
    print("의도 분포:", dict(cnt))


if __name__ == "__main__":
    main()
