import os
import json
import random
from collections import Counter

from PIL import Image

from backend.core.classes import AUG_IMAGE_DIR, CLASSES, DATA_DIR, SPLIT_JSON
from backend.core.kb_parser import kb_for_classes
from backend.experiments.data_io import read_json, write_json
from backend.experiments.data_prep.sample_builder import (augment, build_control, build_full, build_name,
                                                          build_symptom, make_sample)

SEED = 42

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
EVAL_QUESTION = Q_FULL[0]

INTENTS = [
    ("full", Q_FULL, build_full),
    ("control", Q_CONTROL, build_control),
    ("symptom", Q_SYMPTOM, build_symptom),
    ("name", Q_NAME, build_name),
]


# 학습 샘플의 질문 문장으로 의도별 개수를 센다
def count_intents(train: list) -> Counter:
    cnt = Counter()
    for s in train:
        q = s["messages"][0]["content"]
        for name, qs, _ in INTENTS:
            if any(x in q for x in qs):
                cnt[name] += 1
                break
    return cnt


# 플랫폼 학습셋과 평가셋을 만든다
def main():
    os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
    split = read_json(SPLIT_JSON)
    kb = kb_for_classes()

    ref = {label: {name: builder(label, kb[label]) for name, _, builder in INTENTS} for label in CLASSES}
    write_json(ref, os.path.join(DATA_DIR, "target_answers.json"))

    rng = random.Random(SEED)
    train = []
    for row in split["train"]:
        label, key, src = row["label_ko"], row["key"], row["image_path"]
        kk = kb[label]
        train.append(make_sample(src, rng.choice(Q_FULL), build_full(label, kk)))

        img = Image.open(src).convert("RGB")
        aug_dir = os.path.join(AUG_IMAGE_DIR, key)
        os.makedirs(aug_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        for a, (_, qs, builder) in enumerate(INTENTS[1:], 1):
            ap = os.path.join(aug_dir, f"{base}_aug{a}.jpg")
            augment(img, rng).save(ap, "JPEG", quality=92)
            train.append(make_sample(os.path.abspath(ap), rng.choice(qs), builder(label, kk)))

    rng.shuffle(train)
    write_json(train, os.path.join(DATA_DIR, "train.json"))

    with open(os.path.join(DATA_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
        for row in split["test"]:
            label = row["label_ko"]
            f.write(json.dumps({"image": row["image_path"], "label_ko": label, "key": row["key"],
                                "question": EVAL_QUESTION,
                                "gold_answer": build_full(label, kb[label])},
                               ensure_ascii=False) + "\n")

    print(f"train {len(train)}개 / test {len(split['test'])}개")
    print("의도 분포:", dict(count_intents(train)))


if __name__ == "__main__":
    main()
