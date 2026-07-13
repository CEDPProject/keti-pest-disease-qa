"""교차 데이터셋 검증 — 학습에 쓰지 않은 완전히 다른 출처(Mendeley Citrus)로 일반화 검사.

우리 학습 데이터(감귤 병해충 사진, 한국)와 겹치는 병해만 사용:
  Fruits/Canker   → 궤양병      (78장)
  Fruits/Scab     → 더뎅이병     (15장)
  Leaves/Melanose → 검은점무늬병  (13장)

목적: test 25장에서 나온 92%가 '진짜 병반 이해'인지, '우리 데이터셋 스타일 암기'인지 판별.
주의: Mendeley는 한국 감귤(온주밀감)이 아니라 오렌지/라임 계열 → 도메인 시프트 존재.
      성능 하락이 곧 '지름길 학습'을 뜻하지는 않지만, 유지되면 강력한 일반화 증거.
"""

import os
import sys
import glob
import json
import random
import argparse

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import accuracy_score, classification_report

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(VLM_FT)
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa
from classes import MERGED_MODEL_DIR  # noqa

KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}
from classes import MENDELEY_DIR as DS  # noqa
QUESTION = "이 감귤 사진을 보고 어떤 병해충인지 진단하고, 관찰되는 증상과 방제 방법을 알려주세요."

# (폴더, 우리 클래스 키, 한글명)
GROUPS = [
    (os.path.join(DS, "Fruits", "Canker"), "canker", "궤양병"),
    (os.path.join(DS, "Fruits", "Scab"), "scab", "더뎅이병"),
    (os.path.join(DS, "Leaves", "Melanose"), "black_spot", "검은점무늬병"),
]
MAX_PER_CLASS = 30
SEED = 42


def ask(model, proc, path):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": path, "max_pixels": 768 * 768},
        {"type": "text", "text": QUESTION}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img = Image.open(path).convert("RGB")
    inp = proc(text=[txt], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=200, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=MERGED_MODEL_DIR)
    args = ap.parse_args()

    rng = random.Random(SEED)
    items = []
    for folder, key, ko in GROUPS:
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        files = files[:MAX_PER_CLASS]
        items += [(f, key, ko) for f in files]
        print(f"  {ko:12s} ({os.path.basename(folder):10s}): {len(files)}장 사용")
    print(f"총 {len(items)}장\n")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(args.model_dir, max_pixels=768 * 768)

    golds, preds, rows = [], [], []
    for i, (path, key, ko) in enumerate(items, 1):
        ans = ask(model, proc, path)
        pk = extract_label(ans)
        golds.append(key)
        preds.append(pk)
        rows.append({"image": os.path.basename(path), "gold": ko,
                     "pred": KEY_TO_LABEL.get(pk, "불명확"), "ok": pk == key, "answer": ans})
        if i % 10 == 0 or i == len(items):
            acc = sum(g == p for g, p in zip(golds, preds)) / len(golds)
            print(f"  [{i:3d}/{len(items)}] 누적 정확도 {acc:.3f}")

    labels = sorted(set(golds))
    acc = accuracy_score(golds, preds)
    print(f"\n{'='*62}\n[교차 데이터셋(Mendeley) 결과]  N={len(golds)}")
    print(f"  Accuracy = {acc:.3f}   (우리 test 셋: 0.920)")
    print(f"{'='*62}\n")
    print(classification_report(golds, preds, labels=labels,
                                target_names=[KEY_TO_LABEL[l] for l in labels],
                                zero_division=0, digits=3))

    from collections import Counter
    print("실제 예측 분포:", dict(Counter(KEY_TO_LABEL.get(p, "불명확") for p in preds)))

    out = os.path.join(VLM_FT, "results", "cross_dataset.json")
    json.dump({"n": len(golds), "accuracy": acc, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
