"""의도별 정확도 + 의도 간 일관성 + 학명 환각 검사.

발견: "진단" 의도에서 92%였지만, "방제/증상/병명" 의도에서는 다른 병명을 답하는 경우가 있음.
데모에서 사용자가 "약 뭐 써?"라고 물으면 엉뚱한 병의 약제를 받을 수 있으므로 정량화한다.

  test 25장 × 의도 4종 = 100회 추론
  - 의도별 accuracy / macro-F1
  - 의도 간 일관성: 4개 의도가 모두 같은 병명을 예측한 비율
  - 학명 환각: KB에 학명이 '미확정'인 클래스(검은무늬병)에 학명을 생성했는지

결과: results/intent_consistency.json + 콘솔 요약
"""

import os
import sys
import json
import argparse
from collections import Counter

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import f1_score, accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa
from classes import MERGED_MODEL_DIR  # noqa
from kb_parser import kb_for_classes  # noqa

from classes import TEST_JSONL  # noqa
RESULT_DIR = os.path.join(VLM_FT, "results")
KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}
CLASS_KEYS = list(LABEL_TO_KEY.values())

INTENTS = [
    ("진단", "이 감귤 사진을 보고 어떤 병해충인지 진단하고, 관찰되는 증상과 방제 방법을 알려주세요."),
    ("병명", "이거 무슨 병이야?"),
    ("방제", "이 병은 어떻게 방제해야 하나요?"),
    ("증상", "이 감귤에 어떤 증상이 보이나요?"),
]


def ask(model, proc, image_path, question):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image_path, "max_pixels": 768 * 768},
        {"type": "text", "text": question}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img = Image.open(image_path).convert("RGB")
    inp = proc(text=[txt], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=300, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=MERGED_MODEL_DIR)
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(args.model_dir, max_pixels=768 * 768)
    kb = kb_for_classes()
    # KB 상 학명이 미확정인 클래스
    no_sci = {lab for lab, k in kb.items() if (not k["scientific"]) or "미확정" in k["scientific"]}

    test = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")]
    golds = [r["key"] for r in test]

    per_intent = {name: [] for name, _ in INTENTS}
    records = []
    halluc = []

    for i, r in enumerate(test, 1):
        row = {"image": os.path.basename(r["image"]), "gold": r["label_ko"],
               "preds": {}, "answers": {}}
        for name, q in INTENTS:
            ans = ask(model, proc, r["image"], q)
            pk = extract_label(ans)
            per_intent[name].append(pk)
            row["preds"][name] = KEY_TO_LABEL.get(pk, "불명확")
            row["answers"][name] = ans  # 원문 저장 → 추출기 변경 시 재채점 가능
            # 학명 환각 검사: KB에 학명 없는 클래스인데 '학명'을 생성
            if r["label_ko"] in no_sci and "학명" in ans:
                halluc.append({"image": row["image"], "gold": r["label_ko"],
                               "intent": name, "snippet": ans[:120]})
        preds = list(row["preds"].values())
        row["consistent"] = len(set(preds)) == 1
        row["all_correct"] = all(p == r["label_ko"] for p in preds)
        records.append(row)
        flag = "" if row["all_correct"] else ("불일치" if not row["consistent"] else "전부오답")
        print(f"  [{i:2d}/25] {r['label_ko']:12s} {flag}  {row['preds']}")

    # 지표
    summary = {"per_intent": {}, "consistency": {}, "hallucination": halluc}
    print("\n=== 의도별 정확도 ===")
    for name, _ in INTENTS:
        p = per_intent[name]
        acc = accuracy_score(golds, p)
        f1 = f1_score(golds, p, labels=CLASS_KEYS, average="macro", zero_division=0)
        summary["per_intent"][name] = {"accuracy": acc, "macro_f1": f1}
        print(f"  {name}: acc={acc:.3f}  macro-F1={f1:.3f}")

    n_cons = sum(r["consistent"] for r in records)
    n_allok = sum(r["all_correct"] for r in records)
    summary["consistency"] = {
        "all_intents_same_prediction": n_cons, "n": len(records),
        "all_intents_correct": n_allok,
        "consistency_rate": n_cons / len(records),
        "all_correct_rate": n_allok / len(records),
    }
    print(f"\n=== 의도 간 일관성 ===")
    print(f"  4개 의도가 모두 같은 병명 예측 : {n_cons}/{len(records)} = {n_cons/len(records):.1%}")
    print(f"  4개 의도가 모두 정답           : {n_allok}/{len(records)} = {n_allok/len(records):.1%}")

    print(f"\n=== 학명 환각 (KB에 학명 없는 클래스: {', '.join(no_sci) or '없음'}) ===")
    if halluc:
        for h in halluc[:6]:
            print(f"  {h['gold']} [{h['intent']}] : {h['snippet'][:80]}...")
        print(f"  총 {len(halluc)}건")
    else:
        print("  없음")

    summary["records"] = records
    out = os.path.join(RESULT_DIR, "intent_consistency.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
