import os
import sys
import json
import argparse

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo (web/test_demo.py 기준)
CORE_DIR = os.path.join(REPO_DIR, "backend")
sys.path.insert(0, os.path.join(CORE_DIR, "common"))
sys.path.insert(0, os.path.join(CORE_DIR, "rag"))
from kb_parser import kb_for_classes  # noqa
from label_extract import extract_label, LABEL_TO_KEY  # noqa

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
from classes import TEST_JSONL  # noqa
RESULT_DIR = os.path.join(REPO_DIR, "research", "results")
DOCS_DIR = os.path.join(REPO_DIR, "research", "results")
KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}

Q_DIAG = "이 감귤 사진을 보고 어떤 병해충인지 진단하고, 관찰되는 증상과 방제 방법을 알려주세요."
INTENT_QS = [
    ("진단", Q_DIAG),
    ("병명", "이거 무슨 병이야?"),
    ("방제", "이 병은 어떻게 방제해야 하나요?"),
    ("증상", "이 감귤에 어떤 증상이 보이나요?"),
]


def load(model_dir, adapter):
    if adapter:
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        p = AutoProcessor.from_pretrained(BASE_MODEL, max_pixels=768 * 768)
    else:
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
        p = AutoProcessor.from_pretrained(model_dir, max_pixels=768 * 768)
    return m.eval(), p


def ask(model, proc, image_path, question, max_new=400):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image_path, "max_pixels": 768 * 768},
        {"type": "text", "text": question}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img = Image.open(image_path).convert("RGB")
    inp = proc(text=[txt], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    from classes import MERGED_MODEL_DIR  # noqa
    ap.add_argument("--model_dir", default=MERGED_MODEL_DIR)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    model, proc = load(args.model_dir, args.adapter)
    kb = kb_for_classes()
    test = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")]

    # ---- 1) 전체 25장 진단 ----
    print("[1/3] test 25장 진단 테스트...")
    rows, correct = [], 0
    for i, r in enumerate(test, 1):
        ans = ask(model, proc, r["image"], Q_DIAG)
        pk = extract_label(ans)
        ok = pk == r["key"]
        correct += ok
        kb_hit = KEY_TO_LABEL.get(pk) in kb
        rows.append({"image": os.path.basename(r["image"]), "gold": r["label_ko"],
                     "pred": KEY_TO_LABEL.get(pk, "불명확"), "ok": ok, "kb_hit": kb_hit,
                     "answer": ans})
        print(f"  [{i:2d}/25] {'✅' if ok else '❌'} {r['label_ko']} → {KEY_TO_LABEL.get(pk,'불명확')}")
    acc = correct / len(test)

    # ---- 2) 의도별 응답 (정답 맞힌 이미지 중 서로 다른 클래스 3장) ----
    print("\n[2/3] 의도별 응답 테스트...")
    samples, seen = [], set()
    for r, row in zip(test, rows):
        if row["ok"] and r["label_ko"] not in seen:
            samples.append(r)
            seen.add(r["label_ko"])
        if len(samples) == 3:
            break
    intent_out = []
    for r in samples:
        per = {"image": os.path.basename(r["image"]), "gold": r["label_ko"], "answers": {}}
        for name, q in INTENT_QS:
            a = ask(model, proc, r["image"], q, max_new=250)
            per["answers"][name] = a
            print(f"  {r['label_ko']:10s} [{name}] {a[:50].replace(chr(10),' ')}...")
        intent_out.append(per)

    # ---- 3) KB 연결 확인 ----
    kb_ok = sum(1 for x in rows if x["kb_hit"])
    print(f"\n[3/3] KB 연결: {kb_ok}/{len(rows)} 건 성공")

    # ---- 리포트 작성 ----
    md = ["# 데모 실제 테스트 결과", ""]
    md.append(f"- 모델: `{args.adapter or args.model_dir}`")
    md.append(f"- 테스트: held-out test **{len(test)}장**")
    md.append(f"- **진단 정확도: {correct}/{len(test)} = {acc:.1%}**")
    md.append(f"- 병명→공식 KB 연결 성공: {kb_ok}/{len(rows)}")
    md.append("")
    md.append("## 1. 전체 진단 결과")
    md.append("")
    md.append("| # | 이미지 | 정답 | 예측 | 결과 | KB연결 |")
    md.append("|---:|---|---|---|:--:|:--:|")
    for i, x in enumerate(rows, 1):
        md.append(f"| {i} | `{x['image']}` | {x['gold']} | {x['pred']} | "
                  f"{'✅' if x['ok'] else '❌'} | {'✅' if x['kb_hit'] else '—'} |")
    md.append("")

    wrong = [x for x in rows if not x["ok"]]
    md.append(f"### 오답 {len(wrong)}건")
    md.append("")
    if wrong:
        for x in wrong:
            md.append(f"- `{x['image']}` : **{x['gold']}** → {x['pred']} 오분류")
    else:
        md.append("- 없음")
    md.append("")

    md.append("## 2. 의도별 응답 (같은 이미지, 다른 질문)")
    md.append("")
    for per in intent_out:
        md.append(f"### `{per['image']}` (정답: {per['gold']})")
        md.append("")
        for name, q in INTENT_QS:
            md.append(f"**[{name}] 질문:** {q}")
            md.append("")
            md.append("```")
            md.append(per["answers"][name][:600])
            md.append("```")
            md.append("")

    out = os.path.join(DOCS_DIR, "demo_test_report.md")
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n정확도 {acc:.1%} ({correct}/{len(test)}) | 리포트 → {out}")


if __name__ == "__main__":
    main()
