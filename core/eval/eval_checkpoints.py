"""에폭별 LoRA 체크포인트를 test 셋으로 평가 → macro-F1 곡선 → 최적 에폭 선택.

base 모델은 1회만 로딩하고, 각 체크포인트를 named adapter 로 갈아끼우며 평가(빠름).
이미 평가된 에폭(results/pred_lora_ep{N}.jsonl 존재)은 건너뛰고 기존 곡선과 병합한다.
qwenvl-ft 환경에서 실행.

  python eval_checkpoints.py --output_dir .../output/qwen2_5vl_lora [--force]

주의: test 로 에폭을 고르는 것은 경미한 선택 편향 → 전체 곡선을 공개하고 최적점을 보고(리포트에 명시).
"""

import os
import re
import sys
import json
import glob
import argparse

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from label_extract import extract_label  # noqa

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
from classes import TEST_JSONL  # noqa
RESULT_DIR = os.path.join(VLM_FT, "results")
CURVE_JSON = os.path.join(RESULT_DIR, "lora_epoch_curve.json")
QUESTION = "이 감귤 사진을 보고 어떤 병해충인지 진단하고, 관찰되는 증상과 방제 방법을 알려주세요."
CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis",
              "black_spot", "green_mold", "anthracnose"]
STEPS_PER_EPOCH = 48


def ckpt_step(path):
    return int(re.search(r"checkpoint-(\d+)", path).group(1))


def infer(model, processor, image_path):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path, "max_pixels": 768 * 768},
        {"type": "text", "text": QUESTION}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def score(golds, preds):
    return (accuracy_score(golds, preds),
            f1_score(golds, preds, labels=CLASS_KEYS, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--force", action="store_true", help="이미 평가된 에폭도 재평가")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")), key=ckpt_step)
    if not ckpts:
        raise SystemExit(f"체크포인트 없음: {args.output_dir}")

    # 평가 대상 결정 (이미 예측파일 있으면 skip)
    todo, cached = [], []
    for ck in ckpts:
        ep = ckpt_step(ck) // STEPS_PER_EPOCH
        pred_path = os.path.join(RESULT_DIR, f"pred_lora_ep{ep}.jsonl")
        (todo if (args.force or not os.path.exists(pred_path)) else cached).append((ep, ck))
    print(f"전체 체크포인트 {len(ckpts)}개 | 신규평가 {len(todo)}개 {[e for e,_ in todo]} | 캐시재사용 {len(cached)}개 {[e for e,_ in cached]}")

    with open(TEST_JSONL, encoding="utf-8") as f:
        test = [json.loads(l) for l in f]
    golds = [r["key"] for r in test]

    curve = {}
    # 캐시된 에폭은 저장된 예측으로 지표만 재계산
    for ep, ck in cached:
        rows = [json.loads(l) for l in open(os.path.join(RESULT_DIR, f"pred_lora_ep{ep}.jsonl"), encoding="utf-8")]
        g = [r["key"] for r in rows]
        p = [r["pred_key"] for r in rows]
        acc, f1 = score(g, p)
        curve[ep] = {"epoch": ep, "checkpoint": os.path.basename(ck), "accuracy": acc, "macro_f1": f1}

    if todo:
        print("base 로딩(1회)...")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
        processor = AutoProcessor.from_pretrained(BASE_MODEL, max_pixels=768 * 768)

        model = None
        names = []
        for ep, ck in todo:
            name = f"ep{ep}"
            if model is None:
                model = PeftModel.from_pretrained(base, ck, adapter_name=name)
            else:
                model.load_adapter(ck, adapter_name=name)
            names.append((ep, ck, name))
        model.eval()

        for ep, ck, name in names:
            model.set_adapter(name)
            preds, rows = [], []
            for r in test:
                ans = infer(model, processor, r["image"])
                pk = extract_label(ans)
                preds.append(pk)
                rows.append({"image": r["image"], "key": r["key"], "pred_key": pk, "answer": ans})
            acc, f1 = score(golds, preds)
            curve[ep] = {"epoch": ep, "checkpoint": os.path.basename(ck), "accuracy": acc, "macro_f1": f1}
            print(f"  [epoch {ep:2d}] acc={acc:.3f}  macro-F1={f1:.3f}")
            with open(os.path.join(RESULT_DIR, f"pred_lora_ep{ep}.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ordered = [curve[e] for e in sorted(curve)]
    best = max(ordered, key=lambda c: c["macro_f1"])

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(CURVE_JSON, "w", encoding="utf-8") as f:
        json.dump({"curve": ordered, "best": best}, f, ensure_ascii=False, indent=2)

    print("\n=== 에폭별 macro-F1 곡선 ===")
    for c in ordered:
        bar = "█" * int(c["macro_f1"] * 40)
        mark = " " if c["epoch"] == best["epoch"] else ""
        print(f"  ep{c['epoch']:2d}  acc={c['accuracy']:.3f}  F1={c['macro_f1']:.3f}  {bar}{mark}")
    print(f"\n최적: epoch {best['epoch']} (macro-F1={best['macro_f1']:.3f}, acc={best['accuracy']:.3f}) → {best['checkpoint']}")


if __name__ == "__main__":
    main()
