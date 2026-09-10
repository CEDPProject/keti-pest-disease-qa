"""'병명만 답하세요' 프롬프트로 에폭별 LoRA 체크포인트 평가 — 2.5-VL·3.5 공정 비교.

배포 파이프라인(name_rag_pipeline._vlm_name)과 동일한 짧은 병명 프롬프트를 사용해
병명 파싱을 안정화한다(진단+증상 프롬프트는 모델별 출력 형식차로 파싱이 불안정).
기존 결과 파일은 건드리지 않고 nameprompt_* 로 별도 저장한다. qwenvl-ft 환경.

  python eval_name_prompt.py --model_type qwen35 --output_dir ../../models/qwen3_5vl_lora
  python eval_name_prompt.py --model_type qwen25 --output_dir ../../models/qwen2_5vl_lora
"""

import os
import re
import sys
import json
import glob
import argparse

os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from label_extract import extract_label  # noqa
from classes import TEST_JSONL  # noqa

RESULT_DIR = os.path.join(VLM_FT, "results")
NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis",
              "black_spot", "green_mold", "anthracnose"]
STEPS_PER_EPOCH = 48

MODELS = {
    "qwen25": {"base": "Qwen/Qwen2.5-VL-7B-Instruct", "loader": "qwen2_5_vl"},
    "qwen35": {"base": "Qwen/Qwen3.5-4B",             "loader": "auto_mm"},
}


def load_base(cfg):
    if cfg["loader"] == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as M
        return M.from_pretrained(cfg["base"], torch_dtype=torch.bfloat16, device_map="cuda")
    from transformers import AutoModelForMultimodalLM as M
    return M.from_pretrained(cfg["base"], torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)


def infer(model, processor, image_path):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path, "max_pixels": 768 * 768},
        {"type": "text", "text": NAME_PROMPT}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def score(golds, preds):
    return (accuracy_score(golds, preds),
            f1_score(golds, preds, labels=CLASS_KEYS, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", required=True, choices=list(MODELS))
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    cfg = MODELS[args.model_type]
    tag = args.model_type

    ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                   key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    if not ckpts:
        raise SystemExit(f"체크포인트 없음: {args.output_dir}")

    test = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")]
    golds = [r["key"] for r in test]

    print(f"[{tag}] base 로딩: {cfg['base']} ...")
    base = load_base(cfg)
    processor = AutoProcessor.from_pretrained(cfg["base"], trust_remote_code=True)

    model, names = None, []
    for ck in ckpts:
        ep = int(re.search(r"checkpoint-(\d+)", ck).group(1)) // STEPS_PER_EPOCH
        name = f"ep{ep}"
        if model is None:
            model = PeftModel.from_pretrained(base, ck, adapter_name=name)
        else:
            model.load_adapter(ck, adapter_name=name)   # in-place, 반환값 무시
        names.append((ep, ck, name))
    model.eval()

    curve = []
    os.makedirs(RESULT_DIR, exist_ok=True)
    for ep, ck, name in names:
        model.set_adapter(name)
        preds, rows = [], []
        for r in test:
            ans = infer(model, processor, r["image"])
            pk = extract_label(ans)
            preds.append(pk)
            rows.append({"key": r["key"], "pred_key": pk, "answer": ans})
        acc, f1 = score(golds, preds)
        curve.append({"epoch": ep, "accuracy": acc, "macro_f1": f1})
        print(f"  [ep{ep:2d}] acc={acc:.3f}  macro-F1={f1:.3f}")
        with open(os.path.join(RESULT_DIR, f"pred_nameprompt_{tag}_ep{ep}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)

    best = max(curve, key=lambda c: c["macro_f1"])
    with open(os.path.join(RESULT_DIR, f"nameprompt_{tag}_curve.json"), "w", encoding="utf-8") as f:
        json.dump({"curve": curve, "best": best}, f, ensure_ascii=False, indent=2)

    print(f"\n=== [{tag}] '병명만' 프롬프트 macro-F1 곡선 ===")
    for c in curve:
        bar = "█" * int(c["macro_f1"] * 40)
        mark = " " if c["epoch"] == best["epoch"] else ""
        print(f"  ep{c['epoch']:2d}  acc={c['accuracy']:.3f}  F1={c['macro_f1']:.3f}  {bar}{mark}")
    print(f"[{tag}] 최적 ep{best['epoch']} · macro-F1={best['macro_f1']:.3f} · acc={best['accuracy']:.3f}")


if __name__ == "__main__":
    main()
