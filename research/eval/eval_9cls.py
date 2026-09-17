import os
import re
import sys
import json
import glob
import argparse
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(VLM_FT), "backend", "common"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa
KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}

BASE = "Qwen/Qwen3.5-4B"
TEST = os.path.join(VLM_FT, "..", "data", "vlm_finetune_9cls", "test.jsonl")
RESULT_DIR = os.path.join(VLM_FT, "results")
NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot",
              "green_mold", "anthracnose", "normal"]
STEPS_PER_EPOCH = 60


def infer(model, proc, path):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": path, "max_pixels": 768 * 768},
        {"type": "text", "text": NAME_PROMPT}]}]
    try:
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    im = Image.open(path).convert("RGB")
    inp = proc(text=[t], images=[im], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=64, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epoch", type=int, default=0, help="0=전 에폭 스윕")
    args = ap.parse_args()

    cks = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                 key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    if args.epoch:
        cks = [c for c in cks if int(re.search(r"checkpoint-(\d+)", c).group(1)) // STEPS_PER_EPOCH == args.epoch]
    if not cks:
        raise SystemExit("체크포인트 없음")

    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    golds = [r["key"] for r in test]

    print(f"base 로딩: {BASE}")
    base = AutoModelForMultimodalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                    device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

    model, names = None, []
    for ck in cks:
        ep = int(re.search(r"checkpoint-(\d+)", ck).group(1)) // STEPS_PER_EPOCH
        name = f"ep{ep}"
        if model is None:
            model = PeftModel.from_pretrained(base, ck, adapter_name=name)
        else:
            model.load_adapter(ck, adapter_name=name)
        names.append((ep, ck, name))
    model.eval()

    curve = []
    for ep, ck, name in names:
        model.set_adapter(name)
        preds, rows = [], []
        for r in test:
            ans = infer(model, proc, r["image"])
            pk = extract_label(ans)
            preds.append(pk)
            rows.append({"key": r["key"], "pred_key": pk, "answer": ans})
        acc = accuracy_score(golds, preds)
        f1 = f1_score(golds, preds, labels=CLASS_KEYS, average="macro", zero_division=0)
        # 정상 recall
        nidx = [i for i, g in enumerate(golds) if g == "normal"]
        nrec = sum(preds[i] == "normal" for i in nidx) / len(nidx) if nidx else 0
        # 질병(정상 아닌 것) accuracy
        didx = [i for i, g in enumerate(golds) if g != "normal"]
        dacc = sum(preds[i] == golds[i] for i in didx) / len(didx) if didx else 0
        curve.append({"epoch": ep, "accuracy": acc, "macro_f1": f1, "normal_recall": nrec, "disease_acc": dacc})
        print(f"  [ep{ep}] acc={acc:.3f} F1={f1:.3f} | 정상recall {sum(preds[i]=='normal' for i in nidx)}/{len(nidx)} | 질병acc={dacc:.3f}")
        with open(os.path.join(RESULT_DIR, f"pred_9cls_ep{ep}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)

    best = max(curve, key=lambda c: c["macro_f1"])
    json.dump({"curve": curve, "best": best},
              open(os.path.join(RESULT_DIR, "9cls_curve.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n최적 ep{best['epoch']} · macro-F1={best['macro_f1']:.3f} · acc={best['accuracy']:.3f} · "
          f"정상recall={best['normal_recall']:.2f} · 질병acc={best['disease_acc']:.3f}")


if __name__ == "__main__":
    main()
