"""Qwen3.5-9B 균형 8클래스 에폭별 평가 → 최적 에폭 선정.

고정 '병명만' 프롬프트로 test 이미지를 추론(greedy), label_extract로 병명 추출.
지표: macro-F1, accuracy, 정상 recall, 질병 acc, 출처별(원본/NCPMS) acc.
결과: core/results/balanced_curve_9b.json, pred_balanced_9b_ep{N}.json

  python eval_9b_balanced.py --output_dir <adapter_dir> [--epoch N]
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
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa

BASE = "Qwen/Qwen3.5-9B"
TEST = os.path.join(VLM_FT, "..", "data", "vlm_finetune_balanced", "test.jsonl")
RESULT_DIR = os.path.join(VLM_FT, "results")
NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot",
              "green_mold", "anthracnose", "normal"]
STEPS_PER_EPOCH = 233
DONE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train", "eval_9b_balanced.done")


def _source(path, key):
    b = os.path.basename(path)
    if key == "normal":
        return "잎" if "잎" in b else ("열매" if "열매" in b else "기타")
    return "NCPMS" if b.startswith("ncpms_") else "원본"


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
        out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epoch", type=int, default=0, help="0=전 에폭 스윕")
    args = ap.parse_args()

    cks = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                 key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
    if args.epoch:
        cks = [c for c in cks if round(int(re.search(r"checkpoint-(\d+)", c).group(1)) / STEPS_PER_EPOCH) == args.epoch]
    if not cks:
        raise SystemExit("체크포인트 없음")

    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    golds = [r["key"] for r in test]
    srcs = [_source(r["image"], r["key"]) for r in test]

    print(f"[M2] base 로딩: {BASE} · test {len(test)}장", flush=True)
    base = AutoModelForMultimodalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                    device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

    model, names = None, []
    for ck in cks:
        ep = round(int(re.search(r"checkpoint-(\d+)", ck).group(1)) / STEPS_PER_EPOCH)
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
            rows.append({"key": r["key"], "pred_key": pk, "answer": ans,
                         "image": os.path.basename(r["image"]),
                         "source": _source(r["image"], r["key"])})
        acc = accuracy_score(golds, preds)
        f1 = f1_score(golds, preds, labels=CLASS_KEYS, average="macro", zero_division=0)
        nidx = [i for i, g in enumerate(golds) if g == "normal"]
        nrec = sum(preds[i] == "normal" for i in nidx) / len(nidx) if nidx else 0
        didx = [i for i, g in enumerate(golds) if g != "normal"]
        dacc = sum(preds[i] == golds[i] for i in didx) / len(didx) if didx else 0
        src_acc = {}
        for s in ("원본", "NCPMS"):
            si = [i for i in didx if srcs[i] == s]
            src_acc[s] = round(sum(preds[i] == golds[i] for i in si) / len(si), 4) if si else None
        curve.append({"epoch": ep, "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
                      "normal_recall": round(nrec, 3), "disease_acc": round(dacc, 3),
                      "src_acc": src_acc})
        print(f"  [ep{ep}] F1={f1:.4f} acc={acc:.4f} | 정상 {sum(preds[i]=='normal' for i in nidx)}/{len(nidx)} "
              f"| 질병acc={dacc:.3f} | 원본{src_acc['원본']} NCPMS{src_acc['NCPMS']}", flush=True)
        with open(os.path.join(RESULT_DIR, f"pred_balanced_9b_ep{ep}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)

    best = max(curve, key=lambda c: c["macro_f1"])
    json.dump({"model": BASE, "curve": curve, "best": best, "n_test": len(test)},
              open(os.path.join(RESULT_DIR, "balanced_curve_9b.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[M2] 최적 ep{best['epoch']} · macro-F1={best['macro_f1']:.4f} · acc={best['accuracy']:.4f} · "
          f"정상recall={best['normal_recall']:.2f} · 질병acc={best['disease_acc']:.3f} · "
          f"원본{best['src_acc']['원본']} NCPMS{best['src_acc']['NCPMS']}")
    with open(DONE, "w") as f:
        f.write(f"best_ep{best['epoch']}_f1{best['macro_f1']:.4f}\n")


if __name__ == "__main__":
    main()
