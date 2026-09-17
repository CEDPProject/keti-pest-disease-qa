"""교차 데이터셋(Mendeley) 공정 평가 — 2.5·3.5 동일 조건(ep4 + '병명만' 프롬프트).

일반화 검사: 학습에 안 쓴 다른 출처(Mendeley Citrus)의 겹치는 병해로 정확도 측정.
기존 eval_cross_dataset.py(2.5 merged/진단프롬프트, 0.155)와 달리, 같은 프롬프트·에폭으로
두 모델을 나란히 비교한다. qwenvl-ft 환경.

  python eval_cross_nameprompt.py --model_type qwen35 --output_dir ../../models/qwen3_5vl_lora --epoch 4
  python eval_cross_nameprompt.py --model_type qwen25 --output_dir ../../models/qwen2_5vl_lora --epoch 4
"""

import os
import sys
import glob
import json
import random
import argparse
from collections import Counter

os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel
from sklearn.metrics import accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(VLM_FT), "backend", "common"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa
from classes import MENDELEY_DIR as DS  # noqa
KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}

NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
GROUPS = [
    (os.path.join(DS, "Fruits", "Canker"), "canker", "궤양병"),
    (os.path.join(DS, "Fruits", "Scab"), "scab", "더뎅이병"),
    (os.path.join(DS, "Leaves", "Melanose"), "black_spot", "검은점무늬병"),
]
MAX_PER_CLASS = 30
SEED = 42
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


def ask(model, proc, path):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": path, "max_pixels": 768 * 768},
        {"type": "text", "text": NAME_PROMPT}]}]
    try:
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img = Image.open(path).convert("RGB")
    inp = proc(text=[txt], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=64, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", required=True, choices=list(MODELS))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epoch", type=int, default=4)
    args = ap.parse_args()
    cfg = MODELS[args.model_type]
    tag = args.model_type
    ck = os.path.join(args.output_dir, f"checkpoint-{args.epoch * STEPS_PER_EPOCH}")
    if not os.path.isdir(ck):
        raise SystemExit(f"체크포인트 없음: {ck}")

    rng = random.Random(SEED)
    items = []
    for folder, key, ko in GROUPS:
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        files = files[:MAX_PER_CLASS]
        items += [(f, key, ko) for f in files]
        print(f"  {ko:12s}: {len(files)}장")
    print(f"총 {len(items)}장 · [{tag}] ep{args.epoch} · 프롬프트='병명만'\n")

    base = load_base(cfg)
    model = PeftModel.from_pretrained(base, ck).eval()
    proc = AutoProcessor.from_pretrained(cfg["base"], trust_remote_code=True)

    golds, preds, rows = [], [], []
    for i, (path, key, ko) in enumerate(items, 1):
        ans = ask(model, proc, path)
        pk = extract_label(ans)
        golds.append(key)
        preds.append(pk)
        rows.append({"image": os.path.basename(path), "gold": ko,
                     "pred": KEY_TO_LABEL.get(pk, "불명확"), "ok": pk == key, "answer": ans})
        if i % 15 == 0 or i == len(items):
            acc = sum(g == p for g, p in zip(golds, preds)) / len(golds)
            print(f"  [{i:3d}/{len(items)}] 누적 acc {acc:.3f}")

    acc = accuracy_score(golds, preds)
    print(f"\n=== [{tag} ep{args.epoch}] 교차 데이터셋(Mendeley) accuracy = {acc:.3f} (N={len(golds)}) ===")
    for key, ko in [("canker", "궤양병"), ("scab", "더뎅이병"), ("black_spot", "검은점무늬병")]:
        idx = [i for i, g in enumerate(golds) if g == key]
        hit = sum(preds[i] == key for i in idx)
        print(f"    {ko}: recall {hit/len(idx):.3f} ({hit}/{len(idx)})" if idx else f"    {ko}: (없음)")
    print("  예측 분포:", dict(Counter(KEY_TO_LABEL.get(p, "불명확") for p in preds)))

    out = os.path.join(VLM_FT, "results", f"cross_nameprompt_{tag}.json")
    json.dump({"model": tag, "epoch": args.epoch, "n": len(golds), "accuracy": acc, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
