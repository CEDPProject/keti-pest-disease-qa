"""R2: Qwen3.5-4B zero-shot (어댑터 미적용, 파인튜닝 전) test 57장 병명 판정.

기존 평가와 동일 프로토콜: 고정 '병명만' 프롬프트, greedy, label_extract.
산출: acc, macro-F1 (전체 + 출처별) → core/results/zeroshot_qwen35.json
qwenvl-ft 환경. CUDA_VISIBLE_DEVICES로 단일 GPU 지정.
"""
import os
import sys
import json

os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from sklearn.metrics import f1_score, accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from label_extract import extract_label  # noqa

BASE = "Qwen/Qwen3.5-4B"
TEST = os.path.join(VLM_FT, "..", "data", "vlm_finetune_balanced", "test.jsonl")
RESULT_DIR = os.path.join(VLM_FT, "results")
OUT = os.path.join(RESULT_DIR, "zeroshot_qwen35.json")
NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot",
              "green_mold", "anthracnose", "normal"]


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
    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    golds = [r["key"] for r in test]
    srcs = [_source(r["image"], r["key"]) for r in test]

    print(f"[R2] base(zero-shot, 어댑터 없음) 로딩: {BASE} · test {len(test)}장", flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                     device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model.eval()

    preds, rows = [], []
    for i, r in enumerate(test):
        ans = infer(model, proc, r["image"])
        pk = extract_label(ans)
        preds.append(pk)
        rows.append({"key": r["key"], "pred_key": pk, "answer": ans,
                     "source": srcs[i], "image": os.path.basename(r["image"])})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}", flush=True)

    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, labels=CLASS_KEYS, average="macro", zero_division=0)
    # 출처별(질병만: 원본/NCPMS)
    didx = [i for i, g in enumerate(golds) if g != "normal"]
    src_acc = {}
    for s in ("원본", "NCPMS"):
        si = [i for i in didx if srcs[i] == s]
        src_acc[s] = round(sum(preds[i] == golds[i] for i in si) / len(si), 4) if si else None
    # 파싱 실패(unknown) 카운트 — zero-shot 형식 실패 진단
    n_unknown = sum(1 for p in preds if p not in CLASS_KEYS)
    pred_dist = {}
    for p in preds:
        pred_dist[p] = pred_dist.get(p, 0) + 1

    result = {
        "model": BASE, "condition": "zero-shot (no adapter)",
        "protocol": "고정 '병명만' 프롬프트, greedy, label_extract (파인튜닝 평가와 동일)",
        "n_test": len(test),
        "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
        "src_acc_disease": src_acc,
        "n_unknown_parse_fail": n_unknown,
        "pred_distribution": pred_dist,
        "rows": rows,
    }
    json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n★ [R2] zero-shot Qwen3.5-4B · macro-F1={f1:.4f} · acc={acc:.4f} "
          f"· 원본{src_acc['원본']} NCPMS{src_acc['NCPMS']} · unknown={n_unknown}/{len(test)}")
    print(f"예측분포: {pred_dist}")
    print(f"저장: {OUT}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train", "zeroshot_qwen35.done"), "w") as f:
        f.write(f"f1{f1:.4f}_acc{acc:.4f}\n")


if __name__ == "__main__":
    main()
