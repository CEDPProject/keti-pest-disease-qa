"""M3: Qwen3.5-9B top-3 teacher-forcing 스코어링 + 신뢰도(마진) — 논문용.
4B stage1_top3.py 와 동일 방식(8클래스 라벨 length-normalized 로그우도, 결정적),
베이스 9B, 어댑터=최적 에폭(balanced_curve_9b.json에서 자동 판독).

산출: core/results/b_stage1_top3_9b.json
  - score_top1 vs 생성 top-1 일치율 (검증 먼저)
  - top-3 recall (전체 + 출처별)
  - 마진 분포: 정답/오답 케이스 중앙값
"""
import os
import sys
import json

os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(VLM_FT)
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa

BASE = "Qwen/Qwen3.5-9B"
MODEL_DIR = os.path.join(REPO, "models", "qwen3_5_9b_balanced_lora")
CURVE = os.path.join(VLM_FT, "results", "balanced_curve_9b.json")
TEST = os.path.join(REPO, "data", "vlm_finetune_balanced", "test.jsonl")
OUT = os.path.join(VLM_FT, "results", "b_stage1_top3_9b.json")
NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
MAX_PIXELS = 768 * 768
STEPS_PER_EPOCH = 233
CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot",
              "green_mold", "anthracnose", "normal"]
KEY_KO = {v: k for k, v in LABEL_TO_KEY.items()}
KEY_KO["normal"] = "정상"
LABELS = {k: KEY_KO[k] for k in CLASS_KEYS}


def best_adapter():
    best = json.load(open(CURVE, encoding="utf-8"))["best"]
    ep = best["epoch"]
    ck = os.path.join(MODEL_DIR, f"checkpoint-{ep * STEPS_PER_EPOCH}")
    return ep, ck


def _source(path, key):
    b = os.path.basename(path)
    if key == "normal":
        return "잎" if "잎" in b else ("열매" if "열매" in b else "기타")
    return "NCPMS" if b.startswith("ncpms_") else "원본"


def build_prompt(proc):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": None, "max_pixels": MAX_PIXELS},
        {"type": "text", "text": NAME_PROMPT}]}]
    try:
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_labels(model, proc, image, prompt_text):
    im = Image.open(image).convert("RGB")
    enc_p = proc(text=[prompt_text], images=[im], return_tensors="pt").to("cuda")
    Lp = enc_p.input_ids.shape[1]
    scores = {}
    for key, label in LABELS.items():
        enc_f = proc(text=[prompt_text + label], images=[im], return_tensors="pt").to("cuda")
        Lf = enc_f.input_ids.shape[1]
        logits = model(**enc_f).logits[0]
        tgt = enc_f.input_ids[0, Lp:Lf]
        logp = torch.log_softmax(logits[Lp - 1:Lf - 1], dim=-1)
        tok_lp = logp[torch.arange(len(tgt)), tgt]
        scores[key] = float(tok_lp.mean().item())
    return scores


@torch.no_grad()
def generate_top1(model, proc, image, prompt_text):
    im = Image.open(image).convert("RGB")
    inp = proc(text=[prompt_text], images=[im], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return extract_label(ans), ans


def main():
    ep, adapter = best_adapter()
    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    print(f"[M3] test {len(test)}장 · 9B adapter ep{ep} ({adapter})", flush=True)

    base = AutoModelForMultimodalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                    device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter).eval()
    prompt = build_prompt(proc)

    rows = []
    for i, r in enumerate(test):
        gold = r["key"]
        sc = score_labels(model, proc, r["image"], prompt)
        ranked = sorted(sc, key=lambda k: -sc[k])
        gen_key, gen_ans = generate_top1(model, proc, r["image"], prompt)
        rows.append({
            "image": os.path.basename(r["image"]), "gold": gold,
            "source": _source(r["image"], gold),
            "score_top1": ranked[0], "score_top3": ranked[:3],
            "gen_top1": gen_key, "gen_ans": gen_ans,
            "margin_1_2": round(sc[ranked[0]] - sc[ranked[1]], 4),
            "scores": {k: round(v, 4) for k, v in sc.items()},
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}", flush=True)

    n = len(rows)
    score_top1_acc = sum(r["score_top1"] == r["gold"] for r in rows) / n
    top3_recall = sum(r["gold"] in r["score_top3"] for r in rows) / n
    gen_top1_acc = sum(r["gen_top1"] == r["gold"] for r in rows) / n
    agree = sum(r["score_top1"] == r["gen_top1"] for r in rows) / n
    m_correct = sorted(r["margin_1_2"] for r in rows if r["score_top1"] == r["gold"])
    m_wrong = sorted(r["margin_1_2"] for r in rows if r["score_top1"] != r["gold"])

    def _stat(xs):
        if not xs:
            return None
        return {"n": len(xs), "min": xs[0], "median": xs[len(xs)//2], "max": xs[-1],
                "mean": round(sum(xs)/len(xs), 4)}

    src = {}
    for s in ("원본", "NCPMS"):
        sub = [r for r in rows if r["source"] == s]
        src[s] = round(sum(r["gold"] in r["score_top3"] for r in sub) / len(sub), 3) if sub else None

    summary = {
        "model": BASE, "epoch": ep, "n_test": n,
        "score_top1_acc": round(score_top1_acc, 4),
        "gen_top1_acc": round(gen_top1_acc, 4),
        "score_vs_gen_agree": round(agree, 4),
        "top3_recall": round(top3_recall, 4),
        "top3_recall_by_source": src,
        "margin_correct": _stat(m_correct),
        "margin_wrong": _stat(m_wrong),
        "gate_pass": bool(top3_recall >= 0.85),
    }
    json.dump({"summary": summary, "rows": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print(f"  [M3] 9B top-3 스코어링 (ep{ep})")
    print(f"  스코어 top-1 acc : {score_top1_acc:.4f}")
    print(f"  생성   top-1 acc : {gen_top1_acc:.4f}")
    print(f"  스코어=생성 일치 : {agree:.4f}")
    print(f"  top-3 recall   : {top3_recall:.4f} (원본{src['원본']} NCPMS{src['NCPMS']})")
    print(f"  마진 정답케이스: {_stat(m_correct)}")
    print(f"  마진 오답케이스: {_stat(m_wrong)}")
    print(f"  저장: {OUT}")


if __name__ == "__main__":
    main()
