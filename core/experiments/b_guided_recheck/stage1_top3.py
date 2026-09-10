"""STAGE 1: top-3 후보 추출기 + top-3 recall 측정 (B안 게이트).

방법: 각 이미지에 대해 8개 클래스 라벨 문자열을 teacher-forcing 로그우도로 스코어링
      → 상위 3개가 후보. (생성 샘플링 아님, 결정적/재현 가능)
검증: top-1 이 기존 생성평가(0.772)와 일치하는지 먼저 확인.
게이트: top-3 recall >= 0.85 이면 STAGE 2 진행.

결과: core/results/b_stage1_top3.json
"""
import os
import sys
import json
import glob

os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(CORE)
sys.path.insert(0, os.path.join(CORE, "common"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa

BASE = "Qwen/Qwen3.5-4B"
ADAPTER = os.path.join(REPO, "models", "qwen3_5_balanced_lora", "checkpoint-932")  # ep4
TEST = os.path.join(REPO, "data", "vlm_finetune_balanced", "test.jsonl")
OUT = os.path.join(CORE, "results", "b_stage1_top3.json")
NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
MAX_PIXELS = 768 * 768

CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot",
              "green_mold", "anthracnose", "normal"]
KEY_KO = {v: k for k, v in LABEL_TO_KEY.items()}
KEY_KO["normal"] = "정상"
LABELS = {k: KEY_KO[k] for k in CLASS_KEYS}   # 스코어링 대상 문자열(병명)


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
    """각 라벨 문자열의 length-normalized 로그우도 반환 {key: mean_logprob}."""
    im = Image.open(image).convert("RGB")
    enc_p = proc(text=[prompt_text], images=[im], return_tensors="pt").to("cuda")
    Lp = enc_p.input_ids.shape[1]
    scores = {}
    for key, label in LABELS.items():
        enc_f = proc(text=[prompt_text + label], images=[im], return_tensors="pt").to("cuda")
        Lf = enc_f.input_ids.shape[1]
        logits = model(**enc_f).logits[0]                 # [Lf, V]
        tgt = enc_f.input_ids[0, Lp:Lf]                   # 타깃(라벨) 토큰
        logp = torch.log_softmax(logits[Lp - 1:Lf - 1], dim=-1)
        tok_lp = logp[torch.arange(len(tgt)), tgt]
        scores[key] = float(tok_lp.mean().item())          # length-normalized
    return scores


@torch.no_grad()
def generate_top1(model, proc, image, prompt_text):
    """생성평가와 동일 방식 top-1 (검증용)."""
    im = Image.open(image).convert("RGB")
    inp = proc(text=[prompt_text], images=[im], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    ans = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return extract_label(ans), ans


def main():
    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    print(f"test {len(test)}장 · adapter ep4(checkpoint-932)")

    base = AutoModelForMultimodalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                    device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    prompt = build_prompt(proc)

    rows = []
    for i, r in enumerate(test):
        gold = r["key"]
        sc = score_labels(model, proc, r["image"], prompt)
        ranked = sorted(sc, key=lambda k: -sc[k])          # 점수순
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
            print(f"  {i+1}/{len(test)}")

    # 집계
    n = len(rows)
    score_top1_acc = sum(r["score_top1"] == r["gold"] for r in rows) / n
    top3_recall = sum(r["gold"] in r["score_top3"] for r in rows) / n
    gen_top1_acc = sum(r["gen_top1"] == r["gold"] for r in rows) / n
    # 스코어 top1 vs 생성 top1 일치율(검증)
    agree = sum(r["score_top1"] == r["gen_top1"] for r in rows) / n

    # 마진 분포(정답/오답 케이스 분리 — top-1 기준)
    m_correct = sorted(r["margin_1_2"] for r in rows if r["score_top1"] == r["gold"])
    m_wrong = sorted(r["margin_1_2"] for r in rows if r["score_top1"] != r["gold"])

    def _stat(xs):
        if not xs:
            return None
        return {"n": len(xs), "min": xs[0], "median": xs[len(xs)//2], "max": xs[-1],
                "mean": round(sum(xs)/len(xs), 4)}

    # 출처별 top-3 recall
    src = {}
    for s in ("원본", "NCPMS"):
        sub = [r for r in rows if r["source"] == s]
        src[s] = round(sum(r["gold"] in r["score_top3"] for r in sub) / len(sub), 3) if sub else None

    summary = {
        "n_test": n,
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
    print("  STAGE 1 결과")
    print("=" * 60)
    print(f"  스코어 top-1 acc : {score_top1_acc:.4f}")
    print(f"  생성   top-1 acc : {gen_top1_acc:.4f}  (기존 평가 0.772와 비교)")
    print(f"  스코어=생성 일치 : {agree:.4f}")
    print(f"  top-3 recall   : {top3_recall:.4f}   → 게이트(≥0.85) {'통과' if top3_recall>=0.85 else '미달'}")
    print(f"  출처별 top-3     : {src}")
    print(f"  마진(정답케이스) : {_stat(m_correct)}")
    print(f"  마진(오답케이스) : {_stat(m_wrong)}")
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
