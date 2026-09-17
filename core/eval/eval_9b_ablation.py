"""Ablation — 입력 모달리티 3조건 × [zero-shot, fine-tuned].

  (a) 텍스트만   : 이미지 없이 캡션(병명 토큰 제거) → 병명 선택
  (b) 이미지만   : 사진 + 최소 지시문("병명만 답하라.")
  (c) 이미지+텍스트: 사진 + 평가 고정 질문
각 조건 acc + macro-F1 (전체 + 출처별). 캡션은 fusion_captions.json 재사용.
결과: core/results/ablation_9b.json
"""
import os, sys, re, json
os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(VLM_FT)
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa

BASE = "Qwen/Qwen3.5-9B"
MODEL_DIR = os.path.join(REPO, "models", "qwen3_5_9b_balanced_lora")
CURVE = os.path.join(VLM_FT, "results", "balanced_curve_9b.json")
TEST = os.path.join(REPO, "data", "vlm_finetune_balanced", "test.jsonl")
CAPS = os.path.join(VLM_FT, "results", "fusion_captions.json")
OUT = os.path.join(VLM_FT, "results", "ablation_9b.json")
STEPS_PER_EPOCH = 233
MAXPIX = 768 * 768
NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."   # (c)
MIN_PROMPT = "병명만 답하라."                                       # (b)
KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot", "green_mold", "anthracnose", "normal"]
KO = {v: k for k, v in LABEL_TO_KEY.items()}; KO["normal"] = "정상"
DWORDS = list(KO.values()) + ["궤양", "더뎅이", "잿빛", "곰팡이", "수지", "검은점", "탄저", "멜라노스", "푸른곰팡이"]


def clean_caption(c):
    for w in DWORDS:
        c = re.sub(r"\S*" + re.escape(w) + r"\S*", "", c)
    return re.sub(r"\s+", " ", c).strip()


def best_adapter():
    b = json.load(open(CURVE, encoding="utf-8"))["best"]
    return b["epoch"], os.path.join(MODEL_DIR, f"checkpoint-{b['epoch'] * STEPS_PER_EPOCH}")


def _source(path, key):
    b = os.path.basename(path)
    if key == "normal":
        return "잎" if "잎" in b else ("열매" if "열매" in b else "기타")
    return "NCPMS" if b.startswith("ncpms_") else "원본"


@torch.no_grad()
def gen(model, proc, text, image=None):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image, "max_pixels": MAXPIX})
    content.append({"type": "text", "text": text})
    msgs = [{"role": "user", "content": content}]
    try:
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if image is not None:
        im = Image.open(image).convert("RGB")
        inp = proc(text=[t], images=[im], return_tensors="pt").to("cuda")
    else:
        inp = proc(text=[t], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def metrics(golds, preds, srcs):
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, labels=KEYS, average="macro", zero_division=0)
    sd = {}
    for s in ("원본", "NCPMS"):
        idx = [i for i in range(len(golds)) if srcs[i] == s and golds[i] != "normal"]
        sd[s] = round(sum(preds[i] == golds[i] for i in idx) / len(idx), 4) if idx else None
    return {"acc": round(acc, 4), "macro_f1": round(f1, 4), "src": sd}


def text_prompt(cap):
    names = "、".join(KO[k] for k in KEYS)
    return (f"다음은 감귤 사진의 시각적 특징 서술이다. 아래 병명 목록 중 서술과 가장 부합하는 병명 하나만 답하라.\n"
            f"[병명 목록] {names}\n[서술] {cap}\n답(병명 하나):")


def run_conditions(model, proc, test, caps, tag):
    golds = [r["key"] for r in test]
    srcs = [_source(r["image"], r["key"]) for r in test]
    out = {}
    # (a) 텍스트만 (정상은 캡션에 병징 없음 → 캡션 있는 항목만; 캡션 dict 키=basename)
    ga, pa, sa = [], [], []
    for r in test:
        b = os.path.basename(r["image"])
        if b not in caps:
            continue
        cap = clean_caption(caps[b]["caption"])
        pk = extract_label(gen(model, proc, text_prompt(cap), image=None))
        ga.append(r["key"]); pa.append(pk); sa.append(_source(r["image"], r["key"]))
    out["a_text_only"] = metrics(ga, pa, sa) | {"n": len(ga)}
    # (b) 이미지만(최소지시)
    pb = [extract_label(gen(model, proc, MIN_PROMPT, image=r["image"])) for r in test]
    out["b_image_only"] = metrics(golds, pb, srcs) | {"n": len(test)}
    # (c) 이미지+텍스트(고정질문)
    pc = [extract_label(gen(model, proc, NAME_PROMPT, image=r["image"])) for r in test]
    out["c_image_text"] = metrics(golds, pc, srcs) | {"n": len(test)}
    print(f"  [{tag}] a(텍스트)={out['a_text_only']['macro_f1']} "
          f"b(이미지)={out['b_image_only']['macro_f1']} c(이미지+텍스트)={out['c_image_text']['macro_f1']}", flush=True)
    return out


def main():
    ep, adapter = best_adapter()
    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    caps = json.load(open(CAPS, encoding="utf-8"))

    base = AutoModelForMultimodalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                    device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    base.eval()
    print(f"[M1] zero-shot(base) 조건 실행", flush=True)
    zs = run_conditions(base, proc, test, caps, "zero-shot")

    model = PeftModel.from_pretrained(base, adapter).eval()
    print(f"[M1] fine-tuned(ep{ep}) 조건 실행", flush=True)
    ft = run_conditions(model, proc, test, caps, f"fine-tuned ep{ep}")

    res = {"model": BASE, "epoch": ep, "zero_shot": zs, "fine_tuned": ft}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[M1] 저장: {OUT}")


if __name__ == "__main__":
    main()
