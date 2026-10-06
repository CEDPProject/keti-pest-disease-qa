import os, sys, re, json
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
from classes import RESULTS_DIR  # noqa: E402
from classes import CLASSES
from label_extract import LABEL_TO_KEY, extract_label
from kb_retriever import KBRetriever

BASE = "Qwen/Qwen3.5-9B"
MODEL_DIR = os.path.join(REPO, "models", "qwen3_5_9b_balanced_lora")
CURVE = os.path.join(RESULTS_DIR, "balanced_curve_9b.json")
TEST = os.path.join(REPO, "data", "vlm_finetune_balanced", "test.jsonl")
OUT = os.path.join(RESULTS_DIR, "p1_2_hallucination_9b.json")
STEPS_PER_EPOCH = 233
KO = {v: k for k, v in LABEL_TO_KEY.items()}; KO["normal"] = "정상"
K2SEC = {m["key"]: m["kb_section"] for m in CLASSES.values()}
QS = [("학명", "이 병의 학명은 무엇인가요?"), ("약제", "이 병에는 어떤 약제를 사용하나요?"),
      ("설명", "이 병에 대해 자세히 설명해주세요.")]
LATIN = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z]{3,})")


def latin_set(text):
    return set(f"{m.group(1)} {m.group(2)}" for m in LATIN.finditer(text))


def best_adapter():
    b = json.load(open(CURVE, encoding="utf-8"))["best"]
    return b["epoch"], os.path.join(MODEL_DIR, f"checkpoint-{b['epoch'] * STEPS_PER_EPOCH}")


@torch.no_grad()
def vlm(model, proc, img, q):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img, "max_pixels": 768*768},
                                         {"type": "text", "text": q}]}]
    try:
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    im = Image.open(img).convert("RGB")
    inp = proc(text=[t], images=[im], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=200, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ep, adapter = best_adapter()
    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    kb = KBRetriever(device="cpu")
    sci = {k: kb.kb[K2SEC[k]]["scientific"] for k in K2SEC}
    sci_latin = {k: latin_set(sci[k]) for k in sci}
    eng_latin = set()
    for k in K2SEC:
        eng_latin |= latin_set(kb.kb[K2SEC[k]].get("english", ""))  # 영명(학명 아님) → 채점 제외

    base = AutoModelForMultimodalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                    device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter).eval()
    print(f"9B adapter ep{ep} · 질병 test × 3질의", flush=True)

    def classify(text, gold_key):
        found = {n for n in latin_set(text) if n not in eng_latin}  # 영명 제외
        if not found:
            return "none", []
        types = []
        for name in found:
            if name in sci_latin.get(gold_key, set()):
                types.append("correct")
            elif any(name in sci_latin[k] for k in sci_latin):
                types.append("misattribution")
            else:
                types.append("fabrication")
        for lv in ("fabrication", "misattribution", "correct"):
            if lv in types:
                return lv, sorted(found)
        return "none", []

    rowsA, rowsB = [], []
    for i, r in enumerate(test):
        gold = r["key"]; img = r["image"]
        if gold == "normal":
            continue
        for tag, q in QS:
            a = vlm(model, proc, img, q); ca, fa = classify(a, gold)
            rowsA.append({"image": os.path.basename(img), "gold": KO[gold], "q": tag,
                          "answer": a[:250], "latin": fa, "halluc": ca})
            b = kb.build_answer(K2SEC[gold], q); cb, fb = classify(b, gold)
            rowsB.append({"image": os.path.basename(img), "gold": KO[gold], "q": tag,
                          "latin": fb, "halluc": cb})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}", flush=True)

    def agg(rows):
        n = len(rows)
        fab = sum(1 for r in rows if r["halluc"] == "fabrication")
        mis = sum(1 for r in rows if r["halluc"] == "misattribution")
        cor = sum(1 for r in rows if r["halluc"] == "correct")
        non = sum(1 for r in rows if r["halluc"] == "none")
        return {"n": n, "fabrication": fab, "misattribution": mis, "correct": cor, "none": non,
                "halluc_rate": round((fab+mis)/n, 4), "halluc_n": fab+mis}

    res = {"model": BASE, "epoch": ep,
           "condition_A_noRAG": agg(rowsA), "condition_B_RAG": agg(rowsB),
           "note": "학명(라틴 이명) 기준. 정상 클래스 제외(병원균 없음→학명 부재). 질병 49장×3질의(학명/약제/설명)=147 factual queries. P1-2 실측 프로토콜.",
           "A_halluc_examples": [r for r in rowsA if r["halluc"] in ("fabrication", "misattribution")][:8],
           "B_halluc_examples": [r for r in rowsB if r["halluc"] in ("fabrication", "misattribution")]}
    json.dump({"summary": res, "rowsA": rowsA, "rowsB": rowsB},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    A = res["condition_A_noRAG"]; B = res["condition_B_RAG"]
    print("=" * 56); print("  9B 환각률 (학명 기준)")
    print(f"  조건A(9B직접): 환각 {A['halluc_n']}/{A['n']} = {A['halluc_rate']*100:.1f}%")
    print(f"  조건B(9B+RAG): 환각 {B['halluc_n']}/{B['n']} = {B['halluc_rate']*100:.1f}% "
          f"{'0% 확인' if B['halluc_n']==0 else '0아님 → 사례 전부 첨부'}")
    print(f"  저장: {OUT}")


if __name__ == "__main__":
    main()
