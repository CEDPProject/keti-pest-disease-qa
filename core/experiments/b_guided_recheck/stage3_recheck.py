"""유도 감별 재확인 3단계 — 2차 재질문 파이프라인.

1단계 top-3 + 2단계 병징 발췌(가나다순)를 사진과 함께 제시하고 VLM이 다시 관찰해
병명 하나를 답하게 한다. RAG는 판단하지 않고 가이드만 제공.
조건: (a) 전체 (b) 마진기반 선택 (c) 오답 oracle.
안전폴백: 2차 답변이 top-3 밖/파싱실패 → top-1 유지. 결정적(greedy).
결과: core/results/b_stage3_recheck.json
"""
import os
import sys
import json
import random

os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score

random.seed(42); np.random.seed(42); torch.manual_seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(CORE)
sys.path.insert(0, os.path.join(CORE, "common"))
sys.path.insert(0, os.path.join(CORE, "rag"))
from classes import CLASSES  # noqa
from label_extract import extract_label, LABEL_TO_KEY  # noqa
from kb_retriever import KBRetriever  # noqa

BASE = "Qwen/Qwen3.5-4B"
ADAPTER = os.path.join(REPO, "models", "qwen3_5_balanced_lora", "checkpoint-932")
TEST = os.path.join(REPO, "data", "vlm_finetune_balanced", "test.jsonl")
STAGE1 = os.path.join(CORE, "results", "b_stage1_top3.json")
OUT = os.path.join(CORE, "results", "b_stage3_recheck.json")
MAX_PIXELS = 768 * 768

CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot",
              "green_mold", "anthracnose", "normal"]
KO = {v: k for k, v in LABEL_TO_KEY.items()}; KO["normal"] = "정상"
KEY_TO_SECTION = {m["key"]: m["kb_section"] for lab, m in CLASSES.items()}
VISUAL_QUERY = ("병반 반점 딱지 부스럼 융기 돌출 코르크 분화구 검은점 "
                "부패 곰팡이 균사 포자 색깔 표면 모양 크기 테두리")
NORMAL_RULE = "병징에 해당하는 이상 소견(반점·병반·변색·곰팡이 등)이 없으면 정상."


def build_guides(kb):
    g = {}
    for key in CLASS_KEYS:
        if key == "normal":
            g[key] = [NORMAL_RULE]
        else:
            lines = kb.retrieve_lines(KEY_TO_SECTION[key], VISUAL_QUERY, "병징", top_n=3)
            g[key] = [l["text"] for l in lines]
    return g


def second_prompt(proc, top3_keys, guides):
    order = sorted(top3_keys, key=lambda k: KO[k])          # 가나다순
    blocks = []
    for k in order:
        body = " ".join(guides[k])
        blocks.append(f"[{KO[k]}]\n{body}")
    text = ("다음 세 후보 중 하나다. 각 후보의 병징 발췌를 참고하여 "
            "사진을 다시 관찰하고 병명 하나만 답하라.\n\n" + "\n\n".join(blocks))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": None, "max_pixels": MAX_PIXELS},
        {"type": "text", "text": text}]}]
    try:
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return t, order[0]                                        # 첫 후보(위치편향 검사용)


@torch.no_grad()
def gen(model, proc, image, prompt_text):
    im = Image.open(image).convert("RGB")
    inp = proc(text=[prompt_text], images=[im], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=32, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def _metrics(golds, preds, srcs):
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, labels=CLASS_KEYS, average="macro", zero_division=0)
    src = {}
    for s in ("원본", "NCPMS"):
        idx = [i for i in range(len(golds)) if srcs[i] == s and golds[i] != "normal"]
        src[s] = round(sum(preds[i] == golds[i] for i in idx) / len(idx), 3) if idx else None
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "src_acc": src}


def _transitions(golds, base, final):
    w2c = [i for i in range(len(golds)) if base[i] != golds[i] and final[i] == golds[i]]
    c2w = [i for i in range(len(golds)) if base[i] == golds[i] and final[i] != golds[i]]
    normal_flip = [i for i in range(len(golds)) if golds[i] == "normal"
                   and base[i] == "normal" and final[i] != "normal"]
    return {"wrong2correct": len(w2c), "correct2wrong": len(c2w),
            "net": len(w2c) - len(c2w), "normal_to_disease": len(normal_flip),
            "w2c_idx": w2c, "c2w_idx": c2w}


def main():
    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    s1 = {r["image"]: r for r in json.load(open(STAGE1, encoding="utf-8"))["rows"]}

    kb = KBRetriever(device="cpu")
    guides = build_guides(kb)

    base = AutoModelForMultimodalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                    device_map="cuda", trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, ADAPTER).eval()

    rows = []
    for i, r in enumerate(test):
        img_b = os.path.basename(r["image"])
        s = s1[img_b]
        gold = s["gold"]; top1 = s["score_top1"]; top3 = s["score_top3"]; margin = s["margin_1_2"]
        prompt, first_cand = second_prompt(proc, top3, guides)
        raw2 = gen(model, proc, r["image"], prompt)
        key2 = extract_label(raw2)
        applied_valid = key2 in top3
        second = key2 if applied_valid else top1        # 폴백
        rows.append({
            "image": img_b, "gold": gold, "source": s["source"],
            "top1": top1, "top3": top3, "margin": margin,
            "second_raw": raw2, "second_key": key2, "valid": applied_valid,
            "second_final": second, "first_cand": first_cand,
            "hit_first": key2 == first_cand,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}")

    golds = [r["gold"] for r in rows]
    srcs = [r["source"] for r in rows]
    base_pred = [r["top1"] for r in rows]
    margins = [r["margin"] for r in rows]
    wrong_idx = [i for i in range(len(rows)) if base_pred[i] != golds[i]]

    # 조건별 최종 예측
    def cond_a():
        return [r["second_final"] for r in rows]

    def cond_b(t):
        return [rows[i]["second_final"] if margins[i] < t else base_pred[i] for i in range(len(rows))]

    def cond_c():
        return [rows[i]["second_final"] if i in wrong_idx else base_pred[i] for i in range(len(rows))]

    results = {}
    results["baseline"] = {**_metrics(golds, base_pred, srcs)}
    pa = cond_a(); results["a_all"] = {**_metrics(golds, pa, srcs), **_transitions(golds, base_pred, pa)}
    pc = cond_c(); results["c_oracle_wrong13"] = {**_metrics(golds, pc, srcs), **_transitions(golds, base_pred, pc)}

    # (b) 마진 임계 트레이드오프
    b_table = []
    for t in (0.7, 0.9, 1.1):
        applied = [i for i in range(len(rows)) if margins[i] < t]
        n_wrong_covered = sum(1 for i in applied if i in wrong_idx)
        n_correct_risk = sum(1 for i in applied if i not in wrong_idx)
        pb = cond_b(t)
        tr = _transitions(golds, base_pred, pb)
        m = _metrics(golds, pb, srcs)
        b_table.append({"threshold": t, "n_applied": len(applied),
                        "wrong_covered": n_wrong_covered, "correct_at_risk": n_correct_risk,
                        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], **tr,
                        "src_acc": m["src_acc"]})
    results["b_margin_table"] = b_table
    best_b = max(b_table, key=lambda x: x["net"])
    results["b_best"] = best_b

    # 위치편향
    hit_first = sum(r["hit_first"] for r in rows) / len(rows)
    # 약함 판정 케이스(정답=잿빛곰팡이) 2차 성패
    weak = [{"image": r["image"], "gold": KO[r["gold"]], "top1": KO[r["top1"]],
             "second": KO.get(r["second_key"], r["second_key"]), "valid": r["valid"],
             "correct": r["second_final"] == r["gold"]}
            for r in rows if r["gold"] == "gray_mold"]

    summary = {
        "baseline": results["baseline"],
        "a_all": results["a_all"],
        "b_margin_table": b_table,
        "b_best_threshold": best_b["threshold"],
        "c_oracle": results["c_oracle_wrong13"],
        "position_bias_hit_first": round(hit_first, 3),
        "position_bias_expected": round(1/3, 3),
        "graymold_cases": weak,
    }
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("  STAGE 3 결과 요약")
    print("=" * 64)
    b = results["baseline"]; a = results["a_all"]; c = results["c_oracle_wrong13"]
    print(f"  baseline    : F1={b['macro_f1']:.3f} acc={b['accuracy']:.3f} 원본{b['src_acc']['원본']} NCPMS{b['src_acc']['NCPMS']}")
    print(f"  (a) 전체적용: F1={a['macro_f1']:.3f} acc={a['accuracy']:.3f} | 오답→정답 {a['wrong2correct']} 정답→오답 {a['correct2wrong']} 순이득 {a['net']} | 정상→병해 {a['normal_to_disease']}")
    print(f"  (c) oracle  : F1={c['macro_f1']:.3f} acc={c['accuracy']:.3f} | 오답→정답 {c['wrong2correct']} 정답→오답 {c['correct2wrong']} 순이득 {c['net']}")
    print("  (b) 마진 임계 트레이드오프:")
    for r in b_table:
        print(f"     t={r['threshold']}: 적용{r['n_applied']}장(오답{r['wrong_covered']}+정답{r['correct_at_risk']}) "
              f"→ F1={r['macro_f1']:.3f} 순이득{r['net']}")
    print(f"  위치편향(가나다 첫후보 선택율): {hit_first:.3f} (기대 0.333)")
    print(f"  잿빛곰팡이 정답 케이스 2차 성공: {sum(w['correct'] for w in weak)}/{len(weak)}")
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
