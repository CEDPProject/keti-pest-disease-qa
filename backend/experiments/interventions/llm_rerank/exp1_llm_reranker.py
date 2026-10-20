import os

from sklearn.metrics import accuracy_score

from backend.core.classes import KEY_TO_LABEL, RESULTS_DIR
from backend.core.kb_retriever import KBRetriever
from backend.core.label_extract import extract_label
from backend.core.ollama_client import ollama_generate
from backend.experiments.data_io import read_json, write_json
from backend.experiments.kb_text import symptom_guides
from backend.experiments.metrics import image_source, macro_f1, prediction_changes, source_accuracy

CAPS = os.path.join(RESULTS_DIR, "fusion_captions.json")
STAGE1 = os.path.join(RESULTS_DIR, "b_stage1_top3.json")
OUT = os.path.join(RESULTS_DIR, "exp1_llm_reranker.json")
MODEL = "gemma3:12b"
MARGIN_LIMIT = 0.9


# LLM 이 캡션과 후보 병징을 보고 고른 병명을 돌려준다
def ask(caption: str, order_keys: list, guides: dict):
    blocks = "\n".join(f"[{KEY_TO_LABEL[k]}] {guides[k]}" for k in order_keys)
    prompt = ("아래는 감귤 사진의 시각적 특징 서술이다. 이어서 후보 병해 3개의 병징 설명이 있다.\n"
              "서술과 가장 부합하는 후보 **하나만** 병명으로 답하라. 다른 설명 없이 병명만.\n\n"
              f"[관찰된 시각 특징]\n{caption}\n\n[후보]\n{blocks}\n\n답(병명 하나):")
    try:
        r = ollama_generate(prompt, MODEL, {"temperature": 0, "seed": 42, "num_predict": 20}, timeout=120)
    except Exception as e:
        return None, str(e)[:40]
    return extract_label(r), r


# 정확도, macro-F1, 출처별 정확도를 묶어 계산한다
def metrics(golds: list, preds: list, srcs: list) -> dict:
    return {"accuracy": round(accuracy_score(golds, preds), 4), "macro_f1": round(macro_f1(golds, preds), 4),
            "src_acc": source_accuracy(golds, preds, srcs, 3)}


# 기준 대비 오답→정답, 정답→오답 건수와 순이득을 센다
def transitions(golds: list, base: list, final: list) -> dict:
    w2c, c2w = prediction_changes(golds, base, final)
    return {"wrong2correct": len(w2c), "correct2wrong": len(c2w), "net": len(w2c) - len(c2w)}


# 후보 순서를 바꿔 두 번 묻고 두 답이 같을 때만 채택한다
def rerank_rows(caps: dict, s1: dict, guides: dict) -> list:
    rows = []
    for i, (b, c) in enumerate(caps.items()):
        s = s1[b]
        top1, top3 = s["score_top1"], s["score_top3"]
        asc = sorted(top3, key=lambda k: KEY_TO_LABEL[k])
        k1, r1 = ask(c["caption"], asc, guides)
        k2, _ = ask(c["caption"], list(reversed(asc)), guides)
        v1, v2 = k1 in top3, k2 in top3
        agree = (k1 == k2) and v1
        rows.append({"image": b, "gold": s["gold"], "source": image_source(b, s["gold"]),
                     "top1": top1, "top3": top3, "margin": s["margin_1_2"],
                     "pred_asc": k1, "pred_desc": k2, "agree": bool(agree),
                     "valid": bool(v1 and v2), "final": k1 if agree else top1, "raw_asc": r1[:30]})
        if (i + 1) % 15 == 0:
            print(f"  {i+1}/{len(caps)}")
    return rows


# LLM 재순위화를 전체·저마진 조건으로 평가한다
def main():
    caps = read_json(CAPS)
    s1 = {r["image"]: r for r in read_json(STAGE1)["rows"]}
    guides = {k: " ".join(lines) for k, lines in symptom_guides(KBRetriever(device="cpu")).items()}
    rows = rerank_rows(caps, s1, guides)

    golds = [r["gold"] for r in rows]
    srcs = [r["source"] for r in rows]
    base = [r["top1"] for r in rows]
    agree_rate = sum(r["agree"] for r in rows) / len(rows)
    pa = [r["final"] for r in rows]
    pb = [r["final"] if r["margin"] < MARGIN_LIMIT else r["top1"] for r in rows]

    res = {
        "model": MODEL, "n": len(rows),
        "position_agreement": round(agree_rate, 3),
        "baseline": metrics(golds, base, srcs),
        "a_all": {**metrics(golds, pa, srcs), **transitions(golds, base, pa)},
        "b_margin<0.9": {**metrics(golds, pb, srcs), **transitions(golds, base, pb),
                         "n_applied": sum(1 for r in rows if r["margin"] < MARGIN_LIMIT)},
    }
    write_json({"summary": res, "rows": rows}, OUT)

    print("\n" + "=" * 60)
    print(f"  EXP-1 LLM-as-reranker ({MODEL})")
    print("=" * 60)
    print(f"  위치편향 일치율(가나다↔역순): {agree_rate:.3f} ({'양호' if agree_rate >= 0.8 else '편향 심각'})")
    b, a, mb = res["baseline"], res["a_all"], res["b_margin<0.9"]
    print(f"  baseline    F1={b['macro_f1']:.3f} acc={b['accuracy']:.3f}")
    print(f"  (a) 전체    F1={a['macro_f1']:.3f} acc={a['accuracy']:.3f} | 오답→정답 {a['wrong2correct']} "
          f"정답→오답 {a['correct2wrong']} 순이득 {a['net']}")
    print(f"  (b) 마진<0.9 F1={mb['macro_f1']:.3f} acc={mb['accuracy']:.3f} | 적용 {mb['n_applied']} | 순이득 {mb['net']}")
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
