import os
import random

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score

from backend.core.classes import BALANCED_TEST_JSONL, KEY_TO_LABEL, NORMAL_KEY, RESULTS_DIR
from backend.core.kb_retriever import KBRetriever
from backend.core.label_extract import extract_label
from backend.core.vlm import VLMRunner
from backend.experiments.data_io import read_json, read_jsonl, write_json
from backend.experiments.kb_text import symptom_guides
from backend.experiments.metrics import macro_f1, prediction_changes, source_accuracy
from backend.experiments.paper_model import RECHECK_ADAPTER, RECHECK_BASE_MODEL, load_qwen35

STAGE1 = os.path.join(RESULTS_DIR, "b_stage1_top3.json")
OUT = os.path.join(RESULTS_DIR, "b_stage3_recheck.json")


# 후보 3개의 병징 발췌를 붙인 2차 질문 프롬프트를 만든다
def second_prompt(runner: VLMRunner, top3_keys: list, guides: dict):
    order = sorted(top3_keys, key=lambda k: KEY_TO_LABEL[k])
    blocks = [f"[{KEY_TO_LABEL[k]}]\n{' '.join(guides[k])}" for k in order]
    text = ("다음 세 후보 중 하나다. 각 후보의 병징 발췌를 참고하여 "
            "사진을 다시 관찰하고 병명 하나만 답하라.\n\n" + "\n\n".join(blocks))
    return runner.build_prompt(text), order[0]


# 정확도, macro-F1, 출처별 정확도를 묶어 계산한다
def metrics(golds: list, preds: list, srcs: list) -> dict:
    return {"accuracy": round(accuracy_score(golds, preds), 4), "macro_f1": round(macro_f1(golds, preds), 4),
            "src_acc": source_accuracy(golds, preds, srcs, 3)}


# 기준 대비 예측 변화 건수와 정상→질병 뒤집힘 수를 센다
def transitions(golds: list, base: list, final: list) -> dict:
    w2c, c2w = prediction_changes(golds, base, final)
    normal_flip = [i for i in range(len(golds)) if golds[i] == NORMAL_KEY
                   and base[i] == NORMAL_KEY and final[i] != NORMAL_KEY]
    return {"wrong2correct": len(w2c), "correct2wrong": len(c2w),
            "net": len(w2c) - len(c2w), "normal_to_disease": len(normal_flip),
            "w2c_idx": w2c, "c2w_idx": c2w}


# 사진마다 후보 발췌를 보여 주고 다시 물어 2차 답을 기록한다
def recheck_rows(runner: VLMRunner, test: list, s1: dict, guides: dict) -> list:
    rows = []
    for i, r in enumerate(test):
        img_b = os.path.basename(r["image"])
        s = s1[img_b]
        top1, top3 = s["score_top1"], s["score_top3"]
        prompt, first_cand = second_prompt(runner, top3, guides)
        raw2 = runner.generate(prompt, r["image"], 32)
        key2 = extract_label(raw2)
        valid = key2 in top3
        rows.append({
            "image": img_b, "gold": s["gold"], "source": s["source"],
            "top1": top1, "top3": top3, "margin": s["margin_1_2"],
            "second_raw": raw2, "second_key": key2, "valid": valid,
            "second_final": key2 if valid else top1, "first_cand": first_cand,
            "hit_first": key2 == first_cand,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}")
    return rows


# 마진 임계값별 2차 답 적용 결과를 표로 만든다
def margin_table(rows: list, golds: list, srcs: list, base_pred: list, wrong_idx: list) -> list:
    table = []
    for t in (0.7, 0.9, 1.1):
        applied = [i for i in range(len(rows)) if rows[i]["margin"] < t]
        pb = [rows[i]["second_final"] if rows[i]["margin"] < t else base_pred[i] for i in range(len(rows))]
        m = metrics(golds, pb, srcs)
        table.append({"threshold": t, "n_applied": len(applied),
                      "wrong_covered": sum(1 for i in applied if i in wrong_idx),
                      "correct_at_risk": sum(1 for i in applied if i not in wrong_idx),
                      "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                      **transitions(golds, base_pred, pb), "src_acc": m["src_acc"]})
    return table


# 유도 재확인을 전체·마진·오라클 조건으로 평가한다
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    test = read_jsonl(BALANCED_TEST_JSONL)
    s1 = {r["image"]: r for r in read_json(STAGE1)["rows"]}
    guides = symptom_guides(KBRetriever(device="cpu"))

    base, proc = load_qwen35(RECHECK_BASE_MODEL)
    runner = VLMRunner(PeftModel.from_pretrained(base, RECHECK_ADAPTER).eval(), proc, enable_thinking=False)
    rows = recheck_rows(runner, test, s1, guides)

    golds = [r["gold"] for r in rows]
    srcs = [r["source"] for r in rows]
    base_pred = [r["top1"] for r in rows]
    wrong_idx = [i for i in range(len(rows)) if base_pred[i] != golds[i]]

    baseline = metrics(golds, base_pred, srcs)
    pa = [r["second_final"] for r in rows]
    a_all = {**metrics(golds, pa, srcs), **transitions(golds, base_pred, pa)}
    pc = [rows[i]["second_final"] if i in wrong_idx else base_pred[i] for i in range(len(rows))]
    c_oracle = {**metrics(golds, pc, srcs), **transitions(golds, base_pred, pc)}
    b_table = margin_table(rows, golds, srcs, base_pred, wrong_idx)
    best_b = max(b_table, key=lambda x: x["net"])

    hit_first = sum(r["hit_first"] for r in rows) / len(rows)
    weak = [{"image": r["image"], "gold": KEY_TO_LABEL[r["gold"]], "top1": KEY_TO_LABEL[r["top1"]],
             "second": KEY_TO_LABEL.get(r["second_key"], r["second_key"]), "valid": r["valid"],
             "correct": r["second_final"] == r["gold"]}
            for r in rows if r["gold"] == "gray_mold"]

    summary = {
        "baseline": baseline,
        "a_all": a_all,
        "b_margin_table": b_table,
        "b_best_threshold": best_b["threshold"],
        "c_oracle": c_oracle,
        "position_bias_hit_first": round(hit_first, 3),
        "position_bias_expected": round(1 / 3, 3),
        "graymold_cases": weak,
    }
    write_json({"summary": summary, "rows": rows}, OUT)

    print("\n" + "=" * 64)
    print("  STAGE 3 결과 요약")
    print("=" * 64)
    b, a, c = baseline, a_all, c_oracle
    print(f"  baseline    : F1={b['macro_f1']:.3f} acc={b['accuracy']:.3f} "
          f"원본{b['src_acc']['원본']} NCPMS{b['src_acc']['NCPMS']}")
    print(f"  (a) 전체적용: F1={a['macro_f1']:.3f} acc={a['accuracy']:.3f} | 오답→정답 {a['wrong2correct']} "
          f"정답→오답 {a['correct2wrong']} 순이득 {a['net']} | 정상→병해 {a['normal_to_disease']}")
    print(f"  (c) oracle  : F1={c['macro_f1']:.3f} acc={c['accuracy']:.3f} | 오답→정답 {c['wrong2correct']} "
          f"정답→오답 {c['correct2wrong']} 순이득 {c['net']}")
    print("  (b) 마진 임계 트레이드오프:")
    for r in b_table:
        print(f"     t={r['threshold']}: 적용{r['n_applied']}장(오답{r['wrong_covered']}+정답{r['correct_at_risk']}) "
              f"→ F1={r['macro_f1']:.3f} 순이득{r['net']}")
    print(f"  위치편향(가나다 첫후보 선택율): {hit_first:.3f} (기대 0.333)")
    print(f"  잿빛곰팡이 정답 케이스 2차 성공: {sum(w['correct'] for w in weak)}/{len(weak)}")
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
