import os
import argparse
from collections import Counter

from sklearn.metrics import accuracy_score

from backend.core.classes import MERGED_MODEL_DIR, RESULTS_DIR
from backend.core.kb_retriever import KBRetriever
from backend.core.name_rag_pipeline import NameRagPipeline
from backend.core.vlm import VLMRunner, load_qwen25
from backend.experiments.data_io import cross_eval_items, write_json


# KB 에 확정 학명이 없는데 답변이 학명을 말하면 환각으로 본다
def is_sci_hallucination(kb: KBRetriever, out: dict) -> bool:
    if not out["disease"]:
        return False
    k = kb.kb[out["disease"]]
    no_sci = (not k["scientific"]) or "미확정" in k["scientific"]
    return no_sci and ("학명" in out["answer"] and "확정되어 있지 않" not in out["answer"])


# 세트별 진단 정확도, 미해결 건수, 이름 해석 분포를 출력한다
def print_summary(rows: list):
    print(f"\n{'=' * 66}")
    for s in ["ours", "mendeley"]:
        sub = [r for r in rows if r["set"] == s]
        g = [r["key"] for r in sub]
        p = [r["pred_key"] for r in sub]
        unk = sum(1 for r in sub if r["pred_key"] == "unknown")
        print(f"\n[{s}]  N={len(sub)}")
        print(f"  진단 accuracy = {accuracy_score(g, p):.3f}")
        print(f"  불명확(unknown) 처리 = {unk}건")
        print(f"  이름 해석 분포 = {dict(Counter(r['how'] for r in sub))}")
    print(f"\n  학명 환각 = {sum(r['halluc'] for r in rows)}건  (구조상 0이어야 함)")
    print(f"{'=' * 66}")


# 병명 판정 + KB 조회 파이프라인을 평가한다
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=MERGED_MODEL_DIR)
    args = ap.parse_args()

    model, proc = load_qwen25(args.model_dir)
    kb = KBRetriever(device="cuda")
    pipe = NameRagPipeline(VLMRunner(model, proc), kb)

    items = cross_eval_items()
    print(f"평가: 우리 test {sum(i['set'] == 'ours' for i in items)}장 + "
          f"Mendeley {sum(i['set'] == 'mendeley' for i in items)}장\n")

    rows = []
    for i, it in enumerate(items, 1):
        out = pipe.run(it["image"], None)
        rows.append({**it, "pred_key": out.get("class_key", "unknown"), "disease": out["disease"],
                     "how": out["how"], "halluc": is_sci_hallucination(kb, out)})
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    print_summary(rows)
    out = os.path.join(RESULTS_DIR, "name_rag_eval.json")
    write_json(rows, out)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
