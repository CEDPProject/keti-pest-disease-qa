import os

from backend.core.classes import LABEL_TO_KEY, RESULTS_DIR
from backend.core.kb_parser import parse_kb
from backend.experiments.data_io import read_json, write_json
from backend.experiments.metrics import SciNameScorer

IN = os.path.join(RESULTS_DIR, "p1_2_hallucination_9b.json")
NOTE = "영명(Scab diseases/Green mold 등) 오인 제거 후 재채점. 정규식이 영명을 학명으로 잡던 버그 수정(4B p1_2_rescore와 동일)."


# 저장된 환각 결과를 영명을 뺀 학명 기준으로 다시 채점해 덮어쓴다
def main():
    d = read_json(IN)
    scorer = SciNameScorer(parse_kb())
    for key in ("rowsA", "rowsB"):
        for r in d[key]:
            r["latin_all"] = r.get("latin", [])
            r["latin"] = sorted(scorer.scientific_only(set(r["latin_all"])))
            r["halluc"] = scorer.classify(set(r["latin_all"]), LABEL_TO_KEY[r["gold"]])

    halluc = ("fabrication", "misattribution")
    s = d["summary"]
    s["condition_A_noRAG"] = SciNameScorer.aggregate(d["rowsA"])
    s["condition_B_RAG"] = SciNameScorer.aggregate(d["rowsB"])
    s["A_halluc_examples"] = [{"image": r["image"], "gold": r["gold"], "q": r["q"], "halluc": r["halluc"],
                               "latin": r["latin"], "answer": r.get("answer", "")[:180]}
                              for r in d["rowsA"] if r["halluc"] in halluc][:8]
    s["B_halluc_examples"] = [r for r in d["rowsB"] if r["halluc"] in halluc]
    s["rescore_note"] = NOTE
    write_json(d, IN)

    a, b = s["condition_A_noRAG"], s["condition_B_RAG"]
    print("=" * 56)
    print("  9B 환각 재채점 (영명 제외, 학명만)")
    print(f"  조건A(9B직접): 환각 {a['halluc_n']}/{a['n']} = {a['halluc_rate']*100:.1f}% "
          f"(fab{a['fabrication']}+mis{a['misattribution']}, 정답{a['correct']}, 미언급{a['none']})")
    print(f"  조건B(9B+RAG): 환각 {b['halluc_n']}/{b['n']} = {b['halluc_rate']*100:.1f}% "
          f"{'구조상 0 확인' if b['halluc_n'] == 0 else '0아님 → 사례 첨부'}")
    if b["halluc_n"] > 0:
        for r in s["B_halluc_examples"]:
            print("   ", r)


if __name__ == "__main__":
    main()
