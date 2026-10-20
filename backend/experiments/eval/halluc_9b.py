import os

from peft import PeftModel

from backend.core.classes import BALANCED_TEST_JSONL, KEY_TO_LABEL, KEY_TO_SECTION, NORMAL_KEY, RESULTS_DIR
from backend.core.kb_retriever import KBRetriever
from backend.core.vlm import VLMRunner
from backend.experiments.data_io import read_jsonl, write_json
from backend.experiments.metrics import SciNameScorer, latin_names
from backend.experiments.paper_model import PAPER_BASE_MODEL, best_epoch_adapter, load_qwen35

OUT = os.path.join(RESULTS_DIR, "p1_2_hallucination_9b.json")
QUESTIONS = [("학명", "이 병의 학명은 무엇인가요?"), ("약제", "이 병에는 어떤 약제를 사용하나요?"),
             ("설명", "이 병에 대해 자세히 설명해주세요.")]
NOTE = ("학명(라틴 이명) 기준. 정상 클래스 제외(병원균 없음→학명 부재). "
        "질병 49장×3질의(학명/약제/설명)=147 factual queries. P1-2 실측 프로토콜.")


# 답변 하나의 학명 후보를 채점해 결과 행으로 만든다
def score_answer(scorer: SciNameScorer, text: str, gold: str) -> dict:
    names = latin_names(text)
    return {"latin": sorted(scorer.scientific_only(names)), "halluc": scorer.classify(names, gold)}


# 직접 생성(A)과 KB 답변(B)의 학명 환각을 기록한다
def collect_rows(runner: VLMRunner, kb: KBRetriever, scorer: SciNameScorer, test: list):
    rows_a, rows_b = [], []
    for i, r in enumerate(test):
        gold, img = r["key"], r["image"]
        if gold == NORMAL_KEY:
            continue
        for tag, q in QUESTIONS:
            a = runner.ask(img, q, 200)
            sa = score_answer(scorer, a, gold)
            rows_a.append({"image": os.path.basename(img), "gold": KEY_TO_LABEL[gold], "q": tag,
                           "answer": a[:250], "latin": sa["latin"], "halluc": sa["halluc"]})
            sb = score_answer(scorer, kb.build_answer(KEY_TO_SECTION[gold], q), gold)
            rows_b.append({"image": os.path.basename(img), "gold": KEY_TO_LABEL[gold], "q": tag,
                           "latin": sb["latin"], "halluc": sb["halluc"]})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}", flush=True)
    return rows_a, rows_b


# 9B 의 사실 질의 학명 환각률을 측정한다
def main():
    ep, adapter = best_epoch_adapter()
    test = read_jsonl(BALANCED_TEST_JSONL)
    kb = KBRetriever(device="cpu")
    scorer = SciNameScorer(kb.kb)

    base, proc = load_qwen35(PAPER_BASE_MODEL)
    runner = VLMRunner(PeftModel.from_pretrained(base, adapter).eval(), proc, enable_thinking=False)
    print(f"9B adapter ep{ep} · 질병 test × 3질의", flush=True)
    rows_a, rows_b = collect_rows(runner, kb, scorer, test)

    halluc = ("fabrication", "misattribution")
    res = {"model": PAPER_BASE_MODEL, "epoch": ep,
           "condition_A_noRAG": SciNameScorer.aggregate(rows_a),
           "condition_B_RAG": SciNameScorer.aggregate(rows_b),
           "note": NOTE,
           "A_halluc_examples": [r for r in rows_a if r["halluc"] in halluc][:8],
           "B_halluc_examples": [r for r in rows_b if r["halluc"] in halluc]}
    write_json({"summary": res, "rowsA": rows_a, "rowsB": rows_b}, OUT)
    a, b = res["condition_A_noRAG"], res["condition_B_RAG"]
    print("=" * 56)
    print("  9B 환각률 (학명 기준)")
    print(f"  조건A(9B직접): 환각 {a['halluc_n']}/{a['n']} = {a['halluc_rate']*100:.1f}%")
    print(f"  조건B(9B+RAG): 환각 {b['halluc_n']}/{b['n']} = {b['halluc_rate']*100:.1f}% "
          f"{'0% 확인' if b['halluc_n'] == 0 else '0아님 → 사례 전부 첨부'}")
    print(f"  저장: {OUT}")


if __name__ == "__main__":
    main()
