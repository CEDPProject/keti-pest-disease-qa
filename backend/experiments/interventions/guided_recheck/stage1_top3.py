import os

from peft import PeftModel

from backend.core.classes import BALANCED_TEST_JSONL, RESULTS_DIR
from backend.core.name_rag_pipeline import NAME_PROMPT
from backend.core.vlm import VLMRunner
from backend.experiments.data_io import read_jsonl, write_json
from backend.experiments.paper_model import (RECHECK_ADAPTER, RECHECK_BASE_MODEL, load_qwen35, score_test_set,
                                             top3_summary)

OUT = os.path.join(RESULTS_DIR, "b_stage1_top3.json")


# 4B 어댑터로 라벨을 채점해 top-3 후보와 마진을 저장한다
def main():
    test = read_jsonl(BALANCED_TEST_JSONL)
    print(f"test {len(test)}장 · adapter ep4(checkpoint-932)")

    base, proc = load_qwen35(RECHECK_BASE_MODEL)
    runner = VLMRunner(PeftModel.from_pretrained(base, RECHECK_ADAPTER).eval(), proc, enable_thinking=False)
    rows = score_test_set(runner, test, runner.build_prompt(NAME_PROMPT))

    summary = top3_summary(rows)
    write_json({"summary": summary, "rows": rows}, OUT)
    passed = summary["top3_recall"] >= 0.85
    print("\n" + "=" * 60)
    print("  STAGE 1 결과")
    print("=" * 60)
    print(f"  스코어 top-1 acc : {summary['score_top1_acc']:.4f}")
    print(f"  생성   top-1 acc : {summary['gen_top1_acc']:.4f}  (기존 평가 0.772와 비교)")
    print(f"  스코어=생성 일치 : {summary['score_vs_gen_agree']:.4f}")
    print(f"  top-3 recall   : {summary['top3_recall']:.4f}   → 게이트(≥0.85) {'통과' if passed else '미달'}")
    print(f"  출처별 top-3     : {summary['top3_recall_by_source']}")
    print(f"  마진(정답케이스) : {summary['margin_correct']}")
    print(f"  마진(오답케이스) : {summary['margin_wrong']}")
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
