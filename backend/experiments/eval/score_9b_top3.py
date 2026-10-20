import os

from peft import PeftModel

from backend.core.classes import BALANCED_TEST_JSONL, RESULTS_DIR
from backend.core.name_rag_pipeline import NAME_PROMPT
from backend.core.vlm import VLMRunner
from backend.experiments.data_io import read_jsonl, write_json
from backend.experiments.paper_model import (PAPER_BASE_MODEL, best_epoch_adapter, load_qwen35,
                                             score_test_set, top3_summary)

OUT = os.path.join(RESULTS_DIR, "b_stage1_top3_9b.json")


# 9B 최적 어댑터로 8개 라벨을 채점해 마진과 top-3 후보를 저장한다
def main():
    ep, adapter = best_epoch_adapter()
    test = read_jsonl(BALANCED_TEST_JSONL)
    print(f"test {len(test)}장 · 9B adapter ep{ep} ({adapter})", flush=True)

    base, proc = load_qwen35(PAPER_BASE_MODEL)
    runner = VLMRunner(PeftModel.from_pretrained(base, adapter).eval(), proc, enable_thinking=False)
    rows = score_test_set(runner, test, runner.build_prompt(NAME_PROMPT))

    summary = {"model": PAPER_BASE_MODEL, "epoch": ep, **top3_summary(rows)}
    write_json({"summary": summary, "rows": rows}, OUT)
    src = summary["top3_recall_by_source"]
    print("\n" + "=" * 60)
    print(f"  9B top-3 스코어링 (ep{ep})")
    print(f"  스코어 top-1 acc : {summary['score_top1_acc']:.4f}")
    print(f"  생성   top-1 acc : {summary['gen_top1_acc']:.4f}")
    print(f"  스코어=생성 일치 : {summary['score_vs_gen_agree']:.4f}")
    print(f"  top-3 recall   : {summary['top3_recall']:.4f} (원본{src['원본']} NCPMS{src['NCPMS']})")
    print(f"  마진 정답케이스: {summary['margin_correct']}")
    print(f"  마진 오답케이스: {summary['margin_wrong']}")
    print(f"  저장: {OUT}")


if __name__ == "__main__":
    main()
