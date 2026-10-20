import os
import glob
import argparse

from peft import PeftModel
from sklearn.metrics import accuracy_score

from backend.core.classes import BALANCED_TEST_JSONL, LOGS_DIR, NORMAL_KEY, RESULTS_DIR
from backend.core.label_extract import extract_label
from backend.core.name_rag_pipeline import NAME_PROMPT
from backend.core.vlm import VLMRunner
from backend.experiments.data_io import read_jsonl, write_json
from backend.experiments.metrics import image_source, macro_f1, source_accuracy
from backend.experiments.paper_model import PAPER_BASE_MODEL, checkpoint_epoch, checkpoint_step, load_qwen35

DONE = os.path.join(LOGS_DIR, "eval_9b_balanced.done")


# 체크포인트를 step 순으로 고르고 지정 에폭만 남긴다
def select_checkpoints(output_dir: str, epoch: int) -> list:
    cks = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=checkpoint_step)
    if epoch:
        cks = [c for c in cks if checkpoint_epoch(c) == epoch]
    return cks


# 현재 어댑터로 테스트셋을 예측해 에폭 지표와 예측 행을 만든다
def evaluate_epoch(runner: VLMRunner, test: list, ep: int):
    prompt = runner.build_prompt(NAME_PROMPT)
    golds = [r["key"] for r in test]
    srcs = [image_source(r["image"], r["key"]) for r in test]
    preds, rows = [], []
    for r in test:
        ans = runner.generate(prompt, r["image"], 32)
        pk = extract_label(ans)
        preds.append(pk)
        rows.append({"key": r["key"], "pred_key": pk, "answer": ans,
                     "image": os.path.basename(r["image"]),
                     "source": image_source(r["image"], r["key"])})
    acc = accuracy_score(golds, preds)
    f1 = macro_f1(golds, preds)
    nidx = [i for i, g in enumerate(golds) if g == NORMAL_KEY]
    nrec = sum(preds[i] == NORMAL_KEY for i in nidx) / len(nidx) if nidx else 0
    didx = [i for i, g in enumerate(golds) if g != NORMAL_KEY]
    dacc = sum(preds[i] == golds[i] for i in didx) / len(didx) if didx else 0
    src_acc = source_accuracy(golds, preds, srcs, 4)
    print(f"  [ep{ep}] F1={f1:.4f} acc={acc:.4f} | 정상 {sum(preds[i] == NORMAL_KEY for i in nidx)}/{len(nidx)} "
          f"| 질병acc={dacc:.3f} | 원본{src_acc['원본']} NCPMS{src_acc['NCPMS']}", flush=True)
    entry = {"epoch": ep, "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
             "normal_recall": round(nrec, 3), "disease_acc": round(dacc, 3), "src_acc": src_acc}
    return entry, rows


# 에폭별 LoRA 어댑터를 평가해 학습 곡선과 최적 에폭을 저장한다
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epoch", type=int, default=0, help="0=전 에폭 스윕")
    args = ap.parse_args()

    cks = select_checkpoints(args.output_dir, args.epoch)
    if not cks:
        raise SystemExit("체크포인트 없음")

    test = read_jsonl(BALANCED_TEST_JSONL)
    print(f"base 로딩: {PAPER_BASE_MODEL} · test {len(test)}장", flush=True)
    base, proc = load_qwen35(PAPER_BASE_MODEL)

    model, epochs = None, []
    for ck in cks:
        ep = checkpoint_epoch(ck)
        if model is None:
            model = PeftModel.from_pretrained(base, ck, adapter_name=f"ep{ep}")
        else:
            model.load_adapter(ck, adapter_name=f"ep{ep}")
        epochs.append(ep)
    model.eval()
    runner = VLMRunner(model, proc, enable_thinking=False)

    curve = []
    for ep in epochs:
        model.set_adapter(f"ep{ep}")
        entry, rows = evaluate_epoch(runner, test, ep)
        curve.append(entry)
        write_json(rows, os.path.join(RESULTS_DIR, f"pred_balanced_9b_ep{ep}.json"), indent=None)

    best = max(curve, key=lambda c: c["macro_f1"])
    write_json({"model": PAPER_BASE_MODEL, "curve": curve, "best": best, "n_test": len(test)},
               os.path.join(RESULTS_DIR, "balanced_curve_9b.json"))
    print(f"\n최적 ep{best['epoch']} · macro-F1={best['macro_f1']:.4f} · acc={best['accuracy']:.4f} · "
          f"정상recall={best['normal_recall']:.2f} · 질병acc={best['disease_acc']:.3f} · "
          f"원본{best['src_acc']['원본']} NCPMS{best['src_acc']['NCPMS']}")
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(DONE, "w") as f:
        f.write(f"best_ep{best['epoch']}_f1{best['macro_f1']:.4f}\n")


if __name__ == "__main__":
    main()
