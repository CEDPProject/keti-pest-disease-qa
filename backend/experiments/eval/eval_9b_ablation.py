import os

from peft import PeftModel
from sklearn.metrics import accuracy_score

from backend.core.classes import BALANCED_TEST_JSONL, CLASS_KEYS, KEY_TO_LABEL, RESULTS_DIR
from backend.core.label_extract import extract_label
from backend.core.name_rag_pipeline import NAME_PROMPT
from backend.core.vlm import VLMRunner
from backend.experiments.data_io import read_json, read_jsonl, write_json
from backend.experiments.kb_text import clean_caption
from backend.experiments.metrics import image_source, macro_f1, source_accuracy
from backend.experiments.paper_model import PAPER_BASE_MODEL, best_epoch_adapter, load_qwen35

CAPS = os.path.join(RESULTS_DIR, "fusion_captions.json")
OUT = os.path.join(RESULTS_DIR, "ablation_9b.json")
MIN_PROMPT = "병명만 답하라."


# 정확도, macro-F1, 출처별 정확도를 묶어 계산한다
def metrics(golds: list, preds: list, srcs: list) -> dict:
    return {"acc": round(accuracy_score(golds, preds), 4), "macro_f1": round(macro_f1(golds, preds), 4),
            "src": source_accuracy(golds, preds, srcs, 4)}


# 캡션만 보고 병명을 고르게 하는 텍스트 프롬프트를 만든다
def text_prompt(cap: str) -> str:
    names = "、".join(KEY_TO_LABEL[k] for k in CLASS_KEYS)
    return (f"다음은 감귤 사진의 시각적 특징 서술이다. 아래 병명 목록 중 서술과 가장 부합하는 병명 하나만 답하라.\n"
            f"[병명 목록] {names}\n[서술] {cap}\n답(병명 하나):")


# 텍스트만, 이미지만, 이미지 + 질문 세 입력 조건의 성능을 잰다
def run_conditions(runner: VLMRunner, test: list, caps: dict, tag: str) -> dict:
    golds = [r["key"] for r in test]
    srcs = [image_source(r["image"], r["key"]) for r in test]
    out = {}
    ga, pa, sa = [], [], []
    for r in test:
        b = os.path.basename(r["image"])
        if b not in caps:
            continue
        pk = extract_label(runner.ask(None, text_prompt(clean_caption(caps[b]["caption"])), 32))
        ga.append(r["key"])
        pa.append(pk)
        sa.append(image_source(r["image"], r["key"]))
    out["a_text_only"] = metrics(ga, pa, sa) | {"n": len(ga)}
    pb = [extract_label(runner.ask(r["image"], MIN_PROMPT, 32)) for r in test]
    out["b_image_only"] = metrics(golds, pb, srcs) | {"n": len(test)}
    pc = [extract_label(runner.ask(r["image"], NAME_PROMPT, 32)) for r in test]
    out["c_image_text"] = metrics(golds, pc, srcs) | {"n": len(test)}
    print(f"  [{tag}] a(텍스트)={out['a_text_only']['macro_f1']} "
          f"b(이미지)={out['b_image_only']['macro_f1']} c(이미지+텍스트)={out['c_image_text']['macro_f1']}", flush=True)
    return out


# zero-shot 과 파인튜닝 모델에서 입력 조건별 성능을 비교해 저장한다
def main():
    ep, adapter = best_epoch_adapter()
    test = read_jsonl(BALANCED_TEST_JSONL)
    caps = read_json(CAPS)

    base, proc = load_qwen35(PAPER_BASE_MODEL)
    base.eval()
    print("zero-shot(base) 조건 실행", flush=True)
    zs = run_conditions(VLMRunner(base, proc, enable_thinking=False), test, caps, "zero-shot")

    model = PeftModel.from_pretrained(base, adapter).eval()
    print(f"fine-tuned(ep{ep}) 조건 실행", flush=True)
    ft = run_conditions(VLMRunner(model, proc, enable_thinking=False), test, caps, f"fine-tuned ep{ep}")

    write_json({"model": PAPER_BASE_MODEL, "epoch": ep, "zero_shot": zs, "fine_tuned": ft}, OUT)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
