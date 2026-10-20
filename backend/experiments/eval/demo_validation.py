import os
import argparse

from peft import PeftModel

from backend.core.classes import KEY_TO_LABEL, MERGED_MODEL_DIR, PLATFORM_BASE_MODEL, RESULTS_DIR, TEST_JSONL
from backend.core.kb_parser import kb_for_classes
from backend.core.label_extract import extract_label
from backend.core.vlm import VLMRunner, load_qwen25
from backend.experiments.data_io import read_jsonl
from backend.experiments.data_prep.build_dataset import EVAL_QUESTION

INTENT_QS = [
    ("진단", EVAL_QUESTION),
    ("병명", "이거 무슨 병이야?"),
    ("방제", "이 병은 어떻게 방제해야 하나요?"),
    ("증상", "이 감귤에 어떤 증상이 보이나요?"),
]


# 병합 모델 또는 기본 모델 + LoRA 로 실행기를 만든다
def load_runner(model_dir: str, adapter: str) -> VLMRunner:
    if not adapter:
        return VLMRunner(*load_qwen25(model_dir))
    model, proc = load_qwen25(PLATFORM_BASE_MODEL)
    return VLMRunner(PeftModel.from_pretrained(model, adapter).merge_and_unload().eval(), proc)


# 플랫폼 테스트셋 전체를 진단하고 KB 연결 여부를 기록한다
def diagnose_all(runner: VLMRunner, test: list, kb: dict) -> list:
    rows = []
    for i, r in enumerate(test, 1):
        ans = runner.ask(r["image"], EVAL_QUESTION, 400)
        pk = extract_label(ans)
        ok = pk == r["key"]
        rows.append({"image": os.path.basename(r["image"]), "gold": r["label_ko"],
                     "pred": KEY_TO_LABEL.get(pk, "불명확"), "ok": ok,
                     "kb_hit": KEY_TO_LABEL.get(pk) in kb, "answer": ans})
        print(f"  [{i:2d}/{len(test)}] {'O' if ok else 'X'} {r['label_ko']} → {KEY_TO_LABEL.get(pk, '불명확')}")
    return rows


# 맞힌 사진 3장에 의도별 질문을 던져 답변을 모은다
def intent_answers(runner: VLMRunner, test: list, rows: list) -> list:
    samples, seen = [], set()
    for r, row in zip(test, rows):
        if row["ok"] and r["label_ko"] not in seen:
            samples.append(r)
            seen.add(r["label_ko"])
        if len(samples) == 3:
            break
    out = []
    for r in samples:
        per = {"image": os.path.basename(r["image"]), "gold": r["label_ko"], "answers": {}}
        for name, q in INTENT_QS:
            a = runner.ask(r["image"], q, 250)
            per["answers"][name] = a
            print(f"  {r['label_ko']:10s} [{name}] {a[:50].replace(chr(10), ' ')}...")
        out.append(per)
    return out


# 진단 결과와 의도별 답변을 마크다운 리포트로 만든다
def build_report(model_name: str, rows: list, intent_out: list) -> str:
    correct = sum(x["ok"] for x in rows)
    kb_ok = sum(x["kb_hit"] for x in rows)
    md = ["# 데모 실제 테스트 결과", "",
          f"- 모델: `{model_name}`",
          f"- 테스트: held-out test **{len(rows)}장**",
          f"- **진단 정확도: {correct}/{len(rows)} = {correct / len(rows):.1%}**",
          f"- 병명→공식 KB 연결 성공: {kb_ok}/{len(rows)}",
          "", "## 1. 전체 진단 결과", "",
          "| # | 이미지 | 정답 | 예측 | 결과 | KB연결 |",
          "|---:|---|---|---|:--:|:--:|"]
    for i, x in enumerate(rows, 1):
        md.append(f"| {i} | `{x['image']}` | {x['gold']} | {x['pred']} | "
                  f"{'O' if x['ok'] else 'X'} | {'O' if x['kb_hit'] else '—'} |")
    md.append("")

    wrong = [x for x in rows if not x["ok"]]
    md += [f"### 오답 {len(wrong)}건", ""]
    md += [f"- `{x['image']}` : **{x['gold']}** → {x['pred']} 오분류" for x in wrong] or ["- 없음"]
    md += ["", "## 2. 의도별 응답 (같은 이미지, 다른 질문)", ""]
    for per in intent_out:
        md += [f"### `{per['image']}` (정답: {per['gold']})", ""]
        for name, q in INTENT_QS:
            md += [f"**[{name}] 질문:** {q}", "", "```", per["answers"][name][:600], "```", ""]
    return "\n".join(md)


# 서빙 모델의 진단 정확도와 의도별 응답을 점검한다
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=MERGED_MODEL_DIR)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    runner = load_runner(args.model_dir, args.adapter)
    kb = kb_for_classes()
    test = read_jsonl(TEST_JSONL)

    print(f"[1/3] test {len(test)}장 진단 테스트...")
    rows = diagnose_all(runner, test, kb)
    print("\n[2/3] 의도별 응답 테스트...")
    intent_out = intent_answers(runner, test, rows)
    print(f"\n[3/3] KB 연결: {sum(x['kb_hit'] for x in rows)}/{len(rows)} 건 성공")

    out = os.path.join(RESULTS_DIR, "demo_test_report.md")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_report(args.adapter or args.model_dir, rows, intent_out))
    correct = sum(x["ok"] for x in rows)
    print(f"\n정확도 {correct / len(test):.1%} ({correct}/{len(test)}) | 리포트 → {out}")


if __name__ == "__main__":
    main()
