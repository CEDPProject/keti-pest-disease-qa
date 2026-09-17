import os
import sys
import io
import json
import time
import argparse

import ollama
from PIL import Image

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(VLM_FT), "backend", "common"))
from classes import SPLIT_JSON  # noqa
from label_extract import extract_label  # noqa
from metrics import evaluate  # noqa

MODEL = "qwen2.5vl:7b"
from classes import TEST_JSONL  # noqa
RESULT_DIR = os.path.join(VLM_FT, "results")

CANDIDATES = ["궤양병", "더뎅이병", "잿빛곰팡이병", "수지병",
              "검은점무늬병", "녹색곰팡이병", "탄저병"]
INSTRUCTION = (
    "당신은 감귤 병해충 진단 전문가입니다.\n"
    "다음 8가지 후보 중에서 이미지에 가장 가까운 하나로 진단하세요.\n"
    f"후보: {', '.join(CANDIDATES)}\n"
    "반드시 첫 줄에 '진단: <병명>' 형식으로 후보 중 하나를 쓰고, "
    "이어서 관찰된 증상과 방제 방법을 간단히 설명하세요."
)


def img_bytes(path, max_side):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    r = max_side / max(w, h)
    if r < 1:
        im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=90)
    return b.getvalue()


def load_test():
    with open(TEST_JSONL, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def pick_exemplars():
    """클래스별 train 이미지 1장씩 → [(label_ko, path)]"""
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    seen = {}
    for row in split["train"]:
        seen.setdefault(row["label_ko"], row["image_path"])
    return [(lbl, seen[lbl]) for lbl in CANDIDATES if lbl in seen]


def run_zeroshot(test):
    preds = []
    for i, r in enumerate(test, 1):
        t = time.time()
        try:
            resp = ollama.chat(model=MODEL, messages=[
                {"role": "user", "content": INSTRUCTION, "images": [img_bytes(r["image"], 768)]}
            ], options={"temperature": 0, "num_ctx": 4096, "num_predict": 400})
            ans = resp["message"]["content"]
        except Exception as e:
            ans = f"[ERROR] {e}"
        pk = extract_label(ans)
        preds.append({**r, "answer": ans, "pred_key": pk})
        print(f"  [{i:2d}/{len(test)}] gold={r['key']:12s} pred={pk:12s} ({time.time()-t:.1f}s)")
    return preds


def run_fewshot(test, exemplars):
    ex_msgs = []
    for lbl, path in exemplars:
        ex_msgs.append({"role": "user", "content": "이 감귤의 병명은 무엇인가요?",
                        "images": [img_bytes(path, 448)]})
        ex_msgs.append({"role": "assistant", "content": f"진단: {lbl}"})
    preds = []
    for i, r in enumerate(test, 1):
        t = time.time()
        try:
            msgs = ex_msgs + [{
                "role": "user",
                "content": "위 예시들을 참고하여, 이 감귤을 후보 중 하나로 진단하세요.\n" + INSTRUCTION,
                "images": [img_bytes(r["image"], 768)]}]
            resp = ollama.chat(model=MODEL, messages=msgs,
                               options={"temperature": 0, "num_ctx": 8192, "num_predict": 400})
            ans = resp["message"]["content"]
        except Exception as e:
            ans = f"[ERROR] {e}"
        pk = extract_label(ans)
        preds.append({**r, "answer": ans, "pred_key": pk})
        print(f"  [{i:2d}/{len(test)}] gold={r['key']:12s} pred={pk:12s} ({time.time()-t:.1f}s)")
    return preds


def save(preds, name):
    os.makedirs(RESULT_DIR, exist_ok=True)
    p = os.path.join(RESULT_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps({k: v for k, v in r.items() if k != "gold_answer"},
                               ensure_ascii=False) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zeroshot", "fewshot", "both"], default="both")
    args = ap.parse_args()
    test = load_test()
    print(f"테스트 {len(test)}장, 모델 {MODEL}")

    if args.mode in ("zeroshot", "both"):
        print("\n### Zero-shot ###")
        p = save(run_zeroshot(test), "pred_baseline_zeroshot.jsonl")
        evaluate(p, "Baseline Zero-shot (qwen2.5vl:7b)")

    if args.mode in ("fewshot", "both"):
        print("\n### Few-shot (클래스별 예시 1장) ###")
        ex = pick_exemplars()
        print(f"예시 이미지 {len(ex)}장 사용")
        p = save(run_fewshot(test, ex), "pred_baseline_fewshot.jsonl")
        evaluate(p, "Baseline Few-shot (qwen2.5vl:7b)")


if __name__ == "__main__":
    main()
