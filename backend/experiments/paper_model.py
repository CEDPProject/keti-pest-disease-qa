import os
import re

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from backend.core.classes import CLASS_KEYS, KEY_TO_LABEL, MODELS_DIR, PAPER_LORA_DIR, RESULTS_DIR
from backend.core.label_extract import extract_label
from backend.experiments.data_io import read_json
from backend.experiments.metrics import image_source, margin_stats

PAPER_BASE_MODEL = "Qwen/Qwen3.5-9B"
STEPS_PER_EPOCH = 233
CURVE_JSON = os.path.join(RESULTS_DIR, "balanced_curve_9b.json")
RECHECK_BASE_MODEL = "Qwen/Qwen3.5-4B"
RECHECK_ADAPTER = os.path.join(MODELS_DIR, "qwen3_5_balanced_lora", "checkpoint-932")
LABELS = {k: KEY_TO_LABEL[k] for k in CLASS_KEYS}


# Qwen3.5 계열 모델과 프로세서를 bf16 으로 GPU 에 올린다
def load_qwen35(model_id: str):
    model = AutoModelForMultimodalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                     device_map="cuda", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


# 체크포인트 폴더 이름에서 학습 step 수를 읽는다
def checkpoint_step(path: str) -> int:
    return int(re.search(r"checkpoint-(\d+)", path).group(1))


# 체크포인트 step 을 에폭 번호로 바꾼다
def checkpoint_epoch(path: str) -> int:
    return round(checkpoint_step(path) / STEPS_PER_EPOCH)


# 에폭 곡선에서 최적 에폭과 그 어댑터 경로를 찾는다
def best_epoch_adapter():
    ep = read_json(CURVE_JSON)["best"]["epoch"]
    return ep, os.path.join(PAPER_LORA_DIR, f"checkpoint-{ep * STEPS_PER_EPOCH}")


# 테스트 사진마다 8개 라벨을 채점해 결과 행을 만든다
def score_test_set(runner, test: list, prompt: str) -> list:
    rows = []
    for i, r in enumerate(test):
        gold = r["key"]
        sc = runner.score_labels(r["image"], prompt, LABELS)
        ranked = sorted(sc, key=lambda k: -sc[k])
        gen_ans = runner.generate(prompt, r["image"], 32)
        rows.append({
            "image": os.path.basename(r["image"]), "gold": gold,
            "source": image_source(r["image"], gold),
            "score_top1": ranked[0], "score_top3": ranked[:3],
            "gen_top1": extract_label(gen_ans), "gen_ans": gen_ans,
            "margin_1_2": round(sc[ranked[0]] - sc[ranked[1]], 4),
            "scores": {k: round(v, 4) for k, v in sc.items()},
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}", flush=True)
    return rows


# 라벨 채점 결과를 요약한다
def top3_summary(rows: list) -> dict:
    n = len(rows)
    top3_recall = sum(r["gold"] in r["score_top3"] for r in rows) / n
    src = {}
    for s in ("원본", "NCPMS"):
        sub = [r for r in rows if r["source"] == s]
        src[s] = round(sum(r["gold"] in r["score_top3"] for r in sub) / len(sub), 3) if sub else None
    return {
        "n_test": n,
        "score_top1_acc": round(sum(r["score_top1"] == r["gold"] for r in rows) / n, 4),
        "gen_top1_acc": round(sum(r["gen_top1"] == r["gold"] for r in rows) / n, 4),
        "score_vs_gen_agree": round(sum(r["score_top1"] == r["gen_top1"] for r in rows) / n, 4),
        "top3_recall": round(top3_recall, 4),
        "top3_recall_by_source": src,
        "margin_correct": margin_stats(sorted(r["margin_1_2"] for r in rows if r["score_top1"] == r["gold"])),
        "margin_wrong": margin_stats(sorted(r["margin_1_2"] for r in rows if r["score_top1"] != r["gold"])),
        "gate_pass": bool(top3_recall >= 0.85),
    }
