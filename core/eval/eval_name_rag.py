"""새 구조(VLM=병명만 → RAG=KB 정보조회) 검증.

측정
  1) 진단 정확도  — '병명만' 프롬프트로도 기존 0.920 이 유지되는가?
  2) 환각         — 답변 텍스트가 전부 KB 원문인가? (구조상 0이어야 함)
                     특히 KB에 학명이 없는 검은무늬병에 학명이 붙는가?
  3) 이름 해석 분포 — exact / synonym / partial / rag / unknown

Mendeley 교차셋도 함께 돌려, 새 구조가 '틀린 병명'을 어떻게 처리하는지 본다.
"""

import os
import sys
import glob
import json
import random
import argparse

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import accuracy_score
from collections import Counter

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(VLM_FT)
sys.path.insert(0, os.path.join(VLM_FT, "rag"))
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from kb_retriever import KBRetriever  # noqa
from name_rag_pipeline import NameRagPipeline  # noqa
from label_extract import LABEL_TO_KEY  # noqa
from classes import TEST_JSONL  # noqa

from classes import MERGED_MODEL_DIR as MERGED  # noqa
RESULT_DIR = os.path.join(VLM_FT, "results")
from classes import MENDELEY_DIR as DS  # noqa
CROSS = [(os.path.join(DS, "Fruits", "Canker"), "canker", "궤양병", 30),
         (os.path.join(DS, "Fruits", "Scab"), "scab", "더뎅이병", 15),
         (os.path.join(DS, "Leaves", "Melanose"), "black_spot", "검은점무늬병", 13)]


def build_items():
    items = [{"set": "ours", **{k: r[k] for k in ("image", "key", "label_ko")}}
             for r in (json.loads(l) for l in open(TEST_JSONL, encoding="utf-8"))]
    rng = random.Random(42)
    for folder, key, ko, n in CROSS:
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        items += [{"set": "mendeley", "image": f, "key": key, "label_ko": ko} for f in files[:n]]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=MERGED)
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(args.model_dir, max_pixels=768 * 768)
    kb = KBRetriever(device="cuda")
    pipe = NameRagPipeline(model, proc, kb)

    items = build_items()
    print(f"평가: 우리 test {sum(i['set']=='ours' for i in items)}장 + "
          f"Mendeley {sum(i['set']=='mendeley' for i in items)}장\n")

    rows = []
    for i, it in enumerate(items, 1):
        out = pipe.run(it["image"], None)
        pred_key = LABEL_TO_KEY.get(out["disease"], "unknown") if out["disease"] else "unknown"
        # 환각 검사: 답변에 학명이 있는데 KB엔 학명이 없는 경우
        halluc = False
        if out["disease"]:
            k = kb.kb[out["disease"]]
            no_sci = (not k["scientific"]) or "미확정" in k["scientific"]
            halluc = no_sci and ("학명" in out["answer"] and "확정되어 있지 않" not in out["answer"])
        rows.append({**it, "pred_key": pred_key, "disease": out["disease"],
                     "how": out["how"], "halluc": halluc})
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    print(f"\n{'='*66}")
    for s in ["ours", "mendeley"]:
        sub = [r for r in rows if r["set"] == s]
        g = [r["key"] for r in sub]; p = [r["pred_key"] for r in sub]
        unk = sum(1 for r in sub if r["pred_key"] == "unknown")
        print(f"\n[{s}]  N={len(sub)}")
        print(f"  진단 accuracy = {accuracy_score(g,p):.3f}")
        print(f"  불명확(unknown) 처리 = {unk}건")
        print(f"  이름 해석 분포 = {dict(Counter(r['how'] for r in sub))}")
    n_h = sum(r["halluc"] for r in rows)
    print(f"\n  학명 환각 = {n_h}건  (구조상 0이어야 함)")
    print(f"{'='*66}")

    out = os.path.join(RESULT_DIR, "name_rag_eval.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
