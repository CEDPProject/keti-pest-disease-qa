"""VLM-only vs VLM+RAG 선택 비교 — RAG가 실제로 도움이 되는가?

3단계(메모리 절약 위해 모델을 순차 로드/해제):
  Stage 1 (base 모델)   : 병명 없는 시각 관찰문 생성          → obs.json
  Stage 2 (파인튜닝 모델) : (a) 단독 진단  (b) 후보 감별 나열   → diag.json
  Stage 3 (E5)         : 관찰문으로 후보들을 RAG 랭킹 → 최종  → 비교

평가 대상
  - 우리 test 25장 (동일 분포)
  - Mendeley 교차 데이터셋 (다른 출처) — VLM 단독은 0.155

핵심 질문: 파인튜닝 모델의 '이미지 스타일 의존'을, 중립 관찰문 + KB 대조로 보정할 수 있는가?
"""

import os
import sys
import gc
import glob
import json
import random
import argparse

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import accuracy_score

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(VLM_FT)
sys.path.insert(0, os.path.join(os.path.dirname(VLM_FT), "backend", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(VLM_FT), "backend", "rag"))
sys.path.insert(0, os.path.join(os.path.dirname(VLM_FT), "backend", "rag", "_archive_design_a"))
from label_extract import extract_label, LABEL_TO_KEY  # noqa
from classes import TEST_JSONL  # noqa
from vlm_rag_pipeline import (OBSERVE_PROMPT, DIFFERENTIAL_PROMPT,  # noqa
                              extract_candidates, _strip_disease_names,
                              KO_TO_KB, KB_TO_KO, AMBIGUOUS_MARGIN)

BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
from classes import MERGED_MODEL_DIR as MERGED  # noqa
RESULT_DIR = os.path.join(VLM_FT, "results")
CACHE = os.path.join(RESULT_DIR, "rag_selection_cache.json")
DIAG_Q = "이 감귤 사진을 보고 어떤 병해충인지 진단하고, 관찰되는 증상과 방제 방법을 알려주세요."
KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}

from classes import MENDELEY_DIR as DS  # noqa
CROSS = [(os.path.join(DS, "Fruits", "Canker"), "canker", "궤양병", 30),
         (os.path.join(DS, "Fruits", "Scab"), "scab", "더뎅이병", 15),
         (os.path.join(DS, "Leaves", "Melanose"), "black_spot", "검은점무늬병", 13)]


def build_items():
    items = []
    for r in (json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")):
        items.append({"set": "ours", "image": r["image"], "key": r["key"], "label_ko": r["label_ko"]})
    rng = random.Random(42)
    for folder, key, ko, n in CROSS:
        files = sorted(f for f in glob.glob(os.path.join(folder, "*"))
                       if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        for f in files[:n]:
            items.append({"set": "mendeley", "image": f, "key": key, "label_ko": ko})
    return items


def load(path):
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    p = AutoProcessor.from_pretrained(path, max_pixels=768 * 768)
    return m, p


def gen(model, proc, image, prompt, max_new=180):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image, "max_pixels": 768 * 768},
        {"type": "text", "text": prompt}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img = Image.open(image).convert("RGB")
    inp = proc(text=[txt], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)

    items = build_items()
    print(f"평가 대상: 우리 test {sum(i['set']=='ours' for i in items)}장 + "
          f"Mendeley {sum(i['set']=='mendeley' for i in items)}장 = {len(items)}장\n")

    cache = json.load(open(CACHE, encoding="utf-8")) if (os.path.exists(CACHE) and not args.force) else {}

    # ---- Stage 1: base 모델로 중립 관찰문 ----
    if "obs" not in cache:
        print("[Stage 1] base 모델 로딩 → 중립 관찰문 생성")
        m, p = load(BASE)
        obs = []
        for i, it in enumerate(items, 1):
            o = gen(m, p, it["image"], OBSERVE_PROMPT, max_new=170)
            obs.append(o)
            if i % 20 == 0 or i == len(items):
                print(f"  {i}/{len(items)}")
        free(m, p)
        cache["obs"] = obs
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    else:
        print("[Stage 1] 캐시 재사용")

    # ---- Stage 2: 파인튜닝 모델로 단독진단 + 후보감별 ----
    if "diag" not in cache:
        print("\n[Stage 2] 파인튜닝 모델 로딩 → 단독 진단 + 후보 감별")
        m, p = load(MERGED)
        diag, diff = [], []
        for i, it in enumerate(items, 1):
            diag.append(gen(m, p, it["image"], DIAG_Q, max_new=200))
            diff.append(gen(m, p, it["image"], DIFFERENTIAL_PROMPT, max_new=80))
            if i % 20 == 0 or i == len(items):
                print(f"  {i}/{len(items)}")
        free(m, p)
        cache["diag"] = diag
        cache["diff"] = diff
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    else:
        print("[Stage 2] 캐시 재사용")

    # ---- Stage 3: RAG 랭킹 ----
    print("\n[Stage 3] E5 인덱스 로딩 → 관찰문으로 후보 랭킹")
    from kb_index import KBIndex
    idx = KBIndex(device="cuda")

    rows = []
    for it, obs, diag, diff in zip(items, cache["obs"], cache["diag"], cache["diff"]):
        vlm_only = extract_label(diag)
        cands = extract_candidates(diff)
        obs_clean = _strip_disease_names(obs)
        query = obs_clean if len(obs_clean) > 15 else obs

        rag_final, conf, ranking = None, "no_candidate", []
        if cands:
            restrict = [KO_TO_KB[c] for c in cands if c in KO_TO_KB]
            if restrict:
                ranked = idx.search(query, top_k=None, restrict=restrict)
                ranking = [(KB_TO_KO.get(n, n), round(s, 4)) for n, s in ranked]
                rag_final = ranking[0][0]
                conf = ("single_candidate" if len(ranking) == 1
                        else ("ambiguous" if ranking[0][1] - ranking[1][1] < AMBIGUOUS_MARGIN else "high"))
        rows.append({**it, "vlm_only": vlm_only,
                     "candidates": cands, "rag_final": rag_final,
                     "rag_final_key": LABEL_TO_KEY.get(rag_final) if rag_final else "unknown",
                     "confidence": conf, "ranking": ranking, "observation": obs[:300]})

    # ---- 비교 ----
    print(f"\n{'='*70}")
    for setname in ["ours", "mendeley"]:
        sub = [r for r in rows if r["set"] == setname]
        g = [r["key"] for r in sub]
        a = [r["vlm_only"] for r in sub]
        b = [r["rag_final_key"] for r in sub]
        n_cand = sum(len(r["candidates"]) > 1 for r in sub)
        print(f"\n[{setname}]  N={len(sub)}  (후보 2개 이상 제시: {n_cand}건)")
        print(f"  VLM 단독      accuracy = {accuracy_score(g,a):.3f}")
        print(f"  VLM + RAG선택 accuracy = {accuracy_score(g,b):.3f}")
        # RAG가 바꾼 사례
        changed = [(r["label_ko"], KEY_TO_LABEL.get(r["vlm_only"], "?"), r["rag_final"])
                   for r in sub if r["vlm_only"] != r["rag_final_key"]]
        fixed = sum(1 for r in sub if r["vlm_only"] != r["key"] and r["rag_final_key"] == r["key"])
        broke = sum(1 for r in sub if r["vlm_only"] == r["key"] and r["rag_final_key"] != r["key"])
        print(f"  RAG가 예측을 바꾼 건수: {len(changed)}  (오답→정답 {fixed}건 / 정답→오답 {broke}건)")
    print(f"\n{'='*70}")

    out = os.path.join(RESULT_DIR, "rag_selection.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
