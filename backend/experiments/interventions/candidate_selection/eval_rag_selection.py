import os
import gc
import argparse

import torch
from sklearn.metrics import accuracy_score

from backend.core.classes import LABEL_TO_KEY, MERGED_MODEL_DIR, PLATFORM_BASE_MODEL, RESULTS_DIR
from backend.core.label_extract import extract_label
from backend.core.vlm import VLMRunner, load_qwen25
from backend.experiments.data_io import cross_eval_items, read_json, write_json
from backend.experiments.data_prep.build_dataset import EVAL_QUESTION
from backend.experiments.interventions.candidate_selection.kb_index import KBIndex
from backend.experiments.interventions.candidate_selection.vlm_rag_pipeline import (
    AMBIGUOUS_MARGIN, DIFFERENTIAL_PROMPT, KB_TO_KO, KO_TO_KB, OBSERVE_PROMPT, extract_candidates,
    strip_disease_names)

CACHE = os.path.join(RESULTS_DIR, "rag_selection_cache.json")
OUT = os.path.join(RESULTS_DIR, "rag_selection.json")


# 진행 상황을 20장마다 출력한다
def _progress(i: int, n: int):
    if i % 20 == 0 or i == n:
        print(f"  {i}/{n}")


# 모델 하나로 사진별 답변을 생성하고 GPU 메모리를 비운다
def generate_all(model_path: str, items: list, prompts: list) -> list:
    model, proc = load_qwen25(model_path)
    runner = VLMRunner(model, proc)
    outs = [[] for _ in prompts]
    for i, it in enumerate(items, 1):
        for out, (prompt, max_new) in zip(outs, prompts):
            out.append(runner.ask(it["image"], prompt, max_new))
        _progress(i, len(items))
    del runner, model, proc
    gc.collect()
    torch.cuda.empty_cache()
    return outs


# 관찰문으로 후보 병해 순위와 확신도를 정한다
def rank_candidates(idx: KBIndex, obs: str, diff: str):
    cands = extract_candidates(diff)
    obs_clean = strip_disease_names(obs)
    query = obs_clean if len(obs_clean) > 15 else obs
    rag_final, conf, ranking = None, "no_candidate", []
    if cands:
        restrict = [KO_TO_KB[c] for c in cands if c in KO_TO_KB]
        if restrict:
            ranking = [(KB_TO_KO.get(n, n), round(s, 4)) for n, s in idx.search(query, top_k=None, restrict=restrict)]
            rag_final = ranking[0][0]
            conf = ("single_candidate" if len(ranking) == 1
                    else ("ambiguous" if ranking[0][1] - ranking[1][1] < AMBIGUOUS_MARGIN else "high"))
    return cands, rag_final, conf, ranking


# 세트별로 VLM 단독과 RAG 선택의 정확도를 비교 출력한다
def print_comparison(rows: list):
    print(f"\n{'=' * 70}")
    for setname in ["ours", "mendeley"]:
        sub = [r for r in rows if r["set"] == setname]
        g = [r["key"] for r in sub]
        n_cand = sum(len(r["candidates"]) > 1 for r in sub)
        print(f"\n[{setname}]  N={len(sub)}  (후보 2개 이상 제시: {n_cand}건)")
        print(f"  VLM 단독      accuracy = {accuracy_score(g, [r['vlm_only'] for r in sub]):.3f}")
        print(f"  VLM + RAG선택 accuracy = {accuracy_score(g, [r['rag_final_key'] for r in sub]):.3f}")
        changed = [r for r in sub if r["vlm_only"] != r["rag_final_key"]]
        fixed = sum(1 for r in sub if r["vlm_only"] != r["key"] and r["rag_final_key"] == r["key"])
        broke = sum(1 for r in sub if r["vlm_only"] == r["key"] and r["rag_final_key"] != r["key"])
        print(f"  RAG가 예측을 바꾼 건수: {len(changed)}  (오답→정답 {fixed}건 / 정답→오답 {broke}건)")
    print(f"\n{'=' * 70}")


# RAG 후보 선택(설계 A)을 평가한다
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    items = cross_eval_items()
    print(f"평가 대상: 우리 test {sum(i['set'] == 'ours' for i in items)}장 + "
          f"Mendeley {sum(i['set'] == 'mendeley' for i in items)}장 = {len(items)}장\n")
    cache = read_json(CACHE) if (os.path.exists(CACHE) and not args.force) else {}

    if "obs" not in cache:
        print("[Stage 1] base 모델 로딩 → 중립 관찰문 생성")
        cache["obs"], = generate_all(PLATFORM_BASE_MODEL, items, [(OBSERVE_PROMPT, 170)])
        write_json(cache, CACHE, indent=None)
    else:
        print("[Stage 1] 캐시 재사용")

    if "diag" not in cache:
        print("\n[Stage 2] 파인튜닝 모델 로딩 → 단독 진단 + 후보 감별")
        cache["diag"], cache["diff"] = generate_all(MERGED_MODEL_DIR, items,
                                                    [(EVAL_QUESTION, 200), (DIFFERENTIAL_PROMPT, 80)])
        write_json(cache, CACHE, indent=None)
    else:
        print("[Stage 2] 캐시 재사용")

    print("\n[Stage 3] E5 인덱스 로딩 → 관찰문으로 후보 랭킹")
    idx = KBIndex(device="cuda")
    rows = []
    for it, obs, diag, diff in zip(items, cache["obs"], cache["diag"], cache["diff"]):
        cands, rag_final, conf, ranking = rank_candidates(idx, obs, diff)
        rows.append({**it, "vlm_only": extract_label(diag),
                     "candidates": cands, "rag_final": rag_final,
                     "rag_final_key": LABEL_TO_KEY.get(rag_final) if rag_final else "unknown",
                     "confidence": conf, "ranking": ranking, "observation": obs[:300]})

    print_comparison(rows)
    write_json(rows, OUT)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
