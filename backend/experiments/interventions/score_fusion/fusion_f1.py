import os

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.classes import CLASS_KEYS, KEY_TO_LABEL, KEY_TO_SECTION, NORMAL_KEY, RESULTS_DIR
from backend.core.kb_parser import parse_kb
from backend.core.name_rag_pipeline import NAME_PROMPT
from backend.experiments.data_io import read_json, write_json
from backend.experiments.kb_text import NORMAL_PASSAGE, clean_caption
from backend.experiments.metrics import softmax

E5 = "intfloat/multilingual-e5-large"
CAPS = os.path.join(RESULTS_DIR, "fusion_captions.json")
STAGE1 = os.path.join(RESULTS_DIR, "b_stage1_top3.json")
OUT = os.path.join(RESULTS_DIR, "fusion_f1.json")


# 클래스별 병징 문장을 E5 passage 로 임베딩한다
def passage_embeddings(model: SentenceTransformer, kb: dict) -> dict:
    passages = {k: NORMAL_PASSAGE if k == NORMAL_KEY else " ".join(kb[KEY_TO_SECTION[k]]["symptoms"])
                for k in CLASS_KEYS}
    return {k: model.encode([f"passage: {passages[k]}"], normalize_embeddings=True)[0] for k in CLASS_KEYS}


# 캡션과 top-3 후보 병징의 유사도와 정답 순위를 기록한다
def caption_rows(model, caps: dict, s1: dict, pass_emb: dict):
    rows, stds = [], []
    for b, c in caps.items():
        s = s1[b]
        top3, gold = s["score_top3"], s["gold"]
        cap = clean_caption(c["caption"])
        cap_emb = model.encode([f"query: {cap}"], normalize_embeddings=True)[0]
        rag_v1 = {k: float(cap_emb @ pass_emb[k]) for k in CLASS_KEYS}
        v1_p = dict(zip(top3, softmax([rag_v1[k] for k in top3])))
        stds.append(float(np.std(list(v1_p.values()))))
        rank_gold = sorted(top3, key=lambda k: -rag_v1[k]).index(gold) + 1 if gold in top3 else None
        rows.append({
            "image": b, "gold": gold, "top1": s["score_top1"], "top3": top3,
            "margin": s["margin_1_2"], "caption_clean": cap[:300],
            "rag_v1_top3": {k: round(rag_v1[k], 4) for k in top3},
            "rag_v1_softmax": {k: round(v1_p[k], 4) for k in top3},
            "rag_v1_std": round(stds[-1], 4),
            "gold_rag_v1_rank": rank_gold,
        })
    return rows, stds


# 캡션-병징 유사도로 점수 융합 게이트를 진단한다
def main():
    caps = read_json(CAPS)
    s1 = {r["image"]: r for r in read_json(STAGE1)["rows"]}
    model = SentenceTransformer(E5, device="cpu")
    pass_emb = passage_embeddings(model, parse_kb())

    q_emb = model.encode([f"query: {NAME_PROMPT}"], normalize_embeddings=True)[0]
    rag_v2 = {k: float(q_emb @ pass_emb[k]) for k in CLASS_KEYS}
    rows, stds = caption_rows(model, caps, s1, pass_emb)

    n = len(rows)
    std_arr = sorted(stds)
    std_stat = {"min": round(std_arr[0], 4), "median": round(std_arr[n // 2], 4),
                "max": round(std_arr[-1], 4), "mean": round(float(np.mean(std_arr)), 4)}
    wrong = [r for r in rows if r["top1"] != r["gold"] and r["gold"] in r["top3"]]
    gold_rank1 = sum(1 for r in wrong if r["gold_rag_v1_rank"] == 1)
    all_in = [r for r in rows if r["gold"] in r["top3"]]
    all_rank1 = sum(1 for r in all_in if r["gold_rag_v1_rank"] == 1)

    summary = {
        "n": n,
        "v1_top3_softmax_std": std_stat,
        "gate_std_near_zero": bool(std_stat["median"] < 0.02),
        "wrong_n_gold_in_top3": len(wrong),
        "wrong_gold_rag_v1_rank1": gold_rank1,
        "wrong_gold_rank1_ratio": round(gold_rank1 / len(wrong), 3) if wrong else None,
        "all_gold_in_top3": len(all_in),
        "all_gold_rank1_ratio": round(all_rank1 / len(all_in), 3) if all_in else None,
        "chance": 0.333,
        "rag_v2_global": {KEY_TO_LABEL[k]: round(v, 4) for k, v in rag_v2.items()},
    }
    write_json({"summary": summary, "rows": rows}, OUT)

    print("=" * 62)
    print("  STAGE F1 게이트 진단")
    print("=" * 62)
    print(f"  V1 top-3 softmax 표준편차: {std_stat}")
    print(f"    → 중앙값 {std_stat['median']} ({'≈0 수렴, 변별력 없음' if std_stat['median'] < 0.02 else '산포 있음'})")
    print(f"  오답 {len(wrong)}건(정답 top-3 내)에서 정답후보 RAG_V1 1위: "
          f"{gold_rank1}/{len(wrong)} = {summary['wrong_gold_rank1_ratio']} (우연 0.333)")
    print(f"  전체 {len(all_in)}건 정답 RAG_V1 1위 비율: {summary['all_gold_rank1_ratio']}")
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
