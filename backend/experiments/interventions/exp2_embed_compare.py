import os
import itertools

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.classes import DISEASE_KEYS, KEY_TO_SECTION, NORMAL_KEY, RESULTS_DIR
from backend.core.kb_parser import parse_kb
from backend.core.kb_retriever import KBRetriever
from backend.experiments.data_io import read_json, write_json
from backend.experiments.kb_text import NORMAL_PASSAGE, symptom_guides
from backend.experiments.metrics import softmax

CAPS = os.path.join(RESULTS_DIR, "fusion_captions.json")
S1 = os.path.join(RESULTS_DIR, "b_stage1_top3.json")
OUT = os.path.join(RESULTS_DIR, "exp2_embed_compare.json")


# 문장 목록을 정규화 임베딩으로 바꾼다
def encode(model, texts: list, is_e5: bool, is_query: bool):
    if is_e5:
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
    return model.encode(texts, normalize_embeddings=True, batch_size=16)


# 병징 전체와 시각 발췌 두 종류의 passage 를 만든다
def build_passages(kb: dict, retriever: KBRetriever) -> dict:
    full = {k: " ".join(kb[KEY_TO_SECTION[k]]["symptoms"]) for k in DISEASE_KEYS}
    full[NORMAL_KEY] = NORMAL_PASSAGE
    visual = {k: " ".join(lines) for k, lines in symptom_guides(retriever).items() if k != NORMAL_KEY}
    visual[NORMAL_KEY] = NORMAL_PASSAGE
    return {"full": full, "visual": visual}


# 임베딩·passage 조합 하나의 변별력을 평가한다
def evaluate_condition(model, passages: dict, is_e5: bool, caps: dict, s1: dict) -> dict:
    pemb = {k: encode(model, [passages[k]], is_e5, False)[0] for k in DISEASE_KEYS + [NORMAL_KEY]}
    cos = [float(pemb[a] @ pemb[b]) for a, b in itertools.combinations(DISEASE_KEYS, 2)]
    spread = {"mean_pair_cos": round(float(np.mean(cos)), 4), "min": round(min(cos), 4),
              "max": round(max(cos), 4), "std": round(float(np.std(cos)), 4)}

    stds, wrong_rank1, wrong_n, all_rank1, all_n = [], 0, 0, 0, 0
    for b, c in caps.items():
        s = s1[b]
        top3, gold = s["score_top3"], s["gold"]
        cemb = encode(model, [c["caption"]], is_e5, True)[0]
        sc = {k: float(cemb @ pemb[k]) for k in top3}
        stds.append(float(np.std(softmax([sc[k] for k in top3]))))
        if gold in top3:
            all_n += 1
            rank = sorted(top3, key=lambda k: -sc[k]).index(gold) + 1
            all_rank1 += rank == 1
            if s["score_top1"] != gold:
                wrong_n += 1
                wrong_rank1 += rank == 1
    med = sorted(stds)[len(stds) // 2]
    return {"pair_cos": spread, "softmax_std_median": round(med, 4),
            "wrong_gold_rank1": f"{wrong_rank1}/{wrong_n}",
            "wrong_rank1_ratio": round(wrong_rank1 / wrong_n, 3) if wrong_n else None,
            "all_rank1_ratio": round(all_rank1 / all_n, 3) if all_n else None,
            "gate_pass": bool(med >= 0.02 and wrong_n and wrong_rank1 / wrong_n > 0.333)}


# 임베딩과 passage 조합별로 점수 융합 게이트를 비교한다
def main():
    caps = read_json(CAPS)
    s1 = {r["image"]: r for r in read_json(S1)["rows"]}
    retriever = KBRetriever(device="cpu")
    passage_sets = build_passages(parse_kb(), retriever)
    models = {"e5": retriever.model, "bge-m3": SentenceTransformer("BAAI/bge-m3", device="cpu")}

    report = {}
    for mname, model in models.items():
        for ptype, passages in passage_sets.items():
            report[f"{mname}/{ptype}"] = evaluate_condition(model, passages, mname == "e5", caps, s1)
    write_json(report, OUT)

    print("=" * 66)
    print("  EXP-2 임베딩×발췌 비교")
    print("=" * 66)
    print(f"  {'조건':<14}{'쌍cos평균':>9}{'std중앙':>9}{'오답1위':>10}{'게이트':>7}")
    for c, r in report.items():
        print(f"  {c:<14}{r['pair_cos']['mean_pair_cos']:>9}{r['softmax_std_median']:>9}"
              f"{r['wrong_rank1_ratio']:>10}{'통과' if r['gate_pass'] else '미달':>7}")
    any_pass = any(r["gate_pass"] for r in report.values())
    print(f"\n  게이트 통과 조건: {'있음 → 융합 재검토' if any_pass else '없음 → 임베딩·발췌 무관 확정'}")
    print(f"  저장: {OUT}")


if __name__ == "__main__":
    main()
