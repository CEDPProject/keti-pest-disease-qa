import os
import re
import sys
import json

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
from classes import RESULTS_DIR  # noqa: E402
from classes import CLASSES  # noqa
from label_extract import LABEL_TO_KEY  # noqa
from kb_parser import parse_kb  # noqa

E5 = "intfloat/multilingual-e5-large"
CAPS = os.path.join(RESULTS_DIR, "fusion_captions.json")
STAGE1 = os.path.join(RESULTS_DIR, "b_stage1_top3.json")
OUT = os.path.join(RESULTS_DIR, "fusion_f1.json")
FIXED_Q = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."

KO = {v: k for k, v in LABEL_TO_KEY.items()}; KO["normal"] = "정상"
KEY_TO_SECTION = {m["key"]: m["kb_section"] for lab, m in CLASSES.items()}
NORMAL_PASSAGE = "병징에 해당하는 이상 소견(반점 병반 변색 곰팡이 무름)이 없는 건강한 상태."
DISEASE_WORDS = list(KO.values()) + ["궤양", "더뎅이", "잿빛", "곰팡이", "수지",
                                     "검은점", "탄저", "멜라노스", "푸른곰팡이"]


def clean_caption(cap):
    for w in DISEASE_WORDS:                        # 병명 어절 제거(순환 차단)
        cap = re.sub(r"\S*" + re.escape(w) + r"\S*", "", cap)
    return re.sub(r"\s+", " ", cap).strip()


def softmax(x):
    x = np.array(x); e = np.exp(x - x.max()); return e / e.sum()


def main():
    caps = json.load(open(CAPS, encoding="utf-8"))
    s1 = {r["image"]: r for r in json.load(open(STAGE1, encoding="utf-8"))["rows"]}
    kb = parse_kb()

    m = SentenceTransformer(E5, device="cpu")

    # 후보 병징 passage 임베딩 (8클래스)
    keys = list(KO.keys())
    passages = {}
    for k in keys:
        if k == "normal":
            passages[k] = NORMAL_PASSAGE
        else:
            sec = KEY_TO_SECTION[k]
            passages[k] = " ".join(kb[sec]["symptoms"])
    pass_emb = {k: m.encode([f"passage: {passages[k]}"], normalize_embeddings=True)[0] for k in keys}

    # V2 전역: 고정질문 임베딩
    q_emb = m.encode([f"query: {FIXED_Q}"], normalize_embeddings=True)[0]
    rag_v2 = {k: float(q_emb @ pass_emb[k]) for k in keys}   # 이미지 무관

    rows = []
    v1_top3_std = []          # 게이트용
    for b, c in caps.items():
        s = s1[b]
        top3 = s["score_top3"]; gold = s["gold"]
        cap = clean_caption(c["caption"])
        cap_emb = m.encode([f"query: {cap}"], normalize_embeddings=True)[0]
        rag_v1 = {k: float(cap_emb @ pass_emb[k]) for k in keys}

        # top-3 위 softmax
        v1_p = dict(zip(top3, softmax([rag_v1[k] for k in top3])))
        v2_p = dict(zip(top3, softmax([rag_v2[k] for k in top3])))
        v1_top3_std.append(float(np.std(list(v1_p.values()))))

        # 정답후보의 RAG_V1 순위 (top-3 내)
        rank_gold = None
        if gold in top3:
            order = sorted(top3, key=lambda k: -rag_v1[k])
            rank_gold = order.index(gold) + 1

        rows.append({
            "image": b, "gold": gold, "top1": s["score_top1"], "top3": top3,
            "margin": s["margin_1_2"], "caption_clean": cap[:300],
            "rag_v1_top3": {k: round(rag_v1[k], 4) for k in top3},
            "rag_v1_softmax": {k: round(v1_p[k], 4) for k in top3},
            "rag_v1_std": round(v1_top3_std[-1], 4),
            "gold_rag_v1_rank": rank_gold,
        })

    # 진단
    n = len(rows)
    std_arr = sorted(v1_top3_std)
    std_stat = {"min": round(std_arr[0], 4), "median": round(std_arr[n//2], 4),
                "max": round(std_arr[-1], 4), "mean": round(float(np.mean(std_arr)), 4)}
    # 오답 13건에서 정답후보 RAG_V1 1위 비율
    wrong = [r for r in rows if r["top1"] != r["gold"] and r["gold"] in r["top3"]]
    gold_rank1 = sum(1 for r in wrong if r["gold_rag_v1_rank"] == 1)
    # 참고: 전체(정답이 top3에 있는) 케이스에서 정답 RAG 1위 비율
    all_in = [r for r in rows if r["gold"] in r["top3"]]
    all_rank1 = sum(1 for r in all_in if r["gold_rag_v1_rank"] == 1)

    summary = {
        "n": n,
        "v1_top3_softmax_std": std_stat,
        "gate_std_near_zero": bool(std_stat["median"] < 0.02),   # A안 0.018 기준
        "wrong_n_gold_in_top3": len(wrong),
        "wrong_gold_rag_v1_rank1": gold_rank1,
        "wrong_gold_rank1_ratio": round(gold_rank1 / len(wrong), 3) if wrong else None,
        "all_gold_in_top3": len(all_in),
        "all_gold_rank1_ratio": round(all_rank1 / len(all_in), 3) if all_in else None,
        "chance": 0.333,
        "rag_v2_global": {KO[k]: round(v, 4) for k, v in rag_v2.items()},
    }
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("=" * 62)
    print("  STAGE F1 게이트 진단")
    print("=" * 62)
    print(f"  V1 top-3 softmax 표준편차: {std_stat}")
    print(f"    → 중앙값 {std_stat['median']} ({'≈0 수렴, 변별력 없음' if std_stat['median']<0.02 else '산포 있음'})")
    print(f"  오답 {len(wrong)}건(정답 top-3 내)에서 정답후보 RAG_V1 1위: "
          f"{gold_rank1}/{len(wrong)} = {summary['wrong_gold_rank1_ratio']} (우연 0.333)")
    print(f"  전체 {len(all_in)}건 정답 RAG_V1 1위 비율: {summary['all_gold_rank1_ratio']}")
    print(f"\n  저장: {OUT}")


if __name__ == "__main__":
    main()
