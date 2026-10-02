import os
import sys
import json

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
from classes import RESULTS_DIR  # noqa: E402
from classes import CLASSES  # noqa
from kb_retriever import KBRetriever  # noqa

KEY_TO_SECTION = {m["key"]: m["kb_section"] for lab, m in CLASSES.items()}
KO = {"canker": "궤양병", "scab": "더뎅이병", "gray_mold": "잿빛곰팡이병",
      "gummosis": "수지병", "black_spot": "검은점무늬병",
      "green_mold": "녹색곰팡이병", "anthracnose": "탄저병", "normal": "정상"}

# 시각 어휘 질의 (재랭킹용) — 반점형 + 곰팡이형 시각 특징 모두 포괄 (조정1)
VISUAL_QUERY = ("병반 반점 딱지 부스럼 융기 돌출 코르크 분화구 검은점 "
                "부패 곰팡이 균사 포자 색깔 표면 모양 크기 테두리")
NORMAL_RULE = "병징에 해당하는 이상 소견(반점·병반·변색·곰팡이 등)이 없으면 정상."
TOP_N = 3

STAGE1 = os.path.join(RESULTS_DIR, "b_stage1_top3.json")
OUT = os.path.join(RESULTS_DIR, "b_stage2_guides.json")


def extract_guide(kb, key):
    """후보 병해의 병징 섹션에서 시각 특징 문장 발췌(KB 원문)."""
    if key == "normal":
        return {"section": "규칙(정상)", "lines": [NORMAL_RULE], "is_rule": True}
    sec = KEY_TO_SECTION[key]                       # KB 섹션명
    lines = kb.retrieve_lines(sec, VISUAL_QUERY, "병징", top_n=TOP_N)
    return {"section": f"{sec} / 병징", "lines": [l["text"] for l in lines], "is_rule": False}


def build(kb, top3):
    return {KO[k]: extract_guide(kb, k) for k in top3}


def main():
    d = json.load(open(STAGE1, encoding="utf-8"))
    rows = d["rows"]
    kb = KBRetriever(device="cpu")

    # 오답 13건
    wrong = [r for r in rows if r["score_top1"] != r["gold"]]

    # 샘플 5개 선정: 혼동3형제 조합 + 정상포함 조합 우선, 실제 오답에서
    trio = {"canker", "scab", "black_spot"}
    sample_ids = []
    # (1) 혼동3형제 실제 오답
    for r in wrong:
        if len(set(r["score_top3"]) & trio) >= 2:
            sample_ids.append(r["image"]); break
    # (2~4) 나머지 오답에서
    for r in wrong:
        if r["image"] not in sample_ids and len(sample_ids) < 4:
            sample_ids.append(r["image"])
    # (5) 정상 포함 조합 (정상이 후보에 든 케이스 — 정답 케이스에서라도)
    for r in rows:
        if "normal" in r["score_top3"] and r["image"] not in sample_ids:
            sample_ids.append(r["image"]); break

    def _row(img):
        return next(r for r in rows if r["image"] == img)

    samples = []
    for img in sample_ids:
        r = _row(img)
        samples.append({
            "image": img, "gold": KO[r["gold"]], "top1": KO[r["score_top1"]],
            "top3": [KO[k] for k in r["score_top3"]],
            "guides": build(kb, r["score_top3"]),
        })

    # 오답 13건 전체 발췌
    wrong_guides = []
    for r in wrong:
        wrong_guides.append({
            "image": r["image"], "gold": KO[r["gold"]], "top1": KO[r["score_top1"]],
            "top3": [KO[k] for k in r["score_top3"]],
            "guides": build(kb, r["score_top3"]),
        })

    json.dump({"visual_query": VISUAL_QUERY, "normal_rule": NORMAL_RULE,
               "samples": samples, "wrong_all": wrong_guides},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 콘솔: 샘플 5개 출력
    for i, s in enumerate(samples, 1):
        print("=" * 70)
        print(f"[샘플 {i}] {s['image']}  정답={s['gold']} top1={s['top1']} top3={s['top3']}")
        for name, g in s["guides"].items():
            print(f"  ─ {name} ({g['section']})")
            for ln in g["lines"]:
                print(f"      · {ln}")
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
