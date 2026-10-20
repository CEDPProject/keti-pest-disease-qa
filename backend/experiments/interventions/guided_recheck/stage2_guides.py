import os

from backend.core.classes import KEY_TO_LABEL, KEY_TO_SECTION, NORMAL_KEY, RESULTS_DIR
from backend.core.kb_retriever import KBRetriever
from backend.experiments.data_io import read_json, write_json
from backend.experiments.kb_text import NORMAL_RULE, VISUAL_QUERY, symptom_guides

STAGE1 = os.path.join(RESULTS_DIR, "b_stage1_top3.json")
OUT = os.path.join(RESULTS_DIR, "b_stage2_guides.json")
CONFUSED_TRIO = {"canker", "scab", "black_spot"}


# 후보 하나의 병징 발췌를 출처 섹션 표기와 함께 묶는다
def guide_entry(guides: dict, key: str) -> dict:
    if key == NORMAL_KEY:
        return {"section": "규칙(정상)", "lines": guides[key], "is_rule": True}
    return {"section": f"{KEY_TO_SECTION[key]} / 병징", "lines": guides[key], "is_rule": False}


# 채점 결과 행 하나를 후보별 발췌가 붙은 기록으로 바꾼다
def guided_row(guides: dict, r: dict) -> dict:
    return {"image": r["image"], "gold": KEY_TO_LABEL[r["gold"]], "top1": KEY_TO_LABEL[r["score_top1"]],
            "top3": [KEY_TO_LABEL[k] for k in r["score_top3"]],
            "guides": {KEY_TO_LABEL[k]: guide_entry(guides, k) for k in r["score_top3"]}}


# 확인용 샘플 5장을 고른다
def pick_samples(rows: list, wrong: list) -> list:
    ids = []
    for r in wrong:
        if len(set(r["score_top3"]) & CONFUSED_TRIO) >= 2:
            ids.append(r["image"])
            break
    for r in wrong:
        if r["image"] not in ids and len(ids) < 4:
            ids.append(r["image"])
    for r in rows:
        if NORMAL_KEY in r["score_top3"] and r["image"] not in ids:
            ids.append(r["image"])
            break
    return [next(r for r in rows if r["image"] == i) for i in ids]


# top-3 후보별 KB 병징 발췌를 저장한다
def main():
    rows = read_json(STAGE1)["rows"]
    guides = symptom_guides(KBRetriever(device="cpu"))
    wrong = [r for r in rows if r["score_top1"] != r["gold"]]
    samples = [guided_row(guides, r) for r in pick_samples(rows, wrong)]
    wrong_guides = [guided_row(guides, r) for r in wrong]

    write_json({"visual_query": VISUAL_QUERY, "normal_rule": NORMAL_RULE,
                "samples": samples, "wrong_all": wrong_guides}, OUT)
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
