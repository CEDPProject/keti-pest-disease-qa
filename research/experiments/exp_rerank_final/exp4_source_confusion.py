import os
import json
from collections import Counter

from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(os.path.dirname(HERE))
S1 = os.path.join(CORE, "results", "b_stage1_top3.json")
OUT = os.path.join(CORE, "results", "exp4_source_confusion.json")
KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot", "green_mold", "anthracnose", "normal"]
KO = {"canker": "궤양", "scab": "더뎅이", "gray_mold": "잿빛", "gummosis": "수지",
      "black_spot": "검은점", "green_mold": "녹색", "anthracnose": "탄저", "normal": "정상"}
TRIO = {"canker", "scab", "black_spot"}


def _src(img, key):
    if key == "normal":
        return "잎" if "잎" in img else "열매"
    return "NCPMS" if img.startswith("ncpms_") else "원본"


def cm_text(g, p):
    cm = confusion_matrix(g, p, labels=KEYS)
    lines = ["정답\\예측 " + " ".join(f"{KO[k]:>4s}" for k in KEYS)]
    for i, k in enumerate(KEYS):
        lines.append(f"{KO[k]:<5s} " + " ".join(f"{cm[i][j]:>4d}" for j in range(len(KEYS))))
    return lines, cm


def main():
    rows = json.load(open(S1, encoding="utf-8"))["rows"]
    for r in rows:
        r["src"] = _src(r["image"], r["gold"])

    out = {}
    for s in ("원본", "NCPMS"):
        sub = [r for r in rows if (r["src"] == s if s == "원본" else r["src"] == "NCPMS")
               or (s == "원본" and r["src"] in ("원본",))]
        sub = [r for r in rows if r["src"] == s]
        # 정상은 잎/열매라 병해 출처엔 안 들어감 → 병해만
        g = [r["gold"] for r in sub]; p = [r["score_top1"] for r in sub]
        lines, cm = cm_text(g, p)
        # 오답 쌍
        pairs = Counter()
        for gg, pp in zip(g, p):
            if gg != pp:
                pairs[(KO[gg], KO.get(pp, pp))] += 1
        out[s] = {"n": len(sub), "confusion": lines,
                  "top_error_pairs": [{"pair": f"{a}→{b}", "n": c} for (a, b), c in pairs.most_common(3)]}

    # 혼동 3형제 쌍 오류 비중 (전체 오답 중)
    all_wrong = [r for r in rows if r["score_top1"] != r["gold"]]
    trio_wrong = [r for r in all_wrong if r["gold"] in TRIO and r["score_top1"] in TRIO]
    out["trio_analysis"] = {
        "total_wrong": len(all_wrong),
        "trio_pair_wrong": len(trio_wrong),
        "ratio": round(len(trio_wrong) / len(all_wrong), 3) if all_wrong else 0,
        "cases": [{"img": r["image"][:26], "gold": KO[r["gold"]], "pred": KO[r["score_top1"]]} for r in trio_wrong],
    }

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=" * 60); print("  EXP-4 출처별 혼동행렬"); print("=" * 60)
    for s in ("원본", "NCPMS"):
        print(f"\n[{s}] n={out[s]['n']}")
        for ln in out[s]["confusion"]:
            print("  " + ln)
        print("  오답 쌍 top3:", [f"{x['pair']}({x['n']})" for x in out[s]["top_error_pairs"]])
    t = out["trio_analysis"]
    print(f"\n혼동3형제 쌍 오류: {t['trio_pair_wrong']}/{t['total_wrong']} = {t['ratio']} (전체 오답 중 비중)")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
