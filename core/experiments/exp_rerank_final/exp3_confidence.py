"""EXP-3: 신뢰도 곡선 (STAGE1 마진 기반, GPU 불필요).

(a) 임계값 스윕: 감별모드 발동률 / 오답 커버리지 / 후보내 정답 포함률
(b) risk-coverage curve + AURC
(c) 보류 정책: 마진 하위 10/20/30% 보류 시 잔여 acc/F1
결과: core/results/exp3_confidence.json
"""
import os
import json

from sklearn.metrics import f1_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(os.path.dirname(HERE))
S1 = os.path.join(CORE, "results", "b_stage1_top3.json")
OUT = os.path.join(CORE, "results", "exp3_confidence.json")
CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis", "black_spot",
              "green_mold", "anthracnose", "normal"]


def main():
    rows = json.load(open(S1, encoding="utf-8"))["rows"]
    n = len(rows)
    golds = [r["gold"] for r in rows]
    preds = [r["score_top1"] for r in rows]
    mar = [r["margin_1_2"] for r in rows]
    in3 = [r["gold"] in r["score_top3"] for r in rows]
    wrong = [preds[i] != golds[i] for i in range(n)]
    n_wrong = sum(wrong)

    # (a) 임계값 스윕
    sweep = []
    for t in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3):
        trig = [i for i in range(n) if mar[i] < t]
        cov_wrong = sum(1 for i in trig if wrong[i]) / n_wrong if n_wrong else 0
        gold_in3 = sum(1 for i in trig if in3[i]) / len(trig) if trig else 0
        sweep.append({"threshold": t, "trigger_rate": round(len(trig)/n, 3),
                      "n_trigger": len(trig),
                      "wrong_coverage": round(cov_wrong, 3),
                      "gold_in_top3_among_trigger": round(gold_in3, 3)})

    # (b) risk-coverage (신뢰도=마진 내림차순으로 채택)
    order = sorted(range(n), key=lambda i: -mar[i])
    rc = []
    cum_err = 0
    for k in range(1, n + 1):
        i = order[k - 1]
        cum_err += wrong[i]
        rc.append({"coverage": round(k/n, 3), "risk": round(cum_err/k, 4)})
    aurc = round(sum(p["risk"] for p in rc) / n, 4)

    # (c) 보류 정책: 마진 하위 X% 보류
    reject = []
    for pct in (0.1, 0.2, 0.3):
        cut = int(n * pct)
        keep = [order[j] for j in range(n - cut)]      # 마진 상위 (n-cut)개
        g = [golds[i] for i in keep]; p = [preds[i] for i in keep]
        reject.append({"reject_pct": pct, "n_kept": len(keep),
                       "acc": round(accuracy_score(g, p), 4),
                       "macro_f1": round(f1_score(g, p, labels=CLASS_KEYS, average="macro", zero_division=0), 4)})

    # 대표 동작점
    ops = []
    for s in sweep:
        if s["threshold"] in (0.7, 0.9):
            ops.append(f"발동률 {s['trigger_rate']*100:.0f}%로 오답 {s['wrong_coverage']*100:.0f}% 포착, "
                       f"발동케이스 후보내 정답포함 {s['gold_in_top3_among_trigger']*100:.0f}%")

    res = {"n": n, "n_wrong": n_wrong, "baseline_acc": round(accuracy_score(golds, preds), 4),
           "sweep": sweep, "aurc": aurc, "risk_coverage": rc, "reject_policy": reject,
           "operating_points": ops}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("=" * 60); print("  EXP-3 신뢰도 곡선"); print("=" * 60)
    print(f"  baseline acc={res['baseline_acc']} · 오답 {n_wrong}/{n} · AURC={aurc}")
    print(f"  {'임계t':>5}{'발동률':>7}{'오답커버':>8}{'후보내정답':>10}")
    for s in sweep:
        print(f"  {s['threshold']:>5}{s['trigger_rate']:>7}{s['wrong_coverage']:>8}{s['gold_in_top3_among_trigger']:>10}")
    print("  보류정책:")
    for r in reject:
        print(f"    하위{int(r['reject_pct']*100)}% 보류 → 유지 {r['n_kept']}장 acc={r['acc']} F1={r['macro_f1']}")
    print("  대표 동작점:")
    for o in ops:
        print(f"    · {o}")
    print(f"  저장: {OUT}")


if __name__ == "__main__":
    main()
