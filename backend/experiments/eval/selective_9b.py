import os

from sklearn.metrics import accuracy_score

from backend.core.classes import RESULTS_DIR
from backend.experiments.data_io import read_json, write_json
from backend.experiments.metrics import macro_f1

S1 = os.path.join(RESULTS_DIR, "b_stage1_top3_9b.json")
OUT = os.path.join(RESULTS_DIR, "exp3_confidence_9b.json")


# 마진 임계값별 발동률과 오답 포착률을 계산한다
def threshold_sweep(margins: list, wrong: list, in3: list) -> list:
    n, n_wrong = len(margins), sum(wrong)
    sweep = []
    for t in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3):
        trig = [i for i in range(n) if margins[i] < t]
        cov_wrong = sum(1 for i in trig if wrong[i]) / n_wrong if n_wrong else 0
        gold_in3 = sum(1 for i in trig if in3[i]) / len(trig) if trig else 0
        sweep.append({"threshold": t, "trigger_rate": round(len(trig) / n, 3),
                      "n_trigger": len(trig), "wrong_coverage": round(cov_wrong, 3),
                      "gold_in_top3_among_trigger": round(gold_in3, 3)})
    return sweep


# 위험도-커버리지 곡선과 AURC 를 계산한다
def risk_coverage(order: list, wrong: list):
    n = len(order)
    rc, cum_err = [], 0
    for k in range(1, n + 1):
        cum_err += wrong[order[k - 1]]
        rc.append({"coverage": round(k / n, 3), "risk": round(cum_err / k, 4)})
    return rc, round(sum(p["risk"] for p in rc) / n, 4)


# 마진 하위 사진을 보류했을 때의 정확도와 F1 을 계산한다
def reject_policy(order: list, golds: list, preds: list) -> list:
    n = len(order)
    reject = []
    for pct in (0.1, 0.2, 0.3):
        keep = order[:n - int(n * pct)]
        g = [golds[i] for i in keep]
        p = [preds[i] for i in keep]
        reject.append({"reject_pct": pct, "n_kept": len(keep),
                       "acc": round(accuracy_score(g, p), 4), "macro_f1": round(macro_f1(g, p), 4)})
    return reject


# 라벨 채점 마진으로 선택적 응답(보류) 성능을 분석해 저장한다
def main():
    rows = read_json(S1)["rows"]
    n = len(rows)
    golds = [r["gold"] for r in rows]
    preds = [r["score_top1"] for r in rows]
    margins = [r["margin_1_2"] for r in rows]
    in3 = [r["gold"] in r["score_top3"] for r in rows]
    wrong = [preds[i] != golds[i] for i in range(n)]
    n_wrong = sum(wrong)

    sweep = threshold_sweep(margins, wrong, in3)
    order = sorted(range(n), key=lambda i: -margins[i])
    rc, aurc = risk_coverage(order, wrong)
    reject = reject_policy(order, golds, preds)
    ops = [f"발동률 {s['trigger_rate']*100:.0f}%로 오답 {s['wrong_coverage']*100:.0f}% 포착, "
           f"발동케이스 후보내 정답포함 {s['gold_in_top3_among_trigger']*100:.0f}%"
           for s in sweep if s["threshold"] in (0.7, 0.9)]

    res = {"model": "Qwen3.5-9B", "n": n, "n_wrong": n_wrong,
           "baseline_acc": round(accuracy_score(golds, preds), 4),
           "sweep": sweep, "aurc": aurc, "risk_coverage": rc, "reject_policy": reject,
           "operating_points": ops}
    write_json(res, OUT)
    print("=" * 60)
    print("  9B Selective prediction")
    print("=" * 60)
    print(f"  baseline acc={res['baseline_acc']} · 오답 {n_wrong}/{n} · AURC={aurc}")
    for r in reject:
        print(f"    하위{int(r['reject_pct']*100)}% 보류 → 유지 {r['n_kept']}장 acc={r['acc']} F1={r['macro_f1']}")
    print(f"  저장: {OUT}")


if __name__ == "__main__":
    main()
