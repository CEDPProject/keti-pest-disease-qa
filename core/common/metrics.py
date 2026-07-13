"""예측 결과(jsonl: gold_key, pred_key) → macro-F1 / per-class / confusion.

predictions jsonl 각 줄: {"key": gold, "pred_key": pred, ...}
"""

import json
import sys

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

CLASS_KEYS = ["canker", "scab", "gray_mold", "gummosis",
              "black_spot", "green_mold", "anthracnose", "black_blotch"]
KEY_KO = {"canker": "궤양병", "scab": "더뎅이병", "gray_mold": "잿빛곰팡이병",
          "gummosis": "수지병", "black_spot": "검은점무늬병", "green_mold": "녹색곰팡이병",
          "anthracnose": "탄저병", "black_blotch": "검은무늬병"}


def evaluate(pred_path: str, title: str = "") -> dict:
    golds, preds = [], []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            golds.append(r["key"])
            # unknown 은 오답 처리를 위해 별도 라벨 유지
            preds.append(r.get("pred_key", "unknown"))

    labels = CLASS_KEYS
    acc = accuracy_score(golds, preds)
    macro_f1 = f1_score(golds, preds, labels=labels, average="macro", zero_division=0)
    n_unknown = sum(1 for p in preds if p == "unknown")

    print(f"\n{'='*60}")
    print(f"[{title}]  N={len(golds)}")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Macro-F1 : {macro_f1:.3f}")
    print(f"  unknown(추출실패/기타): {n_unknown}")
    print(f"{'='*60}")

    print("\n[클래스별 P/R/F1]")
    rep = classification_report(golds, preds, labels=labels,
                                target_names=[KEY_KO[k] for k in labels],
                                zero_division=0, digits=3)
    print(rep)

    print("[Confusion Matrix] (행=정답, 열=예측)")
    cm = confusion_matrix(golds, preds, labels=labels)
    hdr = "정답\\예측   " + " ".join(f"{KEY_KO[k][:4]:>5s}" for k in labels)
    print(hdr)
    for i, k in enumerate(labels):
        row = " ".join(f"{cm[i][j]:>5d}" for j in range(len(labels)))
        print(f"{KEY_KO[k][:6]:8s} {row}")

    return {"title": title, "n": len(golds), "accuracy": acc,
            "macro_f1": macro_f1, "n_unknown": n_unknown}


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
