"""공통 평가 모듈 — sklearn metrics + Confusion Matrix 시각화"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, class_names):
    """정밀도, 재현율, F1, 정확도 계산"""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1)
    
    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": round(prec[i], 4),
            "recall": round(rec[i], 4),
            "f1": round(f1[i], 4),
            "support": int(support[i]),
        }
    
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    """Confusion Matrix를 PNG로 저장"""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion Matrix 저장: {save_path}")


def print_metrics(metrics, model_name, dataset_name, n_images):
    """공통 출력 포맷"""
    print(f"\n{'='*50}")
    print(f"[테스트 결과 - {model_name}]")
    print(f"데이터: {dataset_name}")
    print(f"이미지 수: {n_images}장")
    print(f"{'-'*50}")
    print(f"클래스별:")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:10s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  ({m['support']}장)")
    print(f"{'-'*50}")
    print(f"Accuracy: {metrics['accuracy']:.2f} | Macro F1: {metrics['macro_f1']:.2f}")
    print(f"{'='*50}")
