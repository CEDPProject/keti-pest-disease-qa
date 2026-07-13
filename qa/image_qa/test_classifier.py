"""경로 A: Classifier 테스트
dima806/orange_fruit_disease_detection → Mendeley Healthy/Canker 2-class 평가
"""

import os
import sys
import csv
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from PIL import Image
from transformers import pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from eval_utils import compute_metrics, plot_confusion_matrix, print_metrics

# ── 설정 ──
MODEL_NAME = "dima806/orange_fruit_disease_detection"
DATASET_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "datasets", "Citrus")
RESULT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "result_md")

# 모델 출력 라벨 → 통합 라벨 매핑
PRED_LABEL_MAP = {
    "healthy": "healthy",
    "citrus canker": "canker",
    "melanose": "melanose",  # 무시 대상
}

# GT 폴더명 → 통합 라벨 (Healthy/Canker만 사용)
GT_FOLDERS = {
    "healthy": "healthy",
    "Canker": "canker",      # Fruits 폴더는 대문자 Canker
    "canker": "canker",      # Leaves 폴더는 소문자 canker
}

CLASS_NAMES = ["healthy", "canker"]


def collect_images():
    """Mendeley 데이터셋에서 Healthy/Canker 이미지만 수집"""
    images = []  # (path, gt_label)
    
    for part in ["Fruits", "Leaves"]:
        part_dir = os.path.join(DATASET_ROOT, part)
        if not os.path.isdir(part_dir):
            continue
        for folder in os.listdir(part_dir):
            gt = GT_FOLDERS.get(folder)
            if gt is None:
                continue
            folder_path = os.path.join(part_dir, folder)
            for fname in sorted(os.listdir(folder_path)):
                fpath = os.path.join(folder_path, fname)
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    images.append((fpath, gt, part))
    
    return images


def run_test():
    print(f"[경로 A] Classifier 테스트 시작")
    print(f"모델: {MODEL_NAME}")
    print(f"데이터셋: Mendeley Citrus (Healthy + Canker)")
    print(f"시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 이미지 수집
    images = collect_images()
    if not images:
        print(f"❌ 이미지가 없습니다. DATASET_ROOT: {DATASET_ROOT}")
        sys.exit(1)
    
    print(f"수집 이미지: {len(images)}장 (Healthy + Canker)")
    
    # 모델 로드
    print(f"모델 로딩 중...")
    classifier = pipeline(
        "image-classification",
        model=MODEL_NAME,
        device=-1,  # CPU (CUDA driver 호환 문제로 CPU 사용)
    )
    print(f"모델 로딩 완료!\n")
    
    # 추론
    y_true = []
    y_pred = []
    rows = []  # CSV 용
    
    start = time.time()
    for i, (path, gt, part) in enumerate(images):
        img = Image.open(path).convert("RGB")
        results = classifier(img, top_k=3)
        
        top = results[0]
        pred_raw = top["label"]
        confidence = round(top["score"], 4)
        
        pred_mapped = PRED_LABEL_MAP.get(pred_raw, "unknown")
        
        # melanose/unknown 예측도 기록하되, 평가는 healthy/canker만
        y_true.append(gt)
        y_pred.append(pred_mapped if pred_mapped in CLASS_NAMES else "unknown")
        
        rows.append({
            "image": os.path.basename(path),
            "part": part,
            "gt": gt,
            "pred_raw": pred_raw,
            "pred_mapped": pred_mapped,
            "confidence": confidence,
            "correct": "✅" if pred_mapped == gt else "❌",
        })
        
        if (i + 1) % 50 == 0 or i == len(images) - 1:
            print(f"  [{i+1}/{len(images)}] 처리 중...")
    
    elapsed = time.time() - start
    print(f"\n추론 완료: {elapsed:.1f}초 ({elapsed/len(images):.2f}초/장)")
    
    # unknown 예측을 처리 — healthy/canker 중 더 가까운 것으로는 하지 않고 그대로 평가
    eval_class_names = CLASS_NAMES + (["unknown"] if "unknown" in y_pred else [])
    
    # 평가
    metrics = compute_metrics(y_true, y_pred, eval_class_names)
    print_metrics(metrics, MODEL_NAME, "Mendeley Citrus (Healthy+Canker)", len(images))
    
    # Confusion Matrix 저장
    os.makedirs(RESULT_DIR, exist_ok=True)
    cm_path = os.path.join(RESULT_DIR, "classifier_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, eval_class_names, cm_path,
                          title=f"Classifier: {MODEL_NAME}")
    
    # CSV 저장
    csv_path = os.path.join(RESULT_DIR, "classifier_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV 저장: {csv_path}")
    
    return metrics, rows, elapsed


if __name__ == "__main__":
    run_test()
