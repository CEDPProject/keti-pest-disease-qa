import os
import csv

from PIL import Image, ImageOps

from backend.core.classes import CLASSES, CLEAN_IMAGE_DIR, MANIFEST_CSV, RAW_IMAGE_DIR
from backend.experiments.data_io import list_images

FIELDS = ["image_path", "key", "label_ko", "orig_name"]


# 클래스 폴더의 원본 사진을 정제해 저장하고 목록 행을 만든다
def clean_class(label_ko: str, key: str) -> list:
    dst_dir = os.path.join(CLEAN_IMAGE_DIR, key)
    os.makedirs(dst_dir, exist_ok=True)
    rows, n_bad = [], 0
    for src in list_images(os.path.join(RAW_IMAGE_DIR, label_ko)):
        try:
            img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        except Exception as e:
            print(f"  [손상 스킵] {os.path.basename(src)}: {e.__class__.__name__}")
            n_bad += 1
            continue
        dst = os.path.join(dst_dir, f"{key}_{len(rows)+1:03d}.jpg")
        img.save(dst, "JPEG", quality=95)
        rows.append({"image_path": os.path.abspath(dst), "key": key, "label_ko": label_ko,
                     "orig_name": os.path.basename(src)})
    print(f"  {label_ko:12s} [{key:12s}]: {len(rows)}장 정제 (손상 {n_bad})")
    return rows


# 플랫폼 원본 사진을 정제해 사진 목록 CSV 를 만든다
def main():
    os.makedirs(CLEAN_IMAGE_DIR, exist_ok=True)
    rows = []
    for label_ko, info in CLASSES.items():
        rows += clean_class(label_ko, info["key"])
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n총 {len(rows)}장 → {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
