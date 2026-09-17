"""원본 이미지(한글 폴더) → ASCII 경로 정제 복사본 + manifest.csv 생성.

- desktop.ini 등 비이미지 제외
- PIL 로 열리는지 검증(손상 파일 스킵), EXIF 회전 보정 후 JPEG로 재저장하여 포맷 정규화
- manifest.csv: image_path(절대), key, label_ko, orig_name
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend", "common"))
import csv
import glob

from PIL import Image, ImageOps

from classes import CLASSES, RAW_IMAGE_DIR, CLEAN_IMAGE_DIR, MANIFEST_CSV

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def build():
    os.makedirs(CLEAN_IMAGE_DIR, exist_ok=True)
    rows = []
    for label_ko, info in CLASSES.items():
        key = info["key"]
        src_dir = os.path.join(RAW_IMAGE_DIR, label_ko)
        dst_dir = os.path.join(CLEAN_IMAGE_DIR, key)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted(
            f for f in glob.glob(os.path.join(src_dir, "*"))
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
        n_ok, n_bad = 0, 0
        for i, src in enumerate(files, 1):
            try:
                img = Image.open(src)
                img = ImageOps.exif_transpose(img).convert("RGB")
            except Exception as e:
                print(f"  [손상 스킵] {os.path.basename(src)}: {e.__class__.__name__}")
                n_bad += 1
                continue
            dst_name = f"{key}_{n_ok+1:03d}.jpg"
            dst = os.path.join(dst_dir, dst_name)
            img.save(dst, "JPEG", quality=95)
            rows.append({
                "image_path": os.path.abspath(dst),
                "key": key,
                "label_ko": label_ko,
                "orig_name": os.path.basename(src),
            })
            n_ok += 1
        print(f"  {label_ko:12s} [{key:12s}]: {n_ok}장 정제 (손상 {n_bad})")

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "key", "label_ko", "orig_name"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n총 {len(rows)}장 → {MANIFEST_CSV}")


if __name__ == "__main__":
    build()
