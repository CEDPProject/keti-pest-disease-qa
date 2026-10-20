import os
import shutil

import numpy as np

from backend.core.classes import BALANCED_DIR, CLASSES, NCPMS_IMAGE_DIR, NORMAL_IMAGE_DIR, RAW_IMAGE_DIR
from backend.experiments.data_io import dhash, list_images

DEST = os.path.join(BALANCED_DIR, "images")
DUP_DIST = 3


# 근접 중복이 아닌 사진만 복사하고 추가한 수를 돌려준다
def add_unique(paths: list, dest_dir: str, hashes: list, prefix: str = None) -> int:
    added = 0
    for p in paths:
        try:
            h = dhash(p)
        except Exception:
            continue
        if any(int(np.count_nonzero(h != k)) <= DUP_DIST for k in hashes):
            continue
        hashes.append(h)
        if prefix:
            dst = os.path.join(dest_dir, f"{prefix}_{added+1:03d}{os.path.splitext(p)[1].lower()}")
        else:
            dst = os.path.join(dest_dir, os.path.basename(p))
        shutil.copy(p, dst)
        added += 1
    return added


# 원본·NCPMS 질병 사진과 정상 사진을 중복 없이 한 폴더로 모은다
def main():
    print(f"{'클래스':<14s}{'원본':>5s}{'+NCPMS신규':>10s}{'최종':>6s}")
    print("-" * 40)
    totals = {}
    for label, meta in CLASSES.items():
        key = meta["key"]
        dest = os.path.join(DEST, key)
        os.makedirs(dest, exist_ok=True)
        hashes = []
        n_orig = add_unique(list_images(os.path.join(RAW_IMAGE_DIR, label)), dest, hashes)
        n_new = add_unique(list_images(os.path.join(NCPMS_IMAGE_DIR, key)), dest, hashes, prefix=f"ncpms_{key}")
        totals[key] = n_orig + n_new
        print(f"{label:<14s}{n_orig:>5d}{n_new:>10d}{totals[key]:>6d}")

    dest = os.path.join(DEST, "normal")
    os.makedirs(dest, exist_ok=True)
    hashes, n = [], 0
    for sub in ("잎_정상_사진", "열매_정상_사진"):
        part = sub.split("_")[0]
        n += add_unique(list_images(os.path.join(NORMAL_IMAGE_DIR, sub)), dest, hashes, prefix=f"normal_{part}")
    totals["normal"] = n
    print(f"{'정상':<14s}{'':>5s}{'':>10s}{n:>6d}")

    print("-" * 40)
    print(f"  통합 위치: {DEST}")
    print(f"  총 {sum(totals.values())}장 · 8클래스")
    under = {k: v for k, v in totals.items() if v < 50}
    print(f"  50장 미만(증강 필요): {under if under else '없음'}")


if __name__ == "__main__":
    main()
