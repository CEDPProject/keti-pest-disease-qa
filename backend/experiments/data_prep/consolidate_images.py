import os
import sys
import shutil

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
from classes import DATA_ROOT, CLASSES  # noqa

ORIG_DISEASE = os.path.join(DATA_ROOT, "raw", "감귤_병해충_사진")
NCPMS = os.path.join(DATA_ROOT, "raw", "ncpms_수집")
NORMAL = os.path.join(DATA_ROOT, "raw", "감귤_정상_사진")
DEST = os.path.join(DATA_ROOT, "vlm_finetune_balanced", "images")
DUP_DIST = 3
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(path, s=8):
    im = Image.open(path).convert("L").resize((s + 1, s), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _list(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMG_EXT))


def _add_unique(paths, dest_dir, hashes, prefix=None, idx_start=1):
    """paths 중 hashes 와 근접중복 아닌 것만 dest_dir 로 복사. 추가된 수 반환."""
    added = 0
    n = idx_start
    for p in paths:
        try:
            h = dhash(p)
        except Exception:
            continue
        if any(int(np.count_nonzero(h != k)) <= DUP_DIST for k in hashes):
            continue
        hashes.append(h)
        if prefix:
            dst = os.path.join(dest_dir, f"{prefix}_{n:03d}{os.path.splitext(p)[1].lower()}")
        else:
            dst = os.path.join(dest_dir, os.path.basename(p))
        shutil.copy(p, dst)
        added += 1
        n += 1
    return added


def main():
    print(f"{'클래스':<14s}{'원본':>5s}{'+NCPMS신규':>10s}{'최종':>6s}")
    print("-" * 40)
    grand = {}
    # 병해 7종
    for label, meta in CLASSES.items():
        key = meta["key"]
        dest = os.path.join(DEST, key)
        os.makedirs(dest, exist_ok=True)
        hashes = []
        # 1) 원본 먼저(풍부한 파일명 유지)
        n_orig = _add_unique(_list(os.path.join(ORIG_DISEASE, label)), dest, hashes)
        # 2) NCPMS 신규만 추가
        n_new = _add_unique(_list(os.path.join(NCPMS, key)), dest, hashes, prefix=f"ncpms_{key}")
        grand[key] = n_orig + n_new
        print(f"{label:<14s}{n_orig:>5d}{n_new:>10d}{grand[key]:>6d}")

    # 정상
    dest = os.path.join(DEST, "normal")
    os.makedirs(dest, exist_ok=True)
    hashes = []
    n = 0
    for sub in ("잎_정상_사진", "열매_정상_사진"):
        part = sub.split("_")[0]
        n += _add_unique(_list(os.path.join(NORMAL, sub)), dest, hashes, prefix=f"normal_{part}")
    grand["normal"] = n
    print(f"{'정상':<14s}{'':>5s}{'':>10s}{n:>6d}")

    print("-" * 40)
    print(f"  통합 위치: {DEST}")
    print(f"  총 {sum(grand.values())}장 · 8클래스")
    under = {k: v for k, v in grand.items() if v < 50}
    print(f"  50장 미만(증강 필요): {under if under else '없음'}")


if __name__ == "__main__":
    main()
