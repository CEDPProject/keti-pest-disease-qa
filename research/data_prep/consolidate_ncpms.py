"""NCPMS 수집 이미지를 원본 클래스 폴더에 통합(근접중복 제거).

원본(data/raw/감귤_병해충_사진/{라벨}) 도 농촌진흥청 출처라 NCPMS 도감컷과
겹칠 수 있다 → dHash 근접중복 검사로 겹치는 것을 버리고, 새로운 것만 편입한다.

  - 원본 vs NCPMS 교차중복 제거 (train/test 누수 방지)
  - NCPMS 내부 근접중복 제거
  - 편입 이미지는 ncpms_{key}_{n}.jpg 로 표시(추적 가능)

실행: python research/data_prep/consolidate_ncpms.py [--apply]
  기본은 리포트만(dry-run). --apply 시 실제 복사.
"""

import os
import sys
import shutil

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend", "common"))
from classes import RAW_IMAGE_DIR, KEY_TO_LABEL  # noqa

NCPMS_ROOT = os.path.join(os.path.dirname(RAW_IMAGE_DIR), "ncpms_수집")
DUP_DIST = 6           # dHash 해밍거리 이하 = 근접중복
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(path, size=8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _imgs(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMG_EXT))


def main(apply=False):
    print(f"{'[APPLY] 실제 복사' if apply else '[DRY-RUN] 리포트만'}\n")
    print(f"{'클래스':<14s}{'원본':>5s}{'NCPMS':>7s}{'중복':>6s}{'신규편입':>9s}{'최종':>6s}")
    print("-" * 52)

    grand = {}
    for key, label in KEY_TO_LABEL.items():
        orig_dir = os.path.join(RAW_IMAGE_DIR, label)
        ncpms_dir = os.path.join(NCPMS_ROOT, key)
        orig = _imgs(orig_dir)
        ncpms = _imgs(ncpms_dir)
        if not ncpms:
            grand[label] = len(orig)
            print(f"{label:<14s}{len(orig):>5d}{0:>7d}{0:>6d}{0:>9d}{len(orig):>6d}")
            continue

        # 기준 해시: 원본 + 편입확정분(중복 누적 방지)
        kept_hashes = [dhash(p) for p in orig]
        dup = 0
        new_files = []
        for p in ncpms:
            try:
                h = dhash(p)
            except Exception:
                continue
            is_dup = any(int(np.count_nonzero(h != kh)) <= DUP_DIST for kh in kept_hashes)
            if is_dup:
                dup += 1
            else:
                kept_hashes.append(h)
                new_files.append(p)

        if apply:
            for i, p in enumerate(new_files, 1):
                ext = os.path.splitext(p)[1].lower()
                dst = os.path.join(orig_dir, f"ncpms_{key}_{i:03d}{ext}")
                shutil.copy(p, dst)

        final = len(orig) + len(new_files)
        grand[label] = final
        print(f"{label:<14s}{len(orig):>5d}{len(ncpms):>7d}{dup:>6d}{len(new_files):>9d}{final:>6d}")

    print("-" * 52)
    print(f"{'합계':<14s}{'':>5s}{'':>7s}{'':>6s}{'':>9s}{sum(grand.values()):>6d}")
    print(f"\n최소 클래스: {min(grand, key=grand.get)} {min(grand.values())}장 · "
          f"최대: {max(grand, key=grand.get)} {max(grand.values())}장")
    under = {k: v for k, v in grand.items() if v < 50}
    if under:
        print(f"\n50장 미만(추가 필요): {under}")
    if not apply:
        print("\n→ 실제 편입하려면:  python research/data_prep/consolidate_ncpms.py --apply")
    return grand


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
