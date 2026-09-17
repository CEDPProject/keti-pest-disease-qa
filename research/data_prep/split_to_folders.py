import os
import sys
import json
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "backend", "common"))
from classes import DATA_ROOT  # noqa

BAL = os.path.join(DATA_ROOT, "vlm_finetune_balanced")
SPLIT = os.path.join(BAL, "split.json")
DEST = os.path.join(BAL, "images_split")


def main():
    with open(SPLIT, encoding="utf-8") as f:
        split = json.load(f)

    # 초기화
    if os.path.exists(DEST):
        shutil.rmtree(DEST)

    counts = {}
    # train: 실사진 train분 + 증강
    for key, rows in split["train"].items():
        d = os.path.join(DEST, "train", key)
        os.makedirs(d, exist_ok=True)
        for r in rows:
            src = r["image"]
            shutil.copy(src, os.path.join(d, os.path.basename(src)))
        counts.setdefault(key, {})["train"] = len(rows)

    # test: 실사진만
    for r in split["test"]:
        key = r["key"]
        d = os.path.join(DEST, "test", key)
        os.makedirs(d, exist_ok=True)
        src = r["image"]
        shutil.copy(src, os.path.join(d, os.path.basename(src)))
        counts.setdefault(key, {})["test"] = counts.get(key, {}).get("test", 0) + 1

    print(f"{'클래스':<14s}{'train':>7s}{'test':>6s}")
    print("-" * 30)
    tt = ttr = tte = 0
    for key in sorted(counts):
        tr = counts[key].get("train", 0)
        te = counts[key].get("test", 0)
        ttr += tr; tte += te
        print(f"{key:<14s}{tr:>7d}{te:>6d}")
    print("-" * 30)
    print(f"{'합계':<14s}{ttr:>7d}{tte:>6d}")
    print(f"\n  위치: {DEST}")
    print("  · train/ = 실사진 train분 + 증강 (60/클래스)")
    print("  · test/  = 실사진만 (증강 없음)")


if __name__ == "__main__":
    main()
