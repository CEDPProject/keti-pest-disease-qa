import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
import csv
import json
import random
from collections import defaultdict

from classes import MANIFEST_CSV, SPLIT_JSON

SEED = 42
TEST_RATIO = 0.2


def make_split():
    by_class = defaultdict(list)
    with open(MANIFEST_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_class[row["key"]].append(row)

    rng = random.Random(SEED)
    train, test = [], []
    print(f"{'클래스':14s} {'전체':>4s} {'train':>6s} {'test':>5s}")
    for key in sorted(by_class):
        items = by_class[key][:]
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(TEST_RATIO * n))
        # train 이 0이 되지 않도록 보정
        n_test = min(n_test, n - 1) if n > 1 else 0
        te, tr = items[:n_test], items[n_test:]
        test.extend(te)
        train.extend(tr)
        warn = "  test 신뢰도 낮음" if len(te) <= 1 else ""
        print(f"{items[0]['label_ko']:14s} {n:>4d} {len(tr):>6d} {len(te):>5d}{warn}")

    split = {
        "seed": SEED,
        "test_ratio": TEST_RATIO,
        "train": train,
        "test": test,
    }
    with open(SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"\ntrain {len(train)}장 / test {len(test)}장 → {SPLIT_JSON}")


if __name__ == "__main__":
    make_split()
