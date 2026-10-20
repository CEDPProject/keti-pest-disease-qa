import csv
import random
from collections import defaultdict

from backend.core.classes import MANIFEST_CSV, SPLIT_JSON
from backend.experiments.data_io import write_json

SEED = 42
TEST_RATIO = 0.2


# 클래스별로 20% 를 테스트로 나눈다
def main():
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
        n_test = min(max(1, round(TEST_RATIO * n)), n - 1) if n > 1 else 0
        te, tr = items[:n_test], items[n_test:]
        test.extend(te)
        train.extend(tr)
        warn = "  test 신뢰도 낮음" if len(te) <= 1 else ""
        print(f"{items[0]['label_ko']:14s} {n:>4d} {len(tr):>6d} {len(te):>5d}{warn}")

    write_json({"seed": SEED, "test_ratio": TEST_RATIO, "train": train, "test": test}, SPLIT_JSON)
    print(f"\ntrain {len(train)}장 / test {len(test)}장 → {SPLIT_JSON}")


if __name__ == "__main__":
    main()
