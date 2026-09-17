import os
import sys
import json

import numpy as np
from PIL import Image

VLM_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(VLM_FT), "backend", "common"))
from classes import SPLIT_JSON  # noqa


def dhash(path, size=8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    return diff.flatten()


def hist(path, bins=8):
    im = Image.open(path).convert("RGB").resize((128, 128))
    a = np.asarray(im).reshape(-1, 3)
    h = []
    for c in range(3):
        cnt, _ = np.histogram(a[:, c], bins=bins, range=(0, 256))
        h.append(cnt / cnt.sum())
    return np.concatenate(h)


def main():
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    train, test = split["train"], split["test"]

    tr_h = {r["image_path"]: dhash(r["image_path"]) for r in train}
    tr_c = {r["image_path"]: hist(r["image_path"]) for r in train}

    print(f"{'test 이미지':38s} {'gold':12s} {'최근접 train':30s} {'dist':>5s} {'histCorr':>9s}  판정")
    print("-" * 115)
    severe, suspect = 0, 0
    rows = []
    for r in test:
        th, tc = dhash(r["image"] if "image" in r else r["image_path"]), None
        tp = r["image_path"]
        th = dhash(tp)
        tc = hist(tp)
        best, bd = None, 999
        for p, h in tr_h.items():
            d = int(np.count_nonzero(th != h))
            if d < bd:
                bd, best = d, p
        corr = float(np.corrcoef(tc, tr_c[best])[0, 1])
        if bd <= 5:
            verdict = "거의 동일(심각)"
            severe += 1
        elif bd <= 10:
            verdict = "매우 유사(의심)"
            suspect += 1
        else:
            verdict = "다른 사진"
        rows.append({"test": os.path.basename(tp), "gold": r["label_ko"],
                     "nearest": os.path.basename(best), "dist": bd, "hist_corr": corr,
                     "verdict": verdict})
        print(f"{os.path.basename(tp):38s} {r['label_ko']:12s} {os.path.basename(best):30s} "
              f"{bd:>5d} {corr:>9.3f}  {verdict}")

    print("-" * 115)
    print(f"총 {len(test)}장 | 거의동일 {severe}장 | 유사 {suspect}장 | 정상 {len(test)-severe-suspect}장")
    if severe:
        print("\n심각한 near-duplicate 누수 존재 → test 성능이 부풀려졌을 가능성 높음. split 재설계 필요.")
    elif suspect:
        print("\n유사 이미지 일부 존재 → 성능 해석 시 주의 필요.")
    else:
        print("\nnear-duplicate 누수 없음 → test 성능 신뢰 가능.")

    out = os.path.join(VLM_FT, "results", "leakage_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"severe": severe, "suspect": suspect, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
