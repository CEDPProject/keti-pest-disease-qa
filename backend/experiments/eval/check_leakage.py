import os

import numpy as np
from PIL import Image

from backend.core.classes import RESULTS_DIR, SPLIT_JSON
from backend.experiments.data_io import dhash, read_json, write_json


# 이미지의 RGB 채널별 정규화 히스토그램을 이어 붙인다
def hist(path: str, bins: int = 8):
    a = np.asarray(Image.open(path).convert("RGB").resize((128, 128))).reshape(-1, 3)
    h = []
    for c in range(3):
        cnt, _ = np.histogram(a[:, c], bins=bins, range=(0, 256))
        h.append(cnt / cnt.sum())
    return np.concatenate(h)


# 해밍 거리로 근접 중복 정도를 판정한다
def verdict_of(dist: int) -> str:
    if dist <= 5:
        return "거의 동일(심각)"
    if dist <= 10:
        return "매우 유사(의심)"
    return "다른 사진"


# 테스트 사진의 근접 중복 누수를 점검한다
def main():
    split = read_json(SPLIT_JSON)
    train, test = split["train"], split["test"]
    tr_h = {r["image_path"]: dhash(r["image_path"]) for r in train}
    tr_c = {r["image_path"]: hist(r["image_path"]) for r in train}

    print(f"{'test 이미지':38s} {'gold':12s} {'최근접 train':30s} {'dist':>5s} {'histCorr':>9s}  판정")
    print("-" * 115)
    rows = []
    for r in test:
        tp = r["image_path"]
        th, tc = dhash(tp), hist(tp)
        best, bd = None, 999
        for p, h in tr_h.items():
            d = int(np.count_nonzero(th != h))
            if d < bd:
                bd, best = d, p
        corr = float(np.corrcoef(tc, tr_c[best])[0, 1])
        verdict = verdict_of(bd)
        rows.append({"test": os.path.basename(tp), "gold": r["label_ko"],
                     "nearest": os.path.basename(best), "dist": bd, "hist_corr": corr,
                     "verdict": verdict})
        print(f"{os.path.basename(tp):38s} {r['label_ko']:12s} {os.path.basename(best):30s} "
              f"{bd:>5d} {corr:>9.3f}  {verdict}")

    severe = sum(r["verdict"] == "거의 동일(심각)" for r in rows)
    suspect = sum(r["verdict"] == "매우 유사(의심)" for r in rows)
    print("-" * 115)
    print(f"총 {len(test)}장 | 거의동일 {severe}장 | 유사 {suspect}장 | 정상 {len(test)-severe-suspect}장")
    if severe:
        print("\n심각한 near-duplicate 누수 존재 → test 성능이 부풀려졌을 가능성 높음. split 재설계 필요.")
    elif suspect:
        print("\n유사 이미지 일부 존재 → 성능 해석 시 주의 필요.")
    else:
        print("\nnear-duplicate 누수 없음 → test 성능 신뢰 가능.")

    out = os.path.join(RESULTS_DIR, "leakage_check.json")
    write_json({"severe": severe, "suspect": suspect, "rows": rows}, out)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
