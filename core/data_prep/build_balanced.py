"""클래스 균형 데이터셋 생성 (NCPMS 실사진 + 정상, 균형증강).

- 실사진 출처: data/raw/ncpms_수집/{키} (7병해, 원본 121장을 포함하는 상위집합) + data/raw/감귤_정상_사진(정상)
- 클래스 내 dHash 근접중복 제거
- 클래스별 stratified split: 테스트는 '실사진만' 홀드아웃(증강 누수 방지)
- 학습셋은 클래스별 TARGET 장까지 균형증강 → 전 클래스 동일 수
- 각 base 이미지에 의도(진단/병명/증상/방제) 라운드로빈 부여 → ShareGPT 샘플

산출물: data/vlm_finetune_balanced/{train.json, test.jsonl, split.json, images_aug/, dataset_info.json}

실행: python core/data_prep/build_balanced.py
"""

import os
import sys
import json
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from classes import RAW_IMAGE_DIR, KEY_TO_LABEL, DATA_ROOT  # noqa
from kb_parser import kb_for_classes  # noqa
from build_dataset import (build_full, build_name, build_control, build_symptom,  # noqa
                           augment, sample as mk_sample)
# 질문 표현: 확장 템플릿(의도별 30개) 사용
sys.path.insert(0, DATA_ROOT)
from question_templates import Q_FULL, Q_NAME, Q_CONTROL, Q_SYMPTOM  # noqa

# ── 설정 ──────────────────────────────────────────────
TARGET = 60                 # 클래스당 학습 base 이미지 목표(균형)
TEST_RATIO = 0.2
TEST_MIN = 5
DUP_DIST = 3                # 클래스 내 근접중복(해밍) 제거 임계
SEED = 2026                 # 분할 랜덤 시드(새로 설정)

OUT = os.path.join(DATA_ROOT, "vlm_finetune_balanced")
IMAGES_ROOT = os.path.join(OUT, "images")            # 통합 실사진(소스, consolidate_images.py 산출)
VLM_ROOT = os.path.join(OUT, "images_vlm")           # 최종: images + images_aug 통합 → train/test
TRAIN_DIR = os.path.join(VLM_ROOT, "train")          # 실사진(train분) + 증강
TEST_DIR = os.path.join(VLM_ROOT, "test")            # 실사진만
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 병해 7키(정상 제외) — NCPMS 폴더명 = classes.py 키
DISEASE_KEYS = [k for k in KEY_TO_LABEL]   # KEY_TO_LABEL 은 7병해
NORMAL_KEY = "normal"


def _dhash(path, s=8):
    im = Image.open(path).convert("L").resize((s + 1, s), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _list(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMG_EXT))


def _dedup(paths):
    """클래스 내 dHash 근접중복 제거."""
    kept, hashes = [], []
    for p in paths:
        try:
            h = _dhash(p)
        except Exception:
            continue
        if any(int(np.count_nonzero(h != k)) <= DUP_DIST for k in hashes):
            continue
        hashes.append(h); kept.append(p)
    return kept


def _collect_real():
    """클래스키 → 통합 실사진 경로 목록(중복제거). 출처: vlm_finetune_balanced/images/{키}/"""
    real = {}
    for key in DISEASE_KEYS + [NORMAL_KEY]:
        real[key] = _dedup(_list(os.path.join(IMAGES_ROOT, key)))
    return real


# ── 정상 답변(무KB) ──
def _normal_full():
    return ("사진에서 관찰되는 특이 병징이 없어 **정상**(건강)으로 보입니다.\n\n"
            "## 관찰 소견\n- 과피 또는 잎 표면에 뚜렷한 병반이 확인되지 않음\n"
            "- 색·형태가 정상 범위\n\n뚜렷한 이상이 없으면 예방적 과원관리를 유지하세요.")
def _normal_name():
    return "이 감귤은 **정상**(건강)으로 보입니다. 뚜렷한 병징이 관찰되지 않습니다."
def _normal_symptom():
    return "뚜렷한 병징이 관찰되지 않습니다. 과피·잎이 **정상**(건강) 상태로 보입니다."


# 의도 라운드로빈 정의: (질문리스트, 답변빌더(label,k))
def _intent_pool(is_normal):
    if is_normal:
        return [
            (Q_FULL, lambda lab, k: _normal_full()),
            (Q_NAME, lambda lab, k: _normal_name()),
            (Q_SYMPTOM, lambda lab, k: _normal_symptom()),
        ]
    return [
        (Q_FULL, build_full),
        (Q_NAME, build_name),
        (Q_SYMPTOM, build_symptom),
        (Q_CONTROL, build_control),
    ]


def _group(fname, key):
    """출처/부위 하위그룹 — 이 그룹 비율대로 train/test 층화."""
    if key == NORMAL_KEY:
        if "잎" in fname:
            return "leaf"
        if "열매" in fname:
            return "fruit"
        return "etc"
    return "ncpms" if fname.startswith("ncpms_") else "orig"   # 영어(NCPMS) vs 한글(원본)


def _split_stratified(paths, key, rng):
    """하위그룹(출처/부위) 비율을 유지하며 test/train 분할."""
    groups = defaultdict(list)
    for p in paths:
        groups[_group(os.path.basename(p), key)].append(p)
    for g in groups.values():
        rng.shuffle(g)
    n = len(paths)
    n_test = min(max(TEST_MIN, round(n * TEST_RATIO)), max(0, n - 1))
    # 그룹별 비례 배정 + 최대잔여법으로 합계 맞춤
    raw = {g: n_test * len(v) / n for g, v in groups.items()}
    alloc = {g: int(raw[g]) for g in groups}
    rem = n_test - sum(alloc.values())
    for g in sorted(groups, key=lambda g: raw[g] - int(raw[g]), reverse=True)[:rem]:
        alloc[g] += 1
    test, train = [], []
    for g, v in groups.items():
        a = min(alloc[g], max(0, len(v) - 1))   # 그룹당 최소 1장은 train에 남김
        test += v[:a]
        train += v[a:]
    return test, train


def main():
    import shutil
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(VLM_ROOT):
        shutil.rmtree(VLM_ROOT)          # 새 시드로 완전 재구성
    rng = random.Random(SEED)
    kb = kb_for_classes()

    real = _collect_real()
    all_keys = DISEASE_KEYS + [NORMAL_KEY]
    label_of = dict(KEY_TO_LABEL); label_of[NORMAL_KEY] = "정상"

    split = {"train": defaultdict(list), "test": []}
    train_samples = []
    report = []

    for key in all_keys:
        paths = real[key][:]
        n = len(paths)
        # 출처(원본 한글/NCPMS 영어) · 부위(정상 잎/열매) 비율 유지 층화분할
        test_paths, train_reals = _split_stratified(paths, key, rng)
        rng.shuffle(train_reals)

        tr_dir = os.path.join(TRAIN_DIR, key); os.makedirs(tr_dir, exist_ok=True)
        te_dir = os.path.join(TEST_DIR, key); os.makedirs(te_dir, exist_ok=True)

        # 테스트: 실사진만 → images_vlm/test/{키}/
        for p in test_paths:
            dst = os.path.join(te_dir, os.path.basename(p))
            shutil.copy(p, dst)
            split["test"].append({"image": os.path.abspath(dst), "label_ko": label_of[key], "key": key})

        # 학습 실사진 → images_vlm/train/{키}/
        base = []
        for p in train_reals:
            dst = os.path.join(tr_dir, os.path.basename(p))
            shutil.copy(p, dst)
            base.append((os.path.abspath(dst), False))
        # 균형 증강 → images_vlm/train/{키}/  (train + aug 한 폴더에 통합)
        i = 0
        while len(base) < TARGET and train_reals:
            src = train_reals[i % len(train_reals)]
            img = Image.open(src).convert("RGB")
            ap = os.path.join(tr_dir, f"aug_{key}_{i+1:03d}.jpg")
            augment(img, rng).save(ap, "JPEG", quality=92)
            base.append((os.path.abspath(ap), True))
            i += 1

        # 각 base 이미지 → 4의도(정상은 3의도) 전부 생성 (이미지 1 : 질문 N)
        pool = _intent_pool(key == NORMAL_KEY)
        kk = kb[label_of[key]] if key != NORMAL_KEY else None
        for img_path, is_aug in base:
            for qs, builder in pool:
                ans = builder(label_of[key], kk)
                train_samples.append(mk_sample(os.path.abspath(img_path), rng.choice(qs), ans))
            split["train"][key].append({"image": os.path.abspath(img_path), "aug": is_aug})

        report.append((label_of[key], n, len(test_paths), len(train_reals), TARGET - len(train_reals) if TARGET > len(train_reals) else 0, len(base)))

    rng.shuffle(train_samples)
    json.dump(train_samples, open(os.path.join(OUT, "train.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "test.jsonl"), "w", encoding="utf-8") as f:
        for r in split["test"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"train": {k: v for k, v in split["train"].items()}, "test": split["test"]},
              open(os.path.join(OUT, "split.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump({"citrus_balanced": {"file_name": "train.json", "formatting": "sharegpt",
               "columns": {"messages": "messages", "images": "images"},
               "tags": {"role_tag": "role", "content_tag": "content",
                        "user_tag": "user", "assistant_tag": "assistant"}}},
              open(os.path.join(OUT, "dataset_info.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"{'클래스':<14s}{'실고유':>6s}{'테스트':>6s}{'train실':>8s}{'증강':>6s}{'train계':>8s}")
    print("-" * 50)
    for lab, n, nt, ntr, naug, base in report:
        print(f"{lab:<14s}{n:>6d}{nt:>6d}{ntr:>8d}{naug:>6d}{base:>8d}")
    print("-" * 50)
    print(f"  학습 샘플 총 {len(train_samples)}개 · 테스트(실사진) {len(split['test'])}장")
    print(f"  출력: {OUT}")


if __name__ == "__main__":
    main()
