import os
import json
import random
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image

from backend.core.classes import BALANCED_DIR, CLASS_KEYS, KEY_TO_LABEL, NORMAL_KEY
from backend.core.kb_parser import kb_for_classes
from backend.experiments.data_io import dhash, list_images, write_json
from backend.experiments.data_prep.question_templates import Q_CONTROL, Q_FULL, Q_NAME, Q_SYMPTOM
from backend.experiments.data_prep.sample_builder import (augment, build_control, build_full, build_name,
                                                          build_symptom, make_sample)

TARGET = 60
TEST_RATIO = 0.2
TEST_MIN = 5
DUP_DIST = 3
SEED = 2026

IMAGES_ROOT = os.path.join(BALANCED_DIR, "images")
VLM_ROOT = os.path.join(BALANCED_DIR, "images_vlm")
TRAIN_DIR = os.path.join(VLM_ROOT, "train")
TEST_DIR = os.path.join(VLM_ROOT, "test")
DATASET_INFO = {"citrus_balanced": {"file_name": "train.json", "formatting": "sharegpt",
                                    "columns": {"messages": "messages", "images": "images"},
                                    "tags": {"role_tag": "role", "content_tag": "content",
                                             "user_tag": "user", "assistant_tag": "assistant"}}}


# 클래스 안에서 dHash 거리가 가까운 근접 중복 사진을 뺀다
def dedup(paths: list) -> list:
    kept, hashes = [], []
    for p in paths:
        try:
            h = dhash(p)
        except Exception:
            continue
        if any(int(np.count_nonzero(h != k)) <= DUP_DIST for k in hashes):
            continue
        hashes.append(h)
        kept.append(p)
    return kept


# 정상 사진용 종합 진단 답변을 돌려준다
def normal_full(label: str, k) -> str:
    return ("사진에서 관찰되는 특이 병징이 없어 **정상**(건강)으로 보입니다.\n\n"
            "## 관찰 소견\n- 과피 또는 잎 표면에 뚜렷한 병반이 확인되지 않음\n"
            "- 색·형태가 정상 범위\n\n뚜렷한 이상이 없으면 예방적 과원관리를 유지하세요.")


# 정상 사진용 병명 답변을 돌려준다
def normal_name(label: str, k) -> str:
    return "이 감귤은 **정상**(건강)으로 보입니다. 뚜렷한 병징이 관찰되지 않습니다."


# 정상 사진용 증상 답변을 돌려준다
def normal_symptom(label: str, k) -> str:
    return "뚜렷한 병징이 관찰되지 않습니다. 과피·잎이 **정상**(건강) 상태로 보입니다."


# 클래스에 적용할 (질문 목록, 답변 생성기) 의도 목록을 고른다
def intent_pool(is_normal: bool) -> list:
    if is_normal:
        return [(Q_FULL, normal_full), (Q_NAME, normal_name), (Q_SYMPTOM, normal_symptom)]
    return [(Q_FULL, build_full), (Q_NAME, build_name), (Q_SYMPTOM, build_symptom), (Q_CONTROL, build_control)]


# 파일 이름으로 층화 기준 하위 그룹을 정한다
def subgroup(fname: str, key: str) -> str:
    if key == NORMAL_KEY:
        if "잎" in fname:
            return "leaf"
        if "열매" in fname:
            return "fruit"
        return "etc"
    return "ncpms" if fname.startswith("ncpms_") else "orig"


# 하위 그룹 비율을 유지하며 테스트·학습으로 나눈다
def split_stratified(paths: list, key: str, rng) -> tuple:
    groups = defaultdict(list)
    for p in paths:
        groups[subgroup(os.path.basename(p), key)].append(p)
    for g in groups.values():
        rng.shuffle(g)
    n = len(paths)
    n_test = min(max(TEST_MIN, round(n * TEST_RATIO)), max(0, n - 1))
    raw = {g: n_test * len(v) / n for g, v in groups.items()}
    alloc = {g: int(raw[g]) for g in groups}
    rem = n_test - sum(alloc.values())
    for g in sorted(groups, key=lambda g: raw[g] - int(raw[g]), reverse=True)[:rem]:
        alloc[g] += 1
    test, train = [], []
    for g, v in groups.items():
        a = min(alloc[g], max(0, len(v) - 1))
        test += v[:a]
        train += v[a:]
    return test, train


# 사진 목록을 대상 폴더로 복사하고 복사본 절대 경로를 돌려준다
def copy_all(paths: list, dest: str) -> list:
    out = []
    for p in paths:
        dst = os.path.join(dest, os.path.basename(p))
        shutil.copy(p, dst)
        out.append(os.path.abspath(dst))
    return out


# 학습 사진이 목표 수가 되도록 증강본을 채운다
def fill_with_augments(key: str, train_reals: list, copied: list, tr_dir: str, rng) -> list:
    base = [(p, False) for p in copied]
    i = 0
    while len(base) < TARGET and train_reals:
        img = Image.open(train_reals[i % len(train_reals)]).convert("RGB")
        ap = os.path.join(tr_dir, f"aug_{key}_{i+1:03d}.jpg")
        augment(img, rng).save(ap, "JPEG", quality=92)
        base.append((os.path.abspath(ap), True))
        i += 1
    return base


# 논문용 균형 데이터셋을 만든다
def main():
    os.makedirs(BALANCED_DIR, exist_ok=True)
    if os.path.exists(VLM_ROOT):
        shutil.rmtree(VLM_ROOT)
    rng = random.Random(SEED)
    kb = kb_for_classes()

    split = {"train": defaultdict(list), "test": []}
    train_samples, report = [], []
    for key in CLASS_KEYS:
        label = KEY_TO_LABEL[key]
        paths = dedup(list_images(os.path.join(IMAGES_ROOT, key)))
        test_paths, train_reals = split_stratified(paths, key, rng)
        rng.shuffle(train_reals)

        tr_dir = os.path.join(TRAIN_DIR, key)
        te_dir = os.path.join(TEST_DIR, key)
        os.makedirs(tr_dir, exist_ok=True)
        os.makedirs(te_dir, exist_ok=True)

        for dst in copy_all(test_paths, te_dir):
            split["test"].append({"image": dst, "label_ko": label, "key": key})
        base = fill_with_augments(key, train_reals, copy_all(train_reals, tr_dir), tr_dir, rng)

        kk = kb[label] if key != NORMAL_KEY else None
        for img_path, is_aug in base:
            for qs, builder in intent_pool(key == NORMAL_KEY):
                ans = builder(label, kk)
                train_samples.append(make_sample(img_path, rng.choice(qs), ans))
            split["train"][key].append({"image": img_path, "aug": is_aug})

        n_aug = TARGET - len(train_reals) if TARGET > len(train_reals) else 0
        report.append((label, len(paths), len(test_paths), len(train_reals), n_aug, len(base)))

    rng.shuffle(train_samples)
    write_json(train_samples, os.path.join(BALANCED_DIR, "train.json"))
    with open(os.path.join(BALANCED_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
        for r in split["test"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json({"train": dict(split["train"]), "test": split["test"]}, os.path.join(BALANCED_DIR, "split.json"))
    write_json(DATASET_INFO, os.path.join(BALANCED_DIR, "dataset_info.json"))

    print(f"{'클래스':<14s}{'실고유':>6s}{'테스트':>6s}{'train실':>8s}{'증강':>6s}{'train계':>8s}")
    print("-" * 50)
    for lab, n, nt, ntr, naug, nbase in report:
        print(f"{lab:<14s}{n:>6d}{nt:>6d}{ntr:>8d}{naug:>6d}{nbase:>8d}")
    print("-" * 50)
    print(f"  학습 샘플 총 {len(train_samples)}개 · 테스트(실사진) {len(split['test'])}장")
    print(f"  출력: {BALANCED_DIR}")


if __name__ == "__main__":
    main()
