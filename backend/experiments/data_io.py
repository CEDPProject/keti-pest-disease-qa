import os
import json
import random

import numpy as np
from PIL import Image

from backend.core.classes import MENDELEY_DIR, TEST_JSONL

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CROSS_SETS = [(os.path.join(MENDELEY_DIR, "Fruits", "Canker"), "canker", "궤양병", 30),
              (os.path.join(MENDELEY_DIR, "Fruits", "Scab"), "scab", "더뎅이병", 15),
              (os.path.join(MENDELEY_DIR, "Leaves", "Melanose"), "black_spot", "검은점무늬병", 13)]


# JSON 파일을 읽는다
def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# 한 줄에 레코드 하나씩 적힌 JSONL 파일을 읽는다
def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# 한글을 그대로 둔 JSON 파일로 저장한다
def write_json(obj, path: str, indent=2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


# 폴더 안 이미지 파일 경로를 이름순으로 돌려준다
def list_images(folder: str, exts=IMG_EXT) -> list:
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts))


# 이미지의 dHash 비트 배열을 계산한다
def dhash(path: str, size: int = 8):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


# 플랫폼 테스트셋과 Mendeley 교차 검증 사진을 묶는다
def cross_eval_items() -> list:
    items = [{"set": "ours", "image": r["image"], "key": r["key"], "label_ko": r["label_ko"]}
             for r in read_jsonl(TEST_JSONL)]
    rng = random.Random(42)
    for folder, key, ko, n in CROSS_SETS:
        files = list_images(folder, exts=(".jpg", ".jpeg", ".png"))
        rng.shuffle(files)
        items += [{"set": "mendeley", "image": f, "key": key, "label_ko": ko} for f in files[:n]]
    return items
