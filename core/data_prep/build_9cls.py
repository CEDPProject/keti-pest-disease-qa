"""9클래스 데이터셋 = 기존 8클래스(질병) + 정상(healthy) 추가.

기존 8클래스 학습/평가 데이터(data/vlm_finetune/train.json, test.jsonl)는 그대로 두고,
정상 40장(잎 20 + 열매 20)을 stratified split → 정상 QA 샘플 생성 → 병합하여
data/vlm_finetune_9cls/ 에 9클래스 train.json + test.jsonl + dataset_info.json 을 만든다.

  정상: KB 없음 → 간단한 '정상/건강' 답변(**정상** 굵게, label_extract가 normal 로 추출).
  train 이미지당 원본→진단(full) + 증강 2장→병명(name)/증상(symptom).  (정상 과대표집 방지)
"""
import os
import sys
import json
import glob
import random
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from PIL import Image, ImageOps, ImageEnhance
from classes import REPO_DIR, DATA_DIR  # noqa

SEED = 42
NORMAL_RAW = os.path.join(REPO_DIR, "data", "raw", "감귤_정상_사진")
PARTS = [("잎_정상_사진", "leaf"), ("열매_정상_사진", "fruit")]
N_TRAIN_PER_PART = 15   # 잎15+열매15=30 train
# 나머지(각 5) → test

OUT = os.path.join(REPO_DIR, "data", "vlm_finetune_9cls")
NORMAL_IMG = os.path.join(OUT, "normal_images")
NORMAL_AUG = os.path.join(OUT, "normal_aug")

Q_FULL = ["이 감귤 사진을 보고 어떤 병해충인지 진단하고, 관찰되는 증상과 방제 방법을 알려주세요.",
          "이 감귤에 어떤 병이 의심되나요? 증상과 관리 방법도 함께 설명해주세요.",
          "사진 속 감귤의 상태를 진단해주세요. 병명, 증상, 방제법을 알려주세요.",
          "이 감귤 이미지에서 보이는 병해충 증상을 설명하고, 의심 병명과 방제 방법을 알려주세요."]
Q_NAME = ["이 감귤은 무슨 병인가요?", "이 사진의 병명만 알려주세요.", "이거 무슨 병이야?"]
Q_SYMPTOM = ["이 감귤에 어떤 증상이 보이나요?", "사진에서 관찰되는 증상을 설명해주세요.", "어떤 병징이 나타나 있나요?"]
EVAL_QUESTION = Q_FULL[0]


def a_full():
    return ("사진 속 감귤에서 뚜렷한 병징(반점·궤양·딱지·곰팡이 등)이 관찰되지 않습니다. "
            "**정상**(건강)으로 보입니다.\n\n"
            "## 관찰 소견\n- 과피 또는 잎 표면에 특이 병반이 확인되지 않음\n- 색·형태가 정상 범위\n\n"
            "## 권장\n- 현재 특별한 방제는 필요하지 않으며, 예방적 과원 위생·통풍 관리를 지속하세요.\n"
            "- 이미지만으로 완전 확진은 어려우니 이상 증상 발견 시 재확인하세요.")


def a_name():
    return "이 감귤은 **정상**(건강)으로 보입니다. 뚜렷한 병징이 관찰되지 않습니다."


def a_symptom():
    return "뚜렷한 병징이 관찰되지 않습니다. 과피·잎이 **정상**(건강) 상태로 보입니다."


def augment(img, rng):
    im = img
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(rng.uniform(-12, 12), resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    w, h = im.size
    scale = rng.uniform(0.85, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x0, y0, x0 + cw, y0 + ch))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.85, 1.15))
    w, h = im.size
    if max(w, h) > 1024:
        r = 1024 / max(w, h)
        im = im.resize((int(w * r), int(h * r)), Image.LANCZOS)
    return im


def sample(image_path, question, answer):
    return {"messages": [{"role": "user", "content": "<image>" + question},
                         {"role": "assistant", "content": answer}], "images": [image_path]}


def main():
    os.makedirs(NORMAL_IMG, exist_ok=True)
    os.makedirs(NORMAL_AUG, exist_ok=True)
    rng = random.Random(SEED)

    # 1) 정상 이미지 정제 + stratified split(part별)
    train_imgs, test_imgs = [], []
    idx = 0
    for sub, part in PARTS:
        files = sorted(f for f in glob.glob(os.path.join(NORMAL_RAW, sub, "*"))
                       if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"))
        clean = []
        for src in files:
            try:
                im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            except Exception:
                continue
            idx += 1
            dst = os.path.join(NORMAL_IMG, f"normal_{idx:03d}.jpg")
            im.save(dst, "JPEG", quality=95)
            clean.append(os.path.abspath(dst))
        rng.shuffle(clean)
        train_imgs += clean[:N_TRAIN_PER_PART]
        test_imgs += clean[N_TRAIN_PER_PART:]
        print(f"  정상[{part}]: 정제 {len(clean)}장 → train {min(N_TRAIN_PER_PART,len(clean))} / test {max(0,len(clean)-N_TRAIN_PER_PART)}")

    # 2) 정상 train 샘플: 원본→full, 증강2→name/symptom
    normal_train = []
    for src in train_imgs:
        normal_train.append(sample(src, rng.choice(Q_FULL), a_full()))
        img = Image.open(src).convert("RGB")
        base = os.path.splitext(os.path.basename(src))[0]
        for a, (qs, builder) in enumerate([(Q_NAME, a_name), (Q_SYMPTOM, a_symptom)], 1):
            ap = os.path.join(NORMAL_AUG, f"{base}_aug{a}.jpg")
            augment(img, rng).save(ap, "JPEG", quality=92)
            normal_train.append(sample(os.path.abspath(ap), rng.choice(qs), builder()))

    # 3) 기존 8클래스 train/test 로드 후 병합
    disease_train = json.load(open(os.path.join(DATA_DIR, "train.json"), encoding="utf-8"))
    train = disease_train + normal_train
    rng.shuffle(train)
    json.dump(train, open(os.path.join(OUT, "train.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "test.jsonl"), "w", encoding="utf-8") as f:
        for line in open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8"):
            f.write(line if line.endswith("\n") else line + "\n")
        for src in test_imgs:
            f.write(json.dumps({"image": src, "label_ko": "정상", "key": "normal",
                                "question": EVAL_QUESTION, "gold_answer": a_full()}, ensure_ascii=False) + "\n")

    # 4) LLaMA-Factory dataset_info
    info = {"citrus_ft_9cls": {"file_name": "train.json", "formatting": "sharegpt",
                               "columns": {"messages": "messages", "images": "images"},
                               "tags": {"role_tag": "role", "content_tag": "content",
                                        "user_tag": "user", "assistant_tag": "assistant"}}}
    json.dump(info, open(os.path.join(OUT, "dataset_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    n_test_disease = sum(1 for _ in open(os.path.join(DATA_DIR, "test.jsonl"), encoding="utf-8"))
    print(f"\n9클래스 완성 → {OUT}")
    print(f"  train {len(train)}개 (질병 {len(disease_train)} + 정상 {len(normal_train)})")
    print(f"  test  {n_test_disease + len(test_imgs)}개 (질병 {n_test_disease} + 정상 {len(test_imgs)})")


if __name__ == "__main__":
    main()
