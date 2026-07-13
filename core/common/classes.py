"""감귤 병해충 사진 8클래스 공용 설정.

- 폴더명(한글) ↔ ASCII 클래스 키 매핑
- 지식베이스(citrus_PD_ver3.md) 섹션명 매핑
- 경로 상수

파인튜닝/평가 전 과정에서 이 파일의 매핑을 단일 소스로 사용한다.
"""

import os

# 이 파일 기준 경로 (core/common/)
COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(COMMON_DIR)
VLM_FT_DIR = CORE_DIR          # 하위호환 별칭 (기존 import 유지)
REPO_DIR = os.path.dirname(CORE_DIR)

# ---- 데이터 루트 (모든 이미지·데이터셋은 repo/data/ 아래로 통합) ----
DATA_ROOT = os.path.join(REPO_DIR, "data")

RAW_IMAGE_DIR = os.path.join(DATA_ROOT, "raw", "감귤_병해충_사진")   # 원본 121장
EXTERNAL_DIR = os.path.join(DATA_ROOT, "external")                  # 외부 공개 데이터셋
MENDELEY_DIR = os.path.join(EXTERNAL_DIR, "mendeley_citrus")        # 교차검증용

# 파인튜닝 파생 데이터
DATA_DIR = os.path.join(DATA_ROOT, "vlm_finetune")
CLEAN_IMAGE_DIR = os.path.join(DATA_DIR, "images")       # 정제 복사본
AUG_IMAGE_DIR = os.path.join(DATA_DIR, "images_aug")     # 증강본
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")
SPLIT_JSON = os.path.join(DATA_DIR, "split.json")
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TEST_JSONL = os.path.join(DATA_DIR, "test.jsonl")

# 학습 산출물 (repo/models/ 로 분리)
MODELS_DIR = os.path.join(REPO_DIR, "models")
LORA_DIR = os.path.join(MODELS_DIR, "qwen2_5vl_lora")
MERGED_MODEL_DIR = os.path.join(MODELS_DIR, "qwen2_5vl_merged")

# 지식베이스 (qa/text_qa/ 로 이동됨)
KB_PATH = os.path.join(REPO_DIR, "qa", "text_qa", "text_info", "citrus_PD_ver3.md")

# 한글 폴더명 → (ASCII 키, KB 섹션명)
# KB 섹션명은 citrus_PD_ver3.md 의 '## ' 헤더와 정확히 일치해야 함.
CLASSES = {
    "궤양병":       {"key": "canker",      "kb_section": "궤양병"},
    "더뎅이병":     {"key": "scab",        "kb_section": "더뎅이병"},
    "잿빛곰팡이병": {"key": "gray_mold",   "kb_section": "잿빛곰팡이병"},
    "수지병":       {"key": "gummosis",    "kb_section": "수지병"},
    "검은점무늬병": {"key": "black_spot",  "kb_section": "검은점무늬병"},
    "녹색곰팡이병": {"key": "green_mold",  "kb_section": "녹색곰팡이병ㆍ푸른곰팡이병"},
    "탄저병":       {"key": "anthracnose", "kb_section": "탄저병"},        # KB 보강 섹션(초안)
    "검은무늬병":   {"key": "black_blotch","kb_section": "검은무늬병"},     # KB 보강 섹션(초안)
}

# 역매핑 (키 → 한글 라벨)
KEY_TO_LABEL = {v["key"]: k for k, v in CLASSES.items()}

# 전문가 검증 필요(초안 KB) 클래스
DRAFT_KB_LABELS = {"탄저병", "검은무늬병"}
