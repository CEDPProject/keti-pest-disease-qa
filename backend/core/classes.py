import os

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CORE_DIR)
REPO_DIR = os.path.dirname(BACKEND_DIR)

DATA_ROOT = os.path.join(REPO_DIR, "data")
RAW_DIR = os.path.join(DATA_ROOT, "raw")
RAW_IMAGE_DIR = os.path.join(RAW_DIR, "감귤_병해충_사진")
NORMAL_IMAGE_DIR = os.path.join(RAW_DIR, "감귤_정상_사진")
NCPMS_IMAGE_DIR = os.path.join(RAW_DIR, "ncpms_수집")
MENDELEY_DIR = os.path.join(DATA_ROOT, "external", "mendeley_citrus")

DATA_DIR = os.path.join(DATA_ROOT, "vlm_finetune")
CLEAN_IMAGE_DIR = os.path.join(DATA_DIR, "images")
AUG_IMAGE_DIR = os.path.join(DATA_DIR, "images_aug")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")
SPLIT_JSON = os.path.join(DATA_DIR, "split.json")
TEST_JSONL = os.path.join(DATA_DIR, "test.jsonl")

BALANCED_DIR = os.path.join(DATA_ROOT, "vlm_finetune_balanced")
BALANCED_TEST_JSONL = os.path.join(BALANCED_DIR, "test.jsonl")

MODELS_DIR = os.path.join(REPO_DIR, "models")
PLATFORM_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MERGED_MODEL_DIR = os.path.join(MODELS_DIR, "qwen2_5vl_merged")
PAPER_LORA_DIR = os.path.join(MODELS_DIR, "qwen3_5_9b_balanced_lora")

KB_PATH = os.path.join(CORE_DIR, "knowledge", "citrus_PD_ver3.md")
KB_CACHE_DIR = os.path.join(MODELS_DIR, "kb_cache")

RESULTS_DIR = os.path.join(REPO_DIR, "qa_docs", "experiment_results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

CLASSES = {
    "궤양병":       {"key": "canker",      "kb_section": "궤양병"},
    "더뎅이병":     {"key": "scab",        "kb_section": "더뎅이병"},
    "잿빛곰팡이병": {"key": "gray_mold",   "kb_section": "잿빛곰팡이병"},
    "수지병":       {"key": "gummosis",    "kb_section": "수지병"},
    "검은점무늬병": {"key": "black_spot",  "kb_section": "검은점무늬병"},
    "녹색곰팡이병": {"key": "green_mold",  "kb_section": "녹색곰팡이병ㆍ푸른곰팡이병"},
    "탄저병":       {"key": "anthracnose", "kb_section": "탄저병"},
}
NORMAL_LABEL = "정상"
NORMAL_KEY = "normal"

DISEASE_KEYS = [info["key"] for info in CLASSES.values()]
CLASS_KEYS = DISEASE_KEYS + [NORMAL_KEY]
LABEL_TO_KEY = {label: info["key"] for label, info in CLASSES.items()} | {NORMAL_LABEL: NORMAL_KEY}
KEY_TO_LABEL = {key: label for label, key in LABEL_TO_KEY.items()}
KEY_TO_SECTION = {info["key"]: info["kb_section"] for info in CLASSES.values()}
SECTION_TO_KEY = {section: key for key, section in KEY_TO_SECTION.items()}

DRAFT_KB_LABELS = {"탄저병"}
