import os
import sys
import csv
import json
import re
import time
import base64
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from eval_utils import compute_metrics, plot_confusion_matrix, print_metrics

# ── 설정 ──
MODEL_NAME = "qwen2.5vl:7b"
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "result_md")
TEST_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "data", "legacy_test_images")
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "data", "external", "mendeley_citrus")

CLASS_NAMES = ["normal", "canker", "red_mite", "aphid"]

# test_images 파일명 → GT 라벨 매핑
LOCAL_GT_MAP = {
    "열매_정상.jpg": "normal",
    "잎_정상.jpg": "normal",
    "열매_궤양병.jpg": "canker",
    "잎_궤양병.jpg": "canker",
    "잎_귤응애.jpg": "red_mite",
    "잎_진딧물.jpg": "aphid",
}

# 시스템 프롬프트
SYSTEM_PROMPT = "당신은 감귤 병해충 전문가입니다. 사용자가 제공하는 감귤 이미지를 분석하여 병해충을 진단하고, 증상 설명과 방제법을 제공합니다."

# 사용자 프롬프트
USER_PROMPT = """이 감귤 이미지를 분석해주세요.

다음 카테고리 중 하나로 분류하고, JSON 형식으로 답변하세요:
- normal: 정상 (병해충 없음)
- canker: 궤양병 (citrus canker)
- red_mite: 귤응애 (citrus red mite)
- aphid: 진딧물 (aphid infestation)

응답 형식:
{
  "classification": "카테고리명",
  "confidence": "high/medium/low",
  "symptoms": "관찰된 증상 설명",
  "treatment": "추천 방제법"
}"""

# Mendeley에서 사용할 샘플 수 (클래스당)
MENDELEY_SAMPLE_PER_CLASS = 10


def collect_test_images():
    """test_images/ 6장 + Mendeley 샘플 수집"""
    images = []  # (path, gt_label, source)
    
    # 1) 로컬 test_images (4클래스)
    for fname, gt in LOCAL_GT_MAP.items():
        fpath = os.path.join(TEST_IMAGES_DIR, fname)
        if os.path.exists(fpath):
            images.append((fpath, gt, "local"))
    
    # 2) Mendeley Healthy/Canker 샘플
    for part in ["Fruits", "Leaves"]:
        for folder, gt in [("healthy", "normal"), ("Canker", "canker"), ("canker", "canker")]:
            folder_path = os.path.join(DATASET_ROOT, part, folder)
            if not os.path.isdir(folder_path):
                continue
            files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            for f in files[:MENDELEY_SAMPLE_PER_CLASS]:
                images.append((os.path.join(folder_path, f), gt, f"mendeley_{part}"))
    
    return images


def parse_vlm_response(text):
    """VLM 응답에서 분류 결과 파싱"""
    result = {
        "classification": "unknown",
        "confidence": "unknown",
        "symptoms": "",
        "treatment": "",
    }
    
    # 1) JSON 파싱 시도
    # JSON 블록 추출 (```json ... ``` 또는 { ... })
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{[^{}]*\"classification\"[^{}]*\}", text, re.DOTALL)
    
    if json_match:
        try:
            parsed = json.loads(json_match.group(1) if "```" in json_match.group(0) else json_match.group(0))
            result["classification"] = parsed.get("classification", "unknown").strip().lower()
            result["confidence"] = parsed.get("confidence", "unknown")
            result["symptoms"] = parsed.get("symptoms", "")
            result["treatment"] = parsed.get("treatment", "")
            return result, True
        except json.JSONDecodeError:
            pass
    
    # 2) 텍스트 키워드 매칭 fallback
    text_lower = text.lower()
    for keyword in CLASS_NAMES:
        if keyword in text_lower:
            result["classification"] = keyword
            break
    
    # 증상/방제 텍스트 추출 시도
    symptoms_match = re.search(r"(?:symptoms|증상)[\":\s]*([^\n\"]+)", text, re.IGNORECASE)
    if symptoms_match:
        result["symptoms"] = symptoms_match.group(1).strip()
    
    treatment_match = re.search(r"(?:treatment|방제)[\":\s]*([^\n\"]+)", text, re.IGNORECASE)
    if treatment_match:
        result["treatment"] = treatment_match.group(1).strip()
    
    return result, False


def _encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime(path):
    ext = os.path.splitext(path)[1].lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")


def run_test():
    print(f"[경로 B] VLM 테스트 시작")
    print(f"모델: {MODEL_NAME} (Ollama)")
    print(f"시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 이미지 수집
    images = collect_test_images()
    if not images:
        print("테스트 이미지가 없습니다.")
        sys.exit(1)
    
    print(f"수집 이미지: {len(images)}장")
    for cls in CLASS_NAMES:
        n = sum(1 for _, gt, _ in images if gt == cls)
        print(f"  {cls}: {n}장")
    print()
    
    # Ollama LLM
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.2,
        num_ctx=4096,
        repeat_penalty=1.2,
    )
    print(f"Ollama 모델 준비 완료: {MODEL_NAME}\n")
    
    # 추론
    y_true = []
    y_pred = []
    detail_rows = []
    json_success = 0
    
    start = time.time()
    for i, (path, gt, source) in enumerate(images):
        img_b64 = _encode_image_base64(path)
        mime = _get_mime(path)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": SYSTEM_PROMPT + "\n\n" + USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            ]
        )
        
        try:
            response_obj = llm.invoke([message])
            response = re.sub(r"<(?:thought|think)>.*?</(?:thought|think)>", "", response_obj.content, flags=re.DOTALL).strip()
        except Exception as e:
            response = f"ERROR: {e}"
        
        # 파싱
        parsed, is_json = parse_vlm_response(response)
        pred = parsed["classification"]
        if pred not in CLASS_NAMES:
            pred = "unknown"
        
        if is_json:
            json_success += 1
        
        y_true.append(gt)
        y_pred.append(pred)
        
        detail_rows.append({
            "image": os.path.basename(path),
            "source": source,
            "gt": gt,
            "pred": pred,
            "confidence": parsed["confidence"],
            "symptoms": parsed["symptoms"][:200],
            "treatment": parsed["treatment"][:200],
            "json_parsed": "" if is_json else "",
            "correct": "" if pred == gt else "",
            "raw_response": response[:500],
        })
        
        status = "" if pred == gt else ""
        print(f"  [{i+1}/{len(images)}] {os.path.basename(path):20s} GT={gt:10s} PRED={pred:10s} {status}")
    
    elapsed = time.time() - start
    print(f"\n추론 완료: {elapsed:.1f}초 ({elapsed/len(images):.1f}초/장)")
    print(f"JSON 파싱 성공률: {json_success}/{len(images)} ({json_success/len(images)*100:.1f}%)")
    
    # 평가
    eval_names = CLASS_NAMES + (["unknown"] if "unknown" in y_pred else [])
    metrics = compute_metrics(y_true, y_pred, eval_names)
    print_metrics(metrics, MODEL_NAME, "Local 6장 + Mendeley 샘플", len(images))
    
    # Confusion Matrix
    os.makedirs(RESULT_DIR, exist_ok=True)
    cm_path = os.path.join(RESULT_DIR, "vlm_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, eval_names, cm_path,
                          title=f"VLM: Qwen2.5-VL-7B")
    
    # 분류 결과 CSV
    csv_path = os.path.join(RESULT_DIR, "vlm_classification_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "source", "gt", "pred", "confidence", "correct", "json_parsed"
        ])
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"  분류 CSV 저장: {csv_path}")
    
    # 정성 평가 CSV (증상/방제 포함)
    detail_csv_path = os.path.join(RESULT_DIR, "vlm_detail_results.csv")
    with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "source", "gt", "pred", "confidence", "symptoms", "treatment", "json_parsed", "correct"
        ])
        writer.writeheader()
        for row in detail_rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"  정성 평가 CSV 저장: {detail_csv_path}")
    
    return metrics, detail_rows, elapsed, json_success


if __name__ == "__main__":
    run_test()
