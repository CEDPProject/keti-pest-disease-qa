"""3가지 방법 실제 실행 예시 수집 스크립트
PPT에 넣을 입력/출력 예시 + 실행 시간 수집
"""
import os
import sys
import time
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TEXT_QA_DIR = os.path.join(PROJECT_DIR, "text_qa")
TEST_IMAGES_DIR = os.path.join(SCRIPT_DIR, "test_images")

results = {"method1": [], "method2": [], "method3": []}

# ═══════════════════════════════════════════
# 방법 1: 텍스트 RAG
# ═══════════════════════════════════════════
print("\n" + "="*70)
print("📝 방법 1: 텍스트 RAG 테스트")
print("="*70)

sys.path.insert(0, TEXT_QA_DIR)

# 방법1 임포트 전에 image_qa/prompts 충돌 방지: 먼저 text_qa의 pest_disease_qa를 임포트
from pest_disease_qa import PestDiseaseQA

qa = PestDiseaseQA(
    data_path=os.path.join(TEXT_QA_DIR, "text_info", "citrus_PD_ver3.md"),
    persist_directory=os.path.join(TEXT_QA_DIR, "chroma_db"),
)

test_qs = [
    "귤응애 방제방법 알려줘",
    "궤양병의 증상과 치료법은?",
    "잎에 노란 반점이 생겼어요",
]

for q in test_qs:
    print(f"\n--- 질문: {q} ---")
    t0 = time.time()
    try:
        answer, sources = qa.query(q)
        elapsed = time.time() - t0
        # 답변 200자로 자르기
        short_answer = answer[:300].replace("\n", " ").strip()
        if len(answer) > 300:
            short_answer += "..."
        results["method1"].append({
            "input": q,
            "output_preview": short_answer,
            "full_length": len(answer),
            "sources_count": len(sources) if sources else 0,
            "time_sec": round(elapsed, 2),
            "status": "SUCCESS",
        })
        print(f"  ✅ 성공 ({elapsed:.2f}초, {len(answer)}자)")
    except Exception as e:
        elapsed = time.time() - t0
        results["method1"].append({
            "input": q,
            "error": str(e),
            "time_sec": round(elapsed, 2),
            "status": "FAILED",
        })
        print(f"  ❌ 실패: {e}")

# ═══════════════════════════════════════════
# 방법 2: 이미지 분류 + RAG
# ═══════════════════════════════════════════
print("\n" + "="*70)
print("🖼️ 방법 2: 이미지 분류 + RAG 테스트")
print("="*70)

# sys.path 순서 중요: image_qa를 text_qa 앞에 넣어서 prompts 충돌 방지
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if TEXT_QA_DIR in sys.path:
    sys.path.remove(TEXT_QA_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.append(TEXT_QA_DIR)

# prompts 캐시 제거 (text_qa의 prompts와 충돌 방지)
if 'prompts' in sys.modules:
    del sys.modules['prompts']

from image_qa import ImagePestDiseaseQA

img_qa = ImagePestDiseaseQA(device="cpu")  # CPU 강제 (cuDNN 이슈)

test_images_m2 = [
    (os.path.join(TEST_IMAGES_DIR, "열매_궤양병.jpg"), "궤양병"),
    (os.path.join(TEST_IMAGES_DIR, "열매_정상.jpg"), "정상"),
    (os.path.join(TEST_IMAGES_DIR, "잎_궤양병.jpg"), "궤양병"),
]

for img_path, gt_label in test_images_m2:
    fname = os.path.basename(img_path)
    print(f"\n--- 이미지: {fname} (GT: {gt_label}) ---")
    t0 = time.time()
    try:
        result = img_qa.query(img_path)
        elapsed = time.time() - t0
        answer_preview = result["final_answer"][:250].replace("\n", " ").strip()
        if len(result["final_answer"]) > 250:
            answer_preview += "..."
        results["method2"].append({
            "input": fname,
            "gt_label": gt_label,
            "predicted_label": result["predicted_label"],
            "confidence": round(result["confidence"], 4),
            "is_healthy": result["is_healthy"],
            "rag_used": result["rag_used"],
            "answer_preview": answer_preview,
            "time_sec": round(elapsed, 2),
            "status": "SUCCESS",
            "correct": (gt_label == "정상" and result["is_healthy"]) or
                       (gt_label != "정상" and not result["is_healthy"]),
        })
        print(f"  ✅ 예측: {result['predicted_label']} ({result['confidence']*100:.1f}%) | {elapsed:.2f}초")
    except Exception as e:
        elapsed = time.time() - t0
        results["method2"].append({
            "input": fname,
            "gt_label": gt_label,
            "error": str(e),
            "time_sec": round(elapsed, 2),
            "status": "FAILED",
        })
        print(f"  ❌ 실패: {e}")

# ═══════════════════════════════════════════
# 방법 3: VLM + RAG
# ═══════════════════════════════════════════
print("\n" + "="*70)
print("🧠 방법 3: VLM + RAG 테스트")
print("="*70)

from vlm_qa import vlm_qa_pipeline

test_images_m3 = [
    (os.path.join(TEST_IMAGES_DIR, "열매_궤양병.jpg"), "이 감귤에 어떤 문제가 있나요?", "궤양병"),
    (os.path.join(TEST_IMAGES_DIR, "잎_정상.jpg"), "이 잎이 건강한가요?", "정상"),
    (os.path.join(TEST_IMAGES_DIR, "잎_귤응애.jpg"), "이 잎에 어떤 병해충이 있나요?", "귤응애"),
]

for img_path, question, gt_label in test_images_m3:
    fname = os.path.basename(img_path)
    print(f"\n--- 이미지: {fname} | 질문: {question} ---")
    t0 = time.time()
    try:
        result = vlm_qa_pipeline(img_path, question)
        elapsed = time.time() - t0
        vlm_answer = result["final_answer"][:300].replace("\n", " ").strip()
        if len(result["final_answer"]) > 300:
            vlm_answer += "..."
        results["method3"].append({
            "input_image": fname,
            "input_question": question,
            "gt_label": gt_label,
            "answer_preview": vlm_answer,
            "time_sec": round(elapsed, 2),
            "status": "SUCCESS",
        })
        print(f"  ✅ 완료 ({elapsed:.2f}초)")
    except Exception as e:
        elapsed = time.time() - t0
        results["method3"].append({
            "input_image": fname,
            "input_question": question,
            "gt_label": gt_label,
            "error": str(e),
            "time_sec": round(elapsed, 2),
            "status": "FAILED",
        })
        print(f"  ❌ 실패: {e}")

# ═══════════════════════════════════════════
# 결과 저장
# ═══════════════════════════════════════════
output_path = os.path.join(PROJECT_DIR, "result_md", "ppt_examples.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*70)
print(f"📊 결과 저장: {output_path}")
print("="*70)

# 요약 출력
for method, data in results.items():
    print(f"\n【{method}】")
    for item in data:
        status = item.get("status", "?")
        t = item.get("time_sec", 0)
        inp = item.get("input", item.get("input_image", "?"))
        print(f"  {status} | {inp} | {t}초")
