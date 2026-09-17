"""
이미지 분류 + RAG 방식 병해충 Q&A 테스트 스크립트
- test_images/ 폴더의 모든 이미지를 자동 순회
- 건강/정상 판정 시 RAG 호출 없이 고정 답변
- 병해충 판정 시 RAG 연동 답변
- 결과를 result_md/classifier_rag_results.md에 저장
"""

import os
import sys
from datetime import datetime

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TEST_IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "data", "legacy_test_images")
RESULT_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "result_md")
RESULT_FILE = os.path.join(RESULT_DIR, "classifier_rag_results.md")

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".webp")


def collect_test_images():
    """test_images/ 폴더에서 이미지 파일 목록 수집"""
    if not os.path.isdir(TEST_IMAGE_DIR):
        print(f"테스트 이미지 폴더가 없습니다: {TEST_IMAGE_DIR}")
        return []
    
    images = []
    for fname in sorted(os.listdir(TEST_IMAGE_DIR)):
        if fname.lower().endswith(SUPPORTED_EXT):
            images.append(os.path.join(TEST_IMAGE_DIR, fname))
            print(f"  {fname}")
    return images


def run_all_test_images():
    """test_images/ 전체 이미지에 대해 classifier + RAG 파이프라인 실행"""
    print("=" * 60)
    print("🧪 이미지 분류 + RAG 파이프라인 테스트")
    print("=" * 60)
    
    # 1. 이미지 수집
    print("\n[1단계] 테스트 이미지 수집")
    image_paths = collect_test_images()
    
    if not image_paths:
        print("테스트 이미지가 없습니다. test_images/ 폴더에 감귤 이미지를 넣어주세요.")
        sys.exit(1)
    
    print(f"\n총 {len(image_paths)}개 이미지 발견\n")
    
    # 2. 파이프라인 실행
    print("[2단계] 파이프라인 실행")
    from image_qa import classifier_rag_pipeline
    
    results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(image_paths)}] {os.path.basename(img_path)}")
        print(f"{'='*60}")
        
        try:
            result = classifier_rag_pipeline(img_path)
            result["status"] = "SUCCESS"
            results.append(result)
        except Exception as e:
            print(f"오류: {e}")
            results.append({
                "image_path": img_path,
                "predicted_label": "ERROR",
                "confidence": 0.0,
                "is_healthy": False,
                "rag_used": False,
                "final_answer": f"오류 발생: {e}",
                "all_scores": [],
                "status": "FAILED",
            })
    
    # 3. 결과 요약 출력
    print_summary(results)
    
    # 4. Markdown 파일로 저장
    save_results_md(results)
    
    return results


def print_summary(results):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    total = len(results)
    success = sum(1 for r in results if r["status"] == "SUCCESS")
    healthy_count = sum(1 for r in results if r.get("is_healthy"))
    rag_count = sum(1 for r in results if r.get("rag_used"))
    
    print(f"  전체: {success}/{total} 성공")
    print(f"  정상 판정 (RAG 미사용): {healthy_count}개")
    print(f"  병해충 판정 (RAG 사용): {rag_count}개")
    
    if success == total:
        print("\n모든 테스트 통과!")
    else:
        print("\n일부 테스트 실패")
        for r in results:
            if r["status"] == "FAILED":
                print(f"  - {os.path.basename(r['image_path'])}: {r['final_answer']}")


def save_results_md(results):
    """결과를 result_md/classifier_rag_results.md에 저장"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    success = sum(1 for r in results if r["status"] == "SUCCESS")
    healthy_count = sum(1 for r in results if r.get("is_healthy"))
    rag_count = sum(1 for r in results if r.get("rag_used"))
    
    lines = []
    lines.append("# 이미지 분류 + RAG 테스트 결과\n")
    lines.append("## 테스트 설정\n")
    lines.append(f"- 테스트 일시: {now}")
    lines.append("- 방식: 이미지 분류 모델 + RAG")
    lines.append("- 이미지 분류 모델: `dima806/orange_fruit_disease_detection` (ViT)")
    lines.append("- LLM: `gemma3:12B` (Ollama)")
    lines.append("- 임베딩: `intfloat/multilingual-e5-large`")
    lines.append(f"- 테스트 이미지 폴더: `image_qa/test_images/` ({total}장)")
    lines.append("- 정상/건강 판정 시 RAG 사용 여부: **사용하지 않음**\n")
    
    lines.append("## 결과 요약\n")
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|---|---|")
    lines.append(f"| 전체 | {success}/{total} 성공 |")
    lines.append(f"| 정상 판정 (RAG 미사용) | {healthy_count}개 |")
    lines.append(f"| 병해충 판정 (RAG 사용) | {rag_count}개 |\n")
    
    lines.append("---\n")
    
    for i, r in enumerate(results, 1):
        fname = os.path.basename(r["image_path"])
        lines.append(f"## 결과 {i}\n")
        lines.append(f"### 입력 이미지\n")
        lines.append(f"- 파일명: `{fname}`\n")
        
        lines.append(f"### 이미지 분류 결과\n")
        lines.append(f"- 예측 라벨: **{r['predicted_label']}**")
        lines.append(f"- 신뢰도: {r['confidence']*100:.1f}%")
        lines.append(f"- 정상 여부: {'정상' if r['is_healthy'] else '병해충'}")
        lines.append(f"- RAG 사용 여부: {'사용' if r['rag_used'] else '미사용'}")
        
        if r.get("all_scores"):
            lines.append(f"\n| 라벨 | 신뢰도 |")
            lines.append(f"|---|---|")
            for s in r["all_scores"]:
                lines.append(f"| {s['label_ko']} ({s['label']}) | {s['score']*100:.1f}% |")
        
        lines.append(f"\n### 최종 답변\n")
        lines.append(f"```")
        lines.append(r["final_answer"])
        lines.append(f"```\n")
        lines.append("---\n")
    
    content = "\n".join(lines)
    
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n📄 결과 저장 완료: {RESULT_FILE}")


if __name__ == "__main__":
    run_all_test_images()
