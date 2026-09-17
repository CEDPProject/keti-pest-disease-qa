import os
import sys
from datetime import datetime

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TEST_IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "data", "legacy_test_images")
RESULT_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "result_md")
RESULT_FILE = os.path.join(RESULT_DIR, "vlm_qa_results.md")

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".webp")

# 테스트 질문 리스트
TEST_QUESTIONS = [
    "이 감귤 상태가 이상한가요?",
    "이 감귤에 병해충 증상이 보이나요?",
    "이 감귤에 어떤 색의 점이나 반점이 보이나요?",
    "이 감귤은 정상인가요, 병든 상태인가요?",
    "의심되는 병명이 있다면 무엇인가요?",
    "왜 그렇게 판단했나요?",
    "어떻게 관리하거나 방제해야 하나요?",
    "추가로 확인해야 할 점은 무엇인가요?",
]


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


def run_vlm_tests_for_all_images():
    """Run all predefined test questions for every image in image_qa/test_images."""
    print("=" * 60)
    print("VLM 기반 이미지-질문 QA 테스트")
    print("=" * 60)
    
    # 1. 이미지 수집
    print("\n[1단계] 테스트 이미지 수집")
    image_paths = collect_test_images()
    
    if not image_paths:
        print("테스트 이미지가 없습니다. test_images/ 폴더에 감귤 이미지를 넣어주세요.")
        sys.exit(1)
    
    total_tests = len(image_paths) * len(TEST_QUESTIONS)
    print(f"\n총 {len(image_paths)}개 이미지 × {len(TEST_QUESTIONS)}개 질문 = {total_tests}개 테스트\n")
    
    # 2. VLM 파이프라인 실행
    print("[2단계] VLM 파이프라인 실행")
    from vlm_qa import vlm_qa_pipeline
    
    all_results = []  # [(image_path, [(question, result_dict), ...]), ...]
    
    for img_idx, img_path in enumerate(image_paths, 1):
        fname = os.path.basename(img_path)
        print(f"\n{'#'*60}")
        print(f"# 이미지 {img_idx}/{len(image_paths)}: {fname}")
        print(f"{'#'*60}")
        
        image_results = []
        for q_idx, question in enumerate(TEST_QUESTIONS, 1):
            test_num = (img_idx - 1) * len(TEST_QUESTIONS) + q_idx
            print(f"\n[{test_num}/{total_tests}] 질문 {q_idx}: {question}")
            
            try:
                result = vlm_qa_pipeline(img_path, question)
                result["status"] = "SUCCESS"
            except Exception as e:
                print(f"오류: {e}")
                result = {
                    "image_path": img_path,
                    "user_question": question,
                    "vlm_response": f"오류: {e}",
                    "rag_used": False,
                    "final_answer": f"오류 발생: {e}",
                    "status": "FAILED",
                }
            
            image_results.append((question, result))
        
        all_results.append((img_path, image_results))
    
    # 3. 결과 요약
    print_summary(all_results)
    
    # 4. Markdown 저장
    save_results_md(all_results)
    
    return all_results


def print_summary(all_results):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    total = 0
    success = 0
    for img_path, image_results in all_results:
        for question, result in image_results:
            total += 1
            if result["status"] == "SUCCESS":
                success += 1
    
    print(f"  전체: {success}/{total} 성공")
    print(f"  이미지 수: {len(all_results)}")
    print(f"  질문 수/이미지: {len(TEST_QUESTIONS)}")
    
    if success == total:
        print("\n모든 테스트 통과!")
    else:
        print("\n일부 테스트 실패")
        for img_path, image_results in all_results:
            for question, result in image_results:
                if result["status"] == "FAILED":
                    print(f"  - {os.path.basename(img_path)} / {question[:20]}...")


def save_results_md(all_results):
    """결과를 result_md/vlm_qa_results.md에 저장"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = sum(len(ir) for _, ir in all_results)
    success = sum(1 for _, ir in all_results for _, r in ir if r["status"] == "SUCCESS")
    
    lines = []
    lines.append("# VLM 기반 이미지-질문 QA 테스트 결과\n")
    lines.append("## 테스트 설정\n")
    lines.append(f"- 테스트 일시: {now}")
    lines.append("- 방식: VLM 기반 이미지-질문 QA (이미지 + 질문을 VLM에 직접 입력)")
    lines.append("- VLM 백엔드: `gemma3:12b` (Ollama, 멀티모달)")
    lines.append(f"- 테스트 이미지 폴더: `image_qa/test_images/` ({len(all_results)}장)")
    lines.append(f"- 테스트 질문 수: {len(TEST_QUESTIONS)}개/이미지")
    lines.append(f"- 총 테스트: {total}개")
    lines.append(f"- 성공: {success}/{total}\n")
    
    lines.append("## 테스트 질문 목록\n")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        lines.append(f"{i}. {q}")
    lines.append("")
    
    lines.append("---\n")
    
    for img_idx, (img_path, image_results) in enumerate(all_results, 1):
        fname = os.path.basename(img_path)
        lines.append(f"## 이미지 {img_idx}: `{fname}`\n")
        
        for q_idx, (question, result) in enumerate(image_results, 1):
            lines.append(f"### 질문 {q_idx}\n")
            lines.append(f"**질문:** {question}\n")
            lines.append(f"**상태:** {'성공' if result['status'] == 'SUCCESS' else '실패'}\n")
            lines.append(f"**VLM 응답:**\n")
            lines.append(f"```")
            lines.append(result["final_answer"])
            lines.append(f"```\n")
            lines.append("---\n")
    
    content = "\n".join(lines)
    
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n결과 저장 완료: {RESULT_FILE}")


if __name__ == "__main__":
    run_vlm_tests_for_all_images()
