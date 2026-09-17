import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from vlm_prompt_builder import build_vlm_prompt
from vlm_inference import generate_vlm_answer

# text_qa 경로 (RAG 보강용)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_QA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "text_qa")


def vlm_qa_pipeline(image_path: str, user_question: str, use_rag: bool = False) -> dict:
    """Run VLM-based image question answering.
    
    Args:
        image_path: 이미지 파일 경로
        user_question: 사용자 질문
        use_rag: RAG 보강 사용 여부 (기본 False)
        
    Returns:
        dict: {image_path, user_question, vlm_response, rag_used, final_answer}
    """
    print(f"\n{'='*60}")
    print(f"🖼️  VLM 분석: {os.path.basename(image_path)}")
    print(f"❓ 질문: {user_question}")
    print(f"{'='*60}")
    
    # 1단계: VLM 프롬프트 생성
    prompt = build_vlm_prompt(user_question)
    
    # 2단계: VLM 추론
    print("💬 VLM 답변 생성 중...")
    vlm_response = generate_vlm_answer(image_path, prompt)
    
    print(f"\n{vlm_response}")
    print(f"{'='*60}\n")
    
    result = {
        "image_path": image_path,
        "user_question": user_question,
        "vlm_response": vlm_response,
        "rag_used": False,
        "final_answer": vlm_response,
    }
    
    return result


def main():
    """단독 실행 테스트"""
    test_image = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "data", "legacy_test_images", "잎_궤양병.jpg")
    if not os.path.exists(test_image):
        print("❌ 테스트 이미지가 없습니다.")
        sys.exit(1)
    
    result = vlm_qa_pipeline(test_image, "이 감귤 상태가 이상한가요?")
    print(f"\n결과: {result['final_answer'][:200]}...")


if __name__ == "__main__":
    main()
