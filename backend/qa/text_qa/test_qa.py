import os
import sys

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 테스트 질문 목록
TEST_QUESTIONS = [
    "검은색 벌레가 잎사귀 위에 있어",   # 모호한 질문 (여러 후보 예상)
    "열매에 검은 점이 보인다",
    "잎에 노란 반점이 생겼어요",
    "귤응애 방제방법 알려줘",           # 특정 병해충 지정
    "궤양병의 증상과 치료법은?"          # 특정 병해 지정
]


def run_test():
    """테스트 실행"""
    print("="*60)
    print("병해충 Q&A 시스템 테스트")
    print("="*60)
    
    # 현재 스크립트 경로 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    
    from pest_disease_qa import PestDiseaseQA
    
    # Q&A 시스템 초기화
    data_path = os.path.join(script_dir, "text_info", "citrus_PD_ver3.md")
    persist_dir = os.path.join(script_dir, "chroma_db")
    
    qa_system = PestDiseaseQA(
        data_path=data_path,
        persist_directory=persist_dir
    )
    
    # 테스트 질문 실행
    results = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {i}/{len(TEST_QUESTIONS)}")
        print(f"{'='*60}")
        
        try:
            answer, sources = qa_system.query(question)
            results.append({
                "question": question,
                "answer": answer[:500] + "..." if len(answer) > 500 else answer,
                "status": "SUCCESS"
            })
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e),
                "status": "FAILED"
            })
    
    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    fail_count = len(results) - success_count
    
    print(f"총 테스트: {len(results)}")
    print(f"성공: {success_count}")
    print(f"실패: {fail_count}")
    
    if fail_count > 0:
        print("\n실패한 테스트:")
        for r in results:
            if r["status"] == "FAILED":
                print(f"  - 질문: {r['question']}")
                print(f"    오류: {r.get('error', 'Unknown')}")
    
    return results


if __name__ == "__main__":
    run_test()
