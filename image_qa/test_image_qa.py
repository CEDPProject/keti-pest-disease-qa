"""
이미지 기반 병해충 Q&A 시스템 테스트 스크립트
1. 테스트 이미지 다운로드 (HuggingFace datasets에서)
2. 이미지 분류기 단독 테스트
3. 이미지 + RAG 연동 테스트
"""

import os
import sys
import json
import urllib.request

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_DIR = os.path.join(SCRIPT_DIR, "test_images")


def download_test_images():
    """테스트용 감귤 이미지 다운로드 (HuggingFace에서)"""
    os.makedirs(TEST_IMAGE_DIR, exist_ok=True)
    
    # 공개 감귤 병해충 이미지 URL (Wikimedia Commons - 자유 이미지)
    test_images = {
        "citrus_canker_test.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Citrus_canker_leaf.jpg/640px-Citrus_canker_leaf.jpg",
        "healthy_orange_test.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Oranges_-_whole-halved-segment.jpg/640px-Oranges_-_whole-halved-segment.jpg",
    }
    
    downloaded = []
    for filename, url in test_images.items():
        filepath = os.path.join(TEST_IMAGE_DIR, filename)
        if os.path.exists(filepath):
            print(f"  ✅ 이미 존재: {filename}")
            downloaded.append(filepath)
            continue
        
        try:
            print(f"  ⬇️  다운로드 중: {filename}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(filepath, "wb") as f:
                    f.write(resp.read())
            print(f"  ✅ 다운로드 완료: {filename}")
            downloaded.append(filepath)
        except Exception as e:
            print(f"  ❌ 다운로드 실패: {filename} - {e}")
    
    return downloaded


def test_classifier_only(image_paths):
    """이미지 분류기 단독 테스트"""
    print("\n" + "=" * 60)
    print("🧪 테스트 1: 이미지 분류기 단독 테스트")
    print("=" * 60)
    
    from image_classifier import CitrusImageClassifier
    
    classifier = CitrusImageClassifier()
    
    results = []
    for img_path in image_paths:
        print(f"\n📷 이미지: {os.path.basename(img_path)}")
        try:
            label_ko, score, all_results = classifier.classify_with_summary(img_path)
            print(f"   판별: {label_ko} (신뢰도: {score*100:.1f}%)")
            for r in all_results:
                bar = "█" * int(r["score"] * 20)
                print(f"   {r['label_ko']:15s} {r['score']*100:5.1f}% {bar}")
            results.append({"image": os.path.basename(img_path), "label": label_ko, "score": score, "status": "SUCCESS"})
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results.append({"image": os.path.basename(img_path), "error": str(e), "status": "FAILED"})
    
    return results


def test_image_qa(image_paths):
    """이미지 + RAG 연동 테스트"""
    print("\n" + "=" * 60)
    print("🧪 테스트 2: 이미지 + RAG 연동 테스트")
    print("=" * 60)
    
    from image_qa import ImagePestDiseaseQA
    
    qa_system = ImagePestDiseaseQA()
    
    results = []
    
    # 테스트 케이스
    test_cases = []
    for img_path in image_paths:
        test_cases.append({"image": img_path, "question": ""})
    
    # 추가 질문 테스트
    if image_paths:
        test_cases.append({"image": image_paths[0], "question": "방제방법만 알려줘"})
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {i}/{len(test_cases)}")
        print(f"이미지: {os.path.basename(case['image'])}")
        if case["question"]:
            print(f"추가 질문: {case['question']}")
        print(f"{'='*60}")
        
        try:
            answer = qa_system.query(case["image"], case["question"])
            results.append({
                "image": os.path.basename(case["image"]),
                "question": case["question"],
                "answer_preview": answer[:300] + "..." if len(answer) > 300 else answer,
                "status": "SUCCESS",
            })
        except Exception as e:
            print(f"❌ 오류: {e}")
            results.append({
                "image": os.path.basename(case["image"]),
                "question": case["question"],
                "error": str(e),
                "status": "FAILED",
            })
    
    return results


def print_summary(classifier_results, qa_results):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    print("\n[이미지 분류기 테스트]")
    c_success = sum(1 for r in classifier_results if r["status"] == "SUCCESS")
    print(f"  성공: {c_success}/{len(classifier_results)}")
    
    print("\n[이미지 + RAG 연동 테스트]")
    q_success = sum(1 for r in qa_results if r["status"] == "SUCCESS")
    print(f"  성공: {q_success}/{len(qa_results)}")
    
    total = len(classifier_results) + len(qa_results)
    total_success = c_success + q_success
    print(f"\n전체: {total_success}/{total} 성공")
    
    if total_success == total:
        print("\n✅ 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패")
        for r in classifier_results + qa_results:
            if r["status"] == "FAILED":
                print(f"  - {r.get('image', 'N/A')}: {r.get('error', 'Unknown')}")


def main():
    print("=" * 60)
    print("🧪 이미지 병해충 Q&A 시스템 테스트")
    print("=" * 60)
    
    # 1. 테스트 이미지 준비
    print("\n[1단계] 테스트 이미지 준비")
    image_paths = download_test_images()
    
    if not image_paths:
        print("❌ 테스트 이미지가 없습니다. 수동으로 test_images/ 폴더에 감귤 이미지를 넣어주세요.")
        sys.exit(1)
    
    print(f"\n준비된 이미지: {len(image_paths)}개")
    
    # 2. 분류기 단독 테스트
    print("\n[2단계] 이미지 분류기 테스트")
    classifier_results = test_classifier_only(image_paths)
    
    # 3. 이미지 + RAG 연동 테스트
    print("\n[3단계] 이미지 + RAG 연동 테스트")
    qa_results = test_image_qa(image_paths)
    
    # 4. 결과 요약
    print_summary(classifier_results, qa_results)


if __name__ == "__main__":
    main()
