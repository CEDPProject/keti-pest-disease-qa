import os

# GPU 설정 (GPU 1번 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from PIL import Image
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
from typing import Dict, List, Tuple, Optional


# 모델 라벨 → 한글 매핑
LABEL_MAP = {
    "citrus canker": "궤양병",
    "healthy": "건강",
    "melanose": "흑점병(멜라노스)",
}


class CitrusImageClassifier:
    """감귤 병해충 이미지 분류기"""
    
    def __init__(
        self,
        model_name: str = "dima806/orange_fruit_disease_detection",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[이미지 분류] 모델 로딩 중: {self.model_name}")
        print(f"[이미지 분류] 디바이스: {self.device}")
        
        self.classifier = pipeline(
            "image-classification",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
        )
        
        print("[이미지 분류] 모델 로딩 완료!")
    
    def classify(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """
        이미지를 분류하여 병해충 예측 결과를 반환
        
        Args:
            image_path: 이미지 파일 경로
            top_k: 상위 K개 결과 반환
            
        Returns:
            [{"label": "citrus canker", "label_ko": "궤양병", "score": 0.95}, ...]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        results = self.classifier(image, top_k=top_k)
        
        classified = []
        for r in results:
            label = r["label"]
            classified.append({
                "label": label,
                "label_ko": LABEL_MAP.get(label, label),
                "score": round(r["score"], 4),
            })
        
        return classified
    
    def classify_with_summary(self, image_path: str) -> Tuple[str, float, List[Dict]]:
        """
        분류 결과를 요약하여 반환
        
        Returns:
            (최상위 한글 라벨, 신뢰도, 전체 결과 리스트)
        """
        results = self.classify(image_path)
        
        top = results[0]
        return top["label_ko"], top["score"], results


def main():
    """단독 실행 테스트"""
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python image_classifier.py <이미지 경로>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    classifier = CitrusImageClassifier()
    
    label_ko, score, all_results = classifier.classify_with_summary(image_path)
    
    print(f"\n{'='*50}")
    print(f"🖼️  이미지: {image_path}")
    print(f"{'='*50}")
    print(f"📌 판별 결과: {label_ko} (신뢰도: {score*100:.1f}%)")
    print(f"\n전체 결과:")
    for r in all_results:
        bar = "█" * int(r['score'] * 30)
        print(f"  {r['label_ko']:15s} {r['score']*100:5.1f}% {bar}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
