"""
이미지 기반 병해충 질의응답 시스템 (Image Q&A System)
- 이미지 분류: dima806/orange_fruit_disease_detection (ViT)
- RAG 연동: text_qa의 PestDiseaseQA 시스템 활용
- LLM: Ollama (gemma3:12B)
- 플로우: 이미지 → 분류 → 분류결과로 RAG 검색 → LLM 상세 답변
"""

import os
import sys
import re

# GPU 설정 (GPU 1번 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from image_classifier import CitrusImageClassifier
from label_utils import is_healthy_label
from prompts import IMAGE_QA_PROMPT, HEALTHY_ANSWER_TEMPLATE

# text_qa 경로 (벡터 저장소 공유)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_QA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "text_qa")


class E5Embeddings(HuggingFaceEmbeddings):
    """E5 모델용 임베딩 래퍼 - query 시 'query: ' prefix 자동 추가"""
    
    def embed_query(self, text: str) -> list:
        return super().embed_query(f"query: {text}")


class ImagePestDiseaseQA:
    """이미지 기반 병해충 질의응답 시스템"""
    
    def __init__(
        self,
        image_model: str = "dima806/orange_fruit_disease_detection",
        embedding_model: str = "intfloat/multilingual-e5-large",
        llm_model: str = "gemma3:12B",
        text_info_path: str = None,
        persist_directory: str = None,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # 기본 경로 설정
        if text_info_path is None:
            text_info_path = os.path.join(TEXT_QA_DIR, "text_info", "citrus_PD_ver3.md")
        if persist_directory is None:
            persist_directory = os.path.join(TEXT_QA_DIR, "chroma_db")
        
        self.text_info_path = text_info_path
        self.persist_directory = persist_directory
        
        print("=" * 60)
        print("🖼️  이미지 기반 병해충 Q&A 시스템 초기화")
        print("=" * 60)
        
        # 1. 이미지 분류기 로드
        self.classifier = CitrusImageClassifier(
            model_name=image_model,
            device=self.device,
        )
        
        # 2. 임베딩 모델 로드
        print(f"[임베딩] {embedding_model} 모델 로딩 중...")
        self.embeddings = E5Embeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[임베딩] 모델 로딩 완료!")
        
        # 3. 벡터 저장소 로드 (text_qa와 공유)
        print(f"[벡터DB] Chroma 벡터 저장소 로딩 중... ({self.persist_directory})")
        if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
            print("[벡터DB] 벡터 저장소가 없습니다. text_qa를 먼저 실행하여 벡터 저장소를 생성해주세요.")
            print(f"  → cd {TEXT_QA_DIR} && python pest_disease_qa.py")
            sys.exit(1)
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        print(f"[벡터DB] 로딩 완료! (총 {self.vectorstore._collection.count()}개 문서)")
        
        # 병해충 목록 구성
        all_docs = self.vectorstore._collection.get(include=["metadatas"])
        disease_names = []
        for meta in all_docs["metadatas"]:
            if meta and "name" in meta:
                disease_names.append(f"- {meta['name']} ({meta.get('type', '')})")
        self.disease_list_str = "\n".join(sorted(set(disease_names)))
        
        # 4. LLM 로드
        print(f"[LLM] {llm_model} 모델 로딩 중...")
        self.llm = OllamaLLM(
            model=llm_model,
            temperature=0.1,
            num_ctx=8192,
            top_p=0.9,
            repeat_penalty=1.2,
        )
        print("[LLM] 모델 로딩 완료!")
        
        # 5. 프롬프트 설정
        self.prompt = PromptTemplate(
            template=IMAGE_QA_PROMPT,
            input_variables=["image_result", "disease_list", "context", "extra_question"],
        )
        
        print("\n[초기화] 이미지 Q&A 시스템 준비 완료!\n")
    
    def query(self, image_path: str, extra_question: str = "") -> dict:
        """
        이미지를 분석하고 건강/병해충 분기에 따라 답변 생성
        
        Args:
            image_path: 감귤 이미지 경로
            extra_question: 추가 질문 (선택, 예: "방제방법만 알려줘")
            
        Returns:
            dict: {image_path, predicted_label, confidence, is_healthy, rag_used, final_answer, all_scores}
        """
        print(f"\n{'='*60}")
        print(f"🖼️  이미지 분석 중: {image_path}")
        print(f"{'='*60}")
        
        # 1단계: 이미지 분류
        label_ko, score, all_results = self.classifier.classify_with_summary(image_path)
        
        print(f"\n📌 1차 판별: {label_ko} (신뢰도: {score*100:.1f}%)")
        for r in all_results:
            bar = "█" * int(r["score"] * 20)
            print(f"   {r['label_ko']:15s} {r['score']*100:5.1f}% {bar}")
        
        # 이미지 분류 결과 텍스트 구성
        image_result_parts = [f"최상위 판별: {label_ko} (신뢰도: {score*100:.1f}%)"]
        for r in all_results:
            image_result_parts.append(f"- {r['label_ko']}({r['label']}): {r['score']*100:.1f}%")
        image_result_str = "\n".join(image_result_parts)
        
        # 2단계: 건강/병해충 분기
        healthy = is_healthy_label(label_ko)
        
        if healthy:
            # 정상 판정 → RAG 호출 없이 고정 답변
            print("\n✅ 정상 판정 → RAG 호출 생략")
            summary_line = ", ".join(
                f"{r['label_ko']}({r['label']}) {r['score']*100:.1f}%"
                for r in all_results
            )
            answer = HEALTHY_ANSWER_TEMPLATE.format(image_result_summary=summary_line)
            print(f"\n{answer}")
            print(f"\n{'='*60}\n")
            
            return {
                "image_path": image_path,
                "predicted_label": label_ko,
                "confidence": score,
                "is_healthy": True,
                "rag_used": False,
                "final_answer": answer,
                "all_scores": all_results,
            }
        
        # 3단계: 병해충 → RAG 검색
        search_query = f"{label_ko} 증상 원인 방제방법 약제"
        
        if extra_question:
            search_query += f" {extra_question}"
        
        print(f"\n🔍 RAG 검색 쿼리: {search_query}")
        source_docs = self.retriever.invoke(search_query)
        context = "\n\n".join(doc.page_content for doc in source_docs)
        
        # 4단계: 프롬프트 생성 & LLM 답변
        extra_q_str = ""
        if extra_question:
            extra_q_str = f"사용자 추가 질문: {extra_question}"
        
        formatted_prompt = self.prompt.format(
            image_result=image_result_str,
            disease_list=self.disease_list_str,
            context=context,
            extra_question=extra_q_str,
        )
        
        print("\n💬 LLM 답변 생성 중...")
        raw_answer = self.llm.invoke(formatted_prompt)
        
        # 모델 내부 추론 태그 제거
        answer = re.sub(
            r"<(?:thought|think)>.*?</(?:thought|think)>", "", raw_answer, flags=re.DOTALL
        ).strip()
        
        print(f"\n{answer}")
        print(f"\n{'='*60}\n")
        
        return {
            "image_path": image_path,
            "predicted_label": label_ko,
            "confidence": score,
            "is_healthy": False,
            "rag_used": True,
            "final_answer": answer,
            "all_scores": all_results,
        }
    
    def interactive_mode(self):
        """대화형 모드 실행"""
        print("\n" + "=" * 60)
        print("🖼️  감귤 병해충 이미지 진단 시스템")
        print("=" * 60)
        print("감귤 이미지 경로를 입력하면 병해충을 진단합니다.")
        print("  - 이미지 경로만 입력: 이미지 분석 + 상세 답변")
        print("  - 이미지 경로 + 질문: 이미지 분석 + 질문 반영 답변")
        print("  - 종료: 'q' 또는 'quit'\n")
        
        while True:
            try:
                user_input = input("🖼️  이미지 경로 (+ 추가 질문): ").strip()
            except (UnicodeDecodeError, EOFError):
                continue
            
            try:
                if user_input.lower() in ("q", "quit", "exit", "종료"):
                    print("\n시스템을 종료합니다. 감사합니다!")
                    break
                
                if not user_input:
                    print("이미지 경로를 입력해주세요.\n")
                    continue
                
                # 입력 파싱: 이미지 경로와 추가 질문 분리
                parts = user_input.split(maxsplit=1)
                image_path = parts[0]
                extra_question = parts[1] if len(parts) > 1 else ""
                
                if not os.path.exists(image_path):
                    print(f"❌ 파일을 찾을 수 없습니다: {image_path}\n")
                    continue
                
                self.query(image_path, extra_question)
                
            except KeyboardInterrupt:
                print("\n\n시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}\n")


# --- 모듈 레벨 편의 함수 ---

_qa_instance = None


def classifier_rag_pipeline(image_path: str, user_question: str = "") -> dict:
    """
    이미지 분류 + RAG 파이프라인 (모듈 레벨 함수).
    ImagePestDiseaseQA 인스턴스를 한 번만 생성하여 재사용한다.
    
    Args:
        image_path: 이미지 파일 경로
        user_question: 추가 질문 (선택)
        
    Returns:
        dict: {image_path, predicted_label, confidence, is_healthy, rag_used, final_answer, all_scores}
    """
    global _qa_instance
    if _qa_instance is None:
        _qa_instance = ImagePestDiseaseQA()
    return _qa_instance.query(image_path, user_question)


def main():
    """메인 함수"""
    qa_system = ImagePestDiseaseQA()
    qa_system.query("../../data/legacy_test_images/잎_궤양병.jpg")


if __name__ == "__main__":
    main()
