"""
병해충 질의응답 시스템 (Pest Disease Q&A System)
- RAG 기반 질의응답 시스템
- LLM: gemma3:12B (Ollama, Instruction-tuned)
- Embedding: intfloat/multilingual-e5-large
- Vector DB: Chroma
"""

import os
import re
from typing import List, Tuple, Dict

# GPU 설정 (GPU 1번 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from prompts import QA_PROMPT


class E5Embeddings(HuggingFaceEmbeddings):
    """E5 모델용 임베딩 래퍼 - query 시 'query: ' prefix 자동 추가"""
    
    def embed_query(self, text: str) -> list:
        return super().embed_query(f"query: {text}")


class PestDiseaseQA:
    """병해충 질의응답 시스템"""
    
    def __init__(
        self,
        data_path: str = "text_info/citrus_PD_ver3.md",
        embedding_model: str = "intfloat/multilingual-e5-large",
        llm_model: str = "gemma3:12B",
        persist_directory: str = "./chroma_db",
        device: str = "cuda"
    ):
        """
        Args:
            data_path: 병해충 데이터 파일 경로
            embedding_model: 임베딩 모델 이름
            llm_model: LLM 모델 이름 (Ollama)
            persist_directory: Chroma DB 저장 경로
            device: 디바이스 (cuda/cpu)
        """
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.persist_directory = persist_directory
        self.device = device if torch.cuda.is_available() else "cpu"
        
        print(f"[설정] GPU 사용: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[설정] 사용 GPU: {torch.cuda.get_device_name(0)}")
        
        # 컴포넌트 초기화
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
        # 대화 히스토리 (멀티턴 대화 지원)
        self.chat_history: List[Dict[str, str]] = []
        
        # 시스템 초기화
        self._initialize()
    
    def _initialize(self):
        """시스템 초기화"""
        print("\n[초기화] 병해충 Q&A 시스템을 초기화합니다...")
        
        # 1. 임베딩 모델 로드
        self._load_embedding_model()
        
        # 2. 벡터 저장소 로드 또는 생성
        self._load_or_create_vectorstore()
        
        # 3. LLM 로드
        self._load_llm()
        
        # 4. QA 체인 생성
        self._create_qa_chain()
        
        print("[초기화] 시스템 초기화 완료!\n")
    
    def _load_embedding_model(self):
        """임베딩 모델 로드 (E5 prefix 적용)"""
        print(f"[임베딩] {self.embedding_model} 모델 로딩 중...")
        self.embeddings = E5Embeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[임베딩] 모델 로딩 완료! (E5 query prefix 적용)")
    
    def _load_document(self) -> str:
        """병해충 데이터 파일 로드"""
        print(f"[데이터] {self.data_path} 파일 로딩 중...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"[데이터] 파일 로딩 완료! (총 {len(content)} 글자)")
        return content
    
    # 충해(해충) 키워드 목록
    PEST_KEYWORDS = [
        "응애", "진딧물", "벌레", "총채벌레", "깍지벌레", "노린재",
        "파리", "나방", "가루이", "매미충", "달팽이"
    ]

    def _split_by_disease_pest(self, content: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        병해충별로 문서를 분리하여 청크 + 메타데이터 생성
        각 병해충 섹션(## 로 시작)을 하나의 청크로 처리
        """
        print("[청킹] 병해충별로 문서 분리 중...")
        
        # ## 를 기준으로 분리 (각 병해충 시작점)
        sections = re.split(r'\n(?=## )', content)
        
        chunks = []
        metadatas = []
        for section in sections:
            section = section.strip()
            if section and section.startswith("## "):
                # 병해충 이름 추출
                disease_name = section.split("\n")[0].replace("## ", "").strip()
                
                # 병해/충해 자동 분류
                is_pest = any(kw in disease_name for kw in self.PEST_KEYWORDS)
                category = "충해" if is_pest else "병해"
                
                # E5 임베딩을 위한 passage prefix 추가
                chunks.append(f"passage: {section}")
                metadatas.append({"name": disease_name, "type": category})
                print(f"  - {disease_name} [{category}]")
        
        print(f"[청킹] 총 {len(chunks)}개의 병해충 청크 생성 완료!")
        return chunks, metadatas
    
    def _load_or_create_vectorstore(self):
        """벡터 저장소 로드 또는 생성 (Chroma)"""
        chroma_exists = os.path.exists(self.persist_directory) and os.listdir(self.persist_directory)
        
        if chroma_exists:
            print(f"[벡터DB] 기존 Chroma 벡터 저장소 로딩 중... ({self.persist_directory})")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"[벡터DB] 기존 벡터 저장소 로딩 완료! (총 {self.vectorstore._collection.count()}개 문서)")
        else:
            print("[벡터DB] 새 Chroma 벡터 저장소 생성 중...")
            
            # 문서 로드 및 청킹
            content = self._load_document()
            chunks, metadatas = self._split_by_disease_pest(content)
            
            # Chroma 벡터 저장소 생성 (메타데이터 포함)
            self.vectorstore = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=self.persist_directory
            )
            print(f"[벡터DB] 벡터 저장소 생성 완료! (총 {len(chunks)}개 문서, 메타데이터 포함)")
    
    def _load_llm(self):
        """LLM 모델 로드 (Ollama)"""
        print(f"[LLM] {self.llm_model} 모델 로딩 중...")
        self.llm = OllamaLLM(
            model=self.llm_model,
            temperature=0.1,
            num_ctx=8192,
            top_p=0.9,
            repeat_penalty=1.2,  # 반복/환각 억제
        )
        print("[LLM] 모델 로딩 완료!")
    
    def _create_qa_chain(self):
        """QA 체인 생성"""
        print("[체인] Q&A 체인 생성 중...")
        
        self.prompt = PromptTemplate(
            template=QA_PROMPT,
            input_variables=["context", "question", "disease_list", "chat_history"]
        )
        
        # Retriever 설정 (상위 5개 검색, 3→5로 확장)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 전체 병해충 목록 구성 (메타데이터에서 추출)
        all_docs = self.vectorstore._collection.get(include=["metadatas"])
        self.disease_names = []
        for meta in all_docs["metadatas"]:
            if meta and "name" in meta:
                self.disease_names.append(f"- {meta['name']} ({meta.get('type', '')})")
        self.disease_list_str = "\n".join(sorted(set(self.disease_names)))
        
        print("[체인] Q&A 체인 생성 완료!")
    
    def query(self, question: str) -> Tuple[str, List[str]]:
        """
        질문에 대한 답변 생성 (대화 히스토리 포함)
        
        Args:
            question: 사용자 질문
            
        Returns:
            Tuple[답변, 관련 문서 리스트]
        """
        print(f"\n{'='*60}")
        print(f"❓ 질문: {question}")
        print(f"{'='*60}")
        
        # 후보 번호 선택 처리 ("1번", "2번" 등 → 이전 후보의 병해충명으로 변환)
        resolved_question = self._resolve_selection(question)
        if resolved_question != question:
            print(f"  → 해석: {resolved_question}")
        
        # 이전 대화 맥락을 포함하여 검색 쿼리 구성
        if self.chat_history:
            # 이전 질문들을 모두 누적하여 검색 → 맥락 유지
            prev_questions = " ".join(turn['question'] for turn in self.chat_history[-3:])
            search_query = f"{prev_questions} {resolved_question}"
        else:
            search_query = resolved_question
        
        # 관련 문서 검색
        source_docs = self.retriever.invoke(search_query)
        
        # 문맥 구성
        context = "\n\n".join(doc.page_content for doc in source_docs)
        
        # 대화 히스토리 문자열 구성 (최근 5턴까지, 답변은 요약)
        history_str = "(첫 질문입니다 - 이전 대화 없음)"
        if self.chat_history:
            history_parts = []
            for i, turn in enumerate(self.chat_history[-5:], 1):
                # 답변이 길면 앞부분만 포함 (500자 제한)
                answer_summary = turn['answer'][:500]
                if len(turn['answer']) > 500:
                    answer_summary += "..."
                history_parts.append(f"[대화 {i}]\n사용자: {turn['question']}\n시스템: {answer_summary}")
            history_str = "\n\n".join(history_parts)
        
        # 프롬프트 생성 (병해충 목록 + 대화 히스토리 주입)
        formatted_prompt = self.prompt.format(
            context=context,
            question=resolved_question,
            disease_list=self.disease_list_str,
            chat_history=history_str
        )
        
        # LLM 답변 생성
        raw_answer = self.llm.invoke(formatted_prompt)
        
        # 모델 내부 추론 태그 제거 (exaone-deep: <thought>, 기타 모델: <think> 등)
        answer = re.sub(r'<(?:thought|think)>.*?</(?:thought|think)>', '', raw_answer, flags=re.DOTALL).strip()
        
        # 대화 히스토리에 현재 턴 추가
        self.chat_history.append({"question": question, "answer": answer})
        
        source_snippets = [doc.page_content[:200] + "..." for doc in source_docs]
        
        print(f"\n{answer}")
        print(f"\n{'='*60}\n")
        
        return answer, source_snippets
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.chat_history.clear()
        print("[히스토리] 대화 기록이 초기화되었습니다.\n")
    
    def _resolve_selection(self, question: str) -> str:
        """
        사용자가 후보 번호(예: '1번', '2번') 또는 병해충명을 선택한 경우,
        이전 대화에서 해당 병해충명을 찾아 질문을 보강한다.
        """
        if not self.chat_history:
            return question
        
        last_answer = self.chat_history[-1].get('answer', '')
        
        # 번호 선택 패턴 ('1번', '1번이야', '1번이 맞는거 같아' 등)
        num_match = re.match(r'^\s*(\d+)\s*번', question)
        if num_match:
            num = int(num_match.group(1))
            # 이전 답변에서 '1. **병명**' 패턴으로 후보 추출
            candidates = re.findall(r'\d+\.\s*\*\*(.+?)\*\*', last_answer)
            if 0 < num <= len(candidates):
                selected = candidates[num - 1]
                return f"{selected}에 대해 상세히 알려줘. 병명, 학명, 영명, 증상, 방제방법, 약제 정보를 모두 포함해줘."
        
        # 병해충명 직접 선택 ('황반병같아', '황반병에 대해 알려줘', '황반병이야' 등)
        # 이전 답변에서 후보로 제시된 병명이 현재 질문에 포함되어 있는지 확인
        candidates = re.findall(r'\*\*(.+?)\*\*', last_answer)
        for candidate in candidates:
            if candidate in question:
                return f"{candidate}에 대해 상세히 알려줘. 병명, 학명, 영명, 증상, 방제방법, 약제 정보를 모두 포함해줘."
        
        return question
    
    def interactive_mode(self):
        """대화형 모드 실행"""
        print("\n" + "="*60)
        print("🍊 감귤 병해충 질의응답 시스템")
        print("="*60)
        print("증상이나 병해충에 대해 질문해주세요.")
        print("  - 종료: 'q' 또는 'quit'")
        print("  - 새 대화: 'reset' 또는 '초기화'\n")
        
        while True:
            try:
                question = input("🔍 질문: ")
                if question is None:
                    continue
                question = question.strip()
            except UnicodeDecodeError:
                print("⚠️ 입력 인코딩 오류가 발생했습니다. 다시 입력해주세요.\n")
                continue
            
            try:
                if question.lower() in ["q", "quit", "exit", "종료"]:
                    print("\n시스템을 종료합니다. 감사합니다!")
                    break
                
                if question.lower() in ["reset", "clear", "초기화", "새대화"]:
                    self.clear_history()
                    continue
                
                if not question:
                    print("질문을 입력해주세요.\n")
                    continue
                
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n\n시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}\n")
    
    def rebuild_vectorstore(self):
        """벡터 저장소 재구축"""
        import shutil
        
        print("[재구축] 기존 벡터 저장소 삭제 중...")
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        
        print("[재구축] 벡터 저장소 재생성 중...")
        self._load_or_create_vectorstore()
        self._create_qa_chain()
        print("[재구축] 완료!")


def main():
    """메인 함수"""
    # 현재 스크립트 경로 기준으로 데이터 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "text_info", "citrus_PD_ver3.md")
    persist_dir = os.path.join(script_dir, "chroma_db")
    
    # Q&A 시스템 초기화
    qa_system = PestDiseaseQA(
        data_path=data_path,
        persist_directory=persist_dir
    )
    
    # 대화형 모드 실행
    qa_system.interactive_mode()


if __name__ == "__main__":
    main()
