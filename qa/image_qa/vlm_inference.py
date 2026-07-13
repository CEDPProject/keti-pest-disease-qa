"""VLM 추론 모듈 - 이미지 + 텍스트 멀티모달 추론

현재 백엔드: Ollama gemma3:12b (멀티모달 지원)
나중에 Agri-LLaVA, LLaVA, Qwen-VL 등으로 교체 가능하도록 함수 인터페이스를 분리.
"""

import os
import re
import base64

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# 기본 VLM 모델 설정
DEFAULT_VLM_MODEL = "gemma3:12b"


def _encode_image_base64(image_path: str) -> str:
    """이미지 파일을 base64로 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_mime_type(image_path: str) -> str:
    """이미지 확장자에 따른 MIME 타입 반환"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/jpeg")


def generate_vlm_answer(
    image_path: str,
    prompt: str,
    model: str = DEFAULT_VLM_MODEL,
    temperature: float = 0.2,
) -> str:
    """Run VLM inference with image and prompt.

    The backend should be easy to replace later with Agri-LLaVA, LLaVA, Qwen-VL, or an API-based VLM.
    
    Args:
        image_path: 이미지 파일 경로
        prompt: VLM에 전달할 텍스트 프롬프트
        model: Ollama 모델 이름
        temperature: 생성 온도
        
    Returns:
        VLM 응답 문자열
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # 이미지 인코딩
    image_b64 = _encode_image_base64(image_path)
    mime_type = _get_image_mime_type(image_path)
    
    # ChatOllama 멀티모달 호출
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=8192,
        repeat_penalty=1.2,
    )
    
    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
    )
    
    response = llm.invoke([message])
    
    # 내부 추론 태그 제거
    answer = re.sub(
        r"<(?:thought|think)>.*?</(?:thought|think)>", "", response.content, flags=re.DOTALL
    ).strip()
    
    return answer
