import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MAX_PIXELS = 768 * 768


# 경로나 PIL 이미지를 RGB 이미지로 연다
def to_rgb(image):
    return Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")


# Qwen2.5-VL 모델과 프로세서를 bf16 으로 GPU 에 올린다
def load_qwen25(model_dir: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(model_dir, max_pixels=MAX_PIXELS)
    return model, processor


# 채팅 템플릿 적용, 탐욕 생성, 라벨 채점을 맡는 VLM 실행기
class VLMRunner:
    # 모델·프로세서와 템플릿 옵션(enable_thinking)을 받는다
    def __init__(self, model, processor, enable_thinking=None, device: str = "cuda"):
        self.model = model
        self.proc = processor
        self.enable_thinking = enable_thinking
        self.device = device

    # 사용자 한 턴(이미지 + 질문)을 채팅 템플릿 문자열로 만든다
    def build_prompt(self, text: str, with_image: bool = True) -> str:
        content = [{"type": "image", "image": None, "max_pixels": MAX_PIXELS}] if with_image else []
        content.append({"type": "text", "text": text})
        options = {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}
        return self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True, **options)

    # 프롬프트와 이미지를 모델 입력 텐서로 바꾼다
    def encode(self, prompt: str, image=None):
        if image is None:
            return self.proc(text=[prompt], return_tensors="pt").to(self.device)
        return self.proc(text=[prompt], images=[to_rgb(image)], return_tensors="pt").to(self.device)

    # 완성된 프롬프트로 탐욕 생성한 답변을 돌려준다
    @torch.no_grad()
    def generate(self, prompt: str, image, max_new_tokens: int) -> str:
        inputs = self.encode(prompt, image)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    # 질문과 이미지(없으면 텍스트만)로 답변을 생성한다
    def ask(self, image, text: str, max_new_tokens: int) -> str:
        return self.generate(self.build_prompt(text, with_image=image is not None), image, max_new_tokens)

    # 각 라벨 문자열의 토큰 평균 로그확률을 구한다
    @torch.no_grad()
    def score_labels(self, image, prompt: str, labels: dict) -> dict:
        im = to_rgb(image)
        prompt_len = self.encode(prompt, im).input_ids.shape[1]
        scores = {}
        for key, label in labels.items():
            enc = self.encode(prompt + label, im)
            full_len = enc.input_ids.shape[1]
            logits = self.model(**enc).logits[0]
            target = enc.input_ids[0, prompt_len:full_len]
            logp = torch.log_softmax(logits[prompt_len - 1:full_len - 1], dim=-1)
            scores[key] = float(logp[torch.arange(len(target)), target].mean().item())
        return scores
