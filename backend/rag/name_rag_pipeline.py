import os
import sys

import torch
from PIL import Image

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
VLM_FT = os.path.dirname(RAG_DIR)
sys.path.insert(0, os.path.join(VLM_FT, "common"))
sys.path.insert(0, RAG_DIR)
from label_extract import extract_label, LABEL_TO_KEY  # noqa
from kb_retriever import KBRetriever  # noqa
from classes import CLASSES, TEST_JSONL  # noqa

KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}
# KB 섹션명 → 우리 8클래스 키 (예: '녹색곰팡이병ㆍ푸른곰팡이병' → 'green_mold')
KB_SECTION_TO_KEY = {v["kb_section"]: v["key"] for v in CLASSES.values()}


def kb_name_to_key(kb_disease: str) -> str:
    """KB 섹션명(또는 클래스 한글명) → 클래스 키. 매핑 없으면 'unknown'."""
    if not kb_disease:
        return "unknown"
    if kb_disease in KB_SECTION_TO_KEY:
        return KB_SECTION_TO_KEY[kb_disease]
    return LABEL_TO_KEY.get(kb_disease, "unknown")

# 병명만 요구하는 프롬프트. (모델이 장황하게 답해도 extract_label 로 병명만 취한다)
NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."


class NameRagPipeline:
    """VLM=병명만, RAG=정보조회."""

    def __init__(self, model, processor, retriever: KBRetriever):
        self.model = model
        self.proc = processor
        self.kb = retriever

    def _vlm_name(self, image):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image, "max_pixels": 768 * 768},
            {"type": "text", "text": NAME_PROMPT}]}]
        txt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        inp = self.proc(text=[txt], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=120, do_sample=False)
        raw = self.proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        key = extract_label(raw)                       # 병명 하나만 취함
        name = KEY_TO_LABEL.get(key)                   # 나머지 텍스트는 버림
        return name, raw

    def run(self, image, user_question: str = None) -> dict:
        name, raw = self._vlm_name(image)

        if name is None:
            # 모델이 학습된 8종 밖의 이름을 뱉었을 수 있음 → 원문에서 후보 제안
            sugg = self.kb.suggest(raw[:40], k=3)
            return {"vlm_raw": raw, "disease": None, "how": "unknown",
                    "suggestions": sugg,
                    "answer": ("## 병명을 특정하지 못했습니다\n\n"
                               "모델이 지식베이스에 없는 병명을 답했습니다. 유사 후보는 다음과 같습니다:\n\n"
                               + "\n".join(f"- {n} (유사도 {s:.3f})" for n, s in sugg)
                               + "\n\n> 사진을 다시 촬영하거나 전문가 확인이 필요합니다.")}

        res = self.kb.resolve(name)
        if res["disease"] is None:
            sugg = self.kb.suggest(name, k=3)
            return {"vlm_raw": raw, "disease": None, "how": "unknown", "suggestions": sugg,
                    "answer": "## 병명을 KB에서 찾지 못했습니다\n\n"
                              + "\n".join(f"- {n} ({s:.3f})" for n, s in sugg)}

        disease = res["disease"]
        answer = self.kb.build_answer(disease, user_question)
        if res["how"] == "rag":
            answer = (f"> 모델이 답한 '{name}' 은 KB에 없어 유사도 {res['score']:.3f} 로 "
                      f"**{disease}** 에 매핑했습니다.\n\n" + answer)
        return {"vlm_raw": raw, "disease": disease, "how": res["how"],
                "score": res["score"], "answer": answer,
                "class_key": kb_name_to_key(disease)}


if __name__ == "__main__":
    import json
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from classes import MERGED_MODEL_DIR as M  # noqa
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        M, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(M, max_pixels=768 * 768)
    pipe = NameRagPipeline(model, proc, KBRetriever(device="cuda"))

    test = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")]
    # 데모: 앞 2개 사례
    picks = test[:2]
    for r in picks:
        for q in [None, "약제 뭐 써요?"]:
            out = pipe.run(r["image"], q)
            print(f"\n{'='*66}\n정답={r['label_ko']} | 질문={q or '(없음)'}")
            print(f"  VLM 원문: {out['vlm_raw'][:70].replace(chr(10),' ')}")
            print(f"  → 병명: {out['disease']} ({out['how']})")
            print(f"  답변 앞부분:\n{out['answer'][:300]}")
