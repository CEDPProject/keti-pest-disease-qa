import json

from backend.core.classes import KEY_TO_LABEL, LABEL_TO_KEY, MERGED_MODEL_DIR, SECTION_TO_KEY, TEST_JSONL
from backend.core.kb_retriever import KBRetriever
from backend.core.label_extract import extract_label
from backend.core.vlm import VLMRunner, load_qwen25

NAME_PROMPT = "이 감귤 사진의 병명은 무엇인가요? 병명만 답하세요."
NAME_MAX_NEW_TOKENS = 120


# KB 섹션명이나 클래스 한글명을 클래스 키로 바꾼다
def kb_name_to_key(kb_disease: str) -> str:
    if not kb_disease:
        return "unknown"
    if kb_disease in SECTION_TO_KEY:
        return SECTION_TO_KEY[kb_disease]
    return LABEL_TO_KEY.get(kb_disease, "unknown")


# VLM 은 병명만 정하고 사실 정보는 KB 에서 가져오는 진단 파이프라인
class NameRagPipeline:
    # VLM 실행기와 KB 검색기를 받는다
    def __init__(self, runner: VLMRunner, retriever: KBRetriever):
        self.runner = runner
        self.kb = retriever

    # 사진에서 병명 하나를 뽑아 모델 원문과 함께 돌려준다
    def predict_name(self, image):
        raw = self.runner.ask(image, NAME_PROMPT, NAME_MAX_NEW_TOKENS)
        return KEY_TO_LABEL.get(extract_label(raw)), raw

    # 병명 판정 후 KB 원문으로 답변을 조립한다
    def run(self, image, user_question: str = None) -> dict:
        name, raw = self.predict_name(image)

        if name is None:
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


# 서빙 모델로 테스트 사진 두 장에 대한 답변을 출력한다
def main():
    model, proc = load_qwen25(MERGED_MODEL_DIR)
    pipe = NameRagPipeline(VLMRunner(model, proc), KBRetriever(device="cuda"))

    with open(TEST_JSONL, encoding="utf-8") as f:
        test = [json.loads(line) for line in f]
    for r in test[:2]:
        for q in [None, "약제 뭐 써요?"]:
            out = pipe.run(r["image"], q)
            print(f"\n{'=' * 66}\n정답={r['label_ko']} | 질문={q or '(없음)'}")
            print(f"  VLM 원문: {out['vlm_raw'][:70].replace(chr(10), ' ')}")
            print(f"  → 병명: {out['disease']} ({out['how']})")
            print(f"  답변 앞부분:\n{out['answer'][:300]}")


if __name__ == "__main__":
    main()
