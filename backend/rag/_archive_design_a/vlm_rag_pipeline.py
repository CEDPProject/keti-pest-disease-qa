"""VLM + RAG 파이프라인 — 여러 병명 후보를 RAG로 대조해 최종 병명을 선택.

개발 계획서 4.7/4.8절의 설계를 실제로 구현한 것.
기존 image_qa/vlm_qa.py 는 use_rag=False 로 RAG 단계가 미구현 상태였다.

흐름
  1) 관찰(observe)   : VLM에게 "병명은 말하지 말고 보이는 것만 서술"시킴  → 시각 관찰문
  2) 감별(differential): VLM에게 의심 병명 후보를 가능성 순으로 나열시킴   → 후보 리스트
  3) 대조(RAG)       : 관찰문을 query 로, 후보들의 KB 병징을 passage 로 코사인 유사도 산출
  4) 선택(select)    : 최고 점수 후보를 최종 병명으로. 1·2위 점수차가 작으면 '애매'로 표시.

순환 논리 방지: 파인튜닝 모델의 진단 답변은 KB 문장을 그대로 재현하므로,
그 텍스트로 RAG를 돌리면 자기 자신을 검색하게 된다. 그래서 1)에서 병명을 배제한
'시각 관찰문'만 뽑아 RAG 질의로 사용한다.
"""

import os
import sys
import re

import torch
from PIL import Image

ARCHIVE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(ARCHIVE_DIR)
VLM_FT = os.path.dirname(RAG_DIR)
sys.path.insert(0, os.path.join(VLM_FT, "common"))
sys.path.insert(0, ARCHIVE_DIR)

from classes import CLASSES, TEST_JSONL  # noqa
from label_extract import CANON, SYNONYM, LABEL_TO_KEY  # noqa
from kb_index import KBIndex  # noqa

# 우리 8클래스 한글명 → KB 섹션명
KO_TO_KB = {ko: v["kb_section"] for ko, v in CLASSES.items()}
KB_TO_KO = {v: k for k, v in KO_TO_KB.items()}

OBSERVE_PROMPT = (
    "이 감귤 사진에서 눈에 보이는 것만 객관적으로 서술하세요.\n"
    "병명은 절대 말하지 마세요. 다음만 구체적으로 쓰세요: "
    "병반의 색, 모양, 크기, 표면 상태(융기/함몰/코르크화), 위치(잎/과실/가지), 주변 조직의 변화."
)
DIFFERENTIAL_PROMPT = (
    "이 사진에서 의심되는 병해충 후보를 가능성이 높은 순서로 최대 3개만 나열하세요.\n"
    "설명 없이 다음 형식으로만 답하세요.\n1. 병명\n2. 병명\n3. 병명"
)
AMBIGUOUS_MARGIN = 0.01  # 1·2위 코사인 점수차가 이보다 작으면 '애매'


def _strip_disease_names(text: str) -> str:
    """관찰문에 병명이 새어나오면 제거(RAG 질의 오염 방지)."""
    for n in CANON + list(SYNONYM.keys()):
        text = text.replace(n, "")
    return re.sub(r"\s+", " ", text).strip()


def extract_candidates(text: str, max_n: int = 3):
    """감별 답변에서 병명 후보를 '등장 순서대로' 중복 없이 추출."""
    t = text.replace(" ", "")
    found = []
    for name in CANON:
        i = t.find(name)
        if i >= 0:
            found.append((i, name))
    for syn, canon in SYNONYM.items():
        i = t.find(syn)
        if i >= 0 and all(canon != n for _, n in found):
            found.append((i, canon))
    found.sort()
    out = []
    for _, n in found:
        if n not in out:
            out.append(n)
    return out[:max_n]


class VLMRagPipeline:
    def __init__(self, model, processor, kb_index: KBIndex):
        self.model = model
        self.proc = processor
        self.idx = kb_index

    def _gen(self, image_path, prompt, max_new=200):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image_path, "max_pixels": 768 * 768},
            {"type": "text", "text": prompt}]}]
        txt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB")
        inp = self.proc(text=[txt], images=[img], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return self.proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    def run(self, image_path):
        # 1) 시각 관찰문 (병명 배제)
        observation_raw = self._gen(image_path, OBSERVE_PROMPT, max_new=180)
        observation = _strip_disease_names(observation_raw)

        # 2) 후보 감별
        differential_raw = self._gen(image_path, DIFFERENTIAL_PROMPT, max_new=80)
        candidates = extract_candidates(differential_raw)

        result = {
            "observation": observation_raw,
            "observation_clean": observation,
            "differential_raw": differential_raw,
            "candidates": candidates,
        }
        if not candidates:
            result.update({"final": None, "confidence": "no_candidate", "ranking": []})
            return result

        # 3) RAG: 관찰문 vs 후보들의 KB 병징
        restrict = [KO_TO_KB[c] for c in candidates if c in KO_TO_KB]
        query = observation if len(observation) > 15 else observation_raw
        ranked_kb = self.idx.search(query, top_k=None, restrict=restrict)
        ranking = [(KB_TO_KO.get(n, n), s) for n, s in ranked_kb]

        # 4) 선택
        final, top_score = ranking[0]
        conf = "high"
        if len(ranking) > 1 and (ranking[0][1] - ranking[1][1]) < AMBIGUOUS_MARGIN:
            conf = "ambiguous"
        if len(candidates) == 1:
            conf = "single_candidate"

        result.update({
            "final": final,
            "final_key": LABEL_TO_KEY.get(final),
            "confidence": conf,
            "ranking": ranking,
            "rag_query": query[:160],
        })
        return result


if __name__ == "__main__":
    import json
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from classes import MERGED_MODEL_DIR as M  # noqa
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        M, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(M, max_pixels=768 * 768)
    pipe = VLMRagPipeline(model, proc, KBIndex(device="cuda"))

    test = [json.loads(l) for l in open(TEST_JSONL, encoding="utf-8")]
    for r in test[:3]:
        out = pipe.run(r["image"])
        print(f"\n=== 정답: {r['label_ko']} ===")
        print(f"  관찰문: {out['observation'][:110]}...")
        print(f"  후보  : {out['candidates']}")
        print(f"  RAG   : {[(n, round(s,4)) for n,s in out['ranking']]}")
        print(f"  최종  : {out['final']} ({out['confidence']})")
