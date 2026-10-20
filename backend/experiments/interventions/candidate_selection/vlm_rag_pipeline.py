import re

from backend.core.classes import CLASSES, LABEL_TO_KEY, MERGED_MODEL_DIR, TEST_JSONL
from backend.core.label_extract import CANON, SYNONYM
from backend.core.vlm import VLMRunner, load_qwen25
from backend.experiments.data_io import read_jsonl
from backend.experiments.interventions.candidate_selection.kb_index import KBIndex

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
AMBIGUOUS_MARGIN = 0.01


# 관찰문에 섞여 나온 병명을 지워 검색 질의 오염을 막는다
def strip_disease_names(text: str) -> str:
    for n in CANON + list(SYNONYM.keys()):
        text = text.replace(n, "")
    return re.sub(r"\s+", " ", text).strip()


# 감별 답변에서 병명 후보를 등장 순서대로 중복 없이 뽑는다
def extract_candidates(text: str, max_n: int = 3) -> list:
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


# VLM 후보 중 관찰문과 KB 병징이 가장 가까운 병명을 고르는 파이프라인
class VLMRagPipeline:
    # VLM 실행기와 KB 색인을 받는다
    def __init__(self, runner: VLMRunner, kb_index: KBIndex):
        self.runner = runner
        self.idx = kb_index

    # 관찰문·후보 감별·KB 검색으로 최종 병명을 고른다
    def run(self, image_path):
        observation_raw = self.runner.ask(image_path, OBSERVE_PROMPT, 180)
        observation = strip_disease_names(observation_raw)
        differential_raw = self.runner.ask(image_path, DIFFERENTIAL_PROMPT, 80)
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

        restrict = [KO_TO_KB[c] for c in candidates if c in KO_TO_KB]
        query = observation if len(observation) > 15 else observation_raw
        ranking = [(KB_TO_KO.get(n, n), s) for n, s in self.idx.search(query, top_k=None, restrict=restrict)]

        final = ranking[0][0]
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


# 테스트 사진 세 장의 선택 과정을 출력한다
def main():
    model, proc = load_qwen25(MERGED_MODEL_DIR)
    pipe = VLMRagPipeline(VLMRunner(model, proc), KBIndex(device="cuda"))
    for r in read_jsonl(TEST_JSONL)[:3]:
        out = pipe.run(r["image"])
        print(f"\n=== 정답: {r['label_ko']} ===")
        print(f"  관찰문: {out['observation'][:110]}...")
        print(f"  후보  : {out['candidates']}")
        print(f"  RAG   : {[(n, round(s, 4)) for n, s in out['ranking']]}")
        print(f"  최종  : {out['final']} ({out['confidence']})")


if __name__ == "__main__":
    main()
