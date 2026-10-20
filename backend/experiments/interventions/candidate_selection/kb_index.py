import os
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.classes import KB_CACHE_DIR
from backend.core.kb_parser import parse_kb

MODEL_NAME = "intfloat/multilingual-e5-large"
CACHE_DIR = os.path.join(KB_CACHE_DIR, "design_a")
CACHE_NPY = os.path.join(CACHE_DIR, "kb_embeddings.npy")
CACHE_META = os.path.join(CACHE_DIR, "kb_meta_designA.json")


# 병해 하나의 검색용 병징 텍스트를 만든다
def _passage_text(name: str, k: dict) -> str:
    parts = [name]
    if k["symptoms"]:
        parts.append("병징: " + " ".join(k["symptoms"]))
    if k["ecology"]:
        parts.append("발생생태: " + " ".join(k["ecology"][:2]))
    return " ".join(parts)


# 병징 임베딩으로 관찰문과 가까운 병해를 찾는 색인
class KBIndex:
    # KB 병해별 임베딩을 캐시에서 불러오거나 새로 만든다
    def __init__(self, device: str = "cuda", rebuild: bool = False):
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.kb = parse_kb()
        self.names = sorted(self.kb.keys())

        if not rebuild and os.path.exists(CACHE_NPY) and os.path.exists(CACHE_META):
            with open(CACHE_META, encoding="utf-8") as f:
                meta = json.load(f)
            if meta["names"] == self.names:
                self.emb = np.load(CACHE_NPY)
                return

        passages = [f"passage: {_passage_text(n, self.kb[n])}" for n in self.names]
        self.emb = self.model.encode(passages, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(CACHE_NPY, self.emb)
        with open(CACHE_META, "w", encoding="utf-8") as f:
            json.dump({"names": self.names}, f, ensure_ascii=False)

    # 질의와 가까운 병해를 후보 범위 안에서 찾는다
    def search(self, query: str, top_k: int = 5, restrict: list = None):
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        sims = self.emb @ q
        idxs = range(len(self.names))
        if restrict:
            allow = set(restrict)
            idxs = [i for i in idxs if self.names[i] in allow]
        scored = sorted(((self.names[i], float(sims[i])) for i in idxs), key=lambda x: -x[1])
        return scored[:top_k] if top_k else scored


# 예시 관찰문으로 상위 병해 검색 결과를 출력한다
def main():
    idx = KBIndex(device="cpu")
    print(f"KB 인덱스: {len(idx.names)}종\n")
    for q in ["과실 표면에 움푹 들어간 갈색 병반과 검은 점이 무리지어 있다",
              "잎 뒷면이 부풀어 코르크화된 황갈색 분화구 모양 병반",
              "과실에 회백색 곰팡이가 뒤덮여 물러 썩었다"]:
        print(f"[질의] {q}")
        for n, s in idx.search(q, top_k=3):
            print(f"    {s:.4f}  {n}")
        print()


if __name__ == "__main__":
    main()
