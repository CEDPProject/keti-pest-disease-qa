import os
import sys
import json

import numpy as np
from sentence_transformers import SentenceTransformer

ARCHIVE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(ARCHIVE_DIR)
VLM_FT = os.path.dirname(RAG_DIR)
sys.path.insert(0, os.path.join(VLM_FT, "common"))
from kb_parser import parse_kb  # noqa

MODEL_NAME = "intfloat/multilingual-e5-large"
CACHE_NPY = os.path.join(ARCHIVE_DIR, "kb_embeddings.npy")
CACHE_META = os.path.join(ARCHIVE_DIR, "kb_meta_designA.json")


def _passage_text(name: str, k: dict) -> str:
    """검색 대상 텍스트 — 병징 중심(시각 관찰문과 매칭되도록)."""
    parts = [name]
    if k["symptoms"]:
        parts.append("병징: " + " ".join(k["symptoms"]))
    if k["ecology"]:
        parts.append("발생생태: " + " ".join(k["ecology"][:2]))
    return " ".join(parts)


class KBIndex:
    def __init__(self, device: str = "cuda", rebuild: bool = False):
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        kb = parse_kb()
        self.names = sorted(kb.keys())
        self.kb = kb

        if not rebuild and os.path.exists(CACHE_NPY) and os.path.exists(CACHE_META):
            meta = json.load(open(CACHE_META, encoding="utf-8"))
            if meta["names"] == self.names:
                self.emb = np.load(CACHE_NPY)
                return

        passages = [f"passage: {_passage_text(n, kb[n])}" for n in self.names]
        self.emb = self.model.encode(passages, normalize_embeddings=True,
                                     batch_size=8, show_progress_bar=False)
        np.save(CACHE_NPY, self.emb)
        json.dump({"names": self.names}, open(CACHE_META, "w", encoding="utf-8"),
                  ensure_ascii=False)

    def search(self, query: str, top_k: int = 5, restrict: list = None):
        """restrict: 후보 병명 리스트로 검색 범위 제한(후보 중 최근접 선택용)."""
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        sims = self.emb @ q  # 정규화됨 → 코사인
        idxs = range(len(self.names))
        if restrict:
            allow = set(restrict)
            idxs = [i for i in idxs if self.names[i] in allow]
        scored = sorted(((self.names[i], float(sims[i])) for i in idxs),
                        key=lambda x: -x[1])
        return scored[:top_k] if top_k else scored


if __name__ == "__main__":
    idx = KBIndex(device="cpu")
    print(f"KB 인덱스: {len(idx.names)}종\n")
    for q in ["과실 표면에 움푹 들어간 갈색 병반과 검은 점이 무리지어 있다",
              "잎 뒷면이 부풀어 코르크화된 황갈색 분화구 모양 병반",
              "과실에 회백색 곰팡이가 뒤덮여 물러 썩었다"]:
        print(f"[질의] {q}")
        for n, s in idx.search(q, top_k=3):
            print(f"    {s:.4f}  {n}")
        print()
