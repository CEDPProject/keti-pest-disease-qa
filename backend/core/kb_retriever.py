import os
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.classes import KB_CACHE_DIR
from backend.core.kb_parser import parse_kb
from backend.core.label_extract import SYNONYM

MODEL_NAME = "intfloat/multilingual-e5-large"
CHUNK_NPY = os.path.join(KB_CACHE_DIR, "kb_chunks.npy")
LINE_NPY = os.path.join(KB_CACHE_DIR, "kb_lines.npy")
NAME_NPY = os.path.join(KB_CACHE_DIR, "kb_names.npy")
META = os.path.join(KB_CACHE_DIR, "kb_meta.json")
CACHE_VERSION = 2

RESOLVE_THRESHOLD = 0.86
TOP_LINES = 3
LIST_SECTIONS = {"방제약제"}

SECTIONS = [("symptoms", "병징"), ("ecology", "발생생태"),
            ("control", "방제 및 과원관리"), ("pesticides", "방제약제")]


# 섹션 청크 텍스트에서 '병명 섹션:' 머리를 뗀 본문을 돌려준다
def _body(chunk: dict) -> str:
    return chunk["text"].split(":", 1)[1].strip() if ":" in chunk["text"] else chunk["text"]


# 병명 정규화와 섹션·문장 검색, KB 원문 답변 조립을 맡는 검색기
class KBRetriever:
    # KB 를 청크·문장으로 나누고 임베딩 캐시를 준비한다
    def __init__(self, device: str = "cuda", rebuild: bool = False):
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.kb = parse_kb()
        self.names = sorted(self.kb.keys())

        self.chunks, self.lines = [], []
        for name in self.names:
            k = self.kb[name]
            for field, ko in SECTIONS:
                vals = k.get(field) or []
                if not vals:
                    continue
                self.chunks.append({"disease": name, "section": ko,
                                    "text": f"{name} {ko}: " + " ".join(vals)})
                if ko in LIST_SECTIONS:
                    continue
                for v in vals:
                    self.lines.append({"disease": name, "section": ko, "text": v})

        if rebuild or not self._cache_valid():
            self._build()
        else:
            self.chunk_emb = np.load(CHUNK_NPY)
            self.line_emb = np.load(LINE_NPY)
            self.name_emb = np.load(NAME_NPY)

    # 캐시 파일이 현재 KB 구성과 맞는지 확인한다
    def _cache_valid(self):
        if not all(os.path.exists(p) for p in (CHUNK_NPY, LINE_NPY, NAME_NPY, META)):
            return False
        with open(META, encoding="utf-8") as f:
            m = json.load(f)
        return (m.get("version") == CACHE_VERSION and m.get("names") == self.names
                and m.get("n_chunks") == len(self.chunks) and m.get("n_lines") == len(self.lines))

    # 청크·문장·병명 임베딩을 계산해 캐시에 저장한다
    def _build(self):
        self.chunk_emb = self.model.encode([f"passage: {c['text']}" for c in self.chunks],
                                           normalize_embeddings=True, batch_size=16)
        self.line_emb = self.model.encode([f"passage: {l['disease']} {l['text']}" for l in self.lines],
                                          normalize_embeddings=True, batch_size=16)
        self.name_emb = self.model.encode(
            [f"passage: {n} " + " ".join((self.kb[n]["symptoms"] or [])[:2]) for n in self.names],
            normalize_embeddings=True, batch_size=16)
        os.makedirs(KB_CACHE_DIR, exist_ok=True)
        np.save(CHUNK_NPY, self.chunk_emb)
        np.save(LINE_NPY, self.line_emb)
        np.save(NAME_NPY, self.name_emb)
        with open(META, "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "names": self.names,
                       "n_chunks": len(self.chunks), "n_lines": len(self.lines)}, f, ensure_ascii=False)

    # 병명을 KB 표준 병명으로 정규화한다
    def resolve(self, raw_name: str) -> dict:
        if not raw_name:
            return {"disease": None, "how": "unknown", "score": 0.0}
        n = raw_name.strip()
        if n in self.kb:
            return {"disease": n, "how": "exact", "score": 1.0}
        if n in SYNONYM and SYNONYM[n] in self.kb:
            return {"disease": SYNONYM[n], "how": "synonym", "score": 1.0}
        for kb_name in self.names:
            if n in kb_name or kb_name in n:
                return {"disease": kb_name, "how": "partial", "score": 1.0}
        q = self.model.encode([f"query: {n}"], normalize_embeddings=True)[0]
        sims = self.name_emb @ q
        i = int(np.argmax(sims))
        score = float(sims[i])
        if score < RESOLVE_THRESHOLD:
            return {"disease": None, "how": "unknown", "score": score}
        return {"disease": self.names[i], "how": "rag", "score": score}

    # 임베딩 유사도가 높은 KB 병명 후보 k개를 돌려준다
    def suggest(self, raw_name: str, k: int = 3) -> list:
        if not raw_name:
            return []
        q = self.model.encode([f"query: {raw_name}"], normalize_embeddings=True)[0]
        sims = self.name_emb @ q
        return [(self.names[i], float(sims[i])) for i in np.argsort(-sims)[:k]]

    # 한 병해의 섹션 청크를 질문 유사도 순으로 고른다
    def retrieve(self, disease: str, question: str = None, top_k: int = 2) -> list:
        idxs = [i for i, c in enumerate(self.chunks) if c["disease"] == disease]
        if not idxs:
            return []
        if not question or not question.strip():
            return [self.chunks[i] for i in idxs]
        q = self.model.encode([f"query: {question}"], normalize_embeddings=True)[0]
        sims = self.chunk_emb[idxs] @ q
        return [dict(self.chunks[idxs[o]], score=float(sims[o]))
                for o in np.argsort(-sims)[:top_k]]

    # 한 섹션 안의 문장을 질문 유사도 순으로 다시 고른다
    def retrieve_lines(self, disease: str, question: str, section: str, top_n: int = TOP_LINES) -> list:
        idxs = [i for i, l in enumerate(self.lines)
                if l["disease"] == disease and l["section"] == section]
        if not idxs or not question:
            return []
        q = self.model.encode([f"query: {question}"], normalize_embeddings=True)[0]
        sims = self.line_emb[idxs] @ q
        return [dict(self.lines[idxs[o]], score=float(sims[o]))
                for o in np.argsort(-sims)[:top_n]]

    # 모델 생성 없이 KB 원문만으로 마크다운 답변을 조립한다
    def build_answer(self, disease: str, question: str = None) -> str:
        k = self.kb[disease]
        out = [f"## 진단: **{disease}**", ""]
        ident = []
        if k["scientific"] and "미확정" not in k["scientific"]:
            ident.append(f"학명 {k['scientific']}")
        if k["english"] and "가칭" not in k["english"]:
            ident.append(f"영명 {k['english']}")
        out += [f"*{' / '.join(ident)}*", ""] if ident else \
               ["> 이 병해는 KB에 학명이 확정되어 있지 않습니다(전문가 검증 필요).", ""]

        if not question:
            for c in self.retrieve(disease, None):
                out += [f"### {c['section']}", _body(c), ""]
        else:
            secs = self.retrieve(disease, question, top_k=2)
            out += [f"**질문:** {question}", ""]
            top = secs[0]

            if top["section"] in LIST_SECTIONS:
                out += [f"### {top['section']}", _body(top), ""]
            else:
                best = self.retrieve_lines(disease, question, top["section"])
                n_total = sum(1 for l in self.lines
                              if l["disease"] == disease and l["section"] == top["section"])
                out += ["### 질문에 대한 답", ""]
                out += [f"- {l['text']}" for l in best] if best else [_body(top)]
                out.append("")
                if best and len(best) < n_total:
                    out += [f"<details><summary>「{top['section']}」 원문 전체 보기 "
                            f"(문장 {n_total}개)</summary>",
                            "", _body(top), "", "</details>", ""]

            for s in secs[1:]:
                out += [f"<details><summary>참고: 「{s['section']}」</summary>",
                        "", _body(s), "", "</details>", ""]

        out += ["---",
                "<sub>위 정보는 모두 전문가 지식베이스(citrus_PD_ver3.md) 원문에서 조회한 것이며, "
                "모델이 생성한 문장이 아닙니다. 이미지만으로 확정 진단은 어려우므로 현장 확인이 필요합니다.</sub>"]
        return "\n".join(out)


# 이름 해석과 질문별 섹션·문장 검색 결과를 출력한다
def main():
    r = KBRetriever(device="cpu")
    print(f"KB {len(r.names)}종 / 섹션청크 {len(r.chunks)}개 / 문장 {len(r.lines)}개\n")

    print("=== 이름 해석 ===")
    for n in ["탄저병", "흑점병", "녹색곰팡이병", "검은반점병"]:
        res = r.resolve(n)
        print(f"  '{n}' → {res['disease']} ({res['how']}, {res['score']:.3f})")

    print("\n=== 하이브리드 검색: 질문별 상위 섹션 + 문장 ===")
    for q in ["약제 뭐 써요?", "언제 방제해요?", "어떤 증상이 보이나요?", "왜 생기나요?"]:
        s = r.retrieve("궤양병", q, top_k=1)[0]
        print(f"\n  '{q}' → [{s['section']}] {s['score']:.3f}")
        if s["section"] in LIST_SECTIONS:
            print(f"      (목록 전체 반환, {len(_body(s))}자)")
        else:
            for l in r.retrieve_lines("궤양병", q, s["section"], top_n=2):
                print(f"      · {l['score']:.3f}  {l['text'][:60]}...")


if __name__ == "__main__":
    main()
