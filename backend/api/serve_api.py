import io
import os
import base64
import argparse

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from backend.core.classes import DRAFT_KB_LABELS, MERGED_MODEL_DIR, REPO_DIR
from backend.core.kb_retriever import KBRetriever
from backend.core.name_rag_pipeline import NameRagPipeline, kb_name_to_key
from backend.core.ollama_client import ollama_generate
from backend.core.settings import OLLAMA_MODEL, SERVER_HOST, SERVER_PORT
from backend.core.vlm import VLMRunner, load_qwen25

FRONTEND_INDEX = os.path.join(REPO_DIR, "frontend", "index.html")
SYNTHESIZE = True
INTENT_QUESTIONS = {
    "diagnose": "이 감귤을 진단하고 관찰 증상과 방제 방법을 종합해서 설명해줘.",
    "name": "이 감귤이 무슨 병인지, 어떤 특징인지 알려줘.",
    "control": "이 병은 어떻게 방제하고 무슨 약제를 써야 해?",
    "symptom": "이 감귤에 어떤 증상이 관찰되는지 설명해줘.",
}
MOCK_ROTATION = ["궤양병", "검은점무늬병", "녹색곰팡이병", "탄저병", "더뎅이병"]


# KB 근거만으로 답변 문장을 LLM 으로 정리한다(실패 시 None)
def synthesize_answer(disease: str, info: dict, intent: str):
    if not SYNTHESIZE:
        return None
    sci, eng = info.get("scientific"), info.get("english")
    head = f"- 병명: {disease}"
    tail = " / ".join(x for x in [f"학명 {sci}" if sci else None, f"영명 {eng}" if eng else None] if x)
    if tail:
        head += f" ({tail})"
    kb = [head]
    for key, label in [("symptoms", "증상"), ("ecology", "발생 생태"),
                       ("control", "방제/관리"), ("pesticides", "등록 약제(예시)")]:
        vals = info.get(key) or []
        if vals:
            kb.append(f"- {label}: " + (", ".join(vals) if key == "pesticides" else " / ".join(vals)))
    prompt = (
        "당신은 감귤 병해충 상담 도우미입니다. 아래 [지식베이스]에 있는 내용만 근거로, "
        "농가의 질문에 자연스럽고 친절하게 3~5문장으로 답하세요.\n"
        "[규칙] 지식베이스에 없는 사실(학명·약제 이름·수치·날짜 등)을 절대 지어내지 마세요. "
        f"병명은 이미 '{disease}'로 확정됐습니다. 정보가 없는 항목은 언급하지 마세요.\n\n"
        "[지식베이스]\n" + "\n".join(kb) +
        f"\n\n[농가 질문] {INTENT_QUESTIONS.get(intent, INTENT_QUESTIONS['diagnose'])}\n[답변]"
    )
    try:
        return ollama_generate(prompt, OLLAMA_MODEL, {"temperature": 0.3, "num_predict": 400}, timeout=60)
    except Exception as e:
        print(f"[serve] synthesis 실패(원문으로 폴백): {e}")
        return None


# KB 항목을 화면 응답 형식으로 바꾼다
def clean_info(k: dict) -> dict:
    sci = k.get("scientific") or None
    if sci and "미확정" in sci:
        sci = None
    eng = k.get("english") or None
    if eng and "가칭" in eng:
        eng = None
    return {
        "scientific": sci, "english": eng,
        "symptoms": k.get("symptoms") or [],
        "ecology": k.get("ecology") or [],
        "control": k.get("control") or [],
        "pesticides": k.get("pesticides") or [],
    }


# base64 문자열(data URL 포함)을 RGB 이미지로 디코딩한다
def decode_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# 병명 판정, 이름 정규화, KB 조회를 묶어 진단 응답을 만드는 서비스
class DiagnosisService:
    # KB 검색기와 진단 파이프라인(모의 모드면 None)을 받는다
    def __init__(self, retriever: KBRetriever, pipeline: NameRagPipeline = None):
        self.kb = retriever
        self.pipe = pipeline
        self.mock_count = 0

    # 서빙 모델과 KB 를 GPU 에 올린 서비스를 만든다
    @classmethod
    def from_model(cls, model_dir: str):
        print(f"[serve] 모델 로딩: {model_dir}")
        model, proc = load_qwen25(model_dir)
        retriever = KBRetriever(device="cuda")
        service = cls(retriever, NameRagPipeline(VLMRunner(model, proc), retriever))
        print("[serve] 모델 + KB 로딩 완료")
        return service

    # 모델 없이 KB 만 CPU 에 올린 모의 서비스를 만든다
    @classmethod
    def mock_only(cls):
        service = cls(KBRetriever(device="cpu"))
        print("[serve] (mock) KB 만 로딩 — 모델 없음")
        return service

    # 모델 없이 동작하는 모의 모드인지 알려준다
    @property
    def mock(self) -> bool:
        return self.pipe is None

    # 모델 병명을 KB 로 정규화해 진단 응답을 만든다
    def build_response(self, disease_name_from_model: str, raw: str, intent: str) -> dict:
        if disease_name_from_model is None:
            sugg = self.kb.suggest((raw or "")[:40], k=3)
            return {"ok": True, "disease": None, "disease_key": None,
                    "name_resolution": "unknown",
                    "resolution_score": round(sugg[0][1], 4) if sugg else 0.0,
                    "resolved_from": raw.strip()[:20] if raw else None,
                    "vlm_raw": raw, "intent": intent,
                    "suggestions": [[n, round(s, 4)] for n, s in sugg]}

        res = self.kb.resolve(disease_name_from_model)
        if res["disease"] is None:
            sugg = self.kb.suggest(disease_name_from_model, k=3)
            return {"ok": True, "disease": None, "disease_key": None,
                    "name_resolution": "unknown",
                    "resolution_score": round(res["score"], 4),
                    "resolved_from": disease_name_from_model,
                    "vlm_raw": raw, "intent": intent,
                    "suggestions": [[n, round(s, 4)] for n, s in sugg]}

        disease = res["disease"]
        info = clean_info(self.kb.kb[disease])
        resp = {
            "ok": True,
            "disease": disease,
            "disease_key": kb_name_to_key(disease),
            "name_resolution": res["how"],
            "resolution_score": round(res["score"], 4),
            "resolved_from": disease_name_from_model if res["how"] in ("synonym", "partial", "rag") else None,
            "is_draft_kb": disease in DRAFT_KB_LABELS,
            "vlm_raw": raw,
            "intent": intent,
            "info": info,
        }
        resp["synthesis"] = synthesize_answer(disease, info, intent)
        return resp

    # 요청 본문으로 진단 응답과 HTTP 상태 코드를 만든다
    def diagnose(self, body: dict):
        intent = body.get("intent") or "diagnose"
        if self.mock:
            name = MOCK_ROTATION[self.mock_count % len(MOCK_ROTATION)]
            self.mock_count += 1
            return self.build_response(name, f"{name}입니다. (mock — 모델 미로드)", intent), 200
        try:
            img = decode_image(body["image_base64"])
        except Exception as e:
            return {"ok": False, "error": f"이미지 디코딩 실패: {e}"}, 400
        name, raw = self.pipe.predict_name(img)
        return self.build_response(name, raw, intent), 200


app = FastAPI(title="감귤 병해충 진단 데모")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.state.service = None


# 웹 화면을 돌려준다
@app.get("/")
def index():
    return FileResponse(FRONTEND_INDEX)


# 서버 상태와 모델 로딩 여부를 알려준다
@app.get("/health")
def health():
    service = app.state.service
    return {"ok": True, "mock": service.mock, "model_loaded": not service.mock}


# 사진 진단 요청을 받아 병명과 KB 정보를 응답한다
@app.post("/diagnose")
async def diagnose(req: Request):
    payload, status = app.state.service.diagnose(await req.json())
    return JSONResponse(payload, status_code=status)


# 명령행 인자로 서비스를 준비하고 웹 서버를 띄운다
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=SERVER_PORT)
    ap.add_argument("--host", default=SERVER_HOST)
    ap.add_argument("--model_dir", default=MERGED_MODEL_DIR)
    ap.add_argument("--mock", action="store_true", help="모델 없이 배선만 테스트")
    args = ap.parse_args()

    app.state.service = DiagnosisService.mock_only() if args.mock else DiagnosisService.from_model(args.model_dir)
    print(f"[serve] http://{args.host}:{args.port}  (mock={args.mock})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
