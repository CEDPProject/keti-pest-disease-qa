import os
import sys
import io
import json
import base64
import argparse
import urllib.request

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

API_DIR = os.path.dirname(os.path.abspath(__file__))       # backend/api/
REPO_DIR = os.path.dirname(os.path.dirname(API_DIR))       # repo
FRONTEND_DIR = os.path.join(REPO_DIR, "frontend")
sys.path.insert(0, os.path.join(REPO_DIR, "backend", "core"))
from classes import DRAFT_KB_LABELS, MERGED_MODEL_DIR  # noqa
from settings import SERVER_HOST, SERVER_PORT, OLLAMA_URL, OLLAMA_MODEL  # noqa

DEFAULT_MODEL = MERGED_MODEL_DIR
DEFAULT_PORT = SERVER_PORT

# ── LLM 종합(grounded) 설정 ───────────────────────────────────────
# 최종 답을 gemma(Ollama)로 KB 근거만 써서 자연스럽게 종합. 프론트는 원문을 토글로 병기.
# 롤백: SYNTHESIZE = False  → synthesis 미포함 → 프론트가 기존처럼 KB 원문 파트별 표시.
SYNTHESIZE = True
_INTENT_ASK = {
    "diagnose": "이 감귤을 진단하고 관찰 증상과 방제 방법을 종합해서 설명해줘.",
    "name": "이 감귤이 무슨 병인지, 어떤 특징인지 알려줘.",
    "control": "이 병은 어떻게 방제하고 무슨 약제를 써야 해?",
    "symptom": "이 감귤에 어떤 증상이 관찰되는지 설명해줘.",
}


def synthesize_answer(disease: str, info: dict, intent: str):
    """검색된 KB 정보만 근거로 자연스러운 답변 생성(grounded). 실패 시 None → 프론트가 원문 표시."""
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
        f"\n\n[농가 질문] {_INTENT_ASK.get(intent, _INTENT_ASK['diagnose'])}\n[답변]"
    )
    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                             "options": {"temperature": 0.3, "num_predict": 400}}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())["response"].strip()
    except Exception as e:
        print(f"[serve] synthesis 실패(원문으로 폴백): {e}")
        return None

# 전역 파이프라인 (startup 시 로드)
_pipe = None
_kb = None
_kb_name_to_key = None


def load_pipeline(model_dir: str):
    """실제 모델 + KB 로드."""
    global _pipe, _kb, _kb_name_to_key
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from kb_retriever import KBRetriever
    from name_rag_pipeline import NameRagPipeline, kb_name_to_key

    print(f"[serve] 모델 로딩: {model_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(model_dir, max_pixels=768 * 768)
    _kb = KBRetriever(device="cuda")
    _pipe = NameRagPipeline(model, proc, _kb)
    _kb_name_to_key = kb_name_to_key
    print("[serve] 모델 + KB 로딩 완료")


def load_kb_only():
    """--mock: KB 만 로드(CPU) — 응답 구조는 동일, 병명은 고정 예시."""
    global _kb, _kb_name_to_key
    from kb_retriever import KBRetriever
    from name_rag_pipeline import kb_name_to_key
    _kb = KBRetriever(device="cpu")
    _kb_name_to_key = kb_name_to_key
    print("[serve] (mock) KB 만 로딩 — 모델 없음")


def clean_info(k: dict) -> dict:
    """KB 원본 → 프론트 계약. 미확정 학명/가칭 영명은 null 로."""
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


def build_response(disease_name_from_model: str, raw: str, intent: str) -> dict:
    """모델이 낸 병명 → RAG 이름 해석 → 구조화 응답."""
    if disease_name_from_model is None:
        sugg = _kb.suggest((raw or "")[:40], k=3)
        return {"ok": True, "disease": None, "disease_key": None,
                "name_resolution": "unknown",
                "resolution_score": round(sugg[0][1], 4) if sugg else 0.0,
                "resolved_from": raw.strip()[:20] if raw else None,
                "vlm_raw": raw, "intent": intent,
                "suggestions": [[n, round(s, 4)] for n, s in sugg]}

    res = _kb.resolve(disease_name_from_model)
    if res["disease"] is None:
        sugg = _kb.suggest(disease_name_from_model, k=3)
        return {"ok": True, "disease": None, "disease_key": None,
                "name_resolution": "unknown",
                "resolution_score": round(res["score"], 4),
                "resolved_from": disease_name_from_model,
                "vlm_raw": raw, "intent": intent,
                "suggestions": [[n, round(s, 4)] for n, s in sugg]}

    disease = res["disease"]
    info = clean_info(_kb.kb[disease])
    resp = {
        "ok": True,
        "disease": disease,
        "disease_key": _kb_name_to_key(disease),
        "name_resolution": res["how"],
        "resolution_score": round(res["score"], 4),
        "resolved_from": disease_name_from_model if res["how"] in ("synonym", "partial", "rag") else None,
        "is_draft_kb": disease in DRAFT_KB_LABELS,
        "vlm_raw": raw,
        "intent": intent,
        "info": info,
    }
    resp["synthesis"] = synthesize_answer(disease, info, intent)   # None 이면 프론트가 원문 표시
    return resp


def decode_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---- 앱 ----
app = FastAPI(title="감귤 병해충 진단 데모")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MOCK_MODE = False
MOCK_ROTATION = ["궤양병", "검은점무늬병", "녹색곰팡이병", "탄저병", "더뎅이병"]
_mock_i = {"n": 0}


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"ok": True, "mock": MOCK_MODE, "model_loaded": _pipe is not None}


@app.post("/diagnose")
async def diagnose(req: Request):
    body = await req.json()
    intent = body.get("intent") or "diagnose"

    if MOCK_MODE:
        # 모델 없이: 병명을 순환 지정하되, 이름해석·KB조회·구조는 실제와 동일
        name = MOCK_ROTATION[_mock_i["n"] % len(MOCK_ROTATION)]
        _mock_i["n"] += 1
        raw = f"{name}입니다. (mock — 모델 미로드)"
        return JSONResponse(build_response(name, raw, intent))

    try:
        img = decode_image(body["image_base64"])
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"이미지 디코딩 실패: {e}"}, status_code=400)

    # VLM: 병명만 추출 (나머지 생성 텍스트는 폐기)
    name, raw = _pipe._vlm_name(img)
    return JSONResponse(build_response(name, raw, intent))


def main():
    global MOCK_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default=SERVER_HOST)
    ap.add_argument("--model_dir", default=DEFAULT_MODEL)
    ap.add_argument("--mock", action="store_true", help="모델 없이 배선만 테스트")
    args = ap.parse_args()

    MOCK_MODE = args.mock
    if args.mock:
        load_kb_only()
    else:
        load_pipeline(args.model_dir)

    import uvicorn
    print(f"[serve] http://{args.host}:{args.port}  (mock={args.mock})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
