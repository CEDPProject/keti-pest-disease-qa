"""감귤 병해충 진단 데모 웹앱 백엔드 (FastAPI).

  GET  /          : web/frontend/index.html 서빙
  GET  /health    : 상태
  POST /diagnose  : {image_base64, question, intent} → 구조화 진단 JSON

기존 파이프라인(name_rag_pipeline.NameRagPipeline + KBRetriever)을 재사용한다.
  - VLM 은 병명 하나만 결정
  - 학명·증상·약제 등 정보는 KB 원문에서 조회 (모델이 생성하지 않음 → 환각 없음)

실행 (루트 app.py 가 이 main() 을 호출):
  python app.py --port 5100          # 실제 모델 (GPU ~16GB)
  python app.py --port 5100 --mock   # 모델 없이 UI 테스트
"""

import os
import sys
import io
import base64
import argparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))   # web/backend/
WEB_DIR = os.path.dirname(BACKEND_DIR)                     # web/
REPO_DIR = os.path.dirname(WEB_DIR)                        # repo
CORE_DIR = os.path.join(REPO_DIR, "core")
FRONTEND_DIR = os.path.join(WEB_DIR, "frontend")
sys.path.insert(0, os.path.join(CORE_DIR, "common"))
sys.path.insert(0, os.path.join(CORE_DIR, "rag"))
from classes import DRAFT_KB_LABELS, MERGED_MODEL_DIR  # noqa

DEFAULT_MODEL = MERGED_MODEL_DIR
DEFAULT_PORT = 5100

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
    return {
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


def decode_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---- 앱 ----
app = FastAPI(title="감귤 병해충 진단 데모")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MOCK_MODE = False
MOCK_ROTATION = ["궤양병", "검은점무늬병", "녹색곰팡이병", "탄저병", "검은무늬병"]
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
    ap.add_argument("--host", default="0.0.0.0")
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
