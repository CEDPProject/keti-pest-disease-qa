import os
import sys
import argparse

import torch
import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo (web/gradio_app.py 기준)
CORE_DIR = os.path.join(REPO_DIR, "backend")
sys.path.insert(0, os.path.join(CORE_DIR, "common"))
sys.path.insert(0, os.path.join(CORE_DIR, "rag"))
from kb_retriever import KBRetriever  # noqa
from name_rag_pipeline import NameRagPipeline  # noqa

from classes import MERGED_MODEL_DIR as MERGED_DEFAULT  # noqa
_pipe = None


def load(model_dir):
    global _pipe
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(model_dir, max_pixels=768 * 768)
    _pipe = NameRagPipeline(model, proc, KBRetriever(device="cuda"))
    print("모델 + KB 로딩 완료")


def diagnose(image, user_question):
    if image is None:
        return "이미지를 올려주세요.", ""
    out = _pipe.run(image, (user_question or "").strip() or None)

    # 좌측: KB 기반 최종 답변
    left = out["answer"]

    # 우측: 파이프라인 내부 동작(투명성)
    how = {"exact": "KB 정확 일치", "synonym": "동의어 매핑",
           "partial": "부분 일치", "rag": "RAG 최근접 매핑", "unknown": "미확인"}
    right = ["### 🔎 파이프라인 내부", "",
             f"**1) VLM 출력(병명만 취함)**  \n`{out['vlm_raw'][:80].splitlines()[0] if out['vlm_raw'] else '-'}`", "",
             f"**2) 병명 해석**  \n{out['disease'] or '불명확'} — {how.get(out['how'], out['how'])}"]
    if out.get("score") is not None and out["how"] == "rag":
        right.append(f"  (유사도 {out['score']:.3f})")
    if out.get("suggestions"):
        right += ["", "**유사 후보**"] + [f"- {n} ({s:.3f})" for n, s in out["suggestions"]]
    right += ["", "**3) 정보 출처**  \n지식베이스 원문 조회 (모델 생성 아님)"]
    return left, "\n".join(right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=MERGED_DEFAULT)
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    load(args.model_dir)

    with gr.Blocks(title="감귤 병해충 AI 진단") as demo:
        gr.Markdown("# 🍊 감귤 병해충 AI 진단")
        gr.Markdown("**VLM은 병명만 판단하고, 모든 정보(학명·증상·방제·약제)는 전문가 지식베이스에서 조회합니다.**  \n"
                    "모델이 사실을 생성하지 않으므로 환각이 발생하지 않습니다.")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="pil", label="감귤 사진")
                uq = gr.Textbox(label="질문 (선택)",
                                placeholder="예) 약제 뭐 써요? / 어떤 증상인가요? / 왜 생기나요?")
                btn = gr.Button("진단하기", variant="primary")
                gr.Markdown("<sub>⚠️ 이미지만으로 확정 진단은 어렵습니다. 현장 확인·전문가 검토가 필요합니다.  \n"
                            "⚠️ 이 모델은 학습 데이터(121장)와 다른 출처의 사진에서는 정확도가 크게 떨어집니다.</sub>")
            with gr.Column(scale=2):
                # sanitize_html=False : KB 원문을 접는 <details> 태그를 렌더링하기 위함
                out_ans = gr.Markdown(label="진단 결과", sanitize_html=False)
            with gr.Column(scale=1):
                out_dbg = gr.Markdown(label="파이프라인")
        btn.click(diagnose, inputs=[img, uq], outputs=[out_ans, out_dbg])

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
