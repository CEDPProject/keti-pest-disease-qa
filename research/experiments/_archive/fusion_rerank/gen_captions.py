"""STAGE F1 (1/2): 베이스 Qwen2.5-VL-7B 로 test 57장 시각 캡션 생성 (병명 없이).

파인튜닝 안 된 베이스 모델 사용 → 자기 판단 편향의 순환 유입 차단.
캡션은 병명 언급 없이 시각 특징만. greedy. JSON 캐싱.

결과: research/results/fusion_captions.json
"""
import os
import sys
import json

os.environ.setdefault("HF_HOME", "/opt/project_data/users/jhlee/hf_cache")
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(CORE)

BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
TEST = os.path.join(REPO, "data", "vlm_finetune_balanced", "test.jsonl")
OUT = os.path.join(CORE, "results", "fusion_captions.json")
PROMPT = ("이 감귤 사진에서 보이는 시각적 특징을 병명 언급 없이 서술하라. "
          "병반의 모양·색·질감·크기·위치·표면 상태만 구체적으로 적어라.")
MAX_PIXELS = 768 * 768


@torch.no_grad()
def caption(model, proc, path):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": path, "max_pixels": MAX_PIXELS},
        {"type": "text", "text": PROMPT}]}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    im = Image.open(path).convert("RGB")
    inp = proc(text=[t], images=[im], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=200, do_sample=False)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    test = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    print(f"베이스 로딩: {BASE} · test {len(test)}장")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(BASE, max_pixels=MAX_PIXELS)

    caps = {}
    for i, r in enumerate(test):
        b = os.path.basename(r["image"])
        caps[b] = {"image": b, "gold": r["key"], "caption": caption(model, proc, r["image"])}
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}")
    json.dump(caps, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"저장: {OUT}")
    # 샘플 2개
    for b in list(caps)[:2]:
        print(f"\n[{b}] gold={caps[b]['gold']}\n  {caps[b]['caption'][:200]}")


if __name__ == "__main__":
    main()
