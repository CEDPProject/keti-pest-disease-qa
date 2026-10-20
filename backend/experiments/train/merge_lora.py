import os
import argparse

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from backend.core.classes import PLATFORM_BASE_MODEL


# 플랫폼 기본 모델에 LoRA 어댑터를 병합해 서빙용 모델로 저장한다
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("base 로딩...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        PLATFORM_BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
    print("LoRA 병합...")
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(PLATFORM_BASE_MODEL).save_pretrained(args.out)
    print(f"병합 모델 저장 완료 → {args.out}")


if __name__ == "__main__":
    main()
