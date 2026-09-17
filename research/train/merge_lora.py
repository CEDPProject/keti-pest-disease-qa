import os
import argparse

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("base 로딩...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
    print("LoRA 병합...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(BASE_MODEL).save_pretrained(args.out)
    print(f"병합 모델 저장 완료 → {args.out}")


if __name__ == "__main__":
    main()
