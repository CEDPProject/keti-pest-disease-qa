#!/bin/bash
# Qwen2.5-VL-7B LoRA SFT 실행 (qwenvl-ft 환경)
set -euo pipefail
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate qwenvl-ft

export CUDA_VISIBLE_DEVICES=0        # GPU0 사용 (학습 중에는 데모 내림)
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=0

CFG=/home/jhlee/CLUST_KETI/keti-pest-disease-qa/research/train/qwen2_5vl_lora_sft.yaml
LOG=/home/jhlee/CLUST_KETI/keti-pest-disease-qa/research/train/train.log
cd /home/jhlee/tools/LLaMA-Factory

echo "== LoRA SFT 시작 =="; date
llamafactory-cli train "$CFG" 2>&1 | tee "$LOG"
echo "== LoRA SFT 완료 (exit ${PIPESTATUS[0]}) =="; date
