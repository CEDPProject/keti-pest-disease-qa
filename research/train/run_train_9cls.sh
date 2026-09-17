#!/bin/bash
# 감귤 9클래스(질병8 + 정상) Qwen3.5-4B LoRA SFT
set -uo pipefail
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate qwenvl-ft
export CUDA_VISIBLE_DEVICES=2        # GPU2 (여유 GPU)
export HF_HOME=/opt/project_data/users/jhlee/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=0
CFG=/home/jhlee/CLUST_KETI/keti-pest-disease-qa/research/train/qwen3_5_9cls_lora_sft.yaml
LOG=/home/jhlee/CLUST_KETI/keti-pest-disease-qa/research/train/train_9cls.log
cd /home/jhlee/tools/LLaMA-Factory
echo "== 9클래스 LoRA SFT 시작 =="; date
llamafactory-cli train "$CFG" 2>&1 | tee "$LOG"
echo "== 9클래스 LoRA SFT 완료 (exit ${PIPESTATUS[0]}) =="; date
