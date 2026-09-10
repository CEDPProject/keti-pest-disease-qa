#!/bin/bash
# Qwen3.5-4B LoRA SFT 실행 (qwenvl-ft 환경) — 2.5-VL-7B 공정 비교용
# 2.5와 동일 데이터/하이퍼, 베이스 모델·template(qwen3_5_nothink)만 교체.
set -euo pipefail
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate qwenvl-ft

export CUDA_VISIBLE_DEVICES=0              # GPU0 (48G 여유)
export HF_HOME=/opt/project_data/users/jhlee/hf_cache
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=0

CFG=/home/jhlee/CLUST_KETI/keti-pest-disease-qa/core/train/qwen3_5_lora_sft.yaml
LOG=/home/jhlee/CLUST_KETI/keti-pest-disease-qa/core/train/train_qwen35.log
cd /home/jhlee/tools/LLaMA-Factory

echo "== Qwen3.5-4B LoRA SFT 시작 =="; date
llamafactory-cli train "$CFG" 2>&1 | tee "$LOG"
echo "== Qwen3.5-4B LoRA SFT 완료 (exit ${PIPESTATUS[0]}) =="; date
