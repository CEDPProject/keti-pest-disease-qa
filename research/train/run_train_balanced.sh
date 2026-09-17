#!/bin/bash
# 감귤 8클래스 균형 데이터셋 Qwen3.5-4B LoRA SFT (장시간 안전 실행)
set -uo pipefail
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate qwenvl-ft
export CUDA_VISIBLE_DEVICES=1        # GPU1 단일 (safety env가 없는 GPU. GPU0·2는 jwmoon safety 점유)
export HF_HOME=/opt/project_data/users/jhlee/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=0    # 단일 GPU

BASE=/home/jhlee/CLUST_KETI/keti-pest-disease-qa/research/train
CFG=$BASE/qwen3_5_balanced_lora_sft.yaml
LOG=$BASE/train_balanced.log
MARKER=$BASE/train_balanced.done

cd /home/jhlee/tools/LLaMA-Factory
{
  echo "== 균형 8클래스 LoRA SFT 시작 =="; date
  llamafactory-cli train "$CFG"
  code=$?
  echo "== 완료 (exit $code) =="; date
  echo "$code" > "$MARKER"
} 2>&1 | tee "$LOG"
