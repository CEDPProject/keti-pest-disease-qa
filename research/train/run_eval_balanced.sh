#!/bin/bash
# 균형 8클래스 학습 완료 후 에폭별 자동 평가 → 최적 에폭 선정
set -uo pipefail
source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate qwenvl-ft
export CUDA_VISIBLE_DEVICES=1        # GPU1 (safety 없는 GPU. 학습 끝나면 비어있음)
export HF_HOME=/opt/project_data/users/jhlee/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE=/home/jhlee/CLUST_KETI/keti-pest-disease-qa
LOG=$BASE/core/train/eval_balanced.log
{
  echo "== 에폭별 평가 시작 =="; date
  cd $BASE/core/eval
  python eval_balanced.py --output_dir $BASE/models/qwen3_5_balanced_lora
  echo "== 평가 완료 =="; date
} 2>&1 | tee "$LOG"
