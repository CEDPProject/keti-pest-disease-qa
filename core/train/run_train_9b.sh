#!/bin/bash
# 감귤 8클래스 균형 데이터셋 Qwen3.5-9B LoRA SFT (논문 메인, 장시간 안전 실행)
# cron+flock 로 호출됨. 완료 시 MARKER에 exit code 기록.
set -uo pipefail

BASE=/home/jhlee/CLUST_KETI/keti-pest-disease-qa/core/train
MARKER=$BASE/train_9b.done
LOG=$BASE/train_9b.log

# 이미 완료됐으면 아무것도 안 함 (cron 재발화 대비)
[ -f "$MARKER" ] && exit 0

source /home/jhlee/miniconda3/etc/profile.d/conda.sh
conda activate qwenvl-ft
export CUDA_VISIBLE_DEVICES=0        # GPU0 단일 (현재 비어있음. GPU1 점유중)
export HF_HOME=/opt/project_data/users/jhlee/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=0    # 단일 GPU

CFG=$BASE/qwen3_5_9b_balanced_lora_sft.yaml

cd /home/jhlee/tools/LLaMA-Factory
{
  echo "== Qwen3.5-9B 균형 8클래스 LoRA SFT 시작 =="; date
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  llamafactory-cli train "$CFG"
  code=$?
  echo "== 완료 (exit $code) =="; date
  echo "$code" > "$MARKER"
} >> "$LOG" 2>&1
