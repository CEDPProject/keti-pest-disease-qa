#!/bin/bash
# 감귤 8클래스 균형 데이터셋 Qwen3.5-9B LoRA SFT (논문 메인 모델)
# 사용: bash run_train_9b.sh   (conda env: qwenvl-ft, LLaMA-Factory 필요)
# 완료 시 train_9b.done 에 exit code 기록 → 재실행해도 건너뜀.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# 환경별 설정: .env 에서 셸에 아직 없는 값만 채운다 (.env.example 참고)
if [ -f "$REPO/.env" ]; then
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    [ -z "${!k+x}" ] && export "$k=$v"
  done < "$REPO/.env"
fi
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-$HOME/tools/LLaMA-Factory}"
MARKER="$SCRIPT_DIR/train_9b.done"
LOG="$SCRIPT_DIR/train_9b.log"
[ -f "$MARKER" ] && exit 0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwenvl-ft
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=0

CFG="$(mktemp --suffix=.yaml)"
sed "s#__REPO__#$REPO#g" "$SCRIPT_DIR/qwen3_5_9b_balanced_lora_sft.yaml" > "$CFG"

cd "$LLAMAFACTORY_DIR"
{
  echo "== Qwen3.5-9B 균형 8클래스 LoRA SFT 시작 =="; date
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  llamafactory-cli train "$CFG"
  code=$?
  echo "== 완료 (exit $code) =="; date
  echo "$code" > "$MARKER"
} >> "$LOG" 2>&1
rm -f "$CFG"
