#!/bin/bash
# Qwen2.5-VL-7B LoRA SFT (플랫폼 서빙 모델: 학습 후 merge_lora.py 로 병합)
# 사용: bash run_train.sh   (conda env: qwenvl-ft, LLaMA-Factory 필요)
set -euo pipefail

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
LOG="$SCRIPT_DIR/train.log"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwenvl-ft
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=0

CFG="$(mktemp --suffix=.yaml)"
sed "s#__REPO__#$REPO#g" "$SCRIPT_DIR/qwen2_5vl_lora_sft.yaml" > "$CFG"

cd "$LLAMAFACTORY_DIR"
echo "== LoRA SFT 시작 =="; date
llamafactory-cli train "$CFG" 2>&1 | tee "$LOG"
echo "== LoRA SFT 완료 (exit ${PIPESTATUS[0]}) =="; date
rm -f "$CFG"
