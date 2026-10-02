#!/bin/bash
# 9B 학습 완료 후 논문용 측정 전체 자동 실행.  사용: bash run_eval_9b.sh
# 순서: M2(에폭스윕/메인수치) → M3(top3) → M4(selective) → M5(환각) → M1(ablation)
# 각 단계는 자체 done 마커로 재실행 안전.
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
BASE="$SCRIPT_DIR"
EVAL="$REPO/backend/experiments/eval"
MODEL="$REPO/models/qwen3_5_9b_balanced_lora"
RES="$REPO/qa_docs/experiment_results"
ALLDONE=$BASE/eval_9b_all.done
LOG=$BASE/eval_9b.log
[ -f "$ALLDONE" ] && exit 0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwenvl-ft
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$EVAL"

step () {  # step <marker> <resultfile> <cmd...>
  local marker="$1"; shift
  local result="$1"; shift
  if [ -f "$marker" ]; then echo "[skip] $marker 존재"; return 0; fi
  echo "== 실행: $* =="; date
  "$@"
  if [ -f "$result" ]; then echo "$(date)" > "$marker"; echo "[done] $marker"; else echo "[FAIL] $result 미생성"; return 1; fi
}

{
  echo "########## 9B 논문 측정 체인 시작 ##########"; date
  # M2: 에폭 스윕 → balanced_curve_9b.json (best epoch). 즉시보고①
  step "$BASE/eval_9b_balanced.done" "$RES/balanced_curve_9b.json" \
      python eval_9b_balanced.py --output_dir "$MODEL" || exit 1
  # M3: top-3 스코어링 → b_stage1_top3_9b.json. 즉시보고②
  step "$BASE/m3_9b.done" "$RES/b_stage1_top3_9b.json" python score_9b_top3.py || exit 1
  # M4: selective → exp3_confidence_9b.json (CPU)
  step "$BASE/m4_9b.done" "$RES/exp3_confidence_9b.json" python selective_9b.py || exit 1
  # M5: 환각 2조건 → p1_2_hallucination_9b.json. 즉시보고③
  step "$BASE/m5_9b.done" "$RES/p1_2_hallucination_9b.json" python halluc_9b.py || exit 1
  # M1: ablation 6셀 → ablation_9b.json
  step "$BASE/m1_9b.done" "$RES/ablation_9b.json" python eval_9b_ablation.py || exit 1
  echo "$(date)" > "$ALLDONE"
  echo "########## 9B 측정 체인 전체 완료 ##########"; date
} >> "$LOG" 2>&1
