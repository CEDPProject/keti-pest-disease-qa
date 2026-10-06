import os
import sys
import argparse
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "backend", "core"))
import settings  # noqa: E402,F401  (.env 로딩)
from classes import RESULTS_DIR, LOGS_DIR, MODELS_DIR  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(MODELS_DIR, "qwen3_5_9b_balanced_lora")

# (단계 이름, 실행 스크립트와 인자, 결과 파일) — 위에서부터 순서대로 실행
STEPS = [
    ("eval_9b_balanced", ["eval_9b_balanced.py", "--output_dir", MODEL], "balanced_curve_9b.json"),  # 에폭 스윕 · 최적 에폭
    ("score_9b_top3", ["score_9b_top3.py"], "b_stage1_top3_9b.json"),                               # 라벨 채점 · 마진 · top-3
    ("selective_9b", ["selective_9b.py"], "exp3_confidence_9b.json"),                               # 선택적 응답 (CPU)
    ("halluc_9b", ["halluc_9b.py"], "p1_2_hallucination_9b.json"),                                  # 사실 환각
    ("eval_9b_ablation", ["eval_9b_ablation.py"], "ablation_9b.json"),                              # 입력 조건 비교
]


def run(cmd, env, log):
    script, *script_args = cmd
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, script)] + script_args, cwd=HERE, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    return proc.wait()


def main():
    ap = argparse.ArgumentParser(description="9B 학습 뒤 논문 측정 전체를 순서대로 실행")
    ap.add_argument("--force", action="store_true", help="완료된 단계도 다시 실행")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(os.path.join(LOGS_DIR, "eval_9b.log"), "a", encoding="utf-8") as log:
        for name, cmd, result in STEPS:
            marker = os.path.join(LOGS_DIR, f"{name}.done")
            if os.path.exists(marker) and not args.force:
                print(f"[skip] {name} (완료됨)")
                continue
            print(f"== 실행: {' '.join(cmd)}")
            code = run(cmd, env, log)
            if code != 0 or not os.path.exists(os.path.join(RESULTS_DIR, result)):
                print(f"[FAIL] {name} (exit {code}, 결과 파일 {result})")
                sys.exit(1)
            with open(marker, "w") as f:
                f.write("0\n")
            print(f"[done] {name}")
    print("== 측정 전체 완료")


if __name__ == "__main__":
    main()
