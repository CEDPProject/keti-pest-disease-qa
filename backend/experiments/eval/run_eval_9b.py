import os
import sys
import argparse
import subprocess

from backend.core.classes import LOGS_DIR, PAPER_LORA_DIR, REPO_DIR, RESULTS_DIR
from backend.core.settings import load_env

STEPS = [
    ("eval_9b_balanced", ["--output_dir", PAPER_LORA_DIR], "balanced_curve_9b.json"),
    ("score_9b_top3", [], "b_stage1_top3_9b.json"),
    ("selective_9b", [], "exp3_confidence_9b.json"),
    ("halluc_9b", [], "p1_2_hallucination_9b.json"),
    ("eval_9b_ablation", [], "ablation_9b.json"),
]


# 측정 모듈 하나를 실행하고 출력을 로그에 남긴다
def run(module: str, args: list, env: dict, log) -> int:
    proc = subprocess.Popen([sys.executable, "-m", f"backend.experiments.eval.{module}"] + args,
                            cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
    return proc.wait()


# 논문 측정 단계를 순서대로 실행한다
def main():
    ap = argparse.ArgumentParser(description="9B 학습 뒤 논문 측정 전체를 순서대로 실행")
    ap.add_argument("--force", action="store_true", help="완료된 단계도 다시 실행")
    args = ap.parse_args()

    load_env()
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(os.path.join(LOGS_DIR, "eval_9b.log"), "a", encoding="utf-8") as log:
        for name, step_args, result in STEPS:
            marker = os.path.join(LOGS_DIR, f"{name}.done")
            if os.path.exists(marker) and not args.force:
                print(f"[skip] {name} (완료됨)")
                continue
            print(f"== 실행: {' '.join([name] + step_args)}")
            code = run(name, step_args, env, log)
            if code != 0 or not os.path.exists(os.path.join(RESULTS_DIR, result)):
                print(f"[FAIL] {name} (exit {code}, 결과 파일 {result})")
                sys.exit(1)
            with open(marker, "w") as f:
                f.write("0\n")
            print(f"[done] {name}")
    print("== 측정 전체 완료")


if __name__ == "__main__":
    main()
