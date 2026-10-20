import os
import sys
import argparse
import subprocess
import tempfile

from backend.core.classes import LOGS_DIR, REPO_DIR
from backend.core.settings import load_env

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {
    "paper": "qwen3_5_9b_balanced_lora_sft.yaml",
    "platform": "qwen2_5vl_lora_sft.yaml",
}


# 설정 파일의 __REPO__ 를 저장소 경로로 바꾼 임시 yaml 을 만든다
def resolve_config(name: str) -> str:
    with open(os.path.join(HERE, CONFIGS[name]), encoding="utf-8") as f:
        text = f.read().replace("__REPO__", REPO_DIR)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    return tmp.name


# LLaMA-Factory 로 LoRA 학습을 실행한다
def main():
    ap = argparse.ArgumentParser(description="LLaMA-Factory LoRA 학습")
    ap.add_argument("--model", choices=sorted(CONFIGS), default="paper")
    ap.add_argument("--skip-done", action="store_true", help="이미 완료됐으면 건너뜀 (반복 예약 실행용)")
    args = ap.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)
    marker = os.path.join(LOGS_DIR, f"train_{args.model}.done")
    log_path = os.path.join(LOGS_DIR, f"train_{args.model}.log")
    if args.skip_done and os.path.exists(marker):
        print(f"[skip] {marker} 존재")
        return

    load_env()
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.update(DISABLE_VERSION_CHECK="1", FORCE_TORCHRUN="0",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    llamafactory_dir = os.path.expanduser(env.get("LLAMAFACTORY_DIR", "~/tools/LLaMA-Factory"))

    cfg = resolve_config(args.model)
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.Popen(["llamafactory-cli", "train", cfg], cwd=llamafactory_dir, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            code = proc.wait()
    finally:
        os.remove(cfg)

    if code == 0:
        with open(marker, "w") as f:
            f.write("0\n")
    print(f"== 학습 종료 (exit {code}) · 로그: {log_path}")
    sys.exit(code)


if __name__ == "__main__":
    main()
