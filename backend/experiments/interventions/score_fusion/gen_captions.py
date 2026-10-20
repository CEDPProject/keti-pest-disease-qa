import os

from backend.core.classes import BALANCED_TEST_JSONL, PLATFORM_BASE_MODEL, RESULTS_DIR
from backend.core.vlm import VLMRunner, load_qwen25
from backend.experiments.data_io import read_jsonl, write_json

OUT = os.path.join(RESULTS_DIR, "fusion_captions.json")
PROMPT = ("이 감귤 사진에서 보이는 시각적 특징을 병명 언급 없이 서술하라. "
          "병반의 모양·색·질감·크기·위치·표면 상태만 구체적으로 적어라.")


# 테스트 사진마다 병명 없는 시각 특징 캡션을 만든다
def main():
    test = read_jsonl(BALANCED_TEST_JSONL)
    print(f"베이스 로딩: {PLATFORM_BASE_MODEL} · test {len(test)}장")
    runner = VLMRunner(*load_qwen25(PLATFORM_BASE_MODEL))

    caps = {}
    for i, r in enumerate(test):
        b = os.path.basename(r["image"])
        caps[b] = {"image": b, "gold": r["key"], "caption": runner.ask(r["image"], PROMPT, 200)}
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}")
    write_json(caps, OUT)
    print(f"저장: {OUT}")
    for b in list(caps)[:2]:
        print(f"\n[{b}] gold={caps[b]['gold']}\n  {caps[b]['caption'][:200]}")


if __name__ == "__main__":
    main()
