# 🍊 감귤 병해충 AI 진단 시스템 (keti-pest-disease-qa)

감귤 사진을 올리면 병해충을 **진단**하고 **공식 방제 지식**을 제공하는 AI 시스템.
KETI + 에코브레인 협업 CEDP 과제.

> **아키텍처**: 이미지 → 파인튜닝 Qwen2.5-VL이 **병명만** 결정 → RAG가 지식베이스에서 **정보 조회**
> **핵심 원칙**: 분류는 VLM이, 지식은 KB가 담당 → 모델이 사실을 생성하지 않아 **환각 구조적 차단**

📄 전체 정리: **[docs/03_vlm_finetune/SYSTEM_OVERVIEW.md](docs/03_vlm_finetune/SYSTEM_OVERVIEW.md)** · 상세 리포트: [REPORT.md](docs/03_vlm_finetune/REPORT.md)

---

## 핵심 결과

| 지표 | 값 |
|---|---:|
| 무학습 VLM macro-F1 | 0.038 |
| **파인튜닝 후 (동일 분포 test)** | **0.941 / acc 0.920** |
| ⛔ 다른 출처 사진 (일반화) | **0.155** |

> 파이프라인은 작동하지만, 121장으로는 일반화되지 않는다.
> **쓸 만한 모델을 얻으려면 다양한 출처의 감귤 데이터가 반드시 필요하다** — 이것이 핵심 결론이자 데이터 확보의 근거다.

---

## 폴더 구조

```
keti-pest-disease-qa/
├── app.py            🌐 웹 데모 진입점
├── data/             📦 데이터 (심볼릭 링크 → 데이터 디스크)
├── docs/             📄 모든 문서 (계획·결과·리포트)
├── qa/               💬 기존 QA 시스템 (text_qa / image_qa)
├── core/             🧠 VLM 파인튜닝 파이프라인 (common·data_prep·train·eval·rag·baseline)
├── web/              🌐 데모 웹앱 (backend/ frontend/ gradio_app.py)
├── models/           🤖 학습 산출물 (git 제외)
└── result_md/        📊 PPT·CSV·혼동행렬 등 산출물
```

각 폴더 상세: `core/README.md`, `data/README.md`, `docs/README.md`

---

## 빠른 시작 — 웹 데모

```bash
conda activate qwenvl-ft

# 실제 모델 (GPU ~16GB)
python app.py --port 5100            # → http://<서버IP>:5100

# 모델 없이 UI만 테스트
python app.py --port 5100 --mock
```

백엔드 없이 `web/frontend/index.html`을 브라우저에서 바로 열어도 **Mock 모드**로 전체 UI가 동작한다.

---

## 실행 환경

| conda 환경 | 용도 |
|---|---|
| `qwenvl-ft` | 학습·추론·RAG·웹서버 (torch 2.6+cu124, transformers 5.7, peft, sentence-transformers, fastapi) |
| `cedpllm` | 무학습 baseline(Ollama), 데이터 전처리, 누수 검사 |

`requirements.txt` 는 `qwenvl-ft` 환경 기준. torch 는 CUDA 12.4 휠을 사용한다:
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 재현·상세

파이프라인 재현(데이터 생성 → 학습 → 평가 → 병합)과 결과 해석은
**[docs/03_vlm_finetune/SYSTEM_OVERVIEW.md](docs/03_vlm_finetune/SYSTEM_OVERVIEW.md)** 10절 참조.

## 다음 스텝

1. **감귤 특화 데이터 확보** — 클래스당 200장 이상, 다양한 출처 (유일한 근본 해법)
2. **전문가 검증 지식베이스** — 탄저병·검은무늬병 KB 초안 검증, 검은무늬병 병명 확정
