# 감귤 병해충 AI 진단 · 질의응답 플랫폼

감귤 사진을 올리면 병명을 판정하고, 학명 · 증상 · 방제 · 약제 정보를 전문가 지식베이스 원문을 근거로 답하는 웹 플랫폼입니다. (KETI · 에코브레인)

- **병명 판정**: 파인튜닝한 시각언어모델(VLM, Qwen2.5-VL-7B)이 사진을 보고 병명 하나를 정합니다.
- **근거 있는 답변**: 학명 · 약제명 · 방제 시기 같은 사실 정보는 모델이 만들지 않고 지식베이스(KB) 원문에서 가져옵니다.
- **질문 의도별 답변**: 진단 · 병명 · 증상 · 방제 중 무엇을 물었는지에 맞춰 KB 항목을 골라 보여 줍니다.

## 동작 방식

```
사진 + 질문 의도
  │
  ├─ 1. 병명 판정       VLM 답변에서 병명만 추출 (나머지 문장은 버림)
  ├─ 2. 이름 정규화     정확 일치 → 동의어 → 부분 일치 → 임베딩 유사도
  │                     (임계값 미달이면 가장 가까운 병에 억지로 붙이지 않고 '미해결'로 처리)
  ├─ 3. 지식베이스 조회  학명 · 영명 · 병징 · 발생생태 · 방제 · 약제
  └─ 4. 답변 구성       질문 의도에 맞는 KB 내용 → 경량 LLM(Gemma 3, Ollama)이 문장만 정리
                        (정리에 실패하거나 Ollama가 없으면 KB 원문을 그대로 표시)
```

모델은 병명만 결정하고 사실 정보는 KB에서 복사하므로, 답변에 지어낸 학명이나 약제가 들어가지 않습니다.
병명을 특정하지 못하면 단정하지 않고 유사한 후보 3개를 유사도와 함께 보여 줍니다.

## 화면 구성

| 탭 | 내용 |
|---|---|
| 홈 | 플랫폼 소개 · 문제와 접근 방식 |
| 병해충 AI 진단 | 사진 업로드 + 질문 → 병명 · 학명 · 근거 답변 |
| 데이터셋 | 학습 데이터 출처 · 클래스별 구성 |
| 지식베이스 | 수록 병해충 목록 · 항목 구조 · 병해충별 상세 보기 |
| 활용 기술 | 추론 파이프라인 · 역할 분리 설명 |
| Reference | 사용한 모델 · 프레임워크 · 데이터 출처 |

진단 화면에서 예시 질문(진단 · 병명 · 증상 · 방제)을 고르면 그 의도로 답하고, 질문을 직접 입력하면 종합 진단으로 답합니다.

## 서빙 모델

| 항목 | 내용 |
|---|---|
| 기본 모델 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 학습 방식 | LoRA (r 16 · alpha 32 · 전 선형층), 비전 인코더 · 프로젝터 고정, 이미지 최대 768×768 |
| 학습 데이터 | AI Hub 감귤 병해 사진 121장 중 학습 96장 → 증강 포함 384개 질문-답변 샘플 |
| 학습 클래스 | 질병 8종: 궤양병 · 더뎅이병 · 잿빛곰팡이병 · 수지병 · 검은점무늬병 · 녹색곰팡이병 · 탄저병 · 검은무늬병 |
| 선택 · 병합 | 5 에폭 어댑터(`checkpoint-240`)를 기본 모델에 병합 → `models/qwen2_5vl_merged` |

- 검은무늬병은 현재 코드의 판정 클래스에서 빠졌습니다. 병원균 동정이 확정되지 않았고 검은점무늬병과 이름이 자주 혼동되기 때문입니다. 모델이 검은무늬병이라고 답하면 검은점무늬병으로 정규화합니다.
- 서빙 모델은 정상(건강) 사진을 학습하지 않았습니다.

### 성능과 한계

| 평가 | 결과 |
|---|---|
| 같은 출처 테스트 사진 25장 | 정확도 0.920 · macro-F1 0.941 |
| 다른 출처 사진 58장 (Mendeley Citrus, 궤양병 · 더뎅이병 · 검은점무늬병) | 정확도 0.155 |

- 테스트가 25장뿐이라 정확도의 95% 신뢰구간이 0.750~0.978로 넓습니다.
- 다른 출처 사진(오렌지 · 라임 계열, 촬영 조건도 다름)에서는 크게 떨어집니다. 다양한 출처의 사진을 더 모으는 것이 다음 과제입니다.
- 상세 결과는 내부 문서 `qa_docs/documents/03_reports/REPORT.md`에 있습니다 (저장소 미포함).

## 지식베이스

농촌진흥청 자료를 병해충별로 정리한 문서입니다 (`backend/core/knowledge/citrus_PD_ver3.md`).

- 수록 범위: 병해 13개 항목(녹색곰팡이병 · 푸른곰팡이병은 한 항목) · 해충 23개 항목, 총 36개 항목
- 항목 구조: 학명/영명 · 형태/병징 · 발생생태 및 피해 · 방제 및 과원관리 · 방제약제
- 탄저병 항목은 전문가 검토 전 초안이며, 답변에 초안 표시(`is_draft_kb`)가 붙습니다.
- 웹 화면의 지식베이스 탭은 `frontend/index.html` 안에 넣어 둔 사본을 보여 줍니다(검은무늬병 초안 포함). KB 문서를 고치면 이 사본도 함께 고쳐야 화면에 반영됩니다.

## 빠른 시작

### 1. 설치

```bash
conda create -n qwenvl-ft python=3.11 && conda activate qwenvl-ft
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 설정

서버 주소 · 포트 같은 환경별 설정과 API 키는 저장소 루트의 `.env`에 둡니다. `.env`는 git에서 제외되며, 항목은 `.env.example`에 있습니다.

```bash
cp .env.example .env
```

| 항목 | 기본값 | 설명 |
|---|---|---|
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `5100` | 웹 서버 주소 · 포트 |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `http://localhost:11434/api/generate` / `gemma3:4b` | 답변 문장 정리용 LLM (없어도 동작, KB 원문 표시) |
| `HF_HOME` | — | Hugging Face 모델 캐시 위치 |
| `NCPMS_API_KEY` | (비어 있음) | NCPMS 사진 수집용 OpenAPI 키 (플랫폼 실행에는 필요 없음) |
| `LLAMAFACTORY_DIR` | `~/tools/LLaMA-Factory` | LoRA 학습 프레임워크 위치 (학습할 때만) |

셸에 이미 설정된 환경변수가 `.env`보다 우선합니다.

### 3. 실행

```bash
python app.py                          # 서빙 모델: models/qwen2_5vl_merged
python app.py --model_dir <병합 모델 경로>
python app.py --mock                   # 모델 없이 화면 · API 동작만 확인 (GPU 불필요)
```

브라우저에서 `http://<서버>:5100`으로 접속합니다.
- 모델은 GPU에 올립니다. bf16 가중치만 약 17GB라 그보다 여유 있는 GPU가 필요하며, 사용할 GPU는 `CUDA_VISIBLE_DEVICES`로 정합니다.
- KB 임베딩 모델(`intfloat/multilingual-e5-large`)의 임베딩은 처음 실행할 때 `models/kb_cache/`에 만들어집니다.
- `--mock`은 병명을 정해진 순서로 돌려 가며 내고, 이름 정규화 · KB 조회 · 응답 형식은 실제와 같습니다.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 웹 화면 |
| GET | `/health` | 상태 확인 `{"ok", "mock", "model_loaded"}` |
| POST | `/diagnose` | 진단 · 질의응답 |

`/diagnose` 요청:

```json
{ "image_base64": "<base64 이미지 또는 data URL>", "intent": "diagnose" }
```

`intent`는 `diagnose`(진단 종합) · `name`(병명) · `symptom`(증상) · `control`(방제 · 약제) 중 하나이며, 없으면 `diagnose`입니다.

주요 응답 필드:

| 필드 | 내용 |
|---|---|
| `disease`, `disease_key` | 판정 병명 (KB 표준명), 클래스 키 |
| `name_resolution`, `resolution_score` | 이름 정규화 방식(`exact` · `synonym` · `partial` · `rag` · `unknown`)과 점수 |
| `info` | KB 원문: 학명 · 영명 · 병징 · 발생생태 · 방제 · 약제 (미확정 학명 · 가칭 영명은 `null`) |
| `synthesis` | 경량 LLM이 정리한 답변 (실패 시 `null` → 화면은 KB 원문 표시) |
| `is_draft_kb` | 초안 KB 항목 여부 |
| `vlm_raw` | 모델이 생성한 원문 (병명 추출 근거) |
| `suggestions` | 병명을 특정하지 못했을 때의 유사 후보 3개와 유사도 |

이미지를 해석할 수 없으면 `400`과 `{"ok": false, "error": ...}`를 돌려줍니다.

## 저장소 구조

```
keti-pest-disease-qa/
├── app.py                 플랫폼 실행 진입점
├── .env.example           환경별 설정 견본 (실제 값은 .env, git 제외)
├── frontend/              웹 화면 (index.html 단일 파일)
├── backend/
│   ├── api/               FastAPI 서버 (serve_api.py)
│   ├── core/              진단 · 지식베이스 핵심 모듈
│   │   ├── settings.py          .env 로딩
│   │   ├── classes.py           클래스 정의 · 경로 상수 (단일 소스)
│   │   ├── vlm.py               VLM 실행기 (모델 로딩 · 채팅 템플릿 · 생성)
│   │   ├── label_extract.py     모델 답변 → 병명 추출
│   │   ├── kb_parser.py         KB 문서 파서
│   │   ├── kb_retriever.py      이름 정규화 · 섹션 검색 · KB 답변 조립
│   │   ├── name_rag_pipeline.py 병명 판정 + KB 조회 파이프라인
│   │   ├── ollama_client.py     답변 문장 정리용 Ollama 호출
│   │   └── knowledge/           지식베이스 원문
│   └── experiments/       데이터 구축 · 학습 · 평가 코드 (연구 실험 포함)
│       ├── data_prep/     데이터셋 구축
│       ├── train/         LoRA 학습 · 모델 병합
│       ├── eval/          성능 · 환각 측정
│       └── interventions/ 검색 개입 예비 실험
├── data/                  데이터 (git 제외)
└── models/                학습 모델 · KB 임베딩 캐시 (git 제외)
```

## 데이터

이미지 데이터는 라이선스와 용량 때문에 저장소에 포함하지 않습니다. 서빙 모델을 다시 학습하려면 AI Hub 노지작물 질병 데이터에서 감귤 병해 사진을 받아 `data/raw/감귤_병해충_사진/<병명>/` 아래에 클래스별로 둡니다.

### 서빙 모델 다시 만들기

LoRA 학습은 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)를 씁니다. 모든 명령은 저장소 루트에서 실행합니다.

```bash
python -m backend.experiments.data_prep.build_manifest    # 사진 정제 → data/vlm_finetune/manifest.csv
python -m backend.experiments.data_prep.make_split        # 클래스별 20% 테스트 분할 → split.json
python -m backend.experiments.data_prep.build_dataset     # 의도별 질문-답변 학습셋 → train.json, test.jsonl
python -m backend.experiments.train.train --model platform
python -m backend.experiments.train.merge_lora --adapter models/qwen2_5vl_lora/checkpoint-240 --out models/qwen2_5vl_merged
python -m backend.experiments.eval.demo_validation        # 병합 모델 점검 (진단 정확도 · 의도별 응답)
```

현재 코드는 검은무늬병을 클래스에서 뺐으므로, 다시 만들면 질병 7종 데이터셋이 됩니다. 지금 서빙 중인 모델은 검은무늬병을 포함한 8종으로 학습했습니다.

## 라이선스

MIT License. 이미지 데이터와 지식베이스 원 자료의 이용 조건은 각 출처(AI Hub, 농촌진흥청)를 따릅니다.
