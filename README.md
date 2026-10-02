# 감귤 병해충 AI 진단 · 질의응답 플랫폼

감귤 사진과 질문을 올리면 병명을 판정하고, 학명 · 증상 · 방제 · 약제 정보를 전문가 지식베이스 원문을 근거로 답하는 웹 플랫폼입니다. (KETI · 에코브레인)

- **병명 판정**: 파인튜닝한 시각언어모델(VLM)이 사진을 보고 8개 클래스(질병 7종 + 정상) 중 하나로 판정합니다.
- **근거 있는 답변**: 학명 · 약제명 · 방제 시기 같은 사실 정보는 모델이 만들지 않고 지식베이스(KB) 원문에서 가져옵니다.
- **질문 의도별 답변**: 같은 사진이라도 진단 · 병명 · 증상 · 방제 중 무엇을 물었는지에 맞춰 답을 구성합니다.

## 동작 방식

```
사진 + 질문
  │
  ├─ 1. 병명 판정      VLM이 생성한 답에서 병명만 추출 (나머지 문장은 버림)
  ├─ 2. 이름 정규화    정확 일치 → 동의어 → 부분 일치 → 임베딩 유사도
  │                    (임계값 미달이면 가장 가까운 병에 억지로 붙이지 않고 '미해결')
  ├─ 3. 지식베이스 조회 학명 · 영명 · 병징 · 발생생태 · 방제 · 약제
  └─ 4. 답변 구성      질문 의도에 맞는 KB 내용으로 답변 → 경량 LLM(Gemma 3)이 문장만 정리
                       (정리에 실패하면 KB 원문을 그대로 표시)
```

모델은 병명만 결정하고 사실 정보는 KB에서 복사하므로, 답변에 지어낸 학명이나 약제가 들어갈 수 없습니다.

## 화면 구성

| 탭 | 내용 |
|---|---|
| 병해충 AI 진단 | 사진 업로드 + 질문(직접 입력 또는 예시 선택) → 병명 · 학명 · 근거 답변 |
| 데이터셋 | 학습 데이터 출처 · 클래스별 구성 |
| 지식베이스 | 수록 병해충과 항목 구조 · 예시 |
| 활용 기술 | 추론 파이프라인 · 역할 분리 설명 |
| Reference | 사용한 모델 · 프레임워크 · 데이터 출처 |

## 지식베이스

농촌진흥청 자료를 병해충별로 정리한 문서입니다 (`backend/core/knowledge/citrus_PD_ver3.md`).

- 수록 범위: 병해 14종 · 해충 23종, 총 37종 (녹색곰팡이병 · 푸른곰팡이병은 한 항목으로 정리)
- 항목 구조: 학명/영명 · 형태/병징 · 발생생태 · 방제/과원관리 · 방제약제
- 탄저병 항목은 전문가 검토 전 초안이며, 답변에 초안 표시(`is_draft_kb`)가 붙습니다.

## 빠른 시작

### 1. 설치

```bash
conda create -n qwenvl-ft python=3.11 && conda activate qwenvl-ft
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 설정

API 키 · 서버 주소 · 포트 같은 환경별 설정은 저장소 루트의 `.env`에 둡니다. `.env`는 git에서 제외되며, 항목은 `.env.example`에 있습니다.

```bash
cp .env.example .env
```

| 항목 | 기본값 | 설명 |
|---|---|---|
| `SERVER_HOST` / `SERVER_PORT` | `0.0.0.0` / `5100` | 웹 서버 주소 · 포트 |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `http://localhost:11434/api/generate` / `gemma3:4b` | 답변 문장 정리용 LLM (없어도 동작, KB 원문 표시) |
| `NCPMS_API_KEY` | (비어 있음) | 데이터 수집용 NCPMS OpenAPI 키 |
| `HF_HOME` | — | Hugging Face 모델 캐시 위치 |

셸에 이미 설정된 환경변수가 `.env`보다 우선합니다.

### 3. 실행

```bash
python app.py                          # 서빙 모델: models/qwen2_5vl_merged
python app.py --model_dir <병합 모델 경로>
python app.py --mock                   # 모델 없이 화면 · API 동작만 확인 (GPU 불필요)
```

브라우저에서 `http://<서버>:5100`으로 접속합니다. KB 임베딩(`intfloat/multilingual-e5-large`)은 처음 실행할 때 `models/kb_cache/`에 만들어집니다.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/` | 웹 화면 |
| GET | `/health` | 상태 확인 `{"ok", "mock", "model_loaded"}` |
| POST | `/diagnose` | 진단 · 질의응답 |

`/diagnose` 요청:

```json
{ "image_base64": "<base64 이미지>", "intent": "diagnose" }
```

`intent`는 `diagnose`(진단 종합) · `name`(병명) · `symptom`(증상) · `control`(방제 · 약제) 중 하나입니다.

주요 응답 필드:

| 필드 | 내용 |
|---|---|
| `disease`, `disease_key` | 판정 병명 (KB 표준명), 클래스 키 |
| `name_resolution`, `resolution_score` | 이름 정규화 방식(`exact` · `synonym` · `partial` · `rag` · `unknown`)과 점수 |
| `info` | KB 원문: 학명 · 영명 · 병징 · 발생생태 · 방제 · 약제 |
| `synthesis` | 경량 LLM이 정리한 답변 (실패 시 `null` → 화면은 KB 원문 표시) |
| `is_draft_kb` | 초안 KB 항목 여부 |
| `vlm_raw` | 모델이 생성한 원문 (병명 추출 근거) |

## 저장소 구조

```
keti-pest-disease-qa/
├── app.py                 플랫폼 실행 진입점
├── .env.example           환경별 설정 견본 (실제 값은 .env, git 제외)
├── frontend/              웹 화면 (index.html 단일 파일)
├── backend/
│   ├── api/               FastAPI 서버 (serve_api.py)
│   ├── core/              진단 · 지식베이스 핵심 모듈 (플랫폼과 실험이 함께 사용)
│   │   ├── settings.py          .env 로딩
│   │   ├── classes.py           클래스 정의 · 경로 상수
│   │   ├── label_extract.py     생성 답변 → 병명 추출
│   │   ├── kb_parser.py         KB 문서 파서
│   │   ├── kb_retriever.py      이름 정규화 · 섹션 검색
│   │   ├── name_rag_pipeline.py VLM 판정 + KB 조회 파이프라인
│   │   └── knowledge/           지식베이스 원문
│   └── experiments/       학습 · 평가 · 논문 실험
│       ├── data_prep/     데이터셋 구축
│       ├── train/         LoRA 학습 · 모델 병합
│       ├── eval/          진단 성능 · 환각 · 선택적 응답 측정
│       └── interventions/ 검색 개입 예비 실험
├── data/                  데이터 (git 제외)
└── models/                학습 모델 · KB 임베딩 캐시 (git 제외)
```

## 데이터

이미지 데이터는 라이선스와 용량 때문에 저장소에 포함하지 않습니다. 학습 · 평가를 다시 하려면 아래 출처에서 받아 `data/raw/` 아래에 두어야 합니다.

| 데이터 | 출처 |
|---|---|
| 감귤 병해 · 정상 실사진 | AI Hub 노지작물 질병 데이터 |
| 감귤 병해 실사진 | 국가농작물병해충관리시스템(NCPMS) OpenAPI (`data_prep/ncpms_collect.py`로 수집) |

- **클래스 (8개)**: 궤양병 · 더뎅이병 · 잿빛곰팡이병 · 수지병 · 검은점무늬병 · 녹색곰팡이병 · 탄저병 + 정상. 검은무늬병은 병원균 동정이 확정되지 않았고 검은점무늬병과 명칭이 자주 혼동되어 제외했습니다.
- **로컬 배치**: `data/raw/감귤_병해충_사진/`(AI Hub, 클래스별 폴더), `data/raw/ncpms_수집/`, `data/raw/감귤_정상_사진/`(잎 · 열매)

### 서빙 모델 만들기

플랫폼 서빙 모델(Qwen2.5-VL-7B + LoRA 병합)은 다음 순서로 만듭니다. LoRA 학습은 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)를 씁니다 (설치 위치: `.env`의 `LLAMAFACTORY_DIR`).

```
data_prep/build_manifest.py → make_split.py → build_dataset.py → train/run_train.sh → train/merge_lora.py
```

## 관련 논문

**A Reliable VLM-RAG System for Citrus Disease Diagnosis and Question Answering**
Jihoon Lee, Sungin Hwang, Jaewon Moon — KETI

이 플랫폼의 역할 분리 구조를 바탕으로, 진단의 신뢰성을 검증하고 보완한 연구입니다. 논문 실험은 Qwen3.5-9B + LoRA로 수행했습니다.

- **환각 차단 검증**: 모델이 직접 답하면 사실 질의 147건 중 15.7%에서 학명 · 약제를 지어내거나 다른 병의 것을 가져오지만, KB 기반 답변에서는 0%
- **마진 기반 선택적 응답**: 8개 라벨을 모두 채점해 1 · 2위 점수 차(마진)가 작으면 하나로 단정하지 않고 상위 3개 후보와 증상 발췌를 제시
- **검색 개입 예비 실험**: 검색 텍스트로 진단을 고치려는 4가지 방식이 모두 악화되거나 효과가 없어, 검색을 답변 쪽에만 쓰는 근거가 됨

| 결과 (테스트 실사진 57장) | 값 |
|---|---|
| 진단 정확도 / macro-F1 | 0.825 / 0.833 |
| 사실 환각률 (직접 생성 → KB 기반) | 15.7% → 0% |
| 마진 하위 30% 보류 시 확신 답변 정확도 | 0.825 → 0.90 |
| 상위 3개 후보에 정답 포함 | 94.7% |

데이터는 고유 실사진 291장(질병 252 · 정상 39)이며, 출처와 촬영 부위로 층화한 57장을 테스트로 쓰고 나머지로 학습 샘플 1,860개를 만들었습니다.

<details>
<summary>실험 재현 스크립트</summary>

결과 JSON은 `qa_docs/experiment_results/`에 저장됩니다 (git 제외). 경로는 `backend/experiments/` 기준입니다.

| 단계 | 스크립트 |
|---|---|
| 데이터 수집 · 구축 | `data_prep/ncpms_collect.py` → `consolidate_images.py` → `build_balanced.py` |
| 누수 점검 | `eval/check_leakage.py` |
| 학습 (Qwen3.5-9B) | `train/run_train_9b.sh` |
| 측정 전체 실행 | `train/run_eval_9b.sh` (아래 eval 스크립트를 순서대로 실행) |
| 진단 성능 · 에폭 선택 | `eval/eval_9b_balanced.py` |
| 입력 조건 비교 | `eval/eval_9b_ablation.py` |
| 라벨 채점 · 마진 · 상위 3개 | `eval/score_9b_top3.py` |
| 선택적 응답 | `eval/selective_9b.py` |
| 사실 환각 | `eval/halluc_9b.py`, `eval/halluc_9b_rescore.py` |
| 검색 개입 4종 | `interventions/` (`candidate_selection` · `guided_recheck` · `score_fusion` · `llm_rerank`, 임베딩 교체 `exp2_embed_compare.py`) |

</details>

## 라이선스

MIT License. 이미지 데이터와 지식베이스 원 자료의 이용 조건은 각 출처(AI Hub, NCPMS, 농촌진흥청)를 따릅니다.
