# keti-pest-disease-qa

감귤 병해 진단·질의응답 시스템. 사진과 질문을 넣으면 파인튜닝한 시각언어모델(VLM)이 병명을 판정하고, 학명·증상·방제·약제 같은 사실 정보는 전문가 지식베이스(KB) 원문에서 가져와 답한다.

관련 논문: *A Reliable VLM-RAG System for Citrus Disease Diagnosis and Question Answering* (Jihoon Lee, Sungin Hwang, Jaewon Moon — KETI)

## 핵심 아이디어

- **역할 분리**: VLM은 8개 클래스 중 병명만 결정하고, 사실 필드(학명·약제명·방제 시기)는 KB 원문에서 복사한다. 경량 언어모델은 문장만 다듬고, 실패하면 원문 섹션을 그대로 보여준다.
- **선택적 응답**: 8개 라벨을 teacher forcing으로 모두 채점하고, 1·2위 점수 차(마진)가 기준값보다 작으면 하나로 단정하지 않고 상위 3개 후보를 증상 발췌와 함께 제시한다.

| 결과 (테스트 57장) | 값 |
|---|---|
| 진단 정확도 / macro-F1 | 0.825 / 0.833 |
| 사실 질의 147건 환각률 (직접 생성 → KB 기반) | 15.7% → 0% |
| 마진 하위 30% 보류 시 확신 답변 정확도 | 0.825 → 0.90 |
| 상위 3개 후보에 정답 포함 | 94.7% |

## 저장소 구조

```
keti-pest-disease-qa/
├── app.py                 플랫폼 실행 진입점
├── .env.example           환경별 설정 · 비밀값 견본 (실제 값은 .env, git 제외)
├── frontend/              웹 화면 (index.html 단일 파일)
├── backend/
│   ├── api/               FastAPI 서버 (serve_api.py)
│   ├── core/              플랫폼과 실험이 함께 쓰는 공통 모듈
│   │   ├── settings.py          .env 로딩 (키 · 주소 · 포트)
│   │   ├── classes.py           클래스 정의 · 경로 상수
│   │   ├── label_extract.py     답변 → 병명 추출
│   │   ├── kb_parser.py         KB 문서 파서
│   │   ├── kb_retriever.py      이름 정규화 · 섹션 검색
│   │   ├── name_rag_pipeline.py VLM 판정 + KB 조회 파이프라인
│   │   └── knowledge/           지식베이스 원문 (citrus_PD_ver3.md)
│   └── experiments/       논문 실험
│       ├── data_prep/     데이터셋 구축
│       ├── train/         LoRA 학습 설정 · 실행 스크립트
│       ├── eval/          진단 성능 · 환각 · 선택적 응답 측정
│       ├── interventions/ 검색 개입 예비 실험 4종 (Table III)
│       └── paper/         논문 그림 · 표 생성
├── data/                  데이터 (git 제외, 아래 설명 참고)
└── models/                학습 산출물 · KB 임베딩 캐시 (git 제외)
```

## 사용 데이터셋

이미지 데이터는 라이선스와 용량 때문에 이 저장소에 포함하지 않는다. 아래 출처에서 직접 받아 `data/` 아래에 두어야 실험 코드를 실행할 수 있다.

| 데이터 | 출처 | 용도 |
|---|---|---|
| 감귤 병해 · 정상 실사진 | AI Hub 노지작물 질병 데이터 | 학습 · 평가 |
| 감귤 병해 실사진 | 국가농작물병해충관리시스템(NCPMS) OpenAPI | 학습 · 평가 (`data_prep/ncpms_collect.py`로 수집, 환경변수 `NCPMS_API_KEY` 필요) |
| 감귤 병해충 지식베이스 | 농촌진흥청 자료를 병해별로 정리한 문서 | 답변의 사실 출처 (`backend/core/knowledge/`에 포함) |

- **클래스 (8개)**: 궤양병 · 더뎅이병 · 잿빛곰팡이병 · 수지병 · 검은점무늬병 · 녹색곰팡이병 · 탄저병 + 정상. 검은무늬병은 병원균 동정이 확정되지 않았고 검은점무늬병과 명칭이 자주 혼동되어 제외했다.
- **규모**: 근접중복을 제거한 고유 실사진 291장 (질병 252 · 정상 39). 출처와 촬영 부위로 층화해 57장(질병 49 · 정상 8)을 테스트로 떼고, 나머지 234장에 질문 의도 4종(진단 · 병명 · 증상 · 방제)을 붙여 클래스 균형 증강 후 학습 샘플 1,860개를 만들었다. 증강은 학습 풀에만 적용했다.
- **사실성 평가**: 질병 테스트 사진 49장 × 사실 질문 3개(학명 · 추천 약제 · 일반 설명) = 147건.

실험 코드가 기대하는 로컬 배치:

```
data/
├── raw/감귤_병해충_사진/        AI Hub 원본 (클래스별 폴더)
├── raw/ncpms_수집/              NCPMS 수집본
├── raw/감귤_정상_사진/          정상 사진 (잎 · 열매)
└── vlm_finetune_balanced/       data_prep 스크립트가 생성 (images/, images_vlm/, train.json, test.jsonl)
```

## 실행 환경

```bash
conda create -n qwenvl-ft python=3.11 && conda activate qwenvl-ft
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

API 키, 서버 주소 · 포트 같은 환경별 설정과 비밀값은 저장소 루트의 `.env`에 둔다. `.env`는 git에서 제외되며, 필요한 항목은 `.env.example`에 있다.

```bash
cp .env.example .env      # 값 채우기: NCPMS_API_KEY, SERVER_HOST/PORT, OLLAMA_URL 등
```

- 코드는 `backend/core/settings.py`로 `.env`를 읽는다. 셸에 이미 설정된 환경변수가 우선한다.
- LoRA 학습은 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)를 쓴다. 설치 위치는 `.env`의 `LLAMAFACTORY_DIR`로 지정한다 (기본값 `~/tools/LLaMA-Factory`).
- Hugging Face 모델 캐시 위치는 `HF_HOME`으로 지정한다.
- 답변 문장 다듬기에는 Ollama의 `gemma3:4b`를 쓴다 (없으면 KB 원문 섹션을 그대로 보여준다).

## 플랫폼 실행

```bash
python app.py --port 5100                 # 병합 모델: models/qwen2_5vl_merged
python app.py --port 5100 --model_dir <병합 모델 경로>
python app.py --port 5100 --mock          # 모델 없이 화면 · API 동작만 확인
```

브라우저에서 `http://<서버>:5100` 으로 접속한다. KB 임베딩은 처음 실행할 때 `models/kb_cache/`에 만들어진다.

## 논문 실험 재현

실험 결과 JSON은 `qa_docs/experiment_results/`, 논문 그림 · 표는 `qa_docs/paper/` 아래에 쓰인다 (둘 다 git 제외).

| 단계 | 스크립트 | 논문 |
|---|---|---|
| 데이터 수집 · 정제 | `data_prep/ncpms_collect.py` → `consolidate_images.py` → `build_balanced.py` | §III Setup |
| 누수 점검 | `eval/check_leakage.py` | §III Setup |
| 학습 | `train/run_train_9b.sh` (Qwen3.5-9B + LoRA) | §III Setup |
| 에폭 선택 · 진단 성능 | `eval/eval_9b_balanced.py` | §III Results |
| 입력 조건 ablation | `eval/eval_9b_ablation.py` | Table II |
| 라벨 채점 · 마진 · 상위 3개 | `eval/score_9b_top3.py` | 식 (1), §III Results |
| 선택적 응답 | `eval/selective_9b.py` | Fig. 2 |
| 사실 환각 | `eval/halluc_9b.py`, `eval/halluc_9b_rescore.py` | Table I |
| 검색 개입 4종 | `interventions/candidate_selection/` · `guided_recheck/` · `score_fusion/` · `llm_rerank/`, 임베딩 교체 `interventions/exp2_embed_compare.py` | Table III |
| 그림 · 표 생성 | `paper/make_figures_9b.py`, `paper/make_tables_9b.py` | Fig. 2, Tables |

학습 뒤 측정 단계 전체는 `train/run_eval_9b.sh` 하나로 순서대로 실행된다. 플랫폼 서빙 모델(Qwen2.5-VL)은 `data_prep/build_manifest.py` → `make_split.py` → `build_dataset.py` → `train/run_train.sh` → `train/merge_lora.py` 순서로 만든다.

## 라이선스

MIT License. 이미지 데이터와 지식베이스 원 자료의 이용 조건은 각 출처(AI Hub, NCPMS, 농촌진흥청)를 따른다.
