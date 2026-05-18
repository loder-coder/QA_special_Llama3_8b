# 파이프라인 TODO

최종 데이터 구조가 확정된 뒤 다시 검토할 파이프라인 개선 항목입니다.

## Gold 선정 점수 보완

현재 Gold/Bronze 분리 점수는 주로 아래 기준을 사용합니다.

- 질문-답변 embedding 유사도
- inverse perplexity
- 답변 길이 패널티

실제 데이터가 확정되면 아래 신호를 추가로 반영할지 검토합니다.

- `views` 기반 인기 신호
- `donation` 기반 품질 또는 가치 신호
- 카테고리 균형 가중치
- 카테고리별 최소 Gold 개수

이유: 현재 점수는 짧고 실용적인 답변, 소수 카테고리, 비정형이지만 유효한 사용자 질문보다 문장이 깔끔하고 모델이 처리하기 쉬운 답변을 더 선호할 수 있습니다.

## Pair Label 품질 보완

현재 pair 생성은 embedding similarity 기준만으로 positive/negative label을 만듭니다.

실제 schema가 확정되면 category 또는 intent 관련 필드를 label guard로 사용할지 검토합니다.

- `같은 category + positive similarity 범위`를 더 강한 positive 후보로 사용
- `다른 category + 낮은 similarity`를 더 강한 negative 후보로 사용
- 애매한 row는 억지로 label을 붙이지 않고 보류

안정적인 `intent_id`가 제공되면 category보다 `intent_id`를 pair label 기준으로 우선 사용합니다.

이유: similarity만 기준으로 삼으면, 특히 한국어 사용자 질문에서 표현은 다르지만 의도는 같은 질문을 negative로 잘못 분류할 수 있습니다.

## Intent 학습 Fallback 품질 보완

현재 `intent.train`은 아래 순서로 동작합니다.

- `data/triplets.json`에 row가 있으면 `TripletLoss` 사용
- triplet이 비어 있으면 `data/pairs.json` 기반 q-q pair 학습으로 fallback
- triplet과 pair가 모두 비어 있으면 base intent model 저장

카테고리 대분류가 안정적으로 들어온다면 pair 학습 fallback에도 category guard를 추가할지 검토합니다.

- 같은 category이면서 similarity가 positive 범위인 경우 positive 후보로 사용
- 다른 category이면서 similarity가 낮은 경우 negative 후보로 사용
- 같은 category인데 similarity가 낮거나, 다른 category인데 similarity가 높은 경우는 보류

주의: category는 정답 label이 아니라 오분류를 줄이기 위한 보조 신호로 사용합니다. 안정적인 `intent_id`가 생기면 category보다 `intent_id`를 우선합니다.

## 한국어 Embedding 모델 통일

현재 embedding 모델이 단계마다 다르면 similarity 기준이 서로 어긋날 수 있습니다.

검토 대상:

- `preprocess.build_pairs`
- `intent.infer`
- `rag.embed`
- `rag.retriever`
- semantic cache 검색

우선 추천은 전체 embedding 계열을 `BAAI/bge-m3`로 통일하는 것입니다.

대안:

- `intfloat/multilingual-e5-large`: 다국어 성능은 좋지만 query/passsage prefix 관례를 코드에 반영해야 합니다.
- 한국어 파인튜닝 bge-m3 계열: 실제 데이터로 A/B 평가 후 채택합니다.

원칙: 모델을 바꾸면 기존 FAISS index, intent model, semantic cache는 같은 기준으로 다시 생성해야 합니다.

## API Startup 운영 편의성

현재 API는 시작 시점에 artifact manifest 검증과 pipeline 전체 로드를 한 번에 수행합니다. 보안상 단순하지만, 싱글노드에서는 Redis, FAISS, intent model, LLM adapter 중 하나만 깨져도 API 전체가 뜨지 않을 수 있습니다.

운영 편의성을 위해 아래 구조를 검토합니다.

- `/healthz`: 프로세스가 살아 있는지만 확인
- `/readyz`: Redis, FAISS, intent model, LLM adapter 준비 여부를 각각 확인
- LLM lazy-load 옵션 추가
- startup warmup 옵션 추가
- Docker healthcheck는 `/readyz` 기준으로 설정
- 학습 컨테이너와 API 컨테이너 분리

싱글노드 Docker 운영 예시:

- `altong-train`: preprocess, pair/triplet 생성, intent 학습, RAG index 생성, LoRA 학습
- `altong-api`: artifact mount 후 FastAPI 실행
- `redis`: Redis 전용 컨테이너

원칙: 학습 작업과 API serving은 같은 GPU를 두고 경쟁하지 않게 시간대 또는 컨테이너 역할을 분리합니다.

## Semantic Cache 재판단

현재 semantic cache는 Redis에서 cached query를 최대 1000개 가져온 뒤 매 요청마다 다시 embedding합니다. NVIDIA Blackwell 기반 싱글노드에서도 이 구조는 LLM inference 자원을 잠식할 수 있습니다.

개선 방향:

- cache 저장 시 query embedding도 함께 저장
- API 시작 시 cache embedding index를 메모리 또는 FAISS로 로드
- 새 cache가 추가되면 index에도 append
- cache payload에 `language`, `model_version`, `data_version`, `created_at` 저장
- embedding 모델이나 데이터가 바뀌면 cache namespace 변경
- semantic cache threshold는 보수적으로 유지

주의: semantic cache는 한 번 잘못 재사용되면 틀린 답변을 빠르게 반복할 수 있습니다. source와 version 정보를 함께 저장해 추적 가능하게 만듭니다.

## Retrain 로그 계약 수정

현재 `logs/qa.jsonl`은 보안 목적상 query/answer 원문을 저장하지 않고 hash와 length 중심으로 기록합니다. 반면 `retrain.build_dataset`은 `query`, `answer` 원문을 기대합니다. 이 두 계약은 서로 맞지 않습니다.

수정 방향:

- `logs/qa.jsonl`은 운영 감사용으로 유지
- 원문 query/answer는 기본 운영 로그에 저장하지 않음
- retrain 후보는 별도 파일에 저장
- 예: `logs/retrain_candidates.jsonl`
- retrain 후보 로그에는 PII, prompt injection, secret pattern을 통과한 질문만 저장
- answer는 비워두거나 실패 이유만 저장
- 최종 Gold 반영은 사람이 승인한 `data/retrain_approved.json`만 사용

권장 흐름:

1. API 실패 또는 낮은 신뢰도 응답 발생
2. query가 보안 필터를 통과하면 `logs/retrain_candidates.jsonl`에 기록
3. 운영자가 후보를 검토하고 정답을 작성
4. 승인된 항목만 `data/retrain_approved.json`에 저장
5. `retrain.build_dataset`이 승인 항목을 Gold에 반영

원칙: 일반 운영 로그와 학습 후보 로그를 분리해 보안과 재학습 요구사항이 충돌하지 않게 합니다.
