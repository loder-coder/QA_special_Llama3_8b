# Hybrid Q&A System Runbook

이 문서는 현재 로컬 환경에서 즉시 실행할 수 없다는 전제를 포함한 실행 절차입니다.

## 1. Prerequisites

- Python 3.10 이상
- Redis 서버
- HuggingFace 계정 및 토큰
- `meta-llama/Llama-3-8b` 접근 권한
- 충분한 디스크 공간
- LoRA/QLoRA 학습용 CUDA GPU 권장

현재 코드에는 다음 외부 의존성이 있습니다.

- HuggingFace 모델 다운로드
- SentenceTransformer 모델 다운로드
- FAISS CPU index 생성
- Redis 접속
- LLaMA 3 base model 및 LoRA adapter 로딩

## 2. Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

HuggingFace 로그인이 필요한 경우:

```powershell
huggingface-cli login
```

Redis 예시:

```powershell
docker run --name qa-redis -p 6379:6379 -d redis:7
```

필요하면 환경 변수를 지정합니다.

```powershell
$env:REDIS_URL = "redis://localhost:6379/0"
```

## 3. Data Preparation

`data/raw.json`은 다음 형식의 배열이어야 합니다.

```json
[
  {
    "id": "q_0001",
    "question": "question text",
    "answer": "answer text",
    "category": "category",
    "views": 100,
    "donation": 50
  }
]
```

### 3.1 Raw 데이터 정제 body 설정

정제 대상 raw item은 아래 필드를 반드시 포함해야 합니다. 현재 전처리 코드는 이 필드명을 기준으로 pandas DataFrame을 만들고, 누락된 필드가 있으면 오류를 발생시킵니다.

```json
{
  "id": "q_0001",
  "question": "사용자 질문 원문",
  "answer": "채택 또는 기준 답변 본문",
  "category": "질문 카테고리",
  "views": 100,
  "donation": 50
}
```

필드 의미:

- `id`: 질문 고유 ID
- `question`: 유사 질문 탐지와 intent 학습에 사용할 질문 텍스트
- `answer`: LLM fine-tuning과 RAG 문서에 사용할 답변 본문
- `category`: prompt의 `[카테고리]` 영역에 들어갈 분류값
- `views`: 점수 계산에 사용할 조회수
- `donation`: 점수 계산에 사용할 후원/가치 점수

### 3.2 데이터 정제 방법

전처리 실행:

```powershell
python -m preprocess.pipeline
```

정제 기준:

- `question`, `answer`, `category`의 HTML tag와 중복 공백 제거
- `answer` 길이가 20자 미만이면 제거
- `몰라요`, `모름`, `없음`, `테스트`, `asdf`, 반복 문자 등 의미 없는 답변 제거
- `views`, `donation`을 숫자로 변환하고 실패 시 `0` 처리
- `score = views * 0.3 + donation * 0.7` 계산
- SBERT 기준 질문 유사도 `0.9` 초과 중복 질문 제거
- score 상위 30%를 `data/gold.json`, 나머지를 `data/bronze.json`으로 저장

### 3.3 데이터 정제 이후 실행 순서

정제 이후에는 아래 순서로 학습/인덱싱 artifact를 생성합니다.

```powershell
python -m preprocess.build_pairs
python -m preprocess.build_triplet
python -m intent.train
python -m rag.embed
python -m model.train_lora
```

순서 의미:

- `build_pairs`: gold 질문 간 positive/negative pair 생성
- `build_triplet`: intent 학습용 anchor/positive/negative triplet 생성
- `intent.train`: Q-Q intent 모델 저장
- `rag.embed`: RAG 전용 bge-base embedding과 FAISS index 생성
- `model.train_lora`: bronze 학습 후 gold를 이어서 학습하는 QLoRA adapter 생성

## 4. Full Pipeline Build Order

전체를 처음부터 실행하는 경우 아래 순서로 실행합니다.

```powershell
python -m preprocess.pipeline
python -m preprocess.build_pairs
python -m preprocess.build_triplet
python -m intent.train
python -m rag.embed
python -m model.train_lora
```

각 단계 결과:

- `preprocess.pipeline`: `data/gold.json`, `data/bronze.json`
- `preprocess.build_pairs`: `data/pairs.json`
- `preprocess.build_triplet`: `data/triplets.json`
- `intent.train`: `artifacts/intent_model`
- `rag.embed`: `artifacts/rag/faiss.index`, `artifacts/rag/metadata.json`
- `model.train_lora`: `artifacts/llama_lora`

## 5. API Server

모든 artifact가 생성된 뒤 실행합니다.

```powershell
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

요청:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ask" `
  -ContentType "application/json" `
  -Body '{"q":"질문내용"}'
```

HTTP body:

```json
{
  "q": "질문내용"
}
```

### 5.1 Runtime Query Flow

현재 `/ask` 요청 처리 순서:

1. Prompt injection 패턴 검사
2. Redis exact match 확인
3. Redis에 저장된 기존 질문들과 intent similarity 계산
4. similarity `> 0.9`이면 Redis의 기존 답변 재사용
5. 그 외에는 RAG 검색
6. RAG similarity `< 0.7`이면 빈 context로 LLM 호출
7. LLM 답변 생성
8. 생성된 질문/답변을 Redis에 저장
9. 요청/응답 로그 저장

이 구조의 목적은 LLM 호출을 최대한 줄이는 것입니다.

### 5.2 Traffic Assumption

현재 기준:

- 피크 시 동시 요청: 50~100
- 예상 처리량: QPS 30~80
- 배포 형태: 로컬 싱글 노드

로컬 싱글 노드에서 주의할 점:

- Redis exact cache hit는 가장 빠릅니다.
- Redis semantic cache는 Redis에 저장된 질문 수가 많아질수록 SBERT similarity 계산 비용이 증가합니다.
- QPS 30~80을 안정적으로 처리하려면 Redis cache hit 비율을 높이고, LLM 호출 비율을 낮게 유지해야 합니다.
- LLM 호출이 많아지면 단일 노드에서 병목은 API가 아니라 GPU/모델 inference가 됩니다.
- 운영 전에는 캐시 hit ratio, 평균 latency, LLM 호출 비율을 로그로 확인해야 합니다.

## 6. Retraining Dataset

실패 로그에서 검수 후보를 만들려면:

```powershell
python -m retrain.build_dataset
```

생성 결과:

```text
data/retrain_candidates.json
```

semi-auto 기준:

- 자동: 로그 수집
- 자동: prompt injection 의심 query 필터링
- 자동: 짧은 query 제거
- 자동: query 빈도 기준 정렬
- 수동: 사람이 `approved=true`, `answer`, `category`를 최종 확정
- 수동 승인 후: `data/retrain_approved.json`으로 저장

승인된 데이터만 gold dataset에 반영하려면 Python에서 다음 함수를 호출합니다.

```python
from retrain.build_dataset import add_approved_queries_to_gold

add_approved_queries_to_gold()
```

로그 파일:

```text
logs/qa.jsonl
```

### 6.1 Retraining Period

현재 트래픽 기준(QPS 30~80, 피크 동시 요청 50~100)에서는 재학습 주기를 아래처럼 잡는 것이 현실적입니다.

- 초기 운영 1개월: 주 1회 후보 생성, 사람이 승인한 데이터만 반영
- 안정화 이후: 2주 1회 또는 월 1회
- 장애/오답이 급증한 경우: 즉시 후보 생성 후 수동 승인

자동 수집된 로그를 바로 gold에 넣으면 데이터 poisoning 위험이 있습니다. 따라서 최종 승인 단계는 반드시 수동으로 유지합니다.

## 7. Expected Failure Points

- `python` 명령이 없으면 Python 설치 또는 PATH 설정이 필요합니다.
- Redis가 실행 중이 아니면 `/ask` 요청 전에 cache 초기화 또는 조회에서 실패합니다.
- FAISS index가 없으면 API 시작 시 RAG retriever 초기화가 실패합니다.
- `artifacts/llama_lora`가 없으면 LoRA adapter 로딩이 실패합니다.
- LLaMA 3 접근 권한이 없으면 모델 다운로드가 실패합니다.
- `bitsandbytes`는 Windows/CPU 환경에서 동작이 제한될 수 있습니다.
- `data/raw.json`이 비어 있으면 학습 데이터 생성 단계가 빈 결과를 만들 수 있습니다.

## 8. Local Review Caveat

현재 로컬에서 Python 실행기가 감지되지 않아 문법 컴파일, import 검증, API 기동 검증은 수행하지 못했습니다. 이 문서는 정적 코드 리뷰 기준의 실행 절차입니다.

## 9. Local Single Node Security Notes

현재 기준은 로컬 싱글 노드 실행입니다. 이 전제에서 우선 적용한 개선점:

- `/ask`를 GET query string에서 POST JSON body로 변경해 질문이 URL, 브라우저 history, 프록시 access log에 남는 위험을 줄였습니다.
- 요청 body의 `q`에 `min_length=1`, `max_length=2000` 제한을 추가했습니다.
- `ignore previous instructions` 같은 prompt injection 의심 패턴은 LLM 호출 전에 차단합니다.
- LLM prompt에 `[참고 문서]`와 `[질문]`을 신뢰할 수 없는 데이터로 선언하는 인젝션 방어 지시를 추가했습니다.
- 사용자 질문과 검색 context는 `<<<CONTEXT ... CONTEXT>>>`, `<<<QUESTION ... QUESTION>>>` 경계 안에 넣어 시스템 지시와 데이터 영역을 분리했습니다.
- 데이터 영역 안에 들어온 `[카테고리]`, `[참고 문서]`, `[질문]`, `[답변]` label은 일반 문자열 label로 치환합니다.
- Redis exact cache와 Redis semantic cache를 사용해 LLM 호출 빈도를 줄입니다.

로컬 싱글 노드에서 아직 남아 있는 취약점:

- Redis는 기본값이 `redis://localhost:6379/0`이며 인증/TLS가 강제되지 않습니다. 로컬 외부에 노출하지 않는 전제입니다.
- `logs/qa.jsonl`에는 재학습 후보 분석을 위해 query와 answer를 저장하지만, 민감정보 패턴은 `[REDACTED]`로 마스킹하고 `sensitive_data_detected=true` 로그는 재학습 후보에서 제외합니다.
- FAISS index와 metadata 파일의 무결성 검증은 없습니다. 로컬 artifact 디렉터리 접근 권한을 제한해야 합니다.
- prompt injection 방어는 완화책이며 완전 차단이 아닙니다. 현재는 입력 금칙 패턴 탐지, system prompt 분리, output validation을 적용했습니다. 운영 전에는 실제 공격 로그 기준으로 패턴을 계속 보강해야 합니다.
- API는 `ALTONG_API_KEY`를 필수로 요구하고 `X-API-Key` 인증을 검사합니다. 로컬 무인증 테스트가 꼭 필요할 때만 `ALTONG_ALLOW_UNAUTHENTICATED_LOCAL=true`로 우회합니다. 단일 프로세스 메모리 기반 rate limit도 적용합니다. 로컬 싱글 노드 외부로 노출할 경우 reverse proxy/IP 제한도 함께 적용해야 합니다.

### 9.1 Prompt Injection Policy

현재 차단하는 대표 패턴:

```python
if "ignore previous instructions" in query:
    block()
```

실제 코드는 영어/한국어 prompt injection 의심 표현을 정규식으로 검사합니다. 차단 대상이면 source는 `blocked`, success는 `false`로 로그에 남습니다.

완전 차단은 보장하지 않습니다. 현실적인 다층 방어 기준은 다음 조합입니다.

- 입력 단계: 명시적 injection 패턴 차단
- 검색 단계: RAG context를 신뢰하지 않는 데이터로 취급
- 프롬프트 단계: context/question delimiter 분리
- 출력 단계: 운영 전 금칙 응답 validation 추가
- 재학습 단계: injection 의심 로그는 후보에서 제외

## 10. Git / Log / Dependency Policy

- Git에 올리지 않을 항목: `.env`, `.env.*`, `.venv/`, `logs/`, `artifacts/`, 실제 운영 데이터가 들어간 `data/*.json`
- `/ask` 응답에는 운영 로그 노출을 줄이기 위해 사용자 질문 원문을 반환하지 않습니다.
- `logs/qa.jsonl`에는 재학습 후보 분석을 위해 실패 로그를 남기되, 이메일/전화번호/주민번호/카드번호/API key/token/DB URL 등은 `[REDACTED]`로 마스킹합니다.
- 민감정보가 감지된 로그는 `sensitive_data_detected=true`로 남기고, `retrain.build_dataset` 후보 생성 단계에서 제외합니다.
- 로컬 싱글 노드 기준 Redis 기본값은 `redis://localhost:6379/0`입니다. 외부 Redis로 바꾸는 경우 인증/네트워크 제한을 별도로 적용해야 합니다.
- 재현 가능한 설치가 필요하면 `requirements.lock`을 사용합니다.
