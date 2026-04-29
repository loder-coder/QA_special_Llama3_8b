# Hybrid Q&A System Tutorial Runbook

이 문서는 `altong_ai`를 로컬 싱글 노드 AI 서버로 실행하는 순서형 튜토리얼입니다.

목표 구조:

```text
Spring Boot Backend
  -> Internal IP JSON API
  -> FastAPI /ask
  -> Redis localhost cache
  -> RAG / FAISS
  -> LLM / LoRA model
  -> JSON answer
```

Git에 올리지 않는 데이터, artifact, hash manifest, secret 보관 기준은 `PORTABLE_INDEX.md`를 따릅니다.

## 0. What This Project Does

`/ask` API에 JSON으로 질문을 보내면 답변 JSON을 돌려줍니다.

요청:

```json
{
  "q": "질문내용",
  "language": "ko"
}
```

응답:

```json
{
  "answer": "답변내용",
  "success": true,
  "source": "llm",
  "intent_similarity": 0.0,
  "rag_similarity": 0.82,
  "language": "ko"
}
```

지원 언어:

```text
ko, en, ja, zh, vi
```

## 1. First-Time Setup

### 1.1 Required Tools

필요한 것:

- Python 3.10 이상
- Redis
- HuggingFace 계정과 token
- `meta-llama/Llama-3-8b` 접근 권한
- 충분한 디스크 공간
- LoRA/QLoRA 학습을 하려면 CUDA GPU 권장

확인:

```powershell
python --version
```

실패하면 Python 설치 또는 PATH 설정이 먼저 필요합니다.

### 1.2 Python Environment

프로젝트 폴더에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

재현 가능한 설치가 필요하면:

```powershell
pip install -r requirements.lock
```

### 1.3 HuggingFace Login

LLaMA 모델 다운로드 권한이 필요합니다.

```powershell
huggingface-cli login
```

토큰은 Git에 올리지 않습니다.

### 1.4 Redis Start

Docker가 있으면:

```powershell
docker run --name qa-redis -p 6379:6379 -d redis:7
```

이미 만든 컨테이너가 있으면:

```powershell
docker start qa-redis
```

Redis는 AI 서버 내부에서만 쓰는 전제입니다.

## 2. Environment Variables

서버 실행 전에 PowerShell에 설정합니다.

```powershell
$env:REDIS_URL = "redis://localhost:6379/0"
$env:ALTONG_API_KEY = "local-secret-key"
$env:ALTONG_RATE_LIMIT_PER_MINUTE = "60"
```

임시 로컬 테스트에서 API key 검사를 끄고 싶을 때만 사용합니다.

```powershell
$env:ALTONG_ALLOW_UNAUTHENTICATED_LOCAL = "true"
```

운영에서는 `ALTONG_ALLOW_UNAUTHENTICATED_LOCAL`을 사용하지 않습니다.

## 3. Prepare Data

### 3.1 Raw Data Shape

`data/raw.json`은 배열이어야 합니다.

실제 운영 데이터는 Git에 올리지 않고 `PORTABLE_INDEX.md` 기준으로 보관합니다.

```json
[
  {
    "id": "q_0001",
    "question": "사용자 질문 원문",
    "answer": "기준 답변 본문",
    "category": "질문 카테고리",
    "views": 100,
    "donation": 50
  }
]
```

필드 의미:

- `id`: 질문 고유 ID
- `question`: 유사 질문 탐지와 intent 학습에 사용할 질문
- `answer`: RAG와 LLM 학습에 사용할 답변
- `category`: 답변 생성 prompt의 분류값
- `views`: 점수 계산용 조회수
- `donation`: 점수 계산용 가치 점수

### 3.2 Clean Raw Data

```powershell
python -m preprocess.pipeline
```

결과:

```text
data/Gold.jsonl
data/Bronze.jsonl
data/report.txt
data/preprocess_checkpoint.json
```

정제 과정:

- HTML tag 제거
- 중복 공백 제거
- 20자 미만 또는 1000자 초과 답변은 제거하지 않고 `Length_Penalty`로 감점
- 의미 없는 답변 제거
- `views`, `donation` 숫자화
- 점수 계산
- 유사 질문 중복 제거
- 상위 데이터는 `gold`, 나머지는 `bronze`로 분리

Q_score preprocessing additions:

- null, empty string, control-character, special-character-only row drop
- streaming chunk processing for large JSON / JSONL input
- CPU multiprocessing for cleaning chunks
- Q_score = alpha * Sim(Q, A) + beta * (1/PPL) + gamma * Length_Penalty
- top 10 percent -> `Gold.jsonl`, remainder -> `Bronze.jsonl`
- checkpoint resume through `data/preprocess_checkpoint.json`

Q_score weights are not fixed in code. Set them in `preprocess/scoring_config.yaml` or override them with environment variables:

```powershell
$env:ALTONG_SCORING_ALPHA = "0.55"
$env:ALTONG_SCORING_BETA = "0.35"
$env:ALTONG_SCORING_GAMMA = "0.10"
$env:ALTONG_SCORING_LLAMA_MODEL_PATH = "C:\models\Llama-3-8B"
```

### 3.3 What Changed In Plain Language

이번 전처리 변경은 "좋은 학습 데이터와 참고용 데이터를 자동으로 나누는 작업"입니다.

예전 방식은 조회수(`views`)와 후원값(`donation`)처럼 숫자로 된 인기도를 보고 데이터를 나눴습니다. 새 방식은 질문과 답변의 실제 품질을 더 많이 봅니다. 즉, 사람들이 많이 본 글인지보다 "질문과 답변이 서로 잘 맞는지", "답변 문장이 자연스러운지", "답변 길이가 너무 짧거나 너무 길지 않은지"를 점수로 계산합니다.

새 점수는 `Q_score`라고 부릅니다.

```text
Q_score = 질문-답변 유사도 + 답변 자연스러움 + 답변 길이 점수
```

각 항목의 의미는 다음과 같습니다.

- `Sim(Q, A)`: 질문과 답변이 얼마나 잘 맞는지 보는 점수입니다. 예를 들어 "비밀번호를 잊었어요"라는 질문에 "비밀번호 재설정 방법"을 답하면 높고, 전혀 다른 답변이면 낮습니다.
- `1/PPL`: 답변 문장이 얼마나 자연스러운지 보는 점수입니다. Llama 3 모델이 읽었을 때 어색하고 이상한 문장일수록 점수가 낮아집니다.
- `Length_Penalty`: 답변 길이에 대한 감점입니다. 20자 미만이면 너무 짧다고 보고 감점하고, 1000자를 넘으면 너무 길다고 보고 감점합니다.

최종 결과 파일은 이렇게 나뉩니다.

- `data/Gold.jsonl`: 점수가 높은 상위 10% 데이터입니다. LoRA 학습에 우선 사용합니다.
- `data/Bronze.jsonl`: 나머지 데이터입니다. RAG 검색 인덱스에 넣어 참고 문서처럼 사용합니다.
- `data/report.txt`: Gold와 Bronze가 각각 몇 개이고, 평균 점수와 최소/최대 점수가 얼마인지 보여주는 요약 보고서입니다.
- `data/preprocess_checkpoint.json`: 중간 저장 파일입니다. 1.5GB처럼 큰 파일을 처리하다가 중간에 멈춰도 처음부터 다시 하지 않도록 도와줍니다.
- `data/scored.jsonl`: 점수 계산이 끝난 전체 중간 결과입니다. 최종 분리 전 임시 결과로 보면 됩니다.

대용량 데이터를 안전하게 처리하기 위해 한 번에 전체 파일을 메모리에 올리지 않습니다. 큰 박스를 한꺼번에 들지 않고 작은 묶음으로 나눠 옮기는 것처럼, JSON 데이터를 chunk 단위로 조금씩 읽고 처리합니다.

정제 중 아래 데이터는 학습 오류를 만들 수 있으므로 점수 계산 전에 제외합니다.

- 질문이나 답변이 비어 있는 데이터
- `null` 값이 들어 있는 데이터
- 제어 문자처럼 학습에 방해되는 문자가 들어 있는 데이터
- 특수문자만 있고 실제 글자나 숫자가 없는 데이터

`alpha`, `beta`, `gamma`는 점수 계산에서 어떤 항목을 더 중요하게 볼지 정하는 비율입니다. 코드에 고정하지 않고 `preprocess/scoring_config.yaml` 또는 환경 변수로 바꿀 수 있습니다.

- `alpha`를 키우면 질문과 답변이 서로 맞는지를 더 중요하게 봅니다.
- `beta`를 키우면 답변 문장의 자연스러움을 더 중요하게 봅니다.
- `gamma`를 키우면 답변 길이가 적절한지를 더 중요하게 봅니다.

## 4. Build Training And Search Artifacts

아래 순서대로 실행합니다.

```powershell
python -m preprocess.build_pairs
python -m preprocess.build_triplet
python -m intent.train
python -m rag.embed
python -m model.train_lora
```

각 단계 의미:

- `preprocess.build_pairs`: intent 학습용 질문 pair 생성
- `preprocess.build_triplet`: intent 학습용 triplet 생성
- `intent.train`: 유사 질문 판단 모델 생성
- `rag.embed`: FAISS index와 metadata 생성
- `model.train_lora`: LLaMA LoRA adapter 학습

주요 결과:

```text
data/pairs.json
data/triplets.json
artifacts/intent_model
artifacts/rag/faiss.index
artifacts/rag/metadata.json
artifacts/llama_lora
```

생성된 `data/`, `artifacts/`, logs, hash manifest 보관 기준은 `PORTABLE_INDEX.md`를 따릅니다.

## 5. Start API Server

모든 artifact가 준비된 뒤 서버를 실행합니다.

로컬 테스트:

```powershell
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Spring Boot가 사내 내부망에서 호출해야 하면 AI 서버 내부 IP로 열 수 있습니다.

```powershell
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

이 경우 Windows 방화벽 또는 사내 방화벽에서 Spring Boot 서버 IP만 `8000` 포트 접근을 허용해야 합니다.

## 6. Use The Model

### 6.1 PowerShell Test

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ask" `
  -Headers @{"X-API-Key"="local-secret-key"} `
  -ContentType "application/json" `
  -Body '{"q":"질문내용","language":"ko"}'
```

영어 답변 요청:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ask" `
  -Headers @{"X-API-Key"="local-secret-key"} `
  -ContentType "application/json" `
  -Body '{"q":"How do I use this service?","language":"en"}'
```

베트남어 답변 요청:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ask" `
  -Headers @{"X-API-Key"="local-secret-key"} `
  -ContentType "application/json" `
  -Body '{"q":"Cach su dung dich vu nay?","language":"vi"}'
```

### 6.2 Spring Boot Call Shape

Spring Boot에서는 AI 서버 내부 IP로 JSON POST를 보냅니다.

```text
POST http://10.x.x.x:8000/ask
Header: X-API-Key: <ALTONG_API_KEY>
Content-Type: application/json
```

Body:

```json
{
  "q": "질문내용",
  "language": "ko"
}
```

외부 공개 DNS보다 사내 내부 IP 또는 내부 DNS를 우선합니다.

## 7. Runtime Flow

`/ask` 요청은 아래 순서로 처리됩니다.

```text
1. API key 검사
2. rate limit 검사
3. q 길이 검증
4. language 정규화
5. prompt injection 의심 패턴 차단
6. Redis exact cache 확인
7. Redis semantic cache 확인
8. RAG 검색
9. LLM 답변 생성
10. output validation
11. Redis cache 저장
12. logs/qa.jsonl 기록
```

캐시 우선순위:

```text
exact cache -> semantic cache -> RAG + LLM
```

언어별 캐시는 분리됩니다. 같은 질문이라도 `ko`와 `en`은 다른 캐시로 저장됩니다.

## 8. Debug Tutorial

### 8.1 Python Command Not Found

증상:

```text
Python was not found
```

확인:

```powershell
python --version
```

해결:

- Python 설치
- PATH 설정
- Windows App Execution Alias 비활성화 확인

### 8.2 Redis Connection Error

증상:

```text
Connection refused
Error connecting to Redis
```

확인:

```powershell
docker ps
```

해결:

```powershell
docker start qa-redis
$env:REDIS_URL = "redis://localhost:6379/0"
```

### 8.3 API Key Error

증상:

```text
401 invalid api key
503 api key is not configured
```

확인:

```powershell
$env:ALTONG_API_KEY
```

해결:

```powershell
$env:ALTONG_API_KEY = "local-secret-key"
```

요청 header에도 같은 값을 넣습니다.

```text
X-API-Key: local-secret-key
```

### 8.4 FAISS Index Missing

증상:

```text
RAG index files are missing. Run rag/embed.py first.
```

해결:

```powershell
python -m rag.embed
```

그래도 실패하면 `data/Gold.jsonl`, `data/Bronze.jsonl`이 있는지 확인합니다.

### 8.5 LoRA Adapter Missing

증상:

```text
artifacts/llama_lora not found
```

해결:

```powershell
python -m model.train_lora
```

모델 접근 권한 또는 GPU/메모리 문제가 있을 수 있습니다.

### 8.6 HuggingFace Model Download Error

확인:

```powershell
huggingface-cli login
```

점검:

- HuggingFace token이 맞는지
- `meta-llama/Llama-3-8b` 접근 권한이 있는지
- 인터넷 연결이 되는지

### 8.7 Output Blocked

응답 source가 아래처럼 나오면:

```json
{
  "source": "output_blocked",
  "success": false
}
```

의미:

- LLM 답변에 내부 prompt 표식 또는 금칙 문구가 섞였습니다.
- 답변을 사용자에게 보내지 않고 차단한 상태입니다.

확인:

```text
logs/qa.jsonl
```

민감정보는 `[REDACTED]`로 마스킹됩니다.

## 9. Retraining Tutorial

운영 중 실패 로그를 모아 재학습 후보를 만들 수 있습니다.

### 9.1 Build Review Candidates

```powershell
python -m retrain.build_dataset
```

결과:

```text
data/retrain_candidates.json
```

자동 제외:

- prompt injection 의심 query
- 민감정보 포함 query/answer
- 너무 짧은 query
- `sensitive_data_detected=true` 로그

### 9.2 Manual Approval

사람이 후보를 검토하고 승인 파일을 만듭니다.

```text
data/retrain_approved.json
```

승인된 item에는 사람이 확정한 답변과 `approved=true`가 있어야 합니다.

### 9.3 Add Approved Data To Gold

Python 환경에서 실행합니다.

```python
from retrain.build_dataset import add_approved_queries_to_gold

add_approved_queries_to_gold()
```

주의:

- 실패 로그를 자동으로 `gold`에 넣지 않습니다.
- 최종 승인은 반드시 사람이 합니다.
- 웹 검색 결과를 학습 데이터로 넣을 때는 출처와 검증 여부를 따로 확인합니다.

## 10. Security Notes

현재 적용된 방어:

- POST JSON body 사용
- `q` 길이 제한
- API key 검사
- 단일 프로세스 rate limit
- prompt injection 입력 차단
- system prompt와 사용자 데이터 영역 분리
- output validation
- 로그 민감정보 마스킹
- 언어별 Redis cache 분리

아직 주의할 점:

- prompt injection 방어는 완전 차단이 아닙니다.
- Redis는 로컬 내부 사용 전제입니다.
- FAISS artifact hash 검증은 아직 코드에 없습니다.
- rate limit은 메모리 기반이라 다중 worker에서는 공유되지 않습니다.
- 웹 검색 확장은 아직 별도 구현이 필요합니다.

외부 공개보다 사내 내부 IP 연결을 우선합니다.

## 11. File Policy

Git에 올리는 것:

```text
source code
requirements.txt
requirements.lock
RUNBOOK.md
IMPROVEMENTS.md
PORTABLE_INDEX.md
.gitignore
```

Git 밖에 두는 것:

```text
.env
logs/
artifacts/
actual data/*.json
hash manifest
API keys
HuggingFace token
search API keys
```

자세한 기준은 `PORTABLE_INDEX.md`를 따릅니다.

## 12. Quick Command Summary

처음부터 빌드:

```powershell
python -m preprocess.pipeline
python -m preprocess.build_pairs
python -m preprocess.build_triplet
python -m intent.train
python -m rag.embed
python -m model.train_lora
```

서버 실행:

```powershell
$env:REDIS_URL = "redis://localhost:6379/0"
$env:ALTONG_API_KEY = "local-secret-key"
$env:ALTONG_RATE_LIMIT_PER_MINUTE = "60"
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

질문 테스트:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ask" `
  -Headers @{"X-API-Key"="local-secret-key"} `
  -ContentType "application/json" `
  -Body '{"q":"질문내용","language":"ko"}'
```

재학습 후보 생성:

```powershell
python -m retrain.build_dataset
```
