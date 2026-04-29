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
docker run --name qa-redis -p 127.0.0.1:6379:6379 -d redis:7 redis-server --requirepass your_strong_password
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
$env:REDIS_PASSWORD = "your_strong_password"
$env:ALTONG_API_KEY = "local-secret-key"
$env:ALTONG_RATE_LIMIT_PER_MINUTE = "60"
```

임시 로컬 테스트에서 API key 검사를 끄고 싶을 때만 사용합니다.

```powershell
$env:ALTONG_ALLOW_UNAUTHENTICATED_LOCAL = "true"
```

운영에서는 `ALTONG_ALLOW_UNAUTHENTICATED_LOCAL`을 사용하지 않습니다.

프로덕션 환경에서는 아래처럼 지정합니다. 이 값이 `production` 또는 `prod`이면 `ALTONG_ALLOW_UNAUTHENTICATED_LOCAL=true`를 설정해도 API key 우회가 강제로 비활성화됩니다.

```powershell
$env:ALTONG_ENV = "production"
```

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
- `null`, 빈 문자열, 제어 문자, 특수문자만 있는 질문/답변 제거
- 이메일, 전화번호, 주민번호 같은 개인정보 패턴이 있으면 Q_score 계산 전에 즉시 제거
- 20자 미만 또는 1000자 초과 답변은 제거하지 않고 `Length_Penalty`로 감점
- `views`, `donation` 숫자화
- `Sim(Q, A)`, `1/PPL`, `Length_Penalty` 기반 Q_score 계산
- 상위 데이터는 `gold`, 나머지는 `bronze`로 분리

Q_score preprocessing additions:

- null, empty string, control-character, special-character-only row drop
- streaming chunk processing for large JSON / JSONL input
- CPU multiprocessing for cleaning chunks
- Q_score = alpha * Sim(Q, A) + beta * (1/PPL) + gamma * Length_Penalty
- top 10 percent -> `Gold.jsonl`, remainder -> `Bronze.jsonl`
- checkpoint resume through `data/preprocess_checkpoint.json`
- PII detection audit through `logs/security_audit.log`

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
- 이메일, 전화번호, 주민번호가 포함된 데이터

개인정보가 탐지되어 제거된 경우 `logs/security_audit.log`에 감사 기록을 남깁니다. 이 로그에는 원문 개인정보를 남기지 않고, row id, 탐지 종류, 콘텐츠 해시만 남깁니다.

`alpha`, `beta`, `gamma`는 점수 계산에서 어떤 항목을 더 중요하게 볼지 정하는 비율입니다. 코드에 고정하지 않고 `preprocess/scoring_config.yaml` 또는 환경 변수로 바꿀 수 있습니다.

- `alpha`를 키우면 질문과 답변이 서로 맞는지를 더 중요하게 봅니다.
- `beta`를 키우면 답변 문장의 자연스러움을 더 중요하게 봅니다.
- `gamma`를 키우면 답변 길이가 적절한지를 더 중요하게 봅니다.

조금 더 실무적으로 보면 아래처럼 바뀝니다.

```text
Q_score = alpha * Sim(Q, A) + beta * (1/PPL) + gamma * Length_Penalty
```

- `alpha`가 커지면 Gold에 들어갈 데이터가 "질문과 답변이 정확히 짝이 맞는지" 중심으로 뽑힙니다. FAQ처럼 질문-답변 매칭이 중요한 데이터에 유리합니다.
- `alpha`가 너무 크면 문장은 자연스럽지만 질문과 약간 다르게 표현된 좋은 답변이 낮게 평가될 수 있습니다.
- `beta`가 커지면 Gold에 들어갈 데이터가 "문장이 자연스럽고 모델이 읽기 쉬운지" 중심으로 뽑힙니다. LoRA 학습용 답변 품질을 올리는 데 유리합니다.
- `beta`가 너무 크면 질문과 완전히 맞지는 않아도 문장만 자연스러운 답변이 높게 평가될 수 있습니다.
- `gamma`가 커지면 Gold에 들어갈 데이터가 "답변 길이가 너무 짧거나 너무 길지 않은지" 중심으로 뽑힙니다. 짧은 단답이나 지나치게 긴 복붙 답변을 줄이는 데 유리합니다.
- `gamma`가 너무 크면 짧지만 정확한 답변이나 길지만 필요한 설명이 많은 답변이 낮게 평가될 수 있습니다.

처음에는 현재 기본값처럼 `alpha=0.55`, `beta=0.35`, `gamma=0.10`으로 두는 것이 무난합니다. 질문-답변이 엉뚱하게 연결된 데이터가 많으면 `alpha`를 올리고, 문장이 어색한 데이터가 많으면 `beta`를 올리고, 너무 짧거나 긴 답변이 Gold에 많이 섞이면 `gamma`를 올립니다.

### 3.4 Compatibility With The Previous Preprocess

기존 전처리와 새 전처리는 동시에 같은 기준을 중복 적용하지 않습니다. 새 전처리에서는 기존 단계 중 일부를 그대로 유지하고, 일부는 Q_score 방식으로 대체했습니다.

유지된 부분:

- 필수 컬럼 확인: `id`, `question`, `answer`, `category`, `views`, `donation`이 없으면 오류 처리
- HTML tag 제거
- 중복 공백 제거
- `views`, `donation` 숫자 변환
- 질문/답변이 비어 있거나 학습에 위험한 값이면 제거

대체된 부분:

- 기존 `score = views * 0.3 + donation * 0.7` 방식은 더 이상 최종 분류 점수로 쓰지 않습니다.
- 기존에는 20자 미만 답변을 제거했지만, 새 방식에서는 제거하지 않고 `Length_Penalty`로 감점합니다.
- 기존 main preprocess의 SBERT 중복 제거 단계는 새 main preprocess에서 직접 실행하지 않습니다. 현재 Gold/Bronze 분류는 Q_score 기준 상위 10% 분류가 우선입니다.

따라서 현재 기준에서 확인된 핵심 충돌은 없습니다. 다만 운영 정책상 "20자 미만 답변은 무조건 삭제"가 필요해지면 `Length_Penalty` 방식과 목표가 달라지므로 둘 중 하나를 선택해야 합니다. 지금 요구사항은 "길이 조건은 감점"이므로 새 전처리 기준과 맞습니다.

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
- `model.train_lora`: 학습 완료 후 `artifacts/manifest.hash` SHA-256 manifest 생성

주요 결과:

```text
data/pairs.json
data/triplets.json
artifacts/intent_model
artifacts/rag/faiss.index
artifacts/rag/metadata.json
artifacts/llama_lora
artifacts/manifest.hash
```

생성된 `data/`, `artifacts/`, logs, hash manifest 보관 기준은 `PORTABLE_INDEX.md`를 따릅니다.

API 서버는 시작할 때 `artifacts/manifest.hash`와 현재 `artifacts/` 폴더의 SHA-256 해시를 비교합니다. 파일이 바뀌었거나 빠졌거나 새 파일이 추가되면 서버 시작을 중단합니다.

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
invalid username-password pair
```

확인:

```powershell
docker ps
```

해결:

```powershell
docker start qa-redis
$env:REDIS_URL = "redis://localhost:6379/0"
$env:REDIS_PASSWORD = "your_strong_password"
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
- `logs/qa.jsonl` whitelist logging: 원문 질문/답변을 저장하지 않고 hash, 길이, 성공 여부, source, similarity, language만 저장
- PII scanner: 전처리 단계에서 이메일, 전화번호, 주민번호 포함 row를 Q_score 계산 전에 제거
- PII audit: 제거 이벤트를 `logs/security_audit.log`에 원문 없이 기록
- Redis password auth: `REDIS_PASSWORD`로 Redis 인증
- artifact integrity check: `artifacts/manifest.hash` 기준으로 API 시작 시 SHA-256 검증
- 언어별 Redis cache 분리

아직 주의할 점:

- prompt injection 방어는 완전 차단이 아닙니다.
- Redis는 `127.0.0.1` 바인딩과 비밀번호 사용 전제입니다.
- `artifacts/manifest.hash` 자체는 무결성 기준 파일이므로 Git이나 외부 응답에 노출하지 않습니다.
- rate limit은 메모리 기반이라 다중 worker에서는 공유되지 않습니다.
- 웹 검색 확장은 아직 별도 구현이 필요합니다.

외부 공개보다 사내 내부 IP 연결을 우선합니다.

### 10.1 Latest Security Change Summary

이번 보안 업데이트는 로컬 LLM 파이프라인에서 개인정보 유출, 무인증 접근, artifact 변조, 시스템 프롬프트 노출을 줄이기 위한 변경입니다.

변경된 실행 영향:

- Redis는 이제 비밀번호가 필요합니다. `docker run`에서 `redis-server --requirepass your_strong_password`를 사용하고, 서버 실행 전 `$env:REDIS_PASSWORD`를 같은 값으로 설정해야 합니다.
- 운영 환경은 `$env:ALTONG_ENV = "production"` 또는 `"prod"`로 표시합니다. 이 상태에서는 `ALTONG_ALLOW_UNAUTHENTICATED_LOCAL=true`를 설정해도 API key 우회가 동작하지 않습니다.
- `model.train_lora`가 끝나면 `artifacts/manifest.hash`가 생성됩니다. 이 파일은 현재 `artifacts/` 폴더의 SHA-256 목록입니다.
- API 서버는 시작 시 `artifacts/manifest.hash`를 읽고 현재 artifact 파일들과 비교합니다. 파일이 누락, 추가, 변조되면 서버가 시작되지 않습니다.
- 전처리 중 이메일, 전화번호, 주민번호가 포함된 row는 Q_score 계산 전에 제거됩니다.
- 제거된 PII row는 `logs/security_audit.log`에 기록됩니다. 원문 개인정보는 남기지 않고 row id, PII 종류, 콘텐츠 해시만 남깁니다.
- `logs/qa.jsonl`은 whitelist 방식으로 기록됩니다. 원문 질문/답변 대신 `query_sha256`, `answer_sha256`, 길이, 성공 여부, source, similarity, language만 저장합니다.
- 응답에 `system prompt`, `developer message`, `ALTONG_API_KEY`, `REDIS_PASSWORD`, `internal secret`, `시스템 프롬프트` 같은 내부 키워드가 포함되면 `source="output_blocked"`로 차단됩니다.

운영자가 확인할 파일:

```text
preprocess/pipeline.py          PII drop and security audit
cache/redis_client.py           REDIS_PASSWORD authentication
security/api_guard.py           local unauth bypass guard
security/output_validator.py    internal keyword output block
security/artifact_integrity.py  SHA-256 manifest write/verify
model/train_lora.py             manifest generation after training
pipeline/pipeline.py            whitelist qa logging and output guard
api/server.py                   startup integrity check
```

보안상 중요한 운영 순서:

```powershell
python -m preprocess.pipeline
python -m preprocess.build_pairs
python -m preprocess.build_triplet
python -m intent.train
python -m rag.embed
python -m model.train_lora
$env:REDIS_PASSWORD = "your_strong_password"
$env:ALTONG_API_KEY = "local-secret-key"
$env:ALTONG_ENV = "production"
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

`model.train_lora`를 다시 실행하거나 artifact 파일을 교체했다면 `artifacts/manifest.hash`도 새 artifact 기준으로 다시 생성되어야 합니다. manifest가 오래된 상태면 API 서버는 변조 가능성으로 보고 시작을 중단합니다.

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
$env:REDIS_PASSWORD = "your_strong_password"
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
