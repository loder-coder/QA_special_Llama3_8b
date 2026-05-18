# altong_ai 쉬운 운영 안내서

이 문서는 `altong_ai`를 한 대의 AI 서버에서 운영하는 방법을 설명합니다.

목표는 단순합니다.

```text
사용자
  -> Spring Boot 백엔드
  -> altong_ai FastAPI /ask
  -> Redis 캐시 확인
  -> RAG 문서 검색
  -> LLM 답변 생성
  -> JSON 답변 반환
```

현재 코드는 싱글노드 운영을 기본으로 합니다. 다만 LLM 추론은 필요하면 별도 모델 서버로 분리할 수 있게 되어 있습니다.

```text
기본값:
FastAPI가 같은 프로세스에서 로컬 LLaMA/LoRA 모델을 직접 로드

선택값:
FastAPI는 질문 처리만 담당하고, LLM 추론은 HTTP 모델 서버에 요청
```

## 1. 전체 구조를 쉽게 이해하기

각 부품의 역할은 아래와 같습니다.

| 이름 | 쉬운 설명 | 담당 |
| --- | --- | --- |
| Spring Boot | 실제 서비스 백엔드 | 사용자 요청을 받고 AI 서버 `/ask` 호출 |
| FastAPI | AI 서버 입구 | 보안 검사, 캐시 확인, RAG 검색, 답변 조립 |
| Redis | 임시 기억장치 | 같은 질문 또는 비슷한 질문의 답변 재사용 |
| RAG / FAISS | 문서 검색기 | 답변에 참고할 문서 찾기 |
| LLM / LoRA | 답변 생성기 | 최종 자연어 답변 생성 |
| logs | 운영 기록 | 어떤 경로로 답했는지 기록 |
| artifacts | 모델과 인덱스 결과물 | 학습 모델, FAISS 인덱스, hash manifest 보관 |

운영 중 가장 중요한 원칙은 아래입니다.

- Spring Boot는 Redis나 FAISS를 직접 만지지 않습니다.
- Spring Boot는 FastAPI `/ask`만 호출합니다.
- Redis는 외부에 공개하지 않습니다.
- 모델, 인덱스, 로그, secret은 Git에 올리지 않습니다.
- 운영 서버에서는 API key 없이 접근할 수 없게 둡니다.

## 2. 처음 한 번 준비하기

### 2.1 Python 준비

프로젝트 폴더에서 실행합니다.

```powershell
python --version
```

Python이 없다고 나오면 Python 3.10 이상을 먼저 설치합니다.

가상환경을 만듭니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2.2 Redis 준비

Redis는 캐시용입니다. 운영에서는 비밀번호를 걸어야 합니다.

예시:

```powershell
docker run --name altong-redis -p 127.0.0.1:6379:6379 redis redis-server --requirepass your_strong_password
```

서버 실행 전 같은 비밀번호를 환경변수로 넣습니다.

```powershell
$env:REDIS_PASSWORD = "your_strong_password"
```

중요:

- Redis 포트는 `127.0.0.1`에만 묶습니다.
- Redis를 `0.0.0.0`으로 열지 않습니다.
- Spring Boot에서 Redis로 직접 접속하지 않습니다.

### 2.3 운영 환경변수 준비

운영에서는 아래 값을 지정합니다.

```powershell
$env:ALTONG_ENV = "production"
$env:ALTONG_API_KEY = "spring_boot_and_ai_server_shared_secret"
$env:REDIS_PASSWORD = "your_strong_password"
```

`ALTONG_ENV`가 `production` 또는 `prod`이면 로컬 무인증 우회가 꺼집니다.

Spring Boot는 `/ask` 요청에 아래 header를 붙여야 합니다.

```text
X-API-Key: spring_boot_and_ai_server_shared_secret
```

## 3. LLM 실행 방식을 고르기

### 3.1 기본 방식: FastAPI 안에서 로컬 모델 실행

별도 설정을 하지 않으면 기본값입니다.

```powershell
$env:ALTONG_GENERATOR_MODE = "local"
```

이 방식은 구조가 단순합니다. 대신 FastAPI 프로세스가 모델까지 직접 로드하므로 시작 시간이 길고 GPU 메모리를 크게 씁니다.

### 3.2 권장 확장 방식: 모델 서버 분리

운영 안정성을 높이려면 FastAPI와 LLM 추론을 나눕니다.

```text
FastAPI
  - API key 확인
  - prompt injection 차단
  - Redis cache 확인
  - RAG 검색
  - output validation
  - log 기록

Model Server
  - 실제 LLM 추론만 담당
```

FastAPI에서 HTTP 모델 서버를 쓰려면 아래처럼 설정합니다.

```powershell
$env:ALTONG_GENERATOR_MODE = "http"
$env:ALTONG_MODEL_SERVER_URL = "http://127.0.0.1:9000/generate"
$env:ALTONG_MODEL_SERVER_TIMEOUT_SECONDS = "60"
```

모델 서버는 JSON 응답에 아래 중 하나를 넣으면 됩니다.

```json
{
  "answer": "답변 내용"
}
```

또는:

```json
{
  "text": "답변 내용"
}
```

OpenAI 호환 형태의 `choices[0].message.content`도 읽을 수 있습니다.

처음에는 `local`로 검증하고, 운영 안정화 단계에서 `http`로 분리하는 순서를 권장합니다.

## 4. 데이터 준비 순서

AI 서버는 원본 데이터가 바로 답변에 쓰이지 않습니다. 아래 순서로 정제하고 결과물을 만듭니다.

### 4.1 원본 데이터 넣기

원본 데이터는 보통 아래 필드를 가집니다.

```json
{
  "id": 1,
  "question": "질문",
  "answer": "답변",
  "category": "카테고리",
  "views": 0,
  "donation": 0
}
```

### 4.2 데이터 정제

```powershell
python -m preprocess.pipeline
```

이 단계에서 하는 일:

- 너무 짧거나 이상한 질문 제거
- 개인정보가 들어간 데이터 제외
- 중복 질문 정리
- 학습용 데이터와 RAG용 데이터 분리

### 4.3 intent pair 생성

```powershell
python -m preprocess.build_pairs
```

이 단계는 비슷한 질문끼리 묶는 재료를 만듭니다.

### 4.4 triplet 생성

```powershell
python -m preprocess.build_triplet
```

이 단계는 intent 모델 학습에 쓸 `anchor`, `positive`, `negative` 구조를 만듭니다.

### 4.5 intent 모델 학습

```powershell
python -m intent.train
```

triplet 데이터가 있으면 `TripletLoss`로 학습합니다.

triplet이 부족하면 q-q pair 기반 `CosineSimilarityLoss`로 fallback합니다.

pair도 없으면 base intent model을 저장해서 전체 흐름이 멈추지 않게 합니다.

### 4.6 RAG 인덱스 생성

```powershell
python -m rag.embed
```

이 단계는 검색용 FAISS index와 metadata를 만듭니다.

### 4.7 LoRA 학습

```powershell
python -m model.train_lora
```

이 단계는 LLM adapter를 학습합니다.

학습이 끝나면 `artifacts/manifest.hash`도 생성되어야 합니다.

## 5. 서버 실행 순서

### 5.1 Redis 먼저 실행

Redis가 먼저 떠 있어야 합니다.

```powershell
docker start altong-redis
```

### 5.2 환경변수 설정

운영 예시:

```powershell
$env:ALTONG_ENV = "production"
$env:ALTONG_API_KEY = "spring_boot_and_ai_server_shared_secret"
$env:REDIS_PASSWORD = "your_strong_password"
$env:ALTONG_GENERATOR_MODE = "local"
```

HTTP 모델 서버를 쓸 때:

```powershell
$env:ALTONG_GENERATOR_MODE = "http"
$env:ALTONG_MODEL_SERVER_URL = "http://127.0.0.1:9000/generate"
```

### 5.3 모델 서버 실행

`ALTONG_GENERATOR_MODE=local`이면 이 단계는 필요 없습니다.

`ALTONG_GENERATOR_MODE=http`이면 FastAPI를 켜기 전에 모델 서버를 먼저 켭니다.

모델 서버는 아래 정보를 받아 답변을 생성해야 합니다.

```json
{
  "prompt": "최종 프롬프트",
  "question": "질문",
  "context": "RAG 참고 문서",
  "category": "카테고리",
  "language": "ko",
  "max_new_tokens": 512,
  "temperature": 0.2,
  "top_p": 0.9
}
```

### 5.4 FastAPI 실행

```powershell
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

처음에는 내부망에서만 접근 가능하게 방화벽을 제한합니다.

Spring Boot 서버 IP만 `8000` 포트에 접근할 수 있게 두는 것이 좋습니다.

## 6. 정상 동작 확인

로컬 테스트:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ask" `
  -Headers @{ "X-API-Key" = $env:ALTONG_API_KEY } `
  -ContentType "application/json" `
  -Body '{"q":"테스트 질문입니다","language":"ko"}'
```

정상 응답 예시:

```json
{
  "answer": "답변 내용",
  "success": true,
  "source": "llm",
  "intent_similarity": 0.0,
  "rag_similarity": 0.82,
  "language": "ko"
}
```

`source` 의미:

| source | 의미 |
| --- | --- |
| `redis` | 완전히 같은 질문이 캐시에서 재사용됨 |
| `redis_intent` | 비슷한 질문의 답변이 캐시에서 재사용됨 |
| `llm` | RAG 검색 후 LLM이 새 답변을 생성함 |
| `blocked` | prompt injection 의심 요청 차단 |
| `output_blocked` | 답변 보안 검사에서 차단 |
| `validation` | 빈 질문 등 기본 검증 실패 |

## 7. `/ask` 내부 처리 순서

요청 하나는 아래 순서로 처리됩니다.

```text
1. API key 확인
2. rate limit 확인
3. 언어값 정리
4. 빈 질문 차단
5. prompt injection 차단
6. Redis exact cache 확인
7. Redis semantic cache 확인
8. RAG 문서 검색
9. LLM 답변 생성
10. output validation
11. Redis cache 저장
12. logs/qa.jsonl 기록
```

캐시 우선순위는 유지해야 합니다.

```text
exact match
  -> intent similarity 0.9 이상 cache reuse
  -> RAG + LLM
```

이 순서를 바꾸면 답변 품질과 비용 구조가 달라집니다.

## 8. 운영 방법

### 8.1 매일 확인할 것

- FastAPI 서버가 떠 있는지 확인
- Redis가 떠 있는지 확인
- `/ask` 테스트 요청이 성공하는지 확인
- `logs/qa.jsonl`이 계속 쌓이는지 확인
- `source="llm"` 비율이 갑자기 늘지 않았는지 확인
- `output_blocked`가 갑자기 늘지 않았는지 확인

### 8.2 로그 확인

운영 로그:

```text
logs/qa.jsonl
```

보안 감사 로그:

```text
logs/security_audit.log
```

로그에는 질문 원문 대신 hash와 길이 중심으로 남깁니다.

민감정보가 포함된 질문이나 답변은 학습 후보로 바로 쓰지 않습니다.

### 8.3 캐시 운영

Redis cache는 영구 데이터가 아닙니다.

아래 상황에서는 캐시를 비우거나 namespace를 바꾸는 것이 좋습니다.

- embedding 모델 변경
- intent 모델 변경
- RAG 데이터 대량 변경
- LoRA 모델 변경
- 잘못된 답변이 캐시에 들어간 경우

### 8.4 배포 운영

권장 배포 순서:

```text
1. 새 데이터 정제
2. intent / RAG / LoRA artifact 생성
3. 테스트 서버에서 /ask 확인
4. artifact manifest 확인
5. 운영 서버 중지
6. artifact 교체
7. Redis cache 정리 여부 판단
8. 운영 서버 시작
9. Spring Boot에서 테스트 호출
10. 로그 확인
```

운영 중 artifact만 몰래 바꾸지 않습니다.

`artifacts/manifest.hash`와 실제 artifact가 맞지 않으면 서버 시작이 중단될 수 있습니다.

## 9. 추후 관리 방법

### 9.1 데이터 관리

새 질문/답변 데이터는 바로 gold 데이터로 승격하지 않습니다.

권장 흐름:

```text
1. logs에서 학습 후보 수집
2. 개인정보와 prompt injection 의심 데이터 제거
3. 사람이 답변 품질 확인
4. 승인된 데이터만 gold로 승격
5. 정제와 학습 파이프라인 다시 실행
```

자동화해도 마지막 gold 승격은 사람이 확인하는 것이 안전합니다.

### 9.2 모델 관리

모델을 바꾸면 아래 항목을 같이 확인합니다.

- intent model
- RAG embedding model
- FAISS index
- semantic cache
- LoRA adapter
- `artifacts/manifest.hash`

embedding 모델을 바꾸면 기존 FAISS index와 semantic cache는 같은 기준으로 다시 만들어야 합니다.

### 9.3 보안 관리

정기적으로 확인할 것:

- `ALTONG_API_KEY`가 Git에 들어가지 않았는지 확인
- `REDIS_PASSWORD`가 Git에 들어가지 않았는지 확인
- Redis가 외부에 열려 있지 않은지 확인
- FastAPI `8000` 포트가 Spring Boot 서버에서만 접근 가능한지 확인
- `ALTONG_ENV=production`이 운영 서버에 설정되어 있는지 확인

### 9.4 성능 관리

처음에는 아래 숫자를 봅니다.

- 평균 응답 시간
- 가장 느린 응답 시간
- cache hit 비율
- LLM 호출 비율
- GPU 메모리 사용량
- Redis 메모리 사용량

문제가 생기는 방향:

- cache hit이 낮으면 LLM 호출이 늘어납니다.
- LLM 호출이 늘면 GPU 사용량과 응답 시간이 늘어납니다.
- semantic cache가 너무 공격적이면 틀린 답변을 재사용할 수 있습니다.

현재 기준에서는 semantic cache threshold `0.9`를 보수적으로 유지합니다.

## 10. 장애 대응

### 10.1 Redis 연결 실패

증상:

```text
Error connecting to Redis
```

확인:

```powershell
docker ps
$env:REDIS_PASSWORD
```

해결:

- Redis 컨테이너가 떠 있는지 확인
- `REDIS_PASSWORD`가 Redis 실행 비밀번호와 같은지 확인
- Redis 포트가 `127.0.0.1:6379`로 열려 있는지 확인

### 10.2 RAG 파일 없음

증상:

```text
RAG index files are missing. Run rag/embed.py first.
```

해결:

```powershell
python -m rag.embed
```

생성되어야 하는 파일:

```text
artifacts/rag/faiss.index
artifacts/rag/metadata.json
```

### 10.3 manifest 불일치

증상:

```text
artifact manifest verification failed
```

해결:

- artifact 파일을 교체했는지 확인
- 학습 또는 인덱스 생성을 다시 실행
- `artifacts/manifest.hash`가 최신 artifact 기준인지 확인

### 10.4 모델 서버 응답 실패

`ALTONG_GENERATOR_MODE=http`일 때만 해당합니다.

확인:

```powershell
$env:ALTONG_MODEL_SERVER_URL
```

해결:

- 모델 서버가 먼저 떠 있는지 확인
- `/generate` endpoint가 POST JSON을 받는지 확인
- 응답에 `answer`, `text`, `generated_text`, 또는 `choices[0].message.content`가 있는지 확인
- timeout이 너무 짧으면 `ALTONG_MODEL_SERVER_TIMEOUT_SECONDS`를 늘림

### 10.5 답변이 보안 검사에서 차단됨

증상:

```json
{
  "source": "output_blocked",
  "success": false
}
```

확인:

- LLM 답변에 내부 prompt 표식이 섞였는지 확인
- 금칙 문구가 답변에 들어갔는지 확인
- RAG 문서 안에 prompt injection 문장이 들어갔는지 확인

## 11. 운영 체크리스트

처음 설치:

- [ ] Python 가상환경 생성
- [ ] `pip install -r requirements.txt`
- [ ] Redis 비밀번호 설정
- [ ] `ALTONG_API_KEY` 설정
- [ ] `ALTONG_ENV=production` 설정
- [ ] 데이터 정제 실행
- [ ] intent 학습 실행
- [ ] RAG index 생성
- [ ] LoRA 학습 또는 모델 서버 준비
- [ ] `/ask` 테스트 성공

배포 전:

- [ ] artifact manifest 최신 여부 확인
- [ ] Redis cache 유지 또는 삭제 여부 결정
- [ ] Spring Boot가 `X-API-Key`를 보내는지 확인
- [ ] FastAPI 포트 접근 제한 확인
- [ ] 테스트 질문 3개 이상 확인

매주:

- [ ] cache hit 비율 확인
- [ ] 느린 요청 확인
- [ ] `output_blocked` 증가 여부 확인
- [ ] 학습 후보 로그 검토
- [ ] disk 용량 확인

모델 또는 데이터 변경 후:

- [ ] FAISS index 재생성 필요 여부 확인
- [ ] semantic cache 초기화 필요 여부 확인
- [ ] manifest 재생성 확인
- [ ] 운영 반영 전 테스트 서버에서 확인

## 12. 수정된 코드 구조

이번 구조에서는 LLM 답변 생성부가 분리되어 있습니다.

```text
model/prompt.py
  - 공통 prompt 생성 규칙

model/inference.py
  - 로컬 LLaMA/LoRA 직접 실행

model/http_generator.py
  - HTTP 모델 서버 호출

model/generator_factory.py
  - ALTONG_GENERATOR_MODE 값에 따라 local 또는 http 선택

pipeline/pipeline.py
  - cache, RAG, output validation 순서는 유지
  - LLM 생성기만 교체 가능
```

운영자는 보통 아래 둘 중 하나만 고르면 됩니다.

```powershell
$env:ALTONG_GENERATOR_MODE = "local"
```

또는:

```powershell
$env:ALTONG_GENERATOR_MODE = "http"
$env:ALTONG_MODEL_SERVER_URL = "http://127.0.0.1:9000/generate"
```
