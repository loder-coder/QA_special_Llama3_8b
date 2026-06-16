# QA_special_Llama3_8b

Llama3 8B 기반 한국어 QA 시스템.

불필요한 LLM 호출을 줄이는 것을 목표로, 멀티레이어 캐시 구조와 RAG 파이프라인을 조합해 설계했습니다. 비용·응답 속도·환각 감소를 함께 고려한 구조로, 현재 재직 중인 회사 서비스에 연동하는 방향으로 개발 중입니다.

---

## 처리 흐름

```
사용자 요청
  → 프롬프트 인젝션 차단
  → Redis 정확 매칭 캐시
  → Redis 시맨틱 캐시 (임베딩 유사도 기반)
  → RAG 문서 검색 (FAISS)
  → LLM 답변 생성 (Llama3 8B / LoRA)
  → 출력 검증
  → 응답 반환 + 로그 기록
```

캐시 히트가 가능한 요청은 최대한 앞단에서 처리해 LLM 호출을 최소화합니다.

---

## 주요 특징

**멀티레이어 캐시**
정확 매칭 → 시맨틱 유사도(임계값 0.9) → RAG → LLM 순서로 처리합니다. Redis TTL 기반 만료 정책과 만료 키 자동 정리 로직을 구현해 데이터 최신성과 응답 속도를 함께 유지합니다.

**RAG 파이프라인**
FAISS 인덱스 기반 문서 검색으로 LLM이 참조할 컨텍스트를 제공합니다. 프롬프트가 검색된 문서 내부 정보만 활용하도록 제한해 환각 발생을 줄였습니다.

**프롬프트 보안**
프롬프트 인젝션 탐지, 컨텍스트 섹션 이스케이프 처리, 출력 검증 레이어를 단계적으로 구성했습니다.

**LLM 분리 구조**
`ALTONG_GENERATOR_MODE` 환경변수로 로컬 모델 실행과 HTTP 모델 서버 분리를 선택할 수 있습니다. 운영 안정화 단계에서 추론 서버를 분리하기 쉽게 설계했습니다.

**운영 로그**
쿼리와 응답을 SHA256으로 해싱해 저장하며, 응답 경로(`source`)를 함께 기록해 캐시 히트율과 LLM 호출 비율을 추적할 수 있습니다.

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| API 서버 | FastAPI, Uvicorn |
| LLM | Llama3 8B, LoRA (PEFT) |
| 임베딩 | sentence-transformers (BAAI/bge) |
| 문서 검색 | FAISS |
| 캐시 | Redis |
| 데이터 처리 | pandas, numpy |

---

## 프로젝트 구조

```
├── api/            # FastAPI 서버 엔트리포인트
├── pipeline/       # 전체 QA 처리 흐름 (캐시 → RAG → LLM)
├── cache/          # Redis 캐시 클라이언트 (TTL, 시맨틱 캐시)
├── intent/         # 임베딩 유사도 기반 의도 분류
├── rag/            # FAISS 문서 검색 및 인덱스 빌드
├── model/          # LLM 추론, 프롬프트 생성, LoRA 학습
├── preprocess/     # 데이터 정제 및 학습 데이터 생성
├── security/       # 프롬프트 인젝션 방어, 출력 검증
├── retrain/        # 운영 로그 기반 재학습 후보 수집
└── logs/           # 운영 로그 (qa.jsonl, security_audit.log)
```

---

## 빠른 시작

```bash
# 가상환경 설치
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Redis 실행
docker run --name qa-redis -p 127.0.0.1:6379:6379 redis redis-server --requirepass your_password

# 환경변수 설정
export LLAMA_API_KEY=your_api_key
export REDIS_PASSWORD=your_password

# 데이터 준비 및 인덱스 생성
python -m preprocess.pipeline
python -m intent.train
python -m rag.embed

# 서버 실행
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

자세한 운영 방법은 [RUNBOOK.md](./RUNBOOK.md)를 참고하세요.

---

## 응답 예시

```json
{
  "answer": "답변 내용",
  "success": true,
  "source": "redis_intent",
  "intent_similarity": 0.93,
  "rag_similarity": 0.0,
  "language": "ko"
}
```

`source` 값으로 요청이 어느 계층에서 처리됐는지 확인할 수 있습니다.

| source | 의미 |
|--------|------|
| `redis` | 동일 질문 캐시 재사용 |
| `redis_intent` | 유사 질문 캐시 재사용 |
| `llm` | RAG 검색 후 LLM 생성 |
| `blocked` | 프롬프트 인젝션 차단 |
| `output_blocked` | 출력 보안 검사 차단 |
