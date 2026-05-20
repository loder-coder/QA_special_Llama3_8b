# Improvement Notes

이 문서는 현재 `QA_special_Llama3_8b`를 로컬 싱글 노드 AI 서버로 운영한다는 전제에서, 운영 전에 검토하거나 단계적으로 추가할 개선 사항을 정리합니다.

## 1. Target Service Flow

권장 서비스 흐름:

```text
User / Frontend
  -> Spring Boot Backend
  -> Internal IP JSON API
  -> QA_special_Llama3_8b Single Node
     - FastAPI
     - Redis localhost
     - FAISS index
     - LLM / LoRA artifacts
     - logs / retrain workflow
  -> Optional outbound Web Search API
```

핵심 원칙:

- AI 서버는 외부 인터넷에 직접 공개하지 않습니다.
- Spring Boot 백엔드만 AI 서버의 `/ask` API를 호출합니다.
- AI 서버는 입력과 출력을 JSON으로만 주고받습니다.
- Redis는 AI 서버 내부에 두고, model artifact, FAISS index, logs, hash manifest 보관 기준은 `PORTABLE_INDEX.md`를 따릅니다.
- 인터넷 검색은 inbound 공개가 아니라 outbound 검색 API 호출로 확장합니다.

## 2. Internal IP Connection

초기 운영은 외부 DNS보다 사내 내부 IP 연결을 우선합니다.

예시:

```text
http://10.x.x.x:8000/ask
```

Spring Boot 설정 예시:

```properties
ai.api.base-url=http://10.x.x.x:8000
ai.api.key=${ALTONG_AI_API_KEY}
```

권장 보안 설정:

- AI 서버의 `8000` 포트는 Spring Boot 서버 IP에서만 접근 허용
- 사내 전체망에서 접근 가능한 상태로 두지 않기
- AI 서버의 Redis 포트 `6379`는 외부 접근 금지
- FastAPI는 `X-API-Key` 검사 유지

내부 DNS를 쓰는 경우:

```text
http://qa-llama3.internal:8000/ask
```

내부 DNS는 IP 변경 대응에는 좋지만, 외부 공개 DNS와 혼동하지 않아야 합니다.

## 3. JSON Input / Output Contract

요청 body:

```json
{
  "q": "질문내용",
  "language": "ko"
}
```

현재 지원 언어:

```text
ko, en, ja, zh, vi
```

기본 응답:

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

향후 웹 검색 확장 시 권장 응답:

```json
{
  "answer": "답변내용",
  "success": true,
  "source": "web",
  "intent_similarity": 0.0,
  "rag_similarity": 0.42,
  "language": "ko",
  "sources": [
    {
      "title": "문서 제목",
      "url": "https://example.com/page"
    }
  ]
}
```

주의:

- `/ask` 응답에는 사용자 질문 원문을 반환하지 않습니다.
- 로그에는 query와 answer가 남지만 민감정보는 `[REDACTED]` 처리합니다.
- 언어별 캐시가 분리되어야 같은 질문의 한국어/영어 답변이 섞이지 않습니다.

## 4. Web Search Expansion

모델이 모르는 정보는 인터넷 검색 API를 통해 보완할 수 있습니다.

권장 순서:

```text
1. Prompt injection block
2. Redis exact cache
3. Redis semantic cache
4. Internal RAG
5. If RAG confidence is low, call Web Search API
6. LLM answer with web snippets as untrusted context
7. Output validation
8. Cache / log
```

권장 방식:

- LLM이 직접 인터넷을 탐색하게 하지 않습니다.
- 서버 코드가 통제된 검색 API를 호출합니다.
- 검색 결과의 `title`, `snippet`, `url`만 LLM context에 넣습니다.
- 웹 검색 결과도 prompt injection 위험이 있으므로 신뢰하지 않는 데이터로 취급합니다.
- 웹 기반 답변은 `source="web"`과 `sources`를 같이 반환합니다.

검색 API 후보:

```text
Google Custom Search API
Bing Web Search API
SerpAPI
Tavily
```

추가 고려 사항:

- 검색 API key는 `.env` 또는 서버 환경변수로만 관리하고, 포터블/secret 보관 기준은 `PORTABLE_INDEX.md`를 따릅니다.
- Git에 검색 API key 저장 금지
- Redis TTL 캐시로 같은 웹 검색 반복 호출 감소
- 허용 도메인 또는 차단 도메인 정책 검토

## 5. FAISS Artifact Hash Verification

현재 RAG는 FAISS index와 metadata artifact를 신뢰합니다. 실제 artifact 경로, hash manifest 보관 위치, 공개 금지 기준은 `PORTABLE_INDEX.md`에서 관리합니다.

해당 파일이 변조되면 잘못된 문서가 LLM context로 들어갈 수 있습니다.

권장 검증 방식:

```text
1. rag.embed 실행 후 faiss.index hash 생성
2. metadata.json hash 생성
3. hash manifest 파일 저장
4. API 시작 시 현재 artifact hash와 manifest 비교
5. 불일치하면 서버 시작 중단
```

manifest에는 실제 hash 값이 들어가므로 Git, 공개 API 응답, 외부 문서에 포함하지 않습니다.

적용 우선순위:

- 로컬 단독 실험: 낮음
- 사내 운영: 중간
- 여러 사람이 artifact를 교체하거나 배포 자동화가 들어가는 경우: 높음

## 6. Rate Limit Improvement

현재 rate limit은 Python 프로세스 메모리 기반입니다.

장점:

- 단일 프로세스 로컬 운영에서는 단순하고 충분함

한계:

- `uvicorn --workers 4`처럼 worker가 여러 개면 요청 카운터가 공유되지 않음
- 서버 재시작 시 rate limit 기록이 초기화됨
- 여러 서버로 확장하면 제한이 정확하지 않음

개선 방향:

```text
Current: Python memory
Future: Redis based rate limit
```

현재 요구사항이 싱글 노드 + worker 1개라면 메모리 기반을 유지해도 됩니다.

외부 공개 또는 다중 worker가 필요해지면 Redis 기반으로 변경합니다.

## 7. Redis Scope

현재 Redis는 싱글 노드 내부에서만 사용하는 것이 안전합니다.

권장:

```text
REDIS_URL=redis://localhost:6379/0
```

주의:

- Redis를 `0.0.0.0`으로 열지 않기
- 외부 서버에서 Redis에 직접 접근하지 않기
- Spring Boot는 Redis가 아니라 FastAPI `/ask`만 호출
- Redis 데이터는 캐시이므로 영구 source of truth로 취급하지 않기

## 8. Logging And Retraining Safety

로그 기반 재학습은 유용하지만 data poisoning 위험이 있습니다.

현재 기준:

- 자동: 로그 수집
- 자동: prompt injection 의심 query 제외
- 자동: 민감정보 포함 query/answer 제외
- 자동: 후보 정렬
- 수동: 사람이 최종 승인
- 수동 승인 후에만 gold dataset 반영

유지해야 할 원칙:

- 실패 로그를 바로 gold로 넣지 않기
- `approved=true`라도 민감정보가 있으면 제외
- 웹 검색 결과를 학습 데이터로 넣을 때는 출처와 검증 여부를 따로 관리

## 9. Deployment Checklist

초기 내부망 운영 전 체크:

- [ ] AI 서버 고정 내부 IP 설정
- [ ] Spring Boot 서버 IP만 AI 서버 `8000` 포트 접근 허용
- [ ] `QA_API_KEY` 설정
- [ ] Spring Boot에서 `X-API-Key` 헤더 전송
- [ ] Redis는 `localhost:6379`로만 접근
- [ ] 포터블 보관 대상과 Git 제외 대상은 `PORTABLE_INDEX.md` 기준으로 확인
- [ ] `/ask` JSON 요청/응답 계약 확정
- [ ] 지원 언어 목록 확정
- [ ] 로그 민감정보 마스킹 확인

웹 검색 확장 전 체크:

- [ ] 검색 API provider 선택
- [ ] 검색 API key 환경변수 관리
- [ ] 웹 검색 fallback 조건 정의
- [ ] `source="web"` 응답 구조 추가
- [ ] `sources` 응답 필드 추가
- [ ] 웹 검색 결과 prompt injection 방어 문구 추가
- [ ] 웹 검색 결과 TTL 캐시 정책 정의

운영 안정화 후 체크:

- [ ] FAISS artifact hash 검증 추가
- [ ] Redis 기반 rate limit 검토
- [ ] 평균 latency, cache hit ratio, LLM call ratio 로그 집계
- [ ] 재학습 후보 승인 프로세스 문서화

## 10. RAG 대안 구조 검토 — 터미널 기반 문서 읽기

현재 RAG 파이프라인은 싱글 노드에서 LLM과 임베딩 모델이 메모리를 나눠 쓰는 구조입니다.

문제:

- LLM(Llama3 8B)과 임베딩 모델(FAISS + SentenceTransformer)이 동시에 메모리에 올라가면 GPU/RAM 경합이 발생합니다.
- 싱글 노드 환경에서는 임베딩 모델 자체가 병목이 될 수 있습니다.

검토한 대안:

LLM이 직접 파일 시스템을 읽는 방식 (터미널/bash 기반 문서 접근) 을 사고실험 수준에서 검토했습니다. 임베딩 모델을 아예 제거하고, LLM이 필요한 문서를 직접 읽어 컨텍스트를 구성하는 구조입니다. SWE-agent 계열 논문에서 모델이 리눅스 터미널을 직접 조작하며 정보를 수집하는 방식이 이 방향에 해당합니다.

현재 구조를 유지한 이유:

- 컨텍스트 길이 제한: Llama3 8B의 컨텍스트 윈도우 안에 문서 전체를 넣기 어렵습니다.
- 응답 지연: 파일 읽기 → 파싱 → 컨텍스트 조합 과정이 RAG 검색보다 느릴 수 있습니다.
- 운영 리스크: LLM이 파일 시스템에 직접 접근하면 보안 경계가 복잡해집니다.

현재 선택:

임베딩 모델 호출 자체를 줄이기 위해 Redis 시맨틱 캐시 레이어를 앞단에 두는 방식으로 메모리 경합 문제를 완화하고 있습니다. 하드웨어 자원이 확보되거나 멀티 노드 구조로 전환되면 임베딩 서버 분리를 먼저 검토합니다.

향후 검토 방향:

- 임베딩 서버를 별도 프로세스로 분리해 LLM과 메모리 경합을 줄이기
- 컨텍스트 윈도우가 큰 모델로 교체 시 터미널 기반 문서 접근 방식 재검토
- Tool use / function calling 기반으로 LLM이 필요한 문서만 선택적으로 읽는 구조 실험
