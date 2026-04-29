from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cache.redis_client import RedisCache
from intent.infer import IntentMatcher
from model.inference import LlamaAnswerGenerator
from rag.db import RetrievedDocument
from rag.retriever import RagRetriever
from security.output_validator import is_forbidden_output
from security.language_policy import normalize_language
from security.prompt_guard import is_prompt_injection
from security.sensitive_filter import contains_sensitive_data, redact_sensitive_data


@dataclass
class PipelineResult:
    answer: str
    source: str
    success: bool
    intent_similarity: float
    rag_similarity: float
    language: str


class HybridQAPipeline:
    def __init__(
        self,
        cache: RedisCache | None = None,
        intent_matcher: IntentMatcher | None = None,
        retriever: RagRetriever | None = None,
        generator: LlamaAnswerGenerator | None = None,
        log_path: str = "logs/qa.jsonl",
        rag_threshold: float = 0.7,
        cache_similarity_threshold: float = 0.9,
    ) -> None:
        self.cache = cache or RedisCache()
        self.intent_matcher = intent_matcher or IntentMatcher()
        self.retriever = retriever or RagRetriever(top_k=5)
        self.generator = generator or LlamaAnswerGenerator()
        self.log_path = Path(log_path)
        self.rag_threshold = rag_threshold
        self.cache_similarity_threshold = cache_similarity_threshold

    @staticmethod
    def _context_from_documents(documents: list[RetrievedDocument], threshold: float) -> tuple[str, str, float]:
        if not documents:
            return "", "", 0.0
        best_similarity = documents[0].similarity
        if best_similarity < threshold:
            return "", "", best_similarity
        context = "\n\n".join(document.text for document in documents)
        category = documents[0].category
        return context, category, best_similarity

    def _write_log(self, query: str, result: PipelineResult) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        sensitive_data_detected = contains_sensitive_data(query) or contains_sensitive_data(result.answer)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": redact_sensitive_data(query),
            "answer": redact_sensitive_data(result.answer),
            "success": result.success,
            "source": result.source,
            "intent_similarity": result.intent_similarity,
            "rag_similarity": result.rag_similarity,
            "language": result.language,
            "sensitive_data_detected": sensitive_data_detected,
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _find_semantic_cache(self, query: str, language: str) -> tuple[str | None, float]:
        cached_items = self.cache.get_cached_items(language=language)
        if not cached_items:
            return None, 0.0

        cached_queries = [item.query for item in cached_items]
        embeddings = self.intent_matcher.model.encode(
            cached_queries + [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        query_embedding = embeddings[-1]
        cache_embeddings = embeddings[:-1]
        similarities = cache_embeddings @ query_embedding
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])

        if best_similarity >= self.cache_similarity_threshold:
            return cached_items[best_index].answer, best_similarity
        return None, best_similarity

    def ask(self, query: str, language: str = "ko") -> PipelineResult:
        normalized_language = normalize_language(language)
        normalized_query = query.strip()
        if not normalized_query:
            result = PipelineResult("", "validation", False, 0.0, 0.0, normalized_language)
            self._write_log(query, result)
            return result

        if is_prompt_injection(normalized_query):
            result = PipelineResult("요청에 안전하지 않은 프롬프트 지시가 포함되어 답변할 수 없습니다.", "blocked", False, 0.0, 0.0, normalized_language)
            self._write_log(normalized_query, result)
            return result

        cached = self.cache.get(normalized_query, language=normalized_language)
        if cached:
            result = PipelineResult(cached, "redis", True, 0.0, 0.0, normalized_language)
            self._write_log(normalized_query, result)
            return result

        semantic_cached, semantic_similarity = self._find_semantic_cache(normalized_query, language=normalized_language)
        if semantic_cached:
            self.cache.set(normalized_query, semantic_cached, language=normalized_language)
            result = PipelineResult(semantic_cached, "redis_intent", True, semantic_similarity, 0.0, normalized_language)
            self._write_log(normalized_query, result)
            return result

        documents = self.retriever.retrieve(normalized_query)
        context, category, rag_similarity = self._context_from_documents(documents, self.rag_threshold)
        answer = self.generator.generate(normalized_query, context=context, category=category, language=normalized_language)
        success = bool(answer)
        if success and is_forbidden_output(answer):
            result = PipelineResult("답변 안전성 검증을 통과하지 못해 응답할 수 없습니다.", "output_blocked", False, semantic_similarity, rag_similarity, normalized_language)
            self._write_log(normalized_query, result)
            return result

        if success:
            self.cache.set(normalized_query, answer, language=normalized_language)

        result = PipelineResult(answer, "llm", success, semantic_similarity, rag_similarity, normalized_language)
        self._write_log(normalized_query, result)
        return result
