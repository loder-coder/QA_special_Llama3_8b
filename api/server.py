from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

from pipeline.pipeline import HybridQAPipeline
from security.api_guard import require_api_key, require_rate_limit
from security.language_policy import normalize_language


app = FastAPI(title="Hybrid Q&A System")
pipeline = HybridQAPipeline()


class AskRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(default="ko", min_length=2, max_length=10)


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask(payload: AskRequest, request: Request) -> dict:
    require_rate_limit(request)
    language = normalize_language(payload.language)
    result = pipeline.ask(payload.q, language=language)
    return {
        "answer": result.answer,
        "success": result.success,
        "source": result.source,
        "intent_similarity": result.intent_similarity,
        "rag_similarity": result.rag_similarity,
        "language": result.language,
    }
