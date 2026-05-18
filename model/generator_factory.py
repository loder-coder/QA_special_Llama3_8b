from __future__ import annotations

import os
from typing import Protocol

from model.prompt import GenerationConfig


class AnswerGenerator(Protocol):
    def generate(
        self,
        question: str,
        context: str = "",
        category: str = "",
        language: str = "ko",
        config: GenerationConfig | None = None,
    ) -> str:
        ...


def create_answer_generator() -> AnswerGenerator:
    mode = os.getenv("ALTONG_GENERATOR_MODE", "local").strip().lower()
    if mode == "http":
        from model.http_generator import HttpAnswerGenerator

        url = os.getenv("ALTONG_MODEL_SERVER_URL", "").strip()
        if not url:
            raise ValueError("ALTONG_MODEL_SERVER_URL is required when ALTONG_GENERATOR_MODE=http")
        timeout_seconds = float(os.getenv("ALTONG_MODEL_SERVER_TIMEOUT_SECONDS", "60"))
        return HttpAnswerGenerator(url=url, timeout_seconds=timeout_seconds)

    if mode != "local":
        raise ValueError("ALTONG_GENERATOR_MODE must be 'local' or 'http'")

    from model.inference import LlamaAnswerGenerator

    return LlamaAnswerGenerator()
