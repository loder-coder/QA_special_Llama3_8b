from __future__ import annotations

import json
from urllib import request

from model.prompt import GenerationConfig, build_prompt


class HttpAnswerGenerator:
    def __init__(self, url: str, timeout_seconds: float = 60.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _extract_answer(payload: dict) -> str:
        for key in ("answer", "text", "generated_text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"].strip()
                if isinstance(first_choice.get("text"), str):
                    return first_choice["text"].strip()

        return ""

    def generate(
        self,
        question: str,
        context: str = "",
        category: str = "",
        language: str = "ko",
        config: GenerationConfig | None = None,
    ) -> str:
        generation_config = config or GenerationConfig()
        body = {
            "prompt": build_prompt(question=question, context=context, category=category, language=language),
            "question": question,
            "context": context,
            "category": category,
            "language": language,
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
        }
        encoded_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            self.url,
            data=encoded_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
        payload = json.loads(response_body)
        if not isinstance(payload, dict):
            return ""
        return self._extract_answer(payload)
