from __future__ import annotations


SUPPORTED_LANGUAGES = {
    "ko": "Korean",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "vi": "Vietnamese",
}


def normalize_language(language: str | None) -> str:
    value = (language or "ko").strip().lower()
    return value if value in SUPPORTED_LANGUAGES else "ko"


def language_instruction(language: str) -> str:
    normalized = normalize_language(language)
    return f"Answer in {SUPPORTED_LANGUAGES[normalized]}."
