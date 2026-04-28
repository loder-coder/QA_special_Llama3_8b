from __future__ import annotations

import re


FORBIDDEN_OUTPUT_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s*prompt",
    r"developer\s*message",
    r"<<<CONTEXT",
    r"CONTEXT>>>",
    r"<<<QUESTION",
    r"QUESTION>>>",
    r"\[참고 문서\]",
    r"\[질문\]",
    r"\[답변\]",
    r"보안 규칙",
    r"(시스템|개발자)\s*(프롬프트|메시지|지시)",
    r"프롬프트(를|을)?\s*(공개|출력)",
]


def is_forbidden_output(answer: str) -> bool:
    normalized = answer.strip()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in FORBIDDEN_OUTPUT_PATTERNS)
