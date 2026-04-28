from __future__ import annotations

import re


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"system\s*prompt",
    r"developer\s*message",
    r"reveal\s+(the\s+)?prompt",
    r"act\s+as\s+(a\s+)?system",
    r"이전\s*(지시|명령|규칙)\s*(무시|잊어|따르지)",
    r"(시스템|개발자)\s*(프롬프트|메시지|지시)",
    r"프롬프트(를|을)?\s*(공개|출력|보여)",
    r"역할을\s*(바꿔|변경)",
]


def is_prompt_injection(query: str) -> bool:
    normalized = query.strip().lower()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS)
