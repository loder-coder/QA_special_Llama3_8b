from __future__ import annotations

import re


FORBIDDEN_OUTPUT_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s*prompt",
    r"developer\s*message",
    r"internal\s+secret",
    r"internal\s+key",
    r"secret\s+key",
    r"ALTONG_API_KEY",
    r"REDIS_PASSWORD",
    r"HF_TOKEN",
    r"HUGGINGFACE_TOKEN",
    r"BEGIN\s+SYSTEM\s+PROMPT",
    r"END\s+SYSTEM\s+PROMPT",
    r"confidential\s+instruction",
    r"내부\s*비밀",
    r"시스템\s*프롬프트",
    r"관리자\s*지시",
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
