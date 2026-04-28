from __future__ import annotations

import re


SENSITIVE_DATA_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    r"\b01[016789]-?\d{3,4}-?\d{4}\b",
    r"\b\d{2,3}-\d{3,4}-\d{4}\b",
    r"\b\d{6}-[1-4]\d{6}\b",
    r"\b(?:\d[ -]*?){13,19}\b",
    r"\b(?:sk|pk|hf)[_-][A-Za-z0-9_\-]{20,}\b",
    r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b",
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|authorization)\b\s*[:=]\s*\S+",
    r"(?i)\bbearer\s+[A-Za-z0-9._\-]+",
    r"(?i)\b(redis|postgres|mysql|mongodb)://[^\s]+",
    r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    r"\b172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}\b",
    r"\b192\.168\.\d{1,3}\.\d{1,3}\b",
]


def contains_sensitive_data(value: str) -> bool:
    text = str(value)
    return any(re.search(pattern, text) for pattern in SENSITIVE_DATA_PATTERNS)


def redact_sensitive_data(value: str) -> str:
    text = str(value)
    for pattern in SENSITIVE_DATA_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text
