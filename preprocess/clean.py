import re
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ["id", "question", "answer", "category", "views", "donation"]
MEANINGLESS_PATTERNS = [
    r"^\s*$",
    r"^(.)\1{4,}$",
    r"^(몰라요|모름|없음|글쎄요|잘 모르겠어요|감사합니다|thanks?|thank you)\.?$",
    r"^(test|테스트|asdf|qwer|ㅋㅋ+|ㅎㅎ+)$",
]


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"raw data is missing required columns: {missing}")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_meaningless_answer(answer: str) -> bool:
    normalized = clean_text(answer).lower()
    if len(normalized) < 20:
        return True
    return any(re.match(pattern, normalized, flags=re.IGNORECASE) for pattern in MEANINGLESS_PATTERNS)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)
    cleaned = df.copy()
    cleaned["question"] = cleaned["question"].map(clean_text)
    cleaned["answer"] = cleaned["answer"].map(clean_text)
    cleaned["category"] = cleaned["category"].map(clean_text)
    cleaned["views"] = pd.to_numeric(cleaned["views"], errors="coerce").fillna(0)
    cleaned["donation"] = pd.to_numeric(cleaned["donation"], errors="coerce").fillna(0)
    cleaned = cleaned[cleaned["question"].str.len() > 0]
    cleaned = cleaned[~cleaned["answer"].map(is_meaningless_answer)]
    cleaned = cleaned.reset_index(drop=True)
    return cleaned
