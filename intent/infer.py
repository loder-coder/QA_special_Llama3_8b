from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class IntentMatch:
    matched: bool
    answer: str
    similarity: float
    question_id: str | None
    question: str | None


class IntentMatcher:
    def __init__(
        self,
        model_path: str = "artifacts/intent_model",
        fallback_model_name: str = "BAAI/bge-small-en-v1.5",
        data_paths: Iterable[str] = ("data/gold.json", "data/bronze.json"),
        threshold: float = 0.85,
    ) -> None:
        model_source = model_path if Path(model_path).exists() else fallback_model_name
        self.model = SentenceTransformer(model_source)
        self.threshold = threshold
        self.items = self._load_items(data_paths)
        self.embeddings = self._build_embeddings()

    def _load_items(self, data_paths: Iterable[str]) -> list[dict]:
        frames = []
        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                frame = pd.read_json(path)
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return []

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
        return df[["id", "question", "answer"]].to_dict("records")

    def _build_embeddings(self) -> np.ndarray:
        if not self.items:
            return np.empty((0, 0), dtype="float32")
        questions = [str(item["question"]) for item in self.items]
        return self.model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def find(self, query: str) -> IntentMatch:
        if not self.items:
            return IntentMatch(False, "", 0.0, None, None)

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")[0]
        similarities = self.embeddings @ query_embedding
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        item = self.items[best_index]

        return IntentMatch(
            matched=best_similarity >= self.threshold,
            answer=str(item["answer"]),
            similarity=best_similarity,
            question_id=str(item["id"]),
            question=str(item["question"]),
        )
