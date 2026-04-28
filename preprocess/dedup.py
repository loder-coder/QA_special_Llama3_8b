from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def _encode_questions(model: SentenceTransformer, questions: List[str], batch_size: int) -> np.ndarray:
    if not questions:
        return np.empty((0, 0), dtype="float32")
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def remove_duplicates_sbert(
    df: pd.DataFrame,
    model_name: str = "BAAI/bge-small-en-v1.5",
    threshold: float = 0.9,
    batch_size: int = 32,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranked = df.sort_values("score", ascending=False).reset_index(drop=True) if "score" in df.columns else df.reset_index(drop=True)
    questions = ranked["question"].astype(str).tolist()
    model = SentenceTransformer(model_name)
    embeddings = _encode_questions(model, questions, batch_size)

    kept_indices: list[int] = []
    kept_embeddings: list[np.ndarray] = []

    for index, embedding in enumerate(embeddings):
        if not kept_embeddings:
            kept_indices.append(index)
            kept_embeddings.append(embedding)
            continue

        kept_matrix = np.vstack(kept_embeddings)
        max_similarity = float(np.max(kept_matrix @ embedding))
        if max_similarity <= threshold:
            kept_indices.append(index)
            kept_embeddings.append(embedding)

    return ranked.iloc[kept_indices].reset_index(drop=True)
