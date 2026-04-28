from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def build_pairs(
    input_path: str = "data/gold.json",
    output_path: str = "data/pairs.json",
    model_name: str = "BAAI/bge-small-en-v1.5",
    positive_threshold: float = 0.8,
    negative_threshold: float = 0.3,
    batch_size: int = 32,
) -> pd.DataFrame:
    df = pd.read_json(input_path)
    if df.empty:
        pairs = pd.DataFrame(columns=["q1_id", "q2_id", "q1", "q2", "label", "similarity"])
        pairs.to_json(output_path, orient="records", force_ascii=False, indent=2)
        return pairs

    model = SentenceTransformer(model_name)
    questions = df["question"].astype(str).tolist()
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    rows = []
    for left, right in combinations(range(len(df)), 2):
        similarity = float(np.dot(embeddings[left], embeddings[right]))
        if similarity > positive_threshold:
            label = "positive"
        elif similarity < negative_threshold:
            label = "negative"
        else:
            continue
        rows.append(
            {
                "q1_id": df.iloc[left]["id"],
                "q2_id": df.iloc[right]["id"],
                "q1": df.iloc[left]["question"],
                "q2": df.iloc[right]["question"],
                "label": label,
                "similarity": similarity,
            }
        )

    pairs = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_json(output_path, orient="records", force_ascii=False, indent=2)
    return pairs


if __name__ == "__main__":
    build_pairs()
