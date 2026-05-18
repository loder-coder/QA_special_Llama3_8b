from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from preprocess.io import read_records


def build_pairs(
    input_path: str = "data/Gold.jsonl",
    output_path: str = "data/pairs.json",
    model_name: str = "BAAI/bge-m3",
    positive_threshold: Optional[float] = None,
    negative_threshold: float = 0.60,
    batch_size: int = 32,
    positive_min_threshold: float = 0.75,
    positive_max_threshold: float = 0.93,
) -> pd.DataFrame:
    pair_columns = ["q1_id", "q2_id", "q1", "q2", "label", "similarity"]
    if positive_threshold is not None:
        positive_min_threshold = positive_threshold

    df = read_records(input_path)
    if df.empty:
        pairs = pd.DataFrame(columns=pair_columns)
        print(pairs["label"].value_counts(dropna=False))
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
        # TODO: See PIPELINE_TODO.md for post-schema pair label guards.
        if positive_min_threshold <= similarity <= positive_max_threshold:
            label = "positive"
        elif similarity <= negative_threshold:
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

    pairs = pd.DataFrame(rows, columns=pair_columns)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(pairs["label"].value_counts(dropna=False))
    pairs.to_json(output_path, orient="records", force_ascii=False, indent=2)
    return pairs


if __name__ == "__main__":
    build_pairs()
