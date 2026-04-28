from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


def build_triplets(
    pairs_path: str = "data/pairs.json",
    output_path: str = "data/triplets.json",
) -> pd.DataFrame:
    pairs = pd.read_json(pairs_path)
    if pairs.empty:
        triplets = pd.DataFrame(columns=["anchor", "positive", "negative"])
        triplets.to_json(output_path, orient="records", force_ascii=False, indent=2)
        return triplets

    positives: dict[str, list[str]] = defaultdict(list)
    negatives: dict[str, list[str]] = defaultdict(list)

    for row in pairs.to_dict("records"):
        left = str(row["q1"])
        right = str(row["q2"])
        if row["label"] == "positive":
            positives[left].append(right)
            positives[right].append(left)
        elif row["label"] == "negative":
            negatives[left].append(right)
            negatives[right].append(left)

    rows = []
    for anchor, positive_values in positives.items():
        negative_values = negatives.get(anchor, [])
        if not negative_values:
            continue
        for positive in positive_values:
            for negative in negative_values:
                rows.append({"anchor": anchor, "positive": positive, "negative": negative})

    triplets = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    triplets.to_json(output_path, orient="records", force_ascii=False, indent=2)
    return triplets


if __name__ == "__main__":
    build_triplets()
