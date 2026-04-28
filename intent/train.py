from __future__ import annotations

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader


def load_triplet_examples(triplets_path: str) -> list[InputExample]:
    triplets = pd.read_json(triplets_path)
    return [
        InputExample(texts=[row["anchor"], row["positive"], row["negative"]])
        for row in triplets.to_dict("records")
    ]


def train_intent_model(
    triplets_path: str = "data/triplets.json",
    output_dir: str = "artifacts/intent_model",
    model_name: str = "BAAI/bge-small-en-v1.5",
    epochs: int = 1,
    batch_size: int = 16,
    warmup_steps: int = 100,
) -> SentenceTransformer:
    examples = load_triplet_examples(triplets_path)
    if not examples:
        raise ValueError("triplet dataset is empty. Run preprocess/build_pairs.py and preprocess/build_triplet.py first.")

    model = SentenceTransformer(model_name)
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = TripletLoss(model=model)

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    return model


if __name__ == "__main__":
    train_intent_model()
