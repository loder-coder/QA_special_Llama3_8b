from __future__ import annotations

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import CosineSimilarityLoss, TripletLoss
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader


def load_triplet_examples(triplets_path: str) -> list[InputExample]:
    triplets = pd.read_json(triplets_path)
    return [
        InputExample(texts=[row["anchor"], row["positive"], row["negative"]])
        for row in triplets.to_dict("records")
    ]


def load_pair_examples(pairs_path: str) -> list[InputExample]:
    path = Path(pairs_path)
    if not path.exists():
        return []

    pairs = pd.read_json(pairs_path)
    if pairs.empty:
        return []

    label_values = {"positive": 1.0, "negative": 0.0}
    return [
        InputExample(texts=[row["q1"], row["q2"]], label=label_values[row["label"]])
        for row in pairs.to_dict("records")
        if row["label"] in label_values
    ]


def train_intent_model(
    triplets_path: str = "data/triplets.json",
    pairs_path: str = "data/pairs.json",
    output_dir: str = "artifacts/intent_model",
    model_name: str = "BAAI/bge-small-en-v1.5",
    epochs: int = 1,
    batch_size: int = 16,
    warmup_steps: int = 100,
) -> SentenceTransformer:
    examples = load_triplet_examples(triplets_path)
    model = SentenceTransformer(model_name)

    if examples:
        loss = TripletLoss(model=model)
        print(f"Training intent model with TripletLoss: triplets={len(examples)}")
    else:
        examples = load_pair_examples(pairs_path)
        if not examples:
            print("triplet and pair datasets are empty. Saving base intent model without fine-tuning.")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            model.save(output_dir)
            return model
        loss = CosineSimilarityLoss(model=model)
        print(f"Triplet dataset is empty. Falling back to CosineSimilarityLoss: pairs={len(examples)}")

    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

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
