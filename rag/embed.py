from __future__ import annotations

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from rag.db import FaissDocumentStore


def _load_documents(data_paths: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for data_path in data_paths:
        path = Path(data_path)
        if path.exists():
            frame = pd.read_json(path)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["id", "question", "answer", "category"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)


def build_rag_index(
    data_paths: tuple[str, ...] = ("data/gold.json", "data/bronze.json"),
    model_name: str = "BAAI/bge-base-en-v1.5",
    index_path: str = "artifacts/rag/faiss.index",
    metadata_path: str = "artifacts/rag/metadata.json",
    batch_size: int = 32,
) -> FaissDocumentStore:
    df = _load_documents(data_paths)
    if df.empty:
        raise ValueError("RAG dataset is empty. Run preprocess/pipeline.py first.")

    texts = (df["question"].astype(str) + "\n" + df["answer"].astype(str)).tolist()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    metadata = []
    for row, text in zip(df.to_dict("records"), texts):
        metadata.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "category": row.get("category", ""),
                "text": text,
            }
        )

    store = FaissDocumentStore(index_path=index_path, metadata_path=metadata_path)
    store.build(embeddings, metadata)
    store.save()
    return store


if __name__ == "__main__":
    build_rag_index()
