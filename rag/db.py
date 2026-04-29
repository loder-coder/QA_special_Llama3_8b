from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class RetrievedDocument:
    id: str
    question: str
    answer: str
    category: str
    text: str
    similarity: float


class FaissDocumentStore:
    def __init__(self, index_path: str = "artifacts/rag/faiss.index", metadata_path: str = "artifacts/rag/metadata.json") -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D numpy array")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))
        self.index = index
        self.metadata = metadata

    def save(self) -> None:
        if self.index is None:
            raise ValueError("index is not built")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("RAG index files are missing. Run rag/embed.py first.")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedDocument]:
        if self.index is None:
            self.load()
        if self.index is None or not self.metadata:
            return []

        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype("float32"), top_k)
        results: list[RetrievedDocument] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            item = self.metadata[int(index)]
            results.append(
                RetrievedDocument(
                    id=str(item["id"]),
                    question=str(item["question"]),
                    answer=str(item["answer"]),
                    category=str(item.get("category", "")),
                    text=str(item["text"]),
                    similarity=float(score),
                )
            )
        return results
