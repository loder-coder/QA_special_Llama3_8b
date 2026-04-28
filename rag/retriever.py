from __future__ import annotations

from sentence_transformers import SentenceTransformer

from rag.db import FaissDocumentStore, RetrievedDocument


class RagRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        index_path: str = "artifacts/rag/faiss.index",
        metadata_path: str = "artifacts/rag/metadata.json",
        top_k: int = 5,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.store = FaissDocumentStore(index_path=index_path, metadata_path=metadata_path)
        self.store.load()
        self.top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")[0]
        return self.store.search(query_embedding, top_k=self.top_k)
