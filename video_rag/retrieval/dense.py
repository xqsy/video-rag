from __future__ import annotations

from collections.abc import Sequence

from video_rag.config import settings
from video_rag.ingest.embeddings import EmbeddingEncoder
from video_rag.schemas import Chunk, RetrievalResult
from video_rag.storage.chroma import ChromaVectorStore


class DenseRetriever:
    def __init__(
        self,
        embedder: EmbeddingEncoder | None = None,
        store: ChromaVectorStore | None = None,
    ) -> None:
        self.embedder = embedder or EmbeddingEncoder()
        self.store = store or ChromaVectorStore()

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed_passages([chunk.text for chunk in chunks])
        self.store.upsert_chunks(chunks, embeddings)

    def retrieve(
        self,
        question: str,
        video_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed_query(question)
        return self.store.search(
            query_embedding=query_embedding,
            video_id=video_id,
            top_k=top_k or settings.retrieval_top_k,
        )
