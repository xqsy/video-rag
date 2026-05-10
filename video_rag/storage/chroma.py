from __future__ import annotations

from collections.abc import Sequence

from video_rag.config import settings
from video_rag.schemas import Chunk, RetrievalResult


class ChromaVectorStore:
    def __init__(self, collection_name: str | None = None) -> None:
        settings.ensure_directories()
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed. Install project dependencies first.") from exc

        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=[
                {
                    "video_id": chunk.video_id,
                    "start": chunk.start,
                    "end": chunk.end,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: Sequence[float],
        video_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        response = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k or settings.retrieval_top_k,
            where={"video_id": video_id} if video_id else None,
        )
        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[RetrievalResult] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index]
            score = 1.0 - float(distances[index]) if index < len(distances) else 0.0
            results.append(
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        video_id=metadata["video_id"],
                        start=float(metadata["start"]),
                        end=float(metadata["end"]),
                        text=documents[index],
                    ),
                    score=score,
                    rank=index + 1,
                )
            )
        return results

    def list_chunks(self, video_id: str) -> list[Chunk]:
        response = self._collection.get(where={"video_id": video_id}, include=["documents", "metadatas"])
        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])
        chunks: list[Chunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    video_id=metadata["video_id"],
                    start=float(metadata["start"]),
                    end=float(metadata["end"]),
                    text=documents[index],
                )
            )
        chunks.sort(key=lambda item: (item.start, item.end, item.chunk_id))
        return chunks

    def delete_video(self, video_id: str) -> None:
        self._collection.delete(where={"video_id": video_id})
