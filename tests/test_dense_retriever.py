from video_rag.retrieval.dense import DenseRetriever
from video_rag.schemas import Chunk, RetrievalResult


class FakeEmbedder:
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), float(index) + 0.5] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 2.0]


class FakeStore:
    def __init__(self) -> None:
        self.chunks = None
        self.embeddings = None

    def upsert_chunks(self, chunks, embeddings) -> None:
        self.chunks = list(chunks)
        self.embeddings = list(embeddings)

    def search(self, query_embedding, video_id=None, top_k=None):
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="c1",
                    video_id=video_id or "video-1",
                    start=10.0,
                    end=20.0,
                    text="Budget was discussed here.",
                ),
                score=0.9,
                rank=1,
            )
        ]


def test_dense_retriever_indexes_and_queries() -> None:
    store = FakeStore()
    retriever = DenseRetriever(embedder=FakeEmbedder(), store=store)
    chunks = [
        Chunk(chunk_id="c1", video_id="video-1", start=0.0, end=5.0, text="Hello"),
        Chunk(chunk_id="c2", video_id="video-1", start=5.0, end=10.0, text="World"),
    ]

    retriever.index_chunks(chunks)
    results = retriever.retrieve("When was the budget mentioned?", video_id="video-1", top_k=3)

    assert store.chunks == chunks
    assert store.embeddings == [[0.0, 0.5], [1.0, 1.5]]
    assert len(results) == 1
    assert results[0].chunk.video_id == "video-1"
