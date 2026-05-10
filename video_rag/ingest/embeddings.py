from __future__ import annotations

from collections.abc import Sequence

from video_rag.config import settings


class EmbeddingEncoder:
    def __init__(self, model_name: str | None = None, batch_size: int | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None

    @staticmethod
    def format_passage(text: str) -> str:
        return f"passage: {text.strip()}"

    @staticmethod
    def format_query(text: str) -> str:
        return f"query: {text.strip()}"

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        payload = [self.format_passage(text) for text in texts]
        return self._encode(payload)

    def embed_query(self, text: str) -> list[float]:
        payload = self._encode([self.format_query(text)])
        return payload[0]

    def embedding_dimension(self) -> int:
        return len(self.embed_query("dimension probe"))

    def _encode(self, payload: Sequence[str]) -> list[list[float]]:
        if not payload:
            return []
        model = self._get_model()
        embeddings = model.encode(
            list(payload),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install project dependencies first."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model
