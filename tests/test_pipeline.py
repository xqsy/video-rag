from video_rag.pipeline import VideoRAGPipeline
from video_rag.schemas import Chunk, RetrievalResult


class FakeStore:
    def list_chunks(self, video_id: str):
        return [
            Chunk(chunk_id="c1", video_id=video_id, start=0.0, end=10.0, text="Intro"),
            Chunk(chunk_id="c2", video_id=video_id, start=10.0, end=20.0, text="Budget"),
            Chunk(chunk_id="c3", video_id=video_id, start=20.0, end=30.0, text="Decision"),
        ]


class FakeRetriever:
    def __init__(self) -> None:
        self.store = FakeStore()
        self.last_question = None

    def retrieve(self, question: str, video_id: str | None = None, top_k: int | None = None):
        self.last_question = question
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="c2",
                    video_id=video_id or "video-1",
                    start=10.0,
                    end=20.0,
                    text="Budget discussion fragment",
                ),
                score=0.95,
                rank=1,
            )
        ]


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, messages, temperature: float = 0.2, max_tokens: int = 700):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        prompt = messages[-1]["content"]
        if "Сформируй структурированный конспект" in prompt:
            return "Тезисы\n- Итог\nТаймлайн\n- [00:10-00:20] Эпизод"
        if "Собери финальный конспект" in prompt:
            return "Тезисы\n- Финальный итог\nТаймлайн\n- [00:10-00:20] Эпизод"
        return "Бюджет обсуждался в середине записи [00:12]."


def test_pipeline_answer_returns_verified_citations() -> None:
    pipeline = VideoRAGPipeline(retriever=FakeRetriever(), llm=FakeLLM())
    answer = pipeline.answer("Когда обсуждали бюджет?", "video-1")

    assert answer.video_id == "video-1"
    assert answer.citations == ["00:12"]
    assert "[00:12]" in answer.answer_text


def test_pipeline_answer_uses_history_in_retrieval_query() -> None:
    retriever = FakeRetriever()
    pipeline = VideoRAGPipeline(retriever=retriever, llm=FakeLLM())
    pipeline.answer(
        "А что решили потом?",
        "video-1",
        history=[("Когда обсуждали бюджет?", "Бюджет обсуждался [00:12].")],
    )

    assert retriever.last_question is not None
    assert "Текущий вопрос: А что решили потом?" in retriever.last_question
    assert "Пользователь: Когда обсуждали бюджет?" in retriever.last_question


def test_pipeline_summarize_uses_full_video_chunks() -> None:
    pipeline = VideoRAGPipeline(retriever=FakeRetriever(), llm=FakeLLM())
    summary = pipeline.summarize("video-1")

    assert "## Тезисы" in summary
    assert "## Решения" in summary
    assert "## Таймлайн" in summary
    assert "## Тезисы\n- Итог" in summary
    assert "[00:10-00:20]" in summary
