from __future__ import annotations

from collections.abc import Sequence

from video_rag.config import settings
from video_rag.generation.citations import attach_fallback_citations
from video_rag.generation.llm import OpenRouterLLM
from video_rag.generation.prompts import (
    build_retrieval_query,
    build_answer_messages,
    build_reduce_summary_messages,
    build_summary_messages,
    limit_history,
    normalize_answer_text,
    normalize_summary_structure,
    render_chunk_context,
    render_retrieval_context,
)
from video_rag.retrieval.dense import DenseRetriever
from video_rag.schemas import Answer, Chunk


class VideoRAGPipeline:
    def __init__(self, retriever: DenseRetriever | None = None, llm: OpenRouterLLM | None = None) -> None:
        self._attach_fallback_citations = attach_fallback_citations
        self._build_answer_messages = build_answer_messages
        self._build_reduce_summary_messages = build_reduce_summary_messages
        self._build_summary_messages = build_summary_messages
        self._render_chunk_context = render_chunk_context
        self._render_retrieval_context = render_retrieval_context
        self.retriever = retriever or DenseRetriever()
        self.llm = llm or OpenRouterLLM()

    def answer(
        self,
        question: str,
        video_id: str,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Answer:
        limited_history = limit_history(history, settings.history_max_turns)
        retrieval_query = build_retrieval_query(
            question=question,
            history=limited_history,
            max_turns=settings.retrieval_history_turns,
        )
        results = self.retriever.retrieve(question=retrieval_query, video_id=video_id)
        if not results:
            return Answer(
                video_id=video_id,
                question=question,
                answer_text="По этому видео не найдено релевантных фрагментов для ответа.",
                citations=[],
            )

        context = self._render_retrieval_context(results)
        messages = self._build_answer_messages(question=question, context=context, history=limited_history)
        raw_answer = self.llm.generate(
            messages=messages,
            temperature=0.2,
            max_tokens=settings.answer_max_tokens,
        )
        normalized_answer = normalize_answer_text(raw_answer, question)
        answer_text, citations = self._attach_fallback_citations(normalized_answer, results)
        return Answer(
            video_id=video_id,
            question=question,
            answer_text=answer_text,
            citations=citations,
        )

    def summarize(self, video_id: str) -> str:
        chunks = self._collect_chunks_for_summary(video_id=video_id)
        if not chunks:
            return normalize_summary_structure("")

        groups = self._group_chunks(chunks, group_size=settings.summary_group_size)
        partial_summaries = [self._summarize_chunk_group(group) for group in groups]
        if len(partial_summaries) == 1:
            return normalize_summary_structure(partial_summaries[0])
        reduce_messages = self._build_reduce_summary_messages(partial_summaries)
        reduced = self.llm.generate(
            messages=reduce_messages,
            temperature=0.2,
            max_tokens=settings.summary_reduce_max_tokens,
        )
        return normalize_summary_structure(reduced)

    def _collect_chunks_for_summary(self, video_id: str) -> list[Chunk]:
        store = getattr(self.retriever, "store", None)
        if store is None or not hasattr(store, "list_chunks"):
            raise RuntimeError("Retriever store does not support listing chunks for summary generation.")
        return list(store.list_chunks(video_id))

    def _summarize_chunk_group(self, chunks: Sequence[Chunk]) -> str:
        context = self._render_chunk_context(chunks)
        messages = self._build_summary_messages(context=context)
        return normalize_summary_structure(
            self.llm.generate(
                messages=messages,
                temperature=0.2,
                max_tokens=settings.summary_chunk_max_tokens,
            )
        )

    def _group_chunks(self, chunks: Sequence[Chunk], group_size: int) -> list[list[Chunk]]:
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        return [list(chunks[index : index + group_size]) for index in range(0, len(chunks), group_size)]
