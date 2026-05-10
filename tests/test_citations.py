from video_rag.generation.citations import attach_fallback_citations, extract_citations, verify_citations
from video_rag.schemas import Chunk, RetrievalResult


def _result(start: float, end: float, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=f"c-{rank}",
            video_id="video-1",
            start=start,
            end=end,
            text="Budget discussion",
        ),
        score=0.9,
        rank=rank,
    )


def test_extract_citations_deduplicates_timecodes() -> None:
    text = "Бюджет обсуждался [00:10] и снова [00:10], потом [01:05]."
    assert extract_citations(text) == ["00:10", "01:05"]


def test_verify_citations_filters_invalid_ones() -> None:
    results = [_result(8.0, 20.0), _result(60.0, 80.0, rank=2)]
    text = "Ответ с [00:10], [01:05] и [03:33]."
    assert verify_citations(text, results) == ["00:10", "01:05"]


def test_attach_fallback_citations_appends_when_missing() -> None:
    results = [_result(8.0, 20.0), _result(60.0, 80.0, rank=2)]
    answer_text, citations = attach_fallback_citations("Краткий ответ.", results)
    assert citations == ["00:08", "01:00"]
    assert "[00:08]" in answer_text


def test_attach_fallback_citations_skips_no_context_answers() -> None:
    results = [_result(8.0, 20.0), _result(60.0, 80.0, rank=2)]
    answer_text, citations = attach_fallback_citations(
        "В предоставленных фрагментах нет информации о рецепте макарон, поэтому я не могу дать ответ на основе данного контекста.",
        results,
    )
    assert citations == []
    assert "Таймкоды:" not in answer_text


def test_attach_fallback_citations_skips_lower_score_overlapping_result() -> None:
    results = [
        RetrievalResult(
            chunk=Chunk(chunk_id="c-1", video_id="video-1", start=295.0, end=314.0, text="Uncle Frank mention"),
            score=0.91,
            rank=1,
        ),
        RetrievalResult(
            chunk=Chunk(chunk_id="c-2", video_id="video-1", start=256.0, end=308.0, text="Overlapping broader fragment"),
            score=0.72,
            rank=2,
        ),
    ]

    answer_text, citations = attach_fallback_citations("Упоминание найдено.", results)

    assert citations == ["04:55"]
    assert "[04:55]" in answer_text
    assert "[04:16]" not in answer_text


def test_attach_fallback_citations_skips_non_overlapping_generic_chunk_below_relative_threshold() -> None:
    results = [
        RetrievalResult(
            chunk=Chunk(chunk_id="c-1", video_id="video-1", start=295.2, end=314.4, text="there, I can forward the memo to Uncle Frank"),
            score=0.7501,
            rank=1,
        ),
        RetrievalResult(
            chunk=Chunk(chunk_id="c-2", video_id="video-1", start=33.6, end=84.32, text="rise of personal AI assistance with the release of Open Claw"),
            score=0.7044,
            rank=2,
        ),
    ]

    answer_text, citations = attach_fallback_citations("Упоминание найдено.", results)

    assert citations == ["04:55"]
    assert "[00:33]" not in answer_text
