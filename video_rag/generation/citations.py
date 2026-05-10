from __future__ import annotations

from collections.abc import Sequence
import re

from video_rag.config import settings
from video_rag.generation.prompts import format_timestamp
from video_rag.schemas import RetrievalResult


TIMECODE_PATTERN = re.compile(r"\[((?:\d{2}:)?\d{2}:\d{2})\]")
NO_CONTEXT_PATTERNS = (
    "нет информации",
    "не найдено релевантных фрагментов",
    "не могу дать ответ на основе данного контекста",
    "недостаточно данных",
    "в предоставленных фрагментах нет",
)


def parse_timecode(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"Unsupported timecode: {value}")


def extract_citations(text: str) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for match in TIMECODE_PATTERN.findall(text):
        if match not in seen:
            seen.add(match)
            citations.append(match)
    return citations


def verify_citations(text: str, results: Sequence[RetrievalResult], tolerance_seconds: float = 2.0) -> list[str]:
    valid: list[str] = []
    for citation in extract_citations(text):
        seconds = parse_timecode(citation)
        if any(
            result.chunk.start - tolerance_seconds <= seconds <= result.chunk.end + tolerance_seconds
            for result in results
        ):
            valid.append(citation)
    return valid


def _intervals_overlap(left: RetrievalResult, right: RetrievalResult) -> bool:
    return left.chunk.start <= right.chunk.end and right.chunk.start <= left.chunk.end


def _select_fallback_results(
    results: Sequence[RetrievalResult],
    limit: int,
    min_score: float,
    relative_threshold: float,
) -> list[RetrievalResult]:
    if not results or limit <= 0:
        return []

    selected: list[RetrievalResult] = [results[0]]
    best_score = results[0].score
    for result in results[1:]:
        if result.score < min_score:
            continue
        if best_score > 0 and result.score < best_score * relative_threshold:
            continue
        if any(_intervals_overlap(result, existing) for existing in selected):
            continue
        selected.append(result)
        if len(selected) >= limit:
            break
    return selected[:limit]


def attach_fallback_citations(text: str, results: Sequence[RetrievalResult], limit: int | None = None) -> tuple[str, list[str]]:
    verified = verify_citations(text, results)
    if verified:
        return text, verified

    normalized = text.lower()
    if any(pattern in normalized for pattern in NO_CONTEXT_PATTERNS):
        return text, []

    selected_results = _select_fallback_results(
        results,
        limit=limit or settings.fallback_citation_limit,
        min_score=settings.citation_score_threshold,
        relative_threshold=settings.citation_relative_score_threshold,
    )
    fallback = [format_timestamp(result.chunk.start) for result in selected_results]
    if not fallback:
        return text, []
    suffix = " ".join(f"[{item}]" for item in fallback)
    merged = text.rstrip() + f"\n\nТаймкоды: {suffix}"
    return merged, fallback
