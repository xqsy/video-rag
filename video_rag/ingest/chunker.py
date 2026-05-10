from __future__ import annotations

from collections.abc import Iterable
import re

from razdel import sentenize

from video_rag.config import settings
from video_rag.ingest.asr import TranscriptionResult
from video_rag.schemas import Chunk


def normalize_text(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if not compact:
        return ""
    sentences = [item.text.strip() for item in sentenize(compact) if item.text.strip()]
    if sentences:
        return " ".join(sentences)
    return compact


def chunk_transcription(
    transcription: TranscriptionResult,
    window_seconds: float | None = None,
    overlap_seconds: float | None = None,
) -> list[Chunk]:
    window = window_seconds or settings.chunk_window_seconds
    overlap = overlap_seconds or settings.chunk_overlap_seconds
    if window <= 0:
        raise ValueError("window_seconds must be positive")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap_seconds must be >= 0 and < window_seconds")

    segments = transcription.segments
    if not segments:
        return []

    chunks: list[Chunk] = []
    cursor = max(0.0, segments[0].start)
    step = window - overlap
    last_end = max(segment.end for segment in segments)

    while cursor <= last_end:
        window_end = cursor + window
        selected = [segment for segment in segments if segment.end > cursor and segment.start < window_end]
        if selected:
            text = normalize_text(" ".join(segment.text for segment in selected))
            if text:
                start = min(segment.start for segment in selected)
                end = max(segment.end for segment in selected)
                chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{transcription.video_id}-{chunk_index:04d}",
                        video_id=transcription.video_id,
                        start=start,
                        end=end,
                        text=text,
                    )
                )
        if window_end >= last_end:
            break
        cursor += step

    return _deduplicate_adjacent_chunks(chunks)


def _deduplicate_adjacent_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    deduplicated: list[Chunk] = []
    for chunk in chunks:
        if deduplicated and deduplicated[-1].text == chunk.text:
            continue
        deduplicated.append(chunk)
    return deduplicated
