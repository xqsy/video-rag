from video_rag.ingest.asr import TranscriptionResult, TranscriptionSegment
from video_rag.ingest.chunker import chunk_transcription


def test_chunk_transcription_creates_overlapping_chunks() -> None:
    transcription = TranscriptionResult(
        video_id="demo-video",
        source_path="demo.wav",
        audio_path="demo.wav",
        model_name="small",
        profile="draft",
        language="ru",
        duration_seconds=75.0,
        segments=[
            TranscriptionSegment(id=0, start=0.0, end=10.0, text="Первая часть записи."),
            TranscriptionSegment(id=1, start=12.0, end=24.0, text="Вторая часть записи."),
            TranscriptionSegment(id=2, start=30.0, end=42.0, text="Третья часть записи."),
            TranscriptionSegment(id=3, start=48.0, end=60.0, text="Четвертая часть записи."),
            TranscriptionSegment(id=4, start=63.0, end=75.0, text="Пятая часть записи."),
        ],
    )

    chunks = chunk_transcription(transcription, window_seconds=45.0, overlap_seconds=8.0)

    assert len(chunks) >= 2
    assert chunks[0].video_id == "demo-video"
    assert chunks[0].start == 0.0
    assert chunks[0].end == 42.0
    assert "Первая часть записи" in chunks[0].text
