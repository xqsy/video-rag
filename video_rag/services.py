from __future__ import annotations

from pathlib import Path

from video_rag.config import settings
from video_rag.ingest.asr import ASRProfile, TranscriptionResult, WhisperTranscriber
from video_rag.ingest.audio import extract_audio
from video_rag.ingest.chunker import chunk_transcription
from video_rag.retrieval.dense import DenseRetriever
from video_rag.schemas import VideoMeta
from video_rag.storage.sqlite import VideoLibraryRepository
from video_rag.utils import ensure_parent_dir, slugify


class VideoLibraryService:
    def __init__(
        self,
        repository: VideoLibraryRepository | None = None,
        retriever: DenseRetriever | None = None,
        transcriber: WhisperTranscriber | None = None,
    ) -> None:
        settings.ensure_directories()
        self.repository = repository or VideoLibraryRepository()
        self.retriever = retriever or DenseRetriever()
        self.transcriber = transcriber or WhisperTranscriber()

    def save_upload(self, filename: str, payload: bytes) -> Path:
        suffix = Path(filename).suffix or ".bin"
        stem = slugify(Path(filename).stem)
        target = self._unique_video_path(stem, suffix)
        ensure_parent_dir(target)
        target.write_bytes(payload)
        return target

    def ingest_file(
        self,
        source_path: str | Path,
        profile: ASRProfile = "draft",
        language: str | None = None,
        force: bool = False,
        title: str | None = None,
        video_id: str | None = None,
    ) -> VideoMeta:
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        resolved_video_id = self._resolve_video_id(source, preferred=video_id)
        video_title = title or source.stem
        processing_meta = VideoMeta(
            video_id=resolved_video_id,
            source_path=str(source),
            title=video_title,
            status="processing",
        )
        self.repository.upsert_video(processing_meta)

        try:
            audio_path = extract_audio(source, settings.audio_dir / f"{resolved_video_id}.wav")
            transcription = self.transcriber.transcribe(
                audio_path=audio_path,
                video_id=resolved_video_id,
                profile=profile,
                language=language,
                force=force,
            )
            chunks = chunk_transcription(transcription)
            if hasattr(self.retriever.store, "delete_video"):
                self.retriever.store.delete_video(resolved_video_id)
            self.retriever.index_chunks(chunks)
            indexed_meta = VideoMeta(
                video_id=resolved_video_id,
                source_path=str(source),
                title=video_title,
                duration_seconds=transcription.duration_seconds,
                language=transcription.language,
                status="indexed",
            )
            self.repository.upsert_video(indexed_meta)
            return indexed_meta
        except Exception:
            failed_meta = VideoMeta(
                video_id=resolved_video_id,
                source_path=str(source),
                title=video_title,
                status="failed",
            )
            self.repository.upsert_video(failed_meta)
            raise

    def reindex_video(self, video_id: str, profile: ASRProfile = "draft") -> VideoMeta:
        video = self.get_video(video_id)
        if video is None:
            raise FileNotFoundError(f"Video not found in library: {video_id}")
        return self.ingest_file(
            source_path=video.source_path,
            profile=profile,
            force=True,
            title=video.title,
            video_id=video.video_id,
        )

    def list_videos(self) -> list[VideoMeta]:
        return self.repository.list_videos()

    def get_video(self, video_id: str) -> VideoMeta | None:
        return self.repository.get_video(video_id)

    def get_summary(self, video_id: str) -> str | None:
        return self.repository.get_summary(video_id=video_id)

    def save_summary(self, video_id: str, content: str) -> None:
        self.repository.save_summary(video_id=video_id, content=content)

    def delete_video(self, video_id: str) -> None:
        video = self.repository.get_video(video_id)
        if video is None:
            return
        if hasattr(self.retriever.store, "delete_video"):
            self.retriever.store.delete_video(video_id)
        self.repository.delete_video(video_id)
        self._safe_unlink(Path(video.source_path))
        self._safe_unlink(settings.audio_dir / f"{video_id}.wav")
        for transcript_path in self._transcript_paths(video_id):
            self._safe_unlink(transcript_path)

    def load_transcript(self, video_id: str) -> TranscriptionResult | None:
        transcript_paths = self._transcript_paths(video_id)
        if not transcript_paths:
            return None
        latest = sorted(transcript_paths, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        return TranscriptionResult.model_validate_json(latest.read_text(encoding="utf-8"))

    def export_transcript_json(self, video_id: str) -> str | None:
        transcript = self.load_transcript(video_id)
        if transcript is None:
            return None
        return transcript.model_dump_json(indent=2)

    def export_transcript_srt(self, video_id: str) -> str | None:
        transcript = self.load_transcript(video_id)
        if transcript is None:
            return None
        blocks: list[str] = []
        for index, segment in enumerate(transcript.segments, start=1):
            blocks.append(
                "\n".join(
                    [
                        str(index),
                        f"{self._format_srt_timestamp(segment.start)} --> {self._format_srt_timestamp(segment.end)}",
                        segment.text.strip(),
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _resolve_video_id(self, source: Path, preferred: str | None = None) -> str:
        if preferred:
            return preferred
        base = slugify(source.stem)
        candidate = base
        counter = 2
        while True:
            existing = self.repository.get_video(candidate)
            if existing is None or Path(existing.source_path) == source:
                return candidate
            candidate = f"{base}-{counter}"
            counter += 1

    def _unique_video_path(self, stem: str, suffix: str) -> Path:
        candidate = settings.videos_dir / f"{stem}{suffix}"
        counter = 2
        while candidate.exists():
            candidate = settings.videos_dir / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    def _transcript_paths(self, video_id: str) -> list[Path]:
        return list(settings.transcripts_dir.glob(f"{video_id}__*__*.json"))

    def _safe_unlink(self, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except FileNotFoundError:
            return
        if not resolved.exists():
            return
        if settings.data_dir.resolve() not in resolved.parents:
            return
        resolved.unlink(missing_ok=True)

    def _format_srt_timestamp(self, seconds: float) -> str:
        milliseconds = max(0, int(round(seconds * 1000)))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
