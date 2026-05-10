from __future__ import annotations

from pathlib import Path
from typing import Literal
import json

from pydantic import BaseModel, ConfigDict, Field

from video_rag.config import settings
from video_rag.utils import ensure_parent_dir

ASRProfile = Literal["draft", "final"]


class TranscriptionSegment(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class TranscriptionResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    video_id: str
    source_path: str
    audio_path: str
    model_name: str
    profile: ASRProfile
    language: str | None = None
    duration_seconds: float = Field(default=0, ge=0)
    segments: list[TranscriptionSegment]


class WhisperTranscriber:
    def __init__(self) -> None:
        settings.ensure_directories()

    def transcribe(
        self,
        audio_path: str | Path,
        video_id: str,
        profile: ASRProfile = "draft",
        language: str | None = None,
        word_timestamps: bool | None = None,
        force: bool = False,
    ) -> TranscriptionResult:
        source = Path(audio_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Audio file not found: {source}")

        model_name = self._resolve_model_name(profile)
        cache_path = self._cache_path(video_id=video_id, model_name=model_name, profile=profile)
        if cache_path.exists() and not force:
            return self._load_cache(cache_path)

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install project dependencies before transcription."
            ) from exc

        model = WhisperModel(
            model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        segment_iter, info = model.transcribe(
            str(source),
            language=language,
            vad_filter=settings.whisper_vad_filter,
            word_timestamps=settings.whisper_word_timestamps
            if word_timestamps is None
            else word_timestamps,
            beam_size=1,
        )
        segments = [
            TranscriptionSegment(
                id=index,
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
            )
            for index, segment in enumerate(segment_iter)
            if segment.text.strip()
        ]
        result = TranscriptionResult(
            video_id=video_id,
            source_path=str(source),
            audio_path=str(source),
            model_name=model_name,
            profile=profile,
            language=info.language,
            duration_seconds=max((segment.end for segment in segments), default=0),
            segments=segments,
        )
        self._save_cache(cache_path, result)
        return result

    def _resolve_model_name(self, profile: ASRProfile) -> str:
        if profile == "final":
            return settings.whisper_final_model
        return settings.whisper_draft_model

    def _cache_path(self, video_id: str, model_name: str, profile: ASRProfile) -> Path:
        return settings.transcripts_dir / f"{video_id}__{model_name}__{profile}.json"

    def _load_cache(self, cache_path: Path) -> TranscriptionResult:
        return TranscriptionResult.model_validate_json(cache_path.read_text(encoding="utf-8"))

    def _save_cache(self, cache_path: Path, result: TranscriptionResult) -> None:
        ensure_parent_dir(cache_path)
        cache_path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
