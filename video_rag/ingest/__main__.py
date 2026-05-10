from __future__ import annotations

import argparse
from pathlib import Path
import sys

from video_rag.config import settings
from video_rag.ingest.asr import WhisperTranscriber
from video_rag.ingest.audio import extract_audio
from video_rag.ingest.chunker import chunk_transcription
from video_rag.utils import slugify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m video_rag.ingest")
    parser.add_argument("source", help="Path to a local video or audio file")
    parser.add_argument("--profile", choices=["draft", "final"], default="draft")
    parser.add_argument("--language", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audio-path", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        parser.error(f"Source file not found: {source}")

    settings.ensure_directories()
    video_id = slugify(source.stem)
    audio_path = extract_audio(source, args.audio_path)

    transcriber = WhisperTranscriber()
    transcription = transcriber.transcribe(
        audio_path=audio_path,
        video_id=video_id,
        profile=args.profile,
        language=args.language,
        force=args.force,
    )
    chunks = chunk_transcription(transcription)

    print(f"video_id={video_id}")
    print(f"audio_path={audio_path}")
    print(f"segments={len(transcription.segments)}")
    print(f"chunks={len(chunks)}")
    print(f"language={transcription.language or 'unknown'}")
    print(f"duration_seconds={transcription.duration_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
