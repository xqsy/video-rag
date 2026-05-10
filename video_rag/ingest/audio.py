from pathlib import Path
import subprocess

from video_rag.config import settings
from video_rag.utils import ensure_parent_dir, slugify


class AudioExtractionError(RuntimeError):
    pass


def build_audio_output_path(video_path: Path) -> Path:
    video_id = slugify(video_path.stem)
    return settings.audio_dir / f"{video_id}.wav"



def extract_audio(video_path: str | Path, output_path: str | Path | None = None) -> Path:
    source = Path(video_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    settings.ensure_directories()
    target = Path(output_path).expanduser().resolve() if output_path else build_audio_output_path(source)
    ensure_parent_dir(target)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(target),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AudioExtractionError(result.stderr.strip() or "ffmpeg failed to extract audio")

    return target
