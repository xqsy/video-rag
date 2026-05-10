from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Video RAG"
    data_dir: Path = Path("data")
    videos_dir: Path = Path("data/videos")
    audio_dir: Path = Path("data/audio")
    transcripts_dir: Path = Path("data/transcripts")
    chroma_dir: Path = Path("data/chroma")
    chroma_collection_name: str = "video_chunks"
    sqlite_path: Path = Path("data/library.db")
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_draft_model: str = "small"
    whisper_final_model: str = "large-v3-turbo"
    whisper_vad_filter: bool = True
    whisper_word_timestamps: bool = False
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str | None = None
    openrouter_model: str = "deepseek/deepseek-v4-flash"
    request_timeout_seconds: float = 60.0
    answer_max_tokens: int = 1200
    summary_chunk_max_tokens: int = 1800
    summary_reduce_max_tokens: int = 3200
    fallback_citation_limit: int = 2
    citation_score_threshold: float = 0.5
    citation_relative_score_threshold: float = 0.95
    retrieval_top_k: int = 5
    retrieval_history_turns: int = 3
    chunk_window_seconds: float = 45.0
    chunk_overlap_seconds: float = 8.0
    history_max_turns: int = 4
    summary_group_size: int = 6
    embedding_model_name: str = "intfloat/multilingual-e5-base"
    embedding_batch_size: int = 16
    log_level: str = Field(default="INFO")

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.videos_dir,
            self.audio_dir,
            self.transcripts_dir,
            self.chroma_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
