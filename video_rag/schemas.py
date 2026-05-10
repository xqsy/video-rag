from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VideoMeta(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    video_id: str
    source_path: str
    title: str
    duration_seconds: float | None = None
    language: str | None = None
    status: Literal["new", "processing", "indexed", "failed"] = "new"


class Chunk(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    chunk_id: str
    video_id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str


class RetrievalResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)


class Answer(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    video_id: str
    question: str
    answer_text: str
    citations: list[str] = Field(default_factory=list)
