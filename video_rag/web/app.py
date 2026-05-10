from __future__ import annotations

from functools import lru_cache
from mimetypes import guess_type
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from video_rag.generation.citations import parse_timecode
from video_rag.ingest.asr import ASRProfile
from video_rag.pipeline import VideoRAGPipeline
from video_rag.services import VideoLibraryService

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Video RAG Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class HistoryTurn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    video_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    history: list[HistoryTurn] = Field(default_factory=list)


class ReindexRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    profile: ASRProfile = "draft"


class SummaryResponse(BaseModel):
    summary: str | None = None


class VideoPayload(BaseModel):
    video_id: str
    title: str
    source_path: str
    duration_seconds: float | None = None
    language: str | None = None
    status: str


class AnswerPayload(BaseModel):
    answer_text: str
    citations: list[str]


@lru_cache(maxsize=1)
def get_library_service() -> VideoLibraryService:
    return VideoLibraryService()


@lru_cache(maxsize=1)
def get_pipeline() -> VideoRAGPipeline:
    return VideoRAGPipeline()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/videos", response_model=list[VideoPayload])
def list_videos() -> list[VideoPayload]:
    videos = get_library_service().list_videos()
    return [VideoPayload.model_validate(video.model_dump()) for video in videos]


@app.post("/api/videos/upload", response_model=VideoPayload)
def upload_video(
    file: UploadFile = File(...),
    profile: ASRProfile = Form("draft"),
    language: str = Form(""),
) -> VideoPayload:
    filename = file.filename or "video.bin"
    payload = file.file.read()
    file.file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Файл пустой.")

    service = get_library_service()
    try:
        saved_path = service.save_upload(filename, payload)
        video = service.ingest_file(
            saved_path,
            profile=profile,
            language=language.strip() or None,
            title=filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось обработать файл: {exc}") from exc
    return VideoPayload.model_validate(video.model_dump())


@app.post("/api/videos/{video_id}/reindex", response_model=VideoPayload)
def reindex_video(video_id: str, request: ReindexRequest) -> VideoPayload:
    service = get_library_service()
    try:
        video = service.reindex_video(video_id, profile=request.profile)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось переиндексировать видео: {exc}") from exc
    return VideoPayload.model_validate(video.model_dump())


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict[str, str]:
    get_library_service().delete_video(video_id)
    return {"status": "ok"}


@app.get("/api/videos/{video_id}", response_model=VideoPayload)
def get_video(video_id: str) -> VideoPayload:
    video = get_library_service().get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Видео не найдено.")
    return VideoPayload.model_validate(video.model_dump())


@app.get("/api/videos/{video_id}/summary", response_model=SummaryResponse)
def get_summary(video_id: str) -> SummaryResponse:
    summary = get_library_service().get_summary(video_id)
    return SummaryResponse(summary=summary)


@app.post("/api/videos/{video_id}/summary", response_model=SummaryResponse)
def generate_summary(video_id: str) -> SummaryResponse:
    service = get_library_service()
    video = service.get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Видео не найдено.")
    if video.status != "indexed":
        raise HTTPException(status_code=400, detail="Видео ещё не проиндексировано.")
    try:
        summary = get_pipeline().summarize(video_id)
        service.save_summary(video_id, summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации конспекта: {exc}") from exc
    return SummaryResponse(summary=summary)


@app.post("/api/chat/answer", response_model=AnswerPayload)
def answer_question(request: AnswerRequest) -> AnswerPayload:
    service = get_library_service()
    video = service.get_video(request.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Видео не найдено.")
    if video.status != "indexed":
        raise HTTPException(status_code=400, detail="Видео ещё не проиндексировано.")
    history = [(item.question, item.answer) for item in request.history]
    try:
        answer = get_pipeline().answer(
            question=request.question,
            video_id=request.video_id,
            history=history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {exc}") from exc
    return AnswerPayload(answer_text=answer.answer_text, citations=answer.citations)


@app.get("/api/videos/{video_id}/transcript/json")
def download_transcript_json(video_id: str) -> Response:
    service = get_library_service()
    payload = service.export_transcript_json(video_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Транскрипт не найден.")
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{video_id}.json"'},
    )


@app.get("/api/videos/{video_id}/transcript/srt")
def download_transcript_srt(video_id: str) -> Response:
    service = get_library_service()
    payload = service.export_transcript_srt(video_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Транскрипт не найден.")
    return Response(
        content=payload,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{video_id}.srt"'},
    )


@app.get("/api/videos/{video_id}/file")
def stream_video(video_id: str) -> FileResponse:
    video = get_library_service().get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Видео не найдено.")
    source = Path(video.source_path)
    if not source.exists():
        raise HTTPException(status_code=404, detail="Файл видео не найден на диске.")
    media_type = guess_type(str(source))[0] or "application/octet-stream"
    return FileResponse(source, media_type=media_type, filename=source.name)


@app.get("/api/videos/{video_id}/jump/{timecode}")
def jump_timecode(video_id: str, timecode: str) -> dict[str, object]:
    video = get_library_service().get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Видео не найдено.")
    try:
        seconds = int(parse_timecode(timecode))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"video_id": video_id, "timecode": timecode, "seconds": seconds}
