from __future__ import annotations

from pathlib import Path
import sqlite3

from video_rag.config import settings
from video_rag.schemas import VideoMeta
from video_rag.utils import ensure_parent_dir


class VideoLibraryRepository:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else settings.sqlite_path
        ensure_parent_dir(self.db_path)
        self._initialize()

    def _normalize_focus(self, focus: str | None) -> str:
        return (focus or "").strip()

    def upsert_video(self, video: VideoMeta) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO videos (
                    video_id,
                    source_path,
                    title,
                    duration_seconds,
                    language,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    title=excluded.title,
                    duration_seconds=excluded.duration_seconds,
                    language=excluded.language,
                    status=excluded.status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    video.video_id,
                    video.source_path,
                    video.title,
                    video.duration_seconds,
                    video.language,
                    video.status,
                ),
            )

    def get_video(self, video_id: str) -> VideoMeta | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT video_id, source_path, title, duration_seconds, language, status FROM videos WHERE video_id = ?",
                (video_id,),
            ).fetchone()
        return self._row_to_video(row) if row else None

    def list_videos(self) -> list[VideoMeta]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT video_id, source_path, title, duration_seconds, language, status FROM videos ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [self._row_to_video(row) for row in rows]

    def save_summary(self, video_id: str, content: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO summaries (video_id, focus, content)
                VALUES (?, ?, ?)
                ON CONFLICT(video_id, focus) DO UPDATE SET
                    content=excluded.content,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (video_id, "", content),
            )

    def get_summary(self, video_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content FROM summaries WHERE video_id = ? AND focus = ?",
                (video_id, ""),
            ).fetchone()
        return str(row["content"]) if row else None

    def delete_video(self, video_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM summaries WHERE video_id = ?", (video_id,))
            connection.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    duration_seconds REAL,
                    language TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    video_id TEXT NOT NULL,
                    focus TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (video_id, focus)
                )
                """
            )

    def _row_to_video(self, row: sqlite3.Row) -> VideoMeta:
        return VideoMeta(
            video_id=row["video_id"],
            source_path=row["source_path"],
            title=row["title"],
            duration_seconds=row["duration_seconds"],
            language=row["language"],
            status=row["status"],
        )
