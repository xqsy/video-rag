from pathlib import Path

from video_rag.schemas import VideoMeta
from video_rag.storage.sqlite import VideoLibraryRepository


def test_video_library_repository_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "library.db"
    repository = VideoLibraryRepository(db_path)
    video = VideoMeta(
        video_id="demo-video",
        source_path="data/videos/demo.mp4",
        title="Demo video",
        duration_seconds=123.4,
        language="ru",
        status="indexed",
    )

    repository.upsert_video(video)
    loaded = repository.get_video("demo-video")

    assert loaded is not None
    assert loaded.video_id == video.video_id
    assert loaded.status == "indexed"
    assert repository.list_videos()[0].title == "Demo video"


def test_video_library_repository_persists_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "library.db"
    repository = VideoLibraryRepository(db_path)
    repository.upsert_video(
        VideoMeta(
            video_id="demo-video",
            source_path="data/videos/demo.mp4",
            title="Demo video",
            status="indexed",
        )
    )

    repository.save_summary("demo-video", "## Тезисы\n- Общий конспект")

    assert repository.get_summary("demo-video") == "## Тезисы\n- Общий конспект"

    repository.delete_video("demo-video")

    assert repository.get_summary("demo-video") is None
