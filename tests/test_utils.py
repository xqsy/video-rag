from video_rag.utils import slugify


def test_slugify_preserves_russian_names() -> None:
    assert slugify("Привет мир") == "привет-мир"


def test_slugify_normalizes_ascii_names() -> None:
    assert slugify("Budget Review 2025") == "budget-review-2025"
