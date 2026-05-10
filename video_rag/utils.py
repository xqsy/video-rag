from pathlib import Path
import re
import unicodedata


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    cleaned = re.sub(r"[^\w\s-]", "", normalized, flags=re.UNICODE)
    cleaned = re.sub(r"[-\s]+", "-", cleaned, flags=re.UNICODE).strip("-_")
    return cleaned or "video"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
