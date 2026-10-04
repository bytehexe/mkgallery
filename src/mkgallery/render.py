"""Convert the selected images and write the gallery page."""

import hashlib
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, PackageLoader
from PIL import Image

from .errors import GalleryError
from .imaging import load_full
from .model import Analyzed

THUMB_EDGE = 600
FULL_EDGE = 1600
VENDOR_DIR = Path(__file__).parent / "vendor"

LIBRARIES: dict[str, tuple[str, str]] = {
    "justifiedgallery": ("Justified Gallery", "https://github.com/miromannino/Justified-Gallery"),
    "glightbox": ("GLightbox", "https://github.com/biati-digital/glightbox"),
    "jquery": ("jQuery", "https://jquery.com"),
}

_UNSAFE_STEM = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class Photo:
    full: str
    thumb: str
    width: int
    height: int
    caption: str


def safe_stem(title: str) -> str:
    stem = _UNSAFE_STEM.sub("_", title).strip().lstrip(".")
    return stem or "gallery"


def _asset_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:6]
    return f"{_UNSAFE_NAME.sub('_', path.stem)}-{digest}.jpg"


def _caption(when: datetime) -> str:
    return f"{when.day} {when:%B %Y}"


def _summary(selected: Sequence[Analyzed]) -> str:
    count = len(selected)
    text = f"{count} photo{'' if count == 1 else 's'}"
    first = min(i.timestamp for i in selected)
    last = max(i.timestamp for i in selected)
    if (first.year, first.month) == (last.year, last.month):
        return f"{text} · {first:%B %Y}"
    if first.year == last.year:
        return f"{text} · {first:%B} – {last:%B %Y}"
    return f"{text} · {first:%b %Y} – {last:%b %Y}"


def _convert(item: Analyzed, name: str, full_dir: Path, thumb_dir: Path) -> Photo:
    with load_full(item.path) as img:
        full = img.copy()
    full.thumbnail((FULL_EDGE, FULL_EDGE), Image.Resampling.LANCZOS)
    full.save(full_dir / name, "JPEG", quality=88, optimize=True, progressive=True)
    thumb = full.copy()
    thumb.thumbnail((THUMB_EDGE, THUMB_EDGE), Image.Resampling.LANCZOS)
    thumb.save(thumb_dir / name, "JPEG", quality=82, optimize=True, progressive=True)
    return Photo(
        full=f"full/{name}",
        thumb=f"thumbs/{name}",
        width=thumb.width,
        height=thumb.height,
        caption=_caption(item.timestamp),
    )


def _replace(target: Path, staged: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    staged.rename(target)


def render(
    selected: Sequence[Analyzed],
    title: str,
    output_dir: Path,
    log: Callable[[str], None] = lambda _message: None,
) -> Path:
    stem = safe_stem(title)
    assets = output_dir / f"{stem}_assets"
    assets.mkdir(exist_ok=True)

    # Convert into staging directories first: a failure must not wipe the previous gallery.
    staged = {name: assets / f"{name}.new" for name in ("thumbs", "full")}
    for directory in staged.values():
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir()

    photos: list[Photo] = []
    converted: list[Analyzed] = []
    for item in selected:
        try:
            photos.append(_convert(item, _asset_name(item.path), staged["full"], staged["thumbs"]))
            converted.append(item)
        except Exception as exc:  # noqa: BLE001 - e.g. a RAW whose sensor data is damaged
            log(f"Skipped {item.path}: {type(exc).__name__}: {exc}")

    if not photos:
        for directory in staged.values():
            shutil.rmtree(directory, ignore_errors=True)
        raise GalleryError("None of the selected images could be converted.")

    for name, directory in staged.items():
        _replace(assets / name, directory)
    shutil.rmtree(assets / "lib", ignore_errors=True)
    shutil.copytree(VENDOR_DIR, assets / "lib", ignore=shutil.ignore_patterns("VERSIONS.md"))

    env = Environment(loader=PackageLoader("mkgallery", "templates"), autoescape=True)
    html = env.get_template("page.html.j2").render(
        title=title,
        summary=_summary(converted),
        assets_url=quote(f"{stem}_assets"),
        photos=photos,
        libraries=list(LIBRARIES.values()),
    )
    page = output_dir / f"{stem}.html"
    page.write_text(html, encoding="utf-8")
    return page
