"""Wire scan → analyse → select → render, with the safety guards."""

import shutil
from collections.abc import Callable
from pathlib import Path

from .analyze import analyze
from .cache import Cache
from .errors import GalleryError
from .render import render, safe_stem
from .scan import scan
from .select import select_images


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _output_targets(output_dir: Path, title: str) -> list[Path]:
    """Every path render() writes through; links among them are resolved by the caller."""
    stem = safe_stem(title)
    assets = output_dir / f"{stem}_assets"
    return [output_dir / f"{stem}.html", assets, assets / "thumbs", assets / "full", assets / "lib"]


def build_gallery(
    src: Path,
    title: str,
    *,
    count: int = 48,
    output_dir: Path = Path("."),
    cache_dir: Path | None = None,
    jobs: int = 1,
    order: str = "chronological",
    log: Callable[[str], None] = lambda _message: None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    src = src.resolve()
    output_dir = output_dir.resolve()
    if _inside(output_dir, src):
        raise GalleryError(
            f"Output directory {output_dir} is inside the source directory {src}; "
            "nothing is ever written there."
        )
    for target in _output_targets(output_dir, title):
        resolved = target.resolve()
        if _inside(resolved, src):
            raise GalleryError(
                f"{target} would be written inside the source directory {src}; "
                "nothing is ever written there."
            )
        # render() deletes and replaces these paths, so the source must not live below one.
        if _inside(src, resolved):
            raise GalleryError(
                f"The source directory {src} is inside {target}, which would be replaced; "
                "nothing in the source is ever deleted."
            )
    if cache_dir is not None and _inside(cache_dir.resolve(), src):
        raise GalleryError(
            f"Cache directory {cache_dir} is inside the source directory {src}; "
            "nothing is ever written there."
        )
    if shutil.which("exiftool") is None:
        raise GalleryError("exiftool was not found on PATH; please install it.")

    paths = scan(src)
    if not paths:
        raise GalleryError(f"No supported images found below {src}")
    log(f"Found {len(paths)} images")

    with Cache(cache_dir) as cache:
        result = analyze(paths, cache, jobs, progress)
    for path, message in result.failures:
        log(f"Skipped {path}: {message}")
    if not result.items:
        raise GalleryError("None of the images could be read.")

    picks = select_images(result.items, count, order)
    if not picks:
        raise GalleryError("No image passed the quality filters.")
    log(f"Selected {len(picks)} of {len(result.items)} images")

    output_dir.mkdir(parents=True, exist_ok=True)
    return render(picks, title, output_dir, log)
