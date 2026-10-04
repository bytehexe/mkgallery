import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def make_image(
    path: Path,
    *,
    color: tuple[int, int, int] = (120, 90, 60),
    size: tuple[int, int] = (320, 240),
    noise: float = 0.0,
    blur: float = 0.0,
    taken: datetime | None = None,
    orientation: int | None = None,
    seed: int = 0,
) -> Path:
    """Write a synthetic image. ``noise`` ≈ 50 gives entropy above the 6.5 cut-off."""
    rng = np.random.default_rng(seed)
    arr = np.full((size[1], size[0], 3), color, dtype=np.float32)
    if noise:
        arr += rng.normal(0, noise, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if taken is not None:
        exif.get_ifd(0x8769)[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
    if orientation is not None:
        exif[0x0112] = orientation
    img.save(path, exif=exif)
    return path


def tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    """relative path -> (size, mtime_ns, sha1) for every file below root."""
    snapshot = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            snapshot[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
                digest,
            )
    return snapshot
