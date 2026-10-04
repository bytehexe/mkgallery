"""Image loading for analysis (small, fast) and rendering (full, upright)."""

import io
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageOps

RAW_EXTENSIONS = frozenset({".cr2", ".cr3", ".nef", ".arw", ".dng"})
RASTER_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | RASTER_EXTENSIONS


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def _shrink(img: Image.Image, max_size: int) -> Image.Image:
    img.draft("RGB", (max_size, max_size))  # no-op for non-JPEG
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    return img


def load_preview(path: Path, max_size: int = 1024) -> Image.Image:
    """A small RGB version for analysis. RAW files are not developed."""
    if not is_raw(path):
        with Image.open(path) as img:
            return _shrink(img, max_size)

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
            return _shrink(Image.fromarray(rgb), max_size)

    if thumb.format == rawpy.ThumbFormat.JPEG:
        with Image.open(io.BytesIO(thumb.data)) as img:
            return _shrink(img, max_size)
    return _shrink(Image.fromarray(np.asarray(thumb.data)), max_size)


def load_full(path: Path) -> Image.Image:
    """The full-resolution RGB image with orientation applied."""
    if is_raw(path):
        with rawpy.imread(str(path)) as raw:
            # rawpy applies the camera's orientation flag by default.
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        return Image.fromarray(rgb)

    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img)
        return upright.convert("RGB")
