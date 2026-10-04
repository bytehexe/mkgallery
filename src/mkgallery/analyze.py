"""Per-image metrics, ported from mkmapdiary's postprocessors."""

from pathlib import Path

import numpy as np
from imagehash import ImageHash, colorhash, whash
from PIL import Image
from scipy.ndimage import laplace

from .imaging import load_preview
from .model import Metrics


def _bitstring(h: ImageHash) -> str:
    return "".join("1" if bit else "0" for bit in h.hash.flatten())


def compute_metrics(img: Image.Image) -> Metrics:
    gray = img.convert("L")

    small = gray.copy()
    small.thumbnail((1024, 1024))
    arr = np.asarray(small, dtype=np.float32) / 255.0
    laplacian = float(laplace(arr).var())
    contrast = float(arr.std())

    arr256 = np.asarray(gray.resize((256, 256)), dtype=np.float32)
    hist, _ = np.histogram(arr256, bins=256, range=(0, 255))
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    return Metrics(
        laplacian=laplacian,
        contrast=contrast,
        entropy=entropy,
        whash=_bitstring(whash(img)),
        colorhash=_bitstring(colorhash(img)),
    )


def analyze_file(path: Path) -> Metrics:
    return compute_metrics(load_preview(path))
