"""Choose an interesting, varied subset of images.

Ported from mkmapdiary's ``Highlights`` (non-geo path) and its quality,
duplicate and entropy postprocessors.
"""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .model import Analyzed

QUALITY_STDDEV_FACTOR = 2
MIN_ENTROPY = 6.5
DUPLICATE_THRESHOLD = 10


def _bits(strings: Sequence[str]) -> np.ndarray:
    return np.array([[c == "1" for c in s] for s in strings], dtype=np.float32)


def _hamming(bits: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances of 0/1 rows, without an n*n*bits intermediate."""
    inverse = 1.0 - bits
    return bits @ inverse.T + inverse @ bits.T


def _scale(matrix: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; a constant matrix becomes all zeros."""
    low = matrix.min()
    span = matrix.max() - low
    if span == 0:
        return np.zeros_like(matrix)
    return (matrix - low) / span


def _time_matrix(seconds: np.ndarray) -> np.ndarray:
    return np.abs(seconds[:, None] - seconds[None, :])


def quality_scores(items: Sequence[Analyzed]) -> np.ndarray:
    laplacian = np.array([i.metrics.laplacian for i in items], dtype=np.float64)
    contrast = np.array([i.metrics.contrast for i in items], dtype=np.float64)
    return 0.5 * _scale(laplacian) + 0.5 * _scale(contrast)


def low_quality(quality: np.ndarray) -> np.ndarray:
    threshold = quality.mean() - QUALITY_STDDEV_FACTOR * quality.std()
    return quality < threshold


def duplicates(items: Sequence[Analyzed], quality: np.ndarray) -> np.ndarray:
    flagged = np.zeros(len(items), dtype=bool)
    by_day: dict[object, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        by_day[item.timestamp.date()].append(index)

    for indexes in by_day.values():
        if len(indexes) < 2:
            continue
        group = [items[i] for i in indexes]
        hashes = _hamming(_bits([g.metrics.whash for g in group]))
        minutes = _time_matrix(np.array([g.timestamp.timestamp() for g in group])) / 60.0
        distance = (hashes + 0.5 * minutes).astype(np.float64)
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DUPLICATE_THRESHOLD,
            metric="precomputed",
            linkage="complete",
        ).fit_predict(distance)
        for label in set(labels):
            members = [indexes[k] for k in np.flatnonzero(labels == label)]
            if len(members) > 1:
                best = max(members, key=lambda m: (quality[m], -m))
                for member in members:
                    if member != best:
                        flagged[member] = True
    return flagged


def eligible(items: Sequence[Analyzed], quality: np.ndarray) -> list[int]:
    if not items:
        return []
    drop = duplicates(items, quality) | low_quality(quality)
    return [
        i
        for i, item in enumerate(items)
        if not drop[i] and item.metrics.entropy > MIN_ENTROPY
    ]


def _distance(items: Sequence[Analyzed]) -> np.ndarray:
    """Colour distance + time distance, each scaled to [0, 1]."""
    colour = _scale(_hamming(_bits([i.metrics.colorhash for i in items])))
    seconds = np.array([i.timestamp.timestamp() for i in items])
    return colour + _scale(_time_matrix(seconds))
