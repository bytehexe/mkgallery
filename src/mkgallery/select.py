"""Choose an interesting, varied subset of images.

Ported from mkmapdiary's ``Highlights`` (non-geo path) and its quality,
duplicate and entropy postprocessors.
"""

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import dual_annealing
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
    return [i for i, item in enumerate(items) if not drop[i] and item.metrics.entropy > MIN_ENTROPY]


def _distance(items: Sequence[Analyzed]) -> np.ndarray:
    """Colour distance + time distance, each scaled to [0, 1]."""
    colour = _scale(_hamming(_bits([i.metrics.colorhash for i in items])))
    seconds = np.array([i.timestamp.timestamp() for i in items])
    return colour + _scale(_time_matrix(seconds))


_ORDERS = ("chronological", "varied")


def _cluster_best(
    items: Sequence[Analyzed], quality: np.ndarray, keep: list[int], count: int
) -> list[int]:
    """One image per cluster: the highest-quality one."""
    subset = [items[i] for i in keep]
    labels = AgglomerativeClustering(
        n_clusters=count, metric="precomputed", linkage="average"
    ).fit_predict(_distance(subset).astype(np.float64))
    picks = []
    for label in range(count):
        members = [keep[k] for k in np.flatnonzero(labels == label)]
        picks.append(max(members, key=lambda m: (quality[m], -m)))
    return picks


def _vary(picks: list[Analyzed], quality: dict[Path, float]) -> list[Analyzed]:
    """Order so that neighbouring images are as dissimilar as possible.

    Same objective as mkmapdiary: minimise the summed *similarity* between
    consecutive images, then rotate the best image to the second position.
    """
    if len(picks) <= 2:
        return picks
    distance = _distance(picks)
    similarity = distance.max() - distance

    def tour_length(x: np.ndarray) -> float:
        order = np.argsort(x)
        return float(similarity[order, np.roll(order, -1)].sum())

    result = dual_annealing(tour_length, [(0, 1)] * len(picks), seed=42)
    arranged = [picks[i] for i in np.argsort(result.x)]
    best = max(range(len(arranged)), key=lambda i: quality[arranged[i].path])
    shift = (best - 1) % len(arranged)
    return arranged[shift:] + arranged[:shift]


def select_images(
    items: Sequence[Analyzed], count: int, order: str = "chronological"
) -> list[Analyzed]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {_ORDERS}, got {order!r}")
    if not items:
        return []

    ordered = sorted(items, key=lambda i: str(i.path))
    quality = quality_scores(ordered)
    keep = eligible(ordered, quality)
    if len(keep) > count:
        keep = _cluster_best(ordered, quality, keep, count)
    picks = [ordered[i] for i in keep]

    if order == "chronological":
        return sorted(picks, key=lambda i: (i.timestamp, str(i.path)))
    quality_by_path = {ordered[i].path: float(quality[i]) for i in range(len(ordered))}
    return _vary(picks, quality_by_path)
