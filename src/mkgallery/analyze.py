"""Per-image metrics, ported from mkmapdiary's postprocessors."""

import multiprocessing
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from imagehash import ImageHash, colorhash, whash
from PIL import Image
from scipy.ndimage import laplace

from .cache import Cache
from .exif import read_times
from .imaging import load_preview
from .model import Analyzed, Metrics


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


@dataclass
class AnalysisResult:
    items: list[Analyzed]
    failures: list[tuple[Path, str]]


def _safe_metrics(path: Path) -> tuple[Path, Metrics | str]:
    """Never raises, so one broken file cannot abort a whole run."""
    try:
        return path, analyze_file(path)
    except Exception as exc:  # noqa: BLE001 - any decoder error means "skip this file"
        return path, f"{type(exc).__name__}: {exc}"


# "spawn": forking a multi-threaded process (exiftool helper threads) can deadlock.
_CONTEXT = multiprocessing.get_context("spawn")

Worker = Callable[[Path], tuple[Path, Metrics | str]]


def _isolated(path: Path, worker: Worker) -> tuple[Path, Metrics | str]:
    """Run one file in its own process, so a hard crash only costs that file."""
    with ProcessPoolExecutor(max_workers=1, mp_context=_CONTEXT) as pool:
        try:
            return pool.submit(worker, path).result()
        except BrokenProcessPool:
            return path, "the worker process crashed while reading this file"


def _compute(
    paths: Sequence[Path], jobs: int, worker: Worker
) -> Iterator[tuple[Path, Metrics | str]]:
    if jobs <= 1:
        for path in paths:
            yield worker(path)
        return
    finished: set[Path] = set()
    try:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=_CONTEXT) as pool:
            futures = [pool.submit(worker, path) for path in paths]
            for future in as_completed(futures):
                result = future.result()
                finished.add(result[0])
                yield result
    except BrokenProcessPool:
        # A worker died (e.g. a crash inside LibRaw); find the culprit file by file.
        for path in paths:
            if path not in finished:
                yield _isolated(path, worker)


def analyze(
    paths: Sequence[Path],
    cache: Cache,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    worker: Worker = _safe_metrics,
) -> AnalysisResult:
    items: dict[Path, Analyzed] = {}
    misses: list[Path] = []
    for path in paths:
        cached = cache.get(path)
        if cached is not None:
            items[path] = cached
        else:
            misses.append(path)

    failures: list[tuple[Path, str]] = []
    if misses:
        times = read_times(misses)
        for done, (path, outcome) in enumerate(_compute(misses, jobs, worker), start=1):
            if isinstance(outcome, Metrics):
                item = Analyzed(path, times[path], outcome)
                cache.put(item)
                items[path] = item
            else:
                failures.append((path, outcome))
            if progress is not None:
                progress(done, len(misses))

    return AnalysisResult([items[p] for p in paths if p in items], failures)
