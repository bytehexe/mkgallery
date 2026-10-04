import os
from datetime import datetime

from mkgallery import analyze as analyze_mod
from mkgallery.analyze import analyze
from mkgallery.cache import Cache
from tests.conftest import make_image


def _files(tmp_path, n=3):
    return [
        make_image(tmp_path / f"{i}.jpg", noise=40, seed=i, taken=datetime(2025, 1, 1 + i, 9, 0, 0))
        for i in range(n)
    ]


def test_second_run_is_served_from_cache(tmp_path, monkeypatch):
    files = _files(tmp_path)
    calls = []
    real = analyze_mod.analyze_file
    monkeypatch.setattr(analyze_mod, "analyze_file", lambda p: (calls.append(p), real(p))[1])

    with Cache(tmp_path / "cache") as cache:
        first = analyze(files, cache, jobs=1)
    assert len(calls) == 3
    calls.clear()
    with Cache(tmp_path / "cache") as cache:
        second = analyze(files, cache, jobs=1)
    assert calls == []
    assert second.items == first.items
    assert [i.timestamp.day for i in first.items] == [1, 2, 3]


def test_corrupt_image_is_reported_not_fatal(tmp_path):
    files = _files(tmp_path, 2)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with Cache(None) as cache:
        result = analyze([files[0], bad, files[1]], cache, jobs=1)
    assert [i.path for i in result.items] == files
    assert [p for p, _ in result.failures] == [bad]
    assert result.failures[0][1]  # has a message


def test_process_pool_path(tmp_path):
    files = _files(tmp_path)
    seen = []
    with Cache(None) as cache:
        result = analyze(files, cache, jobs=2, progress=lambda d, t: seen.append((d, t)))
    assert [i.path for i in result.items] == files
    assert seen[-1] == (3, 3)


def crashing_worker(path):
    """Top-level so a spawned worker can import it."""
    if path.name == "boom.jpg":
        os._exit(1)
    return analyze_mod._safe_metrics(path)


def test_worker_crash_is_isolated_to_the_bad_file(tmp_path):
    files = _files(tmp_path, 3)
    boom = make_image(tmp_path / "boom.jpg", noise=40)
    with Cache(None) as cache:
        result = analyze(
            [files[0], boom, files[1], files[2]], cache, jobs=2, worker=crashing_worker
        )
    assert [p for p, _ in result.failures] == [boom]
    assert [i.path for i in result.items] == files
