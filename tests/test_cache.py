import os
from datetime import datetime

from mkgallery.cache import Cache
from mkgallery.model import Analyzed, Metrics


def _item(path):
    m = Metrics(0.1, 0.2, 7.0, "0" * 64, "1" * 42)
    return Analyzed(path, datetime(2025, 5, 6, 7, 8, 9), m)


def test_roundtrip_and_persistence(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"abc")
    with Cache(tmp_path / "c") as cache:
        assert cache.get(f) is None
        cache.put(_item(f))
        assert cache.get(f) == _item(f)
    with Cache(tmp_path / "c") as again:
        assert again.get(f) == _item(f)


def test_invalidated_by_mtime_or_size(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"abc")
    with Cache(tmp_path / "c") as cache:
        cache.put(_item(f))
        os.utime(f, ns=(1, 1))
        assert cache.get(f) is None
        cache.put(_item(f))
        f.write_bytes(b"abcd")
        os.utime(f, ns=(1, 1))
        assert cache.get(f) is None


def test_in_memory_cache_does_not_touch_disk(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"abc")
    before = set(tmp_path.iterdir())
    with Cache(None) as cache:
        cache.put(_item(f))
        assert cache.get(f) == _item(f)
    assert set(tmp_path.iterdir()) == before
