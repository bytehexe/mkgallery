import os
from datetime import datetime

from mkgallery.exif import read_times
from tests.conftest import make_image


def test_reads_exif_dates_in_one_call(tmp_path):
    a = make_image(tmp_path / "a.jpg", taken=datetime(2025, 3, 4, 10, 30, 5))
    b = make_image(tmp_path / "b.jpg", taken=datetime(2025, 7, 1, 8, 0, 0))
    times = read_times([a, b])
    assert times[a] == datetime(2025, 3, 4, 10, 30, 5)
    assert times[b] == datetime(2025, 7, 1, 8, 0, 0)


def test_falls_back_to_mtime_without_exif(tmp_path):
    p = make_image(tmp_path / "none.jpg")
    os.utime(p, (1_700_000_000, 1_700_000_000))
    assert read_times([p])[p] == datetime.fromtimestamp(1_700_000_000)


def test_unreadable_file_does_not_break_the_batch(tmp_path):
    good = make_image(tmp_path / "good.jpg", taken=datetime(2024, 1, 2, 3, 4, 5))
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not an image")
    os.utime(bad, (1_600_000_000, 1_600_000_000))
    times = read_times([bad, good])
    assert times[good] == datetime(2024, 1, 2, 3, 4, 5)
    assert times[bad] == datetime.fromtimestamp(1_600_000_000)


def test_chunking(tmp_path):
    paths = [
        make_image(tmp_path / f"{i}.jpg", taken=datetime(2025, 1, 1 + i, 0, 0, 0))
        for i in range(5)
    ]
    times = read_times(paths, chunk_size=2)
    assert [times[p].day for p in paths] == [1, 2, 3, 4, 5]
