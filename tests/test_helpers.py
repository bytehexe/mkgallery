from datetime import datetime

from PIL import Image

from tests.conftest import make_image, tree_snapshot


def test_make_image_writes_jpeg_with_exif_date(tmp_path):
    path = make_image(tmp_path / "a.jpg", taken=datetime(2025, 3, 4, 10, 30, 5))
    with Image.open(path) as img:
        assert img.size == (320, 240)
        exif = img.getexif()
        assert exif.get_ifd(0x8769)[0x9003] == "2025:03:04 10:30:05"


def test_make_image_noise_is_seeded(tmp_path):
    a = make_image(tmp_path / "a.png", noise=40, seed=1)
    b = make_image(tmp_path / "b.png", noise=40, seed=1)
    assert a.read_bytes() == b.read_bytes()


def test_tree_snapshot_detects_changes(tmp_path):
    f = tmp_path / "sub" / "x.txt"
    f.parent.mkdir()
    f.write_text("one")
    before = tree_snapshot(tmp_path)
    assert tree_snapshot(tmp_path) == before
    f.write_text("two")
    assert tree_snapshot(tmp_path) != before
