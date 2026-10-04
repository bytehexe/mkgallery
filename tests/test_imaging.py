import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from mkgallery import imaging
from tests.conftest import make_image


def test_load_preview_limits_long_edge(tmp_path):
    path = make_image(tmp_path / "big.jpg", size=(3000, 2000), noise=30)
    img = imaging.load_preview(path)
    assert img.mode == "RGB"
    assert max(img.size) <= 1024


def test_load_full_applies_exif_orientation(tmp_path):
    path = make_image(tmp_path / "rot.jpg", size=(200, 100), orientation=6)
    assert imaging.load_full(path).size == (100, 200)


def test_is_raw():
    assert imaging.is_raw(Path("x.CR2"))
    assert not imaging.is_raw(Path("x.jpg"))


class _FakeRaw:
    def __init__(self, thumb, rgb, no_thumb_exc):
        self._thumb, self._rgb, self._exc = thumb, rgb, no_thumb_exc
        self.postprocess_kwargs = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_thumb(self):
        if self._thumb is None:
            raise self._exc("no thumbnail")
        return self._thumb

    def postprocess(self, **kwargs):
        self.postprocess_kwargs = kwargs
        return self._rgb


def _fake_rawpy(thumb, rgb):
    class NoThumbnail(Exception):
        pass

    class Unsupported(Exception):
        pass

    raw = _FakeRaw(thumb, rgb, NoThumbnail)
    module = SimpleNamespace(
        imread=lambda _path: raw,
        ThumbFormat=SimpleNamespace(JPEG=1, BITMAP=2),
        LibRawNoThumbnailError=NoThumbnail,
        LibRawUnsupportedThumbnailError=Unsupported,
    )
    return module, raw


def _jpeg_bytes(size=(1600, 1200)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 200, 30)).save(buf, "JPEG")
    return buf.getvalue()


def test_raw_preview_uses_embedded_jpeg(tmp_path, monkeypatch):
    thumb = SimpleNamespace(format=1, data=_jpeg_bytes())
    module, raw = _fake_rawpy(thumb, None)
    monkeypatch.setattr(imaging, "rawpy", module)
    img = imaging.load_preview(tmp_path / "a.cr2")
    assert max(img.size) <= 1024
    assert raw.postprocess_kwargs is None  # no development needed


def test_raw_preview_falls_back_to_half_size_develop(tmp_path, monkeypatch):
    rgb = np.zeros((300, 400, 3), dtype=np.uint8)
    module, raw = _fake_rawpy(None, rgb)
    monkeypatch.setattr(imaging, "rawpy", module)
    img = imaging.load_preview(tmp_path / "a.cr2")
    assert img.size == (400, 300)
    assert raw.postprocess_kwargs["half_size"] is True


def test_raw_full_is_fully_developed(tmp_path, monkeypatch):
    rgb = np.zeros((300, 400, 3), dtype=np.uint8)
    module, raw = _fake_rawpy(None, rgb)
    monkeypatch.setattr(imaging, "rawpy", module)
    img = imaging.load_full(tmp_path / "a.cr2")
    assert img.size == (400, 300)
    assert "half_size" not in raw.postprocess_kwargs
    assert raw.postprocess_kwargs["use_camera_wb"] is True
