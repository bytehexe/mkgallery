from datetime import datetime

import pytest

from mkgallery.build import GalleryError, build_gallery
from mkgallery.model import Analyzed, Metrics
from mkgallery.render import render
from tests.conftest import make_image, tree_snapshot

M = Metrics(0.5, 0.5, 7.5, "0" * 64, "1" * 42)


def _images(root, n=3):
    for i in range(n):
        make_image(root / f"IMG_{i}.jpg", noise=50, seed=i, taken=datetime(2025, 1 + i, 3, 12))


def _item(path):
    return Analyzed(path, datetime(2025, 3, 4, 10), M)


def test_source_that_is_the_assets_dir_is_refused_without_deleting(tmp_path):
    base = tmp_path / "base"
    src = base / "Trip_assets"
    _images(src)
    (src / "thumbs").mkdir()
    (src / "thumbs" / "precious.txt").write_text("keep")
    before = tree_snapshot(src)
    with pytest.raises(GalleryError, match="inside"):
        build_gallery(src, "Trip", output_dir=base)
    assert tree_snapshot(src) == before


def test_source_nested_below_a_replaced_assets_dir_is_refused_without_deleting(tmp_path):
    base = tmp_path / "base"
    src = base / "Trip_assets" / "full" / "sub"
    _images(src)
    before = tree_snapshot(src)
    with pytest.raises(GalleryError, match="inside"):
        build_gallery(src, "Trip", output_dir=base)
    assert tree_snapshot(src) == before


def test_assets_symlink_into_source_is_refused(tmp_path):
    src = tmp_path / "photos"
    _images(src)
    (src / "sub").mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "T_assets").symlink_to(src / "sub")
    before = tree_snapshot(src)
    with pytest.raises(GalleryError, match="inside"):
        build_gallery(src, "T", output_dir=out)
    assert tree_snapshot(src) == before


def test_assets_subdir_symlink_into_source_is_refused(tmp_path):
    src = tmp_path / "photos"
    _images(src)
    (src / "sub").mkdir()
    out = tmp_path / "out"
    (out / "T_assets").mkdir(parents=True)
    (out / "T_assets" / "thumbs").symlink_to(src / "sub")
    before = tree_snapshot(src)
    with pytest.raises(GalleryError, match="inside"):
        build_gallery(src, "T", output_dir=out)
    assert tree_snapshot(src) == before


def test_page_symlink_into_source_is_refused(tmp_path):
    src = tmp_path / "photos"
    _images(src)
    out = tmp_path / "out"
    out.mkdir()
    (out / "T.html").symlink_to(src / "page.html")
    with pytest.raises(GalleryError, match="inside"):
        build_gallery(src, "T", output_dir=out)
    assert not (src / "page.html").exists()


def test_render_skips_unconvertible_image_and_logs(tmp_path):
    good = make_image(tmp_path / "good.jpg", noise=40)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"junk")
    out = tmp_path / "out"
    out.mkdir()
    messages = []
    page = render([_item(bad), _item(good)], "t", out, log=messages.append)
    assert page.read_text(encoding="utf-8").count('class="glightbox"') == 1
    assert len(list((out / "t_assets" / "thumbs").glob("*.jpg"))) == 1
    assert any("bad.jpg" in m for m in messages)


def test_render_keeps_previous_gallery_when_nothing_converts(tmp_path):
    good = make_image(tmp_path / "good.jpg", noise=40)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"junk")
    out = tmp_path / "out"
    out.mkdir()
    page = render([_item(good)], "t", out)
    before = tree_snapshot(out)
    with pytest.raises(GalleryError, match="converted"):
        render([_item(bad)], "t", out)
    assert tree_snapshot(out) == before
    assert page.exists()
