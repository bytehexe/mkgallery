from datetime import datetime
from pathlib import Path

from PIL import Image

from mkgallery import render as render_mod
from mkgallery.model import Analyzed, Metrics
from mkgallery.render import LIBRARIES, render, safe_stem
from tests.conftest import make_image

M = Metrics(0.5, 0.5, 7.5, "0" * 64, "1" * 42)


def item(path: Path, when=datetime(2025, 3, 4, 10, 0, 0)) -> Analyzed:
    return Analyzed(path, when, M)


def test_safe_stem():
    assert safe_stem("2025") == "2025"
    assert safe_stem('a/b\\c:d*e?"f<g>h|i') == "a_b_c_d_e__f_g_h_i"
    assert safe_stem("..hidden") == "hidden"
    assert safe_stem("   ") == "gallery"


def test_render_writes_html_and_assets(tmp_path):
    src = tmp_path / "src"
    a = make_image(src / "a.jpg", size=(3200, 2400), noise=40)
    b = make_image(src / "b.jpg", size=(1000, 1500), noise=40)
    out = tmp_path / "out"
    out.mkdir()
    html_path = render([item(a), item(b, datetime(2025, 12, 2))], "Year 2025", out)

    assert html_path == out / "Year 2025.html"
    assets = out / "Year 2025_assets"
    thumbs = sorted((assets / "thumbs").glob("*.jpg"))
    fulls = sorted((assets / "full").glob("*.jpg"))
    assert len(thumbs) == len(fulls) == 2
    with Image.open(assets / "thumbs" / thumbs[0].name) as t:
        assert max(t.size) == 600
    with Image.open(assets / "full" / fulls[0].name) as f:
        assert max(f.size) == 1600

    html = html_path.read_text(encoding="utf-8")
    assert html.count('class="glightbox"') == 2
    assert "2 photos" in html
    assert "March" in html and "December 2025" in html
    assert (assets / "lib" / "jquery" / "jquery.min.js").exists()
    assert (assets / "lib" / "glightbox" / "LICENSE.md").exists()


def test_same_filename_in_two_folders_does_not_collide(tmp_path):
    a = make_image(tmp_path / "x" / "IMG_0001.jpg", noise=40, seed=1)
    b = make_image(tmp_path / "y" / "IMG_0001.jpg", noise=40, seed=2)
    out = tmp_path / "out"
    out.mkdir()
    render([item(a), item(b)], "t", out)
    assert len(list((out / "t_assets" / "thumbs").glob("*.jpg"))) == 2


def test_title_is_escaped_and_filename_safe(tmp_path):
    a = make_image(tmp_path / "a.jpg", noise=40)
    out = tmp_path / "out"
    out.mkdir()
    html_path = render([item(a)], '<script>alert("x")</script>/..', out)
    assert html_path.parent == out
    html = html_path.read_text(encoding="utf-8")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_rerender_replaces_generated_dirs_only(tmp_path):
    a = make_image(tmp_path / "a.jpg", noise=40)
    out = tmp_path / "out"
    out.mkdir()
    render([item(a)], "t", out)
    keep = out / "t_assets" / "my-notes.txt"
    keep.write_text("mine")
    stale = out / "t_assets" / "thumbs" / "stale.jpg"
    stale.write_bytes(b"x")
    render([item(a)], "t", out)
    assert keep.read_text() == "mine"
    assert not stale.exists()


def test_footer_credits_every_bundled_library(tmp_path):
    vendor = Path(render_mod.__file__).parent / "vendor"
    assert set(LIBRARIES) == {p.name for p in vendor.iterdir() if p.is_dir()}
    a = make_image(tmp_path / "a.jpg", noise=40)
    out = tmp_path / "out"
    out.mkdir()
    html = render([item(a)], "t", out).read_text(encoding="utf-8")
    for display, url in LIBRARIES.values():
        assert display in html and url in html
    assert "<!--" in html  # licence/attribution comment


def test_single_photo_wording(tmp_path):
    a = make_image(tmp_path / "a.jpg", noise=40)
    out = tmp_path / "out"
    out.mkdir()
    html = render([item(a)], "t", out).read_text(encoding="utf-8")
    assert "1 photo " in html or "1 photo<" in html
