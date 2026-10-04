from datetime import datetime

from click.testing import CliRunner

from mkgallery import build as build_mod
from mkgallery.cli import main
from tests.conftest import make_image, tree_snapshot


def _library(tmp_path, n=6):
    src = tmp_path / "photos"
    for i in range(n):
        make_image(
            src / f"folder{i % 2}" / f"IMG_{i:04d}.jpg",
            color=(40 * i, 255 - 40 * i, 90),
            noise=50,
            seed=i,
            taken=datetime(2025, 1 + 2 * i, 3, 12, 0, 0),
        )
    return src


def _run(*args):
    return CliRunner().invoke(main, [str(a) for a in args])


def test_end_to_end_leaves_source_untouched(tmp_path):
    src = _library(tmp_path)
    before = tree_snapshot(src)
    out, cache = tmp_path / "out", tmp_path / "cache"
    result = _run(src, "--title", "Year 2025", "-n", 3, "-o", out, "--cache-dir", cache, "-j", 1)
    assert result.exit_code == 0, result.output
    assert (out / "Year 2025.html").exists()
    assert len(list((out / "Year 2025_assets" / "thumbs").glob("*.jpg"))) == 3
    assert tree_snapshot(src) == before


def test_varied_order_runs(tmp_path):
    src = _library(tmp_path)
    result = _run(
        src, "-t", "v", "-n", 4, "-o", tmp_path / "o", "--no-cache", "-j", 1, "--order", "varied"
    )
    assert result.exit_code == 0, result.output


def test_selection_options_are_passed_to_the_selector(tmp_path, monkeypatch):
    src = _library(tmp_path)
    seen = {}
    real = build_mod.select_images

    def spy(items, count, order, **options):
        seen.update(options)
        return real(items, count, order, **options)

    monkeypatch.setattr(build_mod, "select_images", spy)
    result = _run(
        src, "-t", "o", "-n", 3, "-o", tmp_path / "o", "--no-cache", "-j", 1,
        "--no-event-quota", "--balance", "0.5",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen == {"event_quota": False, "balance": 0.5}


def test_selection_option_defaults(tmp_path, monkeypatch):
    src = _library(tmp_path)
    seen = {}
    real = build_mod.select_images

    def spy(items, count, order, **options):
        seen.update(options)
        return real(items, count, order, **options)

    monkeypatch.setattr(build_mod, "select_images", spy)
    result = _run(src, "-t", "o", "-n", 3, "-o", tmp_path / "o", "--no-cache", "-j", 1)
    assert result.exit_code == 0, result.output
    assert seen == {"event_quota": True, "balance": 0.0}


def test_balance_outside_unit_interval_is_a_usage_error(tmp_path):
    src = _library(tmp_path)
    result = _run(src, "-t", "o", "-o", tmp_path / "o", "--no-cache", "--balance", "1.5")
    assert result.exit_code == 2
    assert "not in the range" in result.output


def test_output_inside_source_is_refused_before_writing(tmp_path):
    src = _library(tmp_path)
    before = tree_snapshot(src)
    result = _run(src, "-t", "x", "-o", src / "out", "--no-cache")
    assert result.exit_code != 0
    assert "inside" in result.output
    assert tree_snapshot(src) == before
    assert not (src / "out").exists()


def test_cache_inside_source_is_refused(tmp_path):
    src = _library(tmp_path)
    result = _run(src, "-t", "x", "-o", tmp_path / "o", "--cache-dir", src / ".cache")
    assert result.exit_code != 0
    assert "inside" in result.output
    assert not (src / ".cache").exists()


def test_empty_source_gives_clear_error(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    result = _run(src, "-t", "x", "-o", tmp_path / "o", "--no-cache")
    assert result.exit_code != 0
    assert "No supported images" in result.output
    assert not (tmp_path / "o").exists()


def test_corrupt_image_is_skipped_with_message(tmp_path):
    src = _library(tmp_path, 4)
    (src / "broken.jpg").write_bytes(b"garbage")
    result = _run(src, "-t", "x", "-n", 3, "-o", tmp_path / "o", "--no-cache", "-j", 1)
    assert result.exit_code == 0, result.output
    assert "Skipped" in result.output and "broken.jpg" in result.output


def test_missing_exiftool_is_reported(tmp_path, monkeypatch):
    src = _library(tmp_path, 2)
    monkeypatch.setattr(build_mod.shutil, "which", lambda _name: None)
    result = _run(src, "-t", "x", "-o", tmp_path / "o", "--no-cache")
    assert result.exit_code != 0
    assert "exiftool" in result.output
