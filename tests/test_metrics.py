from mkgallery.analyze import analyze_file, compute_metrics
from mkgallery.imaging import load_preview
from tests.conftest import make_image


def _metrics(tmp_path, name, **kw):
    return analyze_file(make_image(tmp_path / name, **kw))


def test_noise_has_high_entropy_flat_has_low(tmp_path):
    assert _metrics(tmp_path, "n.png", noise=50).entropy > 6.5
    assert _metrics(tmp_path, "f.png").entropy < 1.0


def test_blur_lowers_sharpness(tmp_path):
    sharp = _metrics(tmp_path, "s.png", noise=50)
    soft = _metrics(tmp_path, "b.png", noise=50, blur=3)
    assert soft.laplacian < sharp.laplacian


def test_hash_strings_have_expected_shape(tmp_path):
    m = _metrics(tmp_path, "a.png", noise=50)
    assert len(m.whash) == 64 and set(m.whash) <= {"0", "1"}
    assert len(m.colorhash) == 42 and set(m.colorhash) <= {"0", "1"}


def test_identical_images_hash_identically(tmp_path):
    a = _metrics(tmp_path, "a.png", noise=50, seed=3)
    b = _metrics(tmp_path, "b.png", noise=50, seed=3)
    assert a == b


def test_flat_image_does_not_produce_nan(tmp_path):
    m = _metrics(tmp_path, "flat.png")
    assert all(v == v for v in (m.laplacian, m.contrast, m.entropy))


def test_compute_metrics_accepts_preview(tmp_path):
    img = load_preview(make_image(tmp_path / "x.jpg", noise=50))
    assert compute_metrics(img).entropy > 6.5
