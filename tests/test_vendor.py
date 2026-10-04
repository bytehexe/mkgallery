from pathlib import Path

import mkgallery

VENDOR = Path(mkgallery.__file__).parent / "vendor"
EXPECTED = {
    "jquery": ["jquery.min.js"],
    "justifiedgallery": ["jquery.justifiedGallery.min.js", "justifiedGallery.min.css"],
    "glightbox": ["glightbox.min.js", "glightbox.min.css"],
}


def test_libraries_and_licences_are_bundled():
    assert {p.name for p in VENDOR.iterdir() if p.is_dir()} == set(EXPECTED)
    for name, files in EXPECTED.items():
        folder = VENDOR / name
        for f in files:
            assert (folder / f).stat().st_size > 1000, f
        licences = list(folder.glob("LICENSE*"))
        assert len(licences) == 1, name
        assert "MIT" in licences[0].read_text(encoding="utf-8", errors="replace")


def test_bundled_scripts_carry_a_copyright_banner():
    for js in VENDOR.rglob("*.js"):
        head = js.read_text(encoding="utf-8", errors="replace")[:600].lower()
        assert any(w in head for w in ("copyright", "(c)", "©", "license")), js.name


def test_versions_file_documents_every_library():
    text = (VENDOR / "VERSIONS.md").read_text(encoding="utf-8")
    for name in EXPECTED:
        assert name in text.lower()
