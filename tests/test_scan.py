import pytest

from mkgallery.scan import scan


def test_scan_finds_supported_files_recursively(tmp_path):
    wanted = ["a.jpg", "sub/b.JPEG", "sub/deep/c.CR2", "d.png", "e.webp", "f.tiff", "g.dng"]
    unwanted = ["notes.txt", "movie.mp4", "sub/.hidden/x.jpg", "._junk.jpg", "h.jpg.bak"]
    for rel in wanted + unwanted:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    found = [p.relative_to(tmp_path).as_posix() for p in scan(tmp_path)]
    assert found == sorted(wanted)


def test_scan_rejects_non_directory(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        scan(f)
