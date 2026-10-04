# mkgallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI, `mkgallery SRC --title T`, that turns a photo collection (≈5000 images, incl. CR2) into `<T>.html` + `<T>_assets/` holding a justified gallery of ~48 automatically chosen images.

**Architecture:** scan → analyse (small previews only, batched exiftool, SQLite cache outside SRC) → select (mkmapdiary's filter + cluster + order algorithm, minus geo) → render (convert only the selected images from the originals, write one Jinja2 page with bundled jQuery/Justified Gallery/GLightbox). Each stage is one module with a plain-function interface; `build.py` wires them, `cli.py` is a thin click wrapper.

**Tech Stack:** Python ≥3.10, hatch, click, Pillow, rawpy, imagehash, numpy, scipy, scikit-learn, pyexiftool, platformdirs, Jinja2; system `exiftool`.

**Spec:** `docs/superpowers/specs/2026-10-04-mkgallery-design.md` (reference implementation of the selection: `../mkmapdiary/src/mkmapdiary/lib/highlights.py`, `postprocessors/{simpleImageQualityAssessment,duplicateDetector,entropyCalculator,imageHasher}.py`).

## Global Constraints

- Source images/directories are never modified; nothing is written inside SRC (output dir **and** cache dir must not be inside or equal to SRC).
- No config read from SRC. Output is exactly `<title>.html` + `<title>_assets/` (subdirs `thumbs/`, `full/`, `lib/`).
- Defaults: `--count 48`, `--order chronological` (other value: `varied`), thumbs ≈600px long edge q82, full ≈1600px long edge q88.
- Selection constants (from mkmapdiary): quality = 0.5·norm(Laplacian var) + 0.5·norm(contrast); bad quality = `< mean − 2·σ`; entropy must be `> 6.5`; duplicates: per day, `whash` Hamming + 0.5·minutes, complete linkage, threshold 10; clustering: average linkage on normalised colour-hash + time distance; `varied` order: `dual_annealing(seed=42)`, best image rotated to 2nd position.
- Analysis uses images ≤1024px only; RAW previews via `rawpy.extract_thumb()` (fallback half-size develop); RAW is fully developed only at render time and only for selected images.
- Timestamps are naive local datetimes; EXIF `DateTimeOriginal`→`CreateDate`→`ModifyDate`, fallback file mtime.
- Bundled libraries (jQuery, Justified Gallery, GLightbox; all MIT): pinned versions in `src/mkgallery/vendor/<name>/`, each with its licence file, copyright banners left intact, listed in the page footer with links.
- No performance work beyond the spec for >5000 images. YAGNI: no geo, maps, pins, config files, video.
- Python tooling via hatch only: `hatch test`, `hatch fmt`, `hatch run types:check`. Never `pip`, bare `python`, or `PYTHONPATH`.
- Commits: stage explicit paths only (never `git add -A`/`.`; the sandbox masks dotfiles in this repo), and end every commit message with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

Inputs most likely to bite that the spec only implies; each has a pinning test in the owning task.

1. **Corrupt/unreadable image in SRC** → skipped with a message, run still completes (Task 6, Task 11).
2. **Empty SRC, or fewer eligible images than `--count`** → clear error for empty; all eligible images shown for fewer (Task 8, Task 11).
3. **Same filename in different folders** (`IMG_0001.jpg` ×2) → both rendered, no overwrite (Task 10).
4. **Title with `/`, quotes, `<script>`** → safe file stem, HTML-escaped page (Task 10).
5. **No EXIF dates / all images identical** → mtime fallback; degenerate normalisation (zero range) doesn't crash or produce NaN (Task 3, Task 4, Task 7, Task 8).
6. **Cache dir inside SRC** → refused (Task 11).

## File Structure

```
pyproject.toml, .gitignore, README.md
src/mkgallery/__init__.py
src/mkgallery/model.py        # Metrics, Analyzed dataclasses
src/mkgallery/imaging.py      # extensions, load_preview, load_full (the only rawpy user)
src/mkgallery/scan.py         # scan(src) -> sorted image paths
src/mkgallery/exif.py         # read_times(paths) via one batched exiftool
src/mkgallery/analyze.py      # compute_metrics, analyze_file, analyze() orchestration
src/mkgallery/cache.py        # SQLite cache of Analyzed
src/mkgallery/select.py       # filters, clustering, ordering
src/mkgallery/render.py       # image conversion + HTML
src/mkgallery/build.py        # build_gallery(): wiring + guards, GalleryError
src/mkgallery/cli.py          # click command
src/mkgallery/templates/page.html.j2
src/mkgallery/vendor/{jquery,justifiedgallery,glightbox}/…  + VERSIONS.md
tests/conftest.py + one test file per module
```

---

### Task 1: Project scaffold and test helpers

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/mkgallery/__init__.py`, `tests/conftest.py`, `tests/test_helpers.py`

**Interfaces:**
- Produces (`tests/conftest.py`):
  - `make_image(path: Path, *, color=(120, 90, 60), size=(320, 240), noise=0.0, blur=0.0, taken: datetime | None = None, orientation: int | None = None, seed=0) -> Path` — writes an image (format from suffix) with optional EXIF `DateTimeOriginal`/`Orientation`.
  - `tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]` — relative path → (size, mtime_ns, sha1) for every file below `root`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mkgallery"
version = "0.1.0"
description = "Summarise a photo collection as a single gallery page"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "click",
    "imagehash",
    "jinja2",
    "numpy",
    "Pillow>=10",
    "platformdirs",
    "pyexiftool",
    "rawpy",
    "scikit-learn",
    "scipy",
]

[project.scripts]
mkgallery = "mkgallery.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/mkgallery"]

[tool.hatch.envs.types]
extra-dependencies = ["mypy>=1.0.0"]

[tool.hatch.envs.types.scripts]
check = "mypy --install-types --non-interactive {args:src/mkgallery tests}"

[tool.hatch.envs.hatch-static-analysis]
dependencies = ["ruff==0.14.10"]
config-path = "none"

[tool.mypy]
disable_error_code = ["import-not-found", "import-untyped"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
```

- [ ] **Step 2: Write `.gitignore` and package init**

`.gitignore`:
```
__pycache__/
*.egg-info/
.venv/
dist/
build/
.mypy_cache/
.ruff_cache/
.pytest_cache/
```

`src/mkgallery/__init__.py`:
```python
"""mkgallery: summarise a photo collection as a single gallery page."""
```

- [ ] **Step 3: Write the failing helper tests**

`tests/test_helpers.py`:
```python
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
```

Also create an empty `tests/__init__.py` so `tests.conftest` imports.

- [ ] **Step 4: Run to verify failure**

Run: `hatch test tests/test_helpers.py`
Expected: FAIL — `ImportError: cannot import name 'make_image'` (hatch will first create its env and install dependencies; if the install fails inside the sandbox, re-run with `allowed_domains` `pypi.org` and `files.pythonhosted.org` and tell the user about `/sandbox`).

- [ ] **Step 5: Write `tests/conftest.py`**

```python
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def make_image(
    path: Path,
    *,
    color: tuple[int, int, int] = (120, 90, 60),
    size: tuple[int, int] = (320, 240),
    noise: float = 0.0,
    blur: float = 0.0,
    taken: datetime | None = None,
    orientation: int | None = None,
    seed: int = 0,
) -> Path:
    """Write a synthetic image. ``noise`` ≈ 50 gives entropy above the 6.5 cut-off."""
    rng = np.random.default_rng(seed)
    arr = np.full((size[1], size[0], 3), color, dtype=np.float32)
    if noise:
        arr += rng.normal(0, noise, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if taken is not None:
        exif.get_ifd(0x8769)[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
    if orientation is not None:
        exif[0x0112] = orientation
    img.save(path, exif=exif)
    return path


def tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    """relative path -> (size, mtime_ns, sha1) for every file below root."""
    snapshot = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            snapshot[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
                digest,
            )
    return snapshot
```

- [ ] **Step 6: Run to verify pass**

Run: `hatch test tests/test_helpers.py`
Expected: 3 passed. (If the EXIF-date test fails because Pillow does not persist `get_ifd` edits, fix `make_image` here — later tasks depend on it.)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/mkgallery/__init__.py tests/__init__.py tests/conftest.py tests/test_helpers.py
git commit -m "Scaffold project and test helpers" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Image loading (`imaging.py`) and scanning (`scan.py`)

**Files:**
- Create: `src/mkgallery/imaging.py`, `src/mkgallery/scan.py`, `tests/test_imaging.py`, `tests/test_scan.py`

**Interfaces:**
- Produces (`imaging.py`): `RAW_EXTENSIONS: frozenset[str]`, `RASTER_EXTENSIONS: frozenset[str]`, `SUPPORTED_EXTENSIONS: frozenset[str]` (all lowercase, with dot); `is_raw(path: Path) -> bool`; `load_preview(path: Path, max_size: int = 1024) -> PIL.Image.Image` (RGB, long edge ≤ `max_size`); `load_full(path: Path) -> PIL.Image.Image` (RGB, orientation applied). Module-level `import rawpy` (tests monkeypatch `mkgallery.imaging.rawpy`).
- Produces (`scan.py`): `scan(src: Path) -> list[Path]` — sorted, recursive, supported extensions only, skips hidden files/dirs; raises `NotADirectoryError` if `src` is not a directory.

- [ ] **Step 1: Write failing tests**

`tests/test_scan.py`:
```python
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
```

`tests/test_imaging.py`:
```python
import io
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
    from pathlib import Path

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
```

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_scan.py tests/test_imaging.py`
Expected: FAIL — `ModuleNotFoundError: mkgallery.imaging`.

- [ ] **Step 3: Implement**

`src/mkgallery/imaging.py`:
```python
"""Image loading for analysis (small, fast) and rendering (full, upright)."""

import io
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageOps

RAW_EXTENSIONS = frozenset({".cr2", ".cr3", ".nef", ".arw", ".dng"})
RASTER_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | RASTER_EXTENSIONS


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def _shrink(img: Image.Image, max_size: int) -> Image.Image:
    img.draft("RGB", (max_size, max_size))  # no-op for non-JPEG
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    return img


def load_preview(path: Path, max_size: int = 1024) -> Image.Image:
    """A small RGB version for analysis. RAW files are not developed."""
    if not is_raw(path):
        with Image.open(path) as img:
            return _shrink(img, max_size)

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
            return _shrink(Image.fromarray(rgb), max_size)

    if thumb.format == rawpy.ThumbFormat.JPEG:
        with Image.open(io.BytesIO(thumb.data)) as img:
            return _shrink(img, max_size)
    return _shrink(Image.fromarray(np.asarray(thumb.data)), max_size)


def load_full(path: Path) -> Image.Image:
    """The full-resolution RGB image with orientation applied."""
    if is_raw(path):
        with rawpy.imread(str(path)) as raw:
            # rawpy applies the camera's orientation flag by default.
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        return Image.fromarray(rgb)

    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img)
        return upright.convert("RGB")
```

`src/mkgallery/scan.py`:
```python
"""Find the images below a source directory (read-only)."""

import os
from pathlib import Path

from .imaging import SUPPORTED_EXTENSIONS


def scan(src: Path) -> list[Path]:
    if not src.is_dir():
        raise NotADirectoryError(src)
    found = []
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in files:
            if name.startswith("."):
                continue
            if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(Path(root) / name)
    return sorted(found)
```

- [ ] **Step 4: Run to verify pass**

Run: `hatch test tests/test_scan.py tests/test_imaging.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mkgallery/imaging.py src/mkgallery/scan.py tests/test_imaging.py tests/test_scan.py
git commit -m "Add image loading and directory scan" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Timestamps via batched exiftool (`exif.py`)

**Files:**
- Create: `src/mkgallery/exif.py`, `tests/test_exif.py`

**Interfaces:**
- Produces: `read_times(paths: Sequence[Path], chunk_size: int = 200) -> dict[Path, datetime]` — every input path gets an entry (EXIF date, else file mtime). Requires `exiftool` on PATH (caller checks). Naive datetimes.

Reference for the exiftool calls: `../mkmapdiary/src/mkmapdiary/tasks/base/exifReader.py` (`et.get_metadata([...])`, tag names like `EXIF:DateTimeOriginal`, `ExifToolExecuteError`).

- [ ] **Step 1: Write failing tests**

`tests/test_exif.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_exif.py`
Expected: FAIL — `ModuleNotFoundError: mkgallery.exif`.

- [ ] **Step 3: Implement**

`src/mkgallery/exif.py`:
```python
"""Capture times from EXIF, read for many files with a single exiftool process."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import exiftool
from exiftool.exceptions import ExifToolExecuteError

_DATE_TAGS = (
    ("Composite:SubSecDateTimeOriginal", "%Y:%m:%d %H:%M:%S.%f"),
    ("EXIF:DateTimeOriginal", "%Y:%m:%d %H:%M:%S"),
    ("Composite:SubSecCreateDate", "%Y:%m:%d %H:%M:%S.%f"),
    ("EXIF:CreateDate", "%Y:%m:%d %H:%M:%S"),
    ("EXIF:ModifyDate", "%Y:%m:%d %H:%M:%S"),
)


def _metadata(et: Any, paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Metadata keyed by SourceFile; a failing chunk is retried file by file."""
    try:
        records = et.get_metadata([str(p) for p in paths])
    except ExifToolExecuteError:
        if len(paths) == 1:
            return {}
        merged: dict[str, dict[str, Any]] = {}
        for path in paths:
            merged.update(_metadata(et, [path]))
        return merged
    return {record["SourceFile"]: record for record in records}


def _exif_time(record: dict[str, Any]) -> datetime | None:
    for tag, fmt in _DATE_TAGS:
        value = record.get(tag)
        if not isinstance(value, str):
            continue
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def read_times(paths: Sequence[Path], chunk_size: int = 200) -> dict[Path, datetime]:
    times: dict[Path, datetime] = {}
    with exiftool.ExifToolHelper() as et:
        for start in range(0, len(paths), chunk_size):
            chunk = paths[start : start + chunk_size]
            records = _metadata(et, chunk)
            for path in chunk:
                record = records.get(str(path))
                taken = _exif_time(record) if record else None
                times[path] = taken or datetime.fromtimestamp(path.stat().st_mtime)
    return times
```

- [ ] **Step 4: Run to verify pass**

Run: `hatch test tests/test_exif.py`
Expected: 4 passed. If the corrupt-file test fails because pyexiftool returns a record for the bad file with an `Error` key instead of raising, that is fine — `_exif_time` finds no date and mtime is used; if it fails for another reason, report the exact exception rather than patching around it.

- [ ] **Step 5: Commit**

```bash
git add src/mkgallery/exif.py tests/test_exif.py
git commit -m "Read capture times with batched exiftool" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Metrics (`model.py`, `analyze.compute_metrics`)

**Files:**
- Create: `src/mkgallery/model.py`, `src/mkgallery/analyze.py` (first part), `tests/test_metrics.py`

**Interfaces:**
- Produces (`model.py`):
  ```python
  @dataclass(frozen=True)
  class Metrics:
      laplacian: float   # variance of the Laplacian of the grey preview (/255)
      contrast: float    # std-dev of the grey preview (/255)
      entropy: float     # Shannon entropy of a 256x256 grey histogram, bits (0..8)
      whash: str         # '0'/'1' string, 64 chars
      colorhash: str     # '0'/'1' string, 42 chars

  @dataclass(frozen=True)
  class Analyzed:
      path: Path
      timestamp: datetime
      metrics: Metrics
  ```
- Produces (`analyze.py`): `compute_metrics(img: Image.Image) -> Metrics`; `analyze_file(path: Path) -> Metrics` (= `compute_metrics(load_preview(path))`).

- [ ] **Step 1: Write failing tests**

`tests/test_metrics.py`:
```python
import pytest
from PIL import Image

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
```
(Unused imports `pytest`, `Image` must not be left in — drop them when writing the file.)

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_metrics.py`
Expected: FAIL — `ModuleNotFoundError: mkgallery.analyze`.

- [ ] **Step 3: Implement**

`src/mkgallery/model.py`:
```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Metrics:
    laplacian: float
    contrast: float
    entropy: float
    whash: str
    colorhash: str


@dataclass(frozen=True)
class Analyzed:
    path: Path
    timestamp: datetime
    metrics: Metrics
```

`src/mkgallery/analyze.py`:
```python
"""Per-image metrics, ported from mkmapdiary's postprocessors."""

from pathlib import Path

import numpy as np
from imagehash import ImageHash, colorhash, whash
from PIL import Image
from scipy.ndimage import laplace

from .imaging import load_preview
from .model import Metrics


def _bitstring(h: ImageHash) -> str:
    return "".join("1" if bit else "0" for bit in h.hash.flatten())


def compute_metrics(img: Image.Image) -> Metrics:
    gray = img.convert("L")

    small = gray.copy()
    small.thumbnail((1024, 1024))
    arr = np.asarray(small, dtype=np.float32) / 255.0
    laplacian = float(laplace(arr).var())
    contrast = float(arr.std())

    arr256 = np.asarray(gray.resize((256, 256)), dtype=np.float32)
    hist, _ = np.histogram(arr256, bins=256, range=(0, 255))
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    return Metrics(
        laplacian=laplacian,
        contrast=contrast,
        entropy=entropy,
        whash=_bitstring(whash(img)),
        colorhash=_bitstring(colorhash(img)),
    )


def analyze_file(path: Path) -> Metrics:
    return compute_metrics(load_preview(path))
```

- [ ] **Step 4: Run to verify pass**

Run: `hatch test tests/test_metrics.py`
Expected: all pass. If the `colorhash` length is not 42, print `len(_bitstring(colorhash(img)))`, update the constraint in the spec and these tests together (imagehash's default is `binbits=3`, 14 bins).

- [ ] **Step 5: Commit**

```bash
git add src/mkgallery/model.py src/mkgallery/analyze.py tests/test_metrics.py
git commit -m "Compute quality, entropy and hash metrics" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Cache (`cache.py`)

**Files:**
- Create: `src/mkgallery/cache.py`, `tests/test_cache.py`

**Interfaces:**
- Consumes: `Analyzed`, `Metrics` from `mkgallery.model`.
- Produces: `Cache(directory: Path | None)` — `None` = in-memory (no persistence). Context manager. `get(path: Path) -> Analyzed | None` (valid only if size and mtime_ns of the file still match); `put(item: Analyzed) -> None`; `close() -> None`.

- [ ] **Step 1: Write failing tests**

`tests/test_cache.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_cache.py`
Expected: FAIL — `ModuleNotFoundError: mkgallery.cache`.

- [ ] **Step 3: Implement**

`src/mkgallery/cache.py`:
```python
"""Persistent cache of per-image analysis, kept outside the source tree."""

import dataclasses
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .model import Analyzed, Metrics

# Bump when the analysis changes so stale entries are ignored.
SCHEMA = 1
_COMMIT_EVERY = 200


class Cache:
    def __init__(self, directory: Path | None) -> None:
        if directory is None:
            target = ":memory:"
        else:
            directory.mkdir(parents=True, exist_ok=True)
            target = str(directory / "cache.sqlite3")
        self._db = sqlite3.connect(target)
        self._db.execute("PRAGMA synchronous=OFF")
        self._table = f"entries_v{SCHEMA}"
        self._db.execute(
            f"CREATE TABLE IF NOT EXISTS {self._table} "
            "(path TEXT PRIMARY KEY, size INTEGER NOT NULL, "
            "mtime_ns INTEGER NOT NULL, payload TEXT NOT NULL)"
        )
        self._pending = 0

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _key(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return str(path.resolve()), stat.st_size, stat.st_mtime_ns

    def get(self, path: Path) -> Analyzed | None:
        key, size, mtime_ns = self._key(path)
        row = self._db.execute(
            f"SELECT size, mtime_ns, payload FROM {self._table} WHERE path = ?", (key,)
        ).fetchone()
        if row is None or (row[0], row[1]) != (size, mtime_ns):
            return None
        data = json.loads(row[2])
        return Analyzed(
            path=path,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metrics=Metrics(**data["metrics"]),
        )

    def put(self, item: Analyzed) -> None:
        key, size, mtime_ns = self._key(item.path)
        payload = json.dumps(
            {
                "timestamp": item.timestamp.isoformat(),
                "metrics": dataclasses.asdict(item.metrics),
            }
        )
        self._db.execute(
            f"INSERT OR REPLACE INTO {self._table} VALUES (?, ?, ?, ?)",
            (key, size, mtime_ns, payload),
        )
        self._pending += 1
        if self._pending >= _COMMIT_EVERY:
            self._db.commit()
            self._pending = 0

    def close(self) -> None:
        self._db.commit()
        self._db.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `hatch test tests/test_cache.py`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mkgallery/cache.py tests/test_cache.py
git commit -m "Add SQLite analysis cache" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Analysis orchestration (`analyze.analyze`)

**Files:**
- Modify: `src/mkgallery/analyze.py` (append)
- Create: `tests/test_analyze.py`

**Interfaces:**
- Consumes: `read_times`, `Cache`, `analyze_file`, `Metrics`, `Analyzed`.
- Produces:
  ```python
  @dataclass
  class AnalysisResult:
      items: list[Analyzed]                    # input order, failures omitted
      failures: list[tuple[Path, str]]         # (path, error message)

  def analyze(paths: Sequence[Path], cache: Cache, jobs: int = 1,
              progress: Callable[[int, int], None] | None = None) -> AnalysisResult
  ```
  `jobs == 1` runs inline; `jobs > 1` uses a `ProcessPoolExecutor`. `progress(done, total)` is called after each *uncached* image.

- [ ] **Step 1: Write failing tests**

`tests/test_analyze.py`:
```python
from datetime import datetime

from mkgallery import analyze as analyze_mod
from mkgallery.analyze import analyze
from mkgallery.cache import Cache
from tests.conftest import make_image


def _files(tmp_path, n=3):
    return [
        make_image(tmp_path / f"{i}.jpg", noise=40, seed=i, taken=datetime(2025, 1, 1 + i, 9, 0, 0))
        for i in range(n)
    ]


def test_second_run_is_served_from_cache(tmp_path, monkeypatch):
    files = _files(tmp_path)
    calls = []
    real = analyze_mod.analyze_file
    monkeypatch.setattr(analyze_mod, "analyze_file", lambda p: (calls.append(p), real(p))[1])

    with Cache(tmp_path / "cache") as cache:
        first = analyze(files, cache, jobs=1)
    assert len(calls) == 3
    calls.clear()
    with Cache(tmp_path / "cache") as cache:
        second = analyze(files, cache, jobs=1)
    assert calls == []
    assert second.items == first.items
    assert [i.timestamp.day for i in first.items] == [1, 2, 3]


def test_corrupt_image_is_reported_not_fatal(tmp_path):
    files = _files(tmp_path, 2)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with Cache(None) as cache:
        result = analyze([files[0], bad, files[1]], cache, jobs=1)
    assert [i.path for i in result.items] == files
    assert [p for p, _ in result.failures] == [bad]
    assert result.failures[0][1]  # has a message


def test_process_pool_path(tmp_path):
    files = _files(tmp_path)
    seen = []
    with Cache(None) as cache:
        result = analyze(files, cache, jobs=2, progress=lambda d, t: seen.append((d, t)))
    assert [i.path for i in result.items] == files
    assert seen[-1] == (3, 3)
```

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_analyze.py`
Expected: FAIL — `ImportError: cannot import name 'analyze'`.

- [ ] **Step 3: Implement** — add imports at the top of `analyze.py` and append:

Imports to add:
```python
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from .cache import Cache
from .exif import read_times
from .model import Analyzed
```

Appended code:
```python
@dataclass
class AnalysisResult:
    items: list[Analyzed]
    failures: list[tuple[Path, str]]


def _safe_metrics(path: Path) -> tuple[Path, Metrics | str]:
    """Never raises, so one broken file cannot abort a whole run."""
    try:
        return path, analyze_file(path)
    except Exception as exc:  # noqa: BLE001 - any decoder error means "skip this file"
        return path, f"{type(exc).__name__}: {exc}"


def _compute(paths: Sequence[Path], jobs: int) -> Iterator[tuple[Path, Metrics | str]]:
    if jobs <= 1:
        for path in paths:
            yield _safe_metrics(path)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_safe_metrics, path) for path in paths]
        for future in as_completed(futures):
            yield future.result()


def analyze(
    paths: Sequence[Path],
    cache: Cache,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    items: dict[Path, Analyzed] = {}
    misses: list[Path] = []
    for path in paths:
        cached = cache.get(path)
        if cached is not None:
            items[path] = cached
        else:
            misses.append(path)

    failures: list[tuple[Path, str]] = []
    if misses:
        times = read_times(misses)
        for done, (path, outcome) in enumerate(_compute(misses, jobs), start=1):
            if isinstance(outcome, Metrics):
                item = Analyzed(path, times[path], outcome)
                cache.put(item)
                items[path] = item
            else:
                failures.append((path, outcome))
            if progress is not None:
                progress(done, len(misses))

    return AnalysisResult([items[p] for p in paths if p in items], failures)
```
Note: `_safe_metrics` must call `analyze_file` through the module global (as written) so the monkeypatch in the test takes effect in the inline path.

- [ ] **Step 4: Run to verify pass**

Run: `hatch test tests/test_analyze.py tests/test_metrics.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/mkgallery/analyze.py tests/test_analyze.py
git commit -m "Orchestrate cached, parallel image analysis" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Selection filters (`select.py`, part 1)

**Files:**
- Create: `src/mkgallery/select.py`, `tests/test_select_filters.py`

**Interfaces:**
- Consumes: `Analyzed`, `Metrics`.
- Produces (all in `select.py`; items are processed in the order given):
  - `quality_scores(items: Sequence[Analyzed]) -> np.ndarray` — shape (n,), each in [0,1]; zero-range metrics yield 0 (never NaN).
  - `low_quality(quality: np.ndarray) -> np.ndarray` — bool mask, `quality < mean - 2*std`.
  - `duplicates(items: Sequence[Analyzed], quality: np.ndarray) -> np.ndarray` — bool mask of images to drop; per calendar day, keep the best-quality image of each near-duplicate cluster (ties → lowest index).
  - `eligible(items: Sequence[Analyzed], quality: np.ndarray) -> list[int]` — indexes passing all three filters (not duplicate, not low quality, `entropy > 6.5`).
  - helpers reused in Task 8: `_bits`, `_hamming`, `_scale`, `_time_matrix`, `_distance`.

- [ ] **Step 1: Write failing tests**

`tests/test_select_filters.py`:
```python
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from mkgallery.model import Analyzed, Metrics
from mkgallery.select import duplicates, eligible, low_quality, quality_scores


def bits(seed: int, n: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join("1" if b else "0" for b in rng.integers(0, 2, n))


def mk(name, when, *, lap=0.5, con=0.5, entropy=7.5, w=None, c=None) -> Analyzed:
    seed = abs(hash(name)) % 10_000
    return Analyzed(
        Path(name),
        when,
        Metrics(lap, con, entropy, w or bits(seed, 64), c or bits(seed + 1, 42)),
    )


T0 = datetime(2025, 6, 1, 12, 0, 0)


def test_quality_is_normalised_and_nan_free():
    items = [mk("a", T0, lap=0.0, con=0.0), mk("b", T0, lap=1.0, con=1.0), mk("c", T0, lap=0.5, con=0.5)]
    q = quality_scores(items)
    assert list(q) == [0.0, 1.0, 0.5]
    same = quality_scores([mk("a", T0), mk("b", T0)])
    assert list(same) == [0.0, 0.0]


def test_low_quality_flags_outliers_only():
    q = np.array([0.5] * 20 + [0.0])
    mask = low_quality(q)
    assert mask[-1] and not mask[:-1].any()
    assert not low_quality(np.array([0.4, 0.5, 0.6])).any()


def test_duplicates_keep_best_of_each_same_day_cluster():
    same_hash = bits(1, 64)
    burst = [
        mk("a", T0, w=same_hash),
        mk("b", T0 + timedelta(seconds=5), w=same_hash),
        mk("c", T0 + timedelta(seconds=9), w=same_hash),
    ]
    other = mk("z", T0 + timedelta(minutes=1), w=bits(99, 64))
    items = burst + [other]
    quality = np.array([0.2, 0.9, 0.4, 0.5])
    flagged = duplicates(items, quality)
    assert list(flagged) == [True, False, True, False]


def test_same_picture_on_different_days_is_not_a_duplicate():
    h = bits(1, 64)
    items = [mk("a", T0, w=h), mk("b", T0 + timedelta(days=1), w=h)]
    assert not duplicates(items, np.array([0.5, 0.5])).any()


def test_eligible_applies_all_three_filters():
    h = bits(5, 64)
    items = [
        mk("good1", T0, lap=0.5, con=0.5),
        mk("good2", T0 + timedelta(days=1), lap=0.6, con=0.6),
        mk("flat", T0 + timedelta(days=2), entropy=3.0),
        mk("dupe", T0 + timedelta(days=3), w=h),
        mk("dupe2", T0 + timedelta(days=3, seconds=3), w=h, lap=0.9, con=0.9),
    ]
    q = quality_scores(items)
    keep = eligible(items, q)
    names = [items[i].path.name for i in keep]
    assert "flat" not in names
    assert "dupe" not in names and "dupe2" in names
    assert "good1" in names and "good2" in names
```
Note: `hash(name)` is randomised per process for strings, which would make test data differ between runs; replace the `seed` line with `seed = sum(map(ord, name))` when writing the file.

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_select_filters.py`
Expected: FAIL — `ModuleNotFoundError: mkgallery.select`.

- [ ] **Step 3: Implement**

`src/mkgallery/select.py`:
```python
"""Choose an interesting, varied subset of images.

Ported from mkmapdiary's ``Highlights`` (non-geo path) and its quality,
duplicate and entropy postprocessors.
"""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .model import Analyzed

QUALITY_STDDEV_FACTOR = 2
MIN_ENTROPY = 6.5
DUPLICATE_THRESHOLD = 10


def _bits(strings: Sequence[str]) -> np.ndarray:
    return np.array([[c == "1" for c in s] for s in strings], dtype=np.float32)


def _hamming(bits: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances of 0/1 rows, without an n*n*bits intermediate."""
    inverse = 1.0 - bits
    return bits @ inverse.T + inverse @ bits.T


def _scale(matrix: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; a constant matrix becomes all zeros."""
    low = matrix.min()
    span = matrix.max() - low
    if span == 0:
        return np.zeros_like(matrix)
    return (matrix - low) / span


def _time_matrix(seconds: np.ndarray) -> np.ndarray:
    return np.abs(seconds[:, None] - seconds[None, :])


def _unit(values: np.ndarray) -> np.ndarray:
    return _scale(values)


def quality_scores(items: Sequence[Analyzed]) -> np.ndarray:
    laplacian = np.array([i.metrics.laplacian for i in items], dtype=np.float64)
    contrast = np.array([i.metrics.contrast for i in items], dtype=np.float64)
    return 0.5 * _unit(laplacian) + 0.5 * _unit(contrast)


def low_quality(quality: np.ndarray) -> np.ndarray:
    threshold = quality.mean() - QUALITY_STDDEV_FACTOR * quality.std()
    return quality < threshold


def duplicates(items: Sequence[Analyzed], quality: np.ndarray) -> np.ndarray:
    flagged = np.zeros(len(items), dtype=bool)
    by_day: dict[object, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        by_day[item.timestamp.date()].append(index)

    for indexes in by_day.values():
        if len(indexes) < 2:
            continue
        group = [items[i] for i in indexes]
        hashes = _hamming(_bits([g.metrics.whash for g in group]))
        minutes = _time_matrix(np.array([g.timestamp.timestamp() for g in group])) / 60.0
        distance = (hashes + 0.5 * minutes).astype(np.float64)
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=DUPLICATE_THRESHOLD,
            metric="precomputed",
            linkage="complete",
        ).fit_predict(distance)
        for label in set(labels):
            members = [indexes[k] for k in np.flatnonzero(labels == label)]
            if len(members) > 1:
                best = max(members, key=lambda m: (quality[m], -m))
                for member in members:
                    if member != best:
                        flagged[member] = True
    return flagged


def eligible(items: Sequence[Analyzed], quality: np.ndarray) -> list[int]:
    if not items:
        return []
    drop = duplicates(items, quality) | low_quality(quality)
    return [
        i
        for i, item in enumerate(items)
        if not drop[i] and item.metrics.entropy > MIN_ENTROPY
    ]


def _distance(items: Sequence[Analyzed]) -> np.ndarray:
    """Colour distance + time distance, each scaled to [0, 1]."""
    colour = _scale(_hamming(_bits([i.metrics.colorhash for i in items])))
    seconds = np.array([i.timestamp.timestamp() for i in items])
    return colour + _scale(_time_matrix(seconds))
```

- [ ] **Step 4: Run to verify pass**

Run: `hatch test tests/test_select_filters.py`
Expected: all pass. If `test_duplicates_keep_best…` fails because the 0.5·minutes term pushes the burst above threshold 10, check the timestamps (seconds apart → ≤0.15 minutes) rather than changing the constants.

- [ ] **Step 5: Commit**

```bash
git add src/mkgallery/select.py tests/test_select_filters.py
git commit -m "Add selection filters: quality, duplicates, entropy" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Clustering and ordering (`select.select_images`)

**Files:**
- Modify: `src/mkgallery/select.py` (append; add imports)
- Create: `tests/test_select.py`

**Interfaces:**
- Consumes: Task 7 functions.
- Produces: `select_images(items: Sequence[Analyzed], count: int, order: str = "chronological") -> list[Analyzed]`. `order` ∈ {`"chronological"`, `"varied"`}, else `ValueError`; `count < 1` → `ValueError`. Input order does not affect the result (items are sorted by path first). Returns ≤ `count` images; all eligible images if there are fewer.

- [ ] **Step 1: Write failing tests**

`tests/test_select.py`:
```python
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from mkgallery.model import Analyzed, Metrics
from mkgallery.select import select_images

START = datetime(2025, 1, 1, 12, 0, 0)


def bits(seed: int, n: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join("1" if b else "0" for b in rng.integers(0, 2, n))


def mk(i: int, *, days=None, quality=0.5, colour_seed=None, entropy=7.5) -> Analyzed:
    when = START + timedelta(days=i * 7 if days is None else days)
    seed = i if colour_seed is None else colour_seed
    return Analyzed(
        Path(f"img{i:03d}.jpg"),
        when,
        Metrics(quality, quality, entropy, bits(1000 + i, 64), bits(seed, 42)),
    )


def varied_items(n):
    # laplacian/contrast spread so quality is not constant
    return [mk(i, quality=0.3 + (i % 5) * 0.1) for i in range(n)]


def test_count_is_honoured_and_unique():
    picks = select_images(varied_items(30), 8)
    assert len(picks) == 8
    assert len({p.path for p in picks}) == 8


def test_fewer_eligible_than_count_returns_all_eligible():
    items = varied_items(5)
    assert len(select_images(items, 48)) == 5


def test_best_quality_wins_each_cluster():
    group_a = [mk(i, days=i, quality=0.1 * (i + 1), colour_seed=1) for i in range(3)]
    group_b = [mk(10 + i, days=300 + i, quality=0.1 * (i + 1), colour_seed=2) for i in range(3)]
    picks = select_images(group_a + group_b, 2)
    assert {p.path.name for p in picks} == {"img002.jpg", "img012.jpg"}


def test_chronological_is_default_and_sorted():
    picks = select_images(varied_items(30), 8)
    times = [p.timestamp for p in picks]
    assert times == sorted(times)


def test_varied_is_deterministic_and_same_set_as_chronological():
    items = varied_items(14)
    shuffled = items[:]
    random.Random(1).shuffle(shuffled)
    a = select_images(items, 8, order="varied")
    b = select_images(shuffled, 8, order="varied")
    assert [p.path for p in a] == [p.path for p in b]
    chrono = select_images(items, 8, order="chronological")
    assert {p.path for p in a} == {p.path for p in chrono}


def test_varied_best_image_is_second():
    items = varied_items(14)
    items[6] = mk(6, quality=10.0)  # clearly the best
    picks = select_images(items, 8, order="varied")
    assert picks[1].path.name == "img006.jpg"


def test_low_entropy_images_are_never_selected():
    items = varied_items(10) + [mk(50, entropy=2.0)]
    assert all(p.path.name != "img050.jpg" for p in select_images(items, 11))


def test_identical_images_collapse_without_nan_errors():
    same = [
        Analyzed(Path(f"s{i}.jpg"), START, Metrics(0.5, 0.5, 7.5, "0" * 64, "1" * 42))
        for i in range(10)
    ]
    assert len(select_images(same, 5)) == 1


def test_single_and_empty_inputs():
    assert select_images([], 5) == []
    one = [mk(0)]
    assert select_images(one, 5, order="varied") == one


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        select_images(varied_items(3), 0)
    with pytest.raises(ValueError):
        select_images(varied_items(3), 3, order="random")
```

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_select.py`
Expected: FAIL — `ImportError: cannot import name 'select_images'`.

- [ ] **Step 3: Implement** — add `from scipy.optimize import dual_annealing` to the imports, then append:

```python
_ORDERS = ("chronological", "varied")


def _cluster_best(
    items: Sequence[Analyzed], quality: np.ndarray, keep: list[int], count: int
) -> list[int]:
    """One image per cluster: the highest-quality one."""
    subset = [items[i] for i in keep]
    labels = AgglomerativeClustering(
        n_clusters=count, metric="precomputed", linkage="average"
    ).fit_predict(_distance(subset).astype(np.float64))
    picks = []
    for label in range(count):
        members = [keep[k] for k in np.flatnonzero(labels == label)]
        picks.append(max(members, key=lambda m: (quality[m], -m)))
    return picks


def _vary(picks: list[Analyzed], quality: dict[object, float]) -> list[Analyzed]:
    """Order so that neighbouring images are as dissimilar as possible.

    Same objective as mkmapdiary: minimise the summed *similarity* between
    consecutive images, then rotate the best image to the second position.
    """
    if len(picks) <= 2:
        return picks
    distance = _distance(picks)
    similarity = distance.max() - distance

    def tour_length(x: np.ndarray) -> float:
        order = np.argsort(x)
        return float(similarity[order, np.roll(order, -1)].sum())

    result = dual_annealing(tour_length, [(0, 1)] * len(picks), seed=42)
    arranged = [picks[i] for i in np.argsort(result.x)]
    best = max(range(len(arranged)), key=lambda i: quality[arranged[i].path])
    shift = (best - 1) % len(arranged)
    return arranged[shift:] + arranged[:shift]


def select_images(
    items: Sequence[Analyzed], count: int, order: str = "chronological"
) -> list[Analyzed]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if order not in _ORDERS:
        raise ValueError(f"order must be one of {_ORDERS}, got {order!r}")
    if not items:
        return []

    ordered = sorted(items, key=lambda i: str(i.path))
    quality = quality_scores(ordered)
    keep = eligible(ordered, quality)
    if len(keep) > count:
        keep = _cluster_best(ordered, quality, keep, count)
    picks = [ordered[i] for i in keep]

    if order == "chronological":
        return sorted(picks, key=lambda i: (i.timestamp, str(i.path)))
    quality_by_path = {ordered[i].path: float(quality[i]) for i in range(len(ordered))}
    return _vary(picks, quality_by_path)
```
(Annotate `quality: dict[Path, float]` and import `Path` if mypy complains about `object`.)

- [ ] **Step 4: Run to verify pass**

Run: `hatch test tests/test_select.py tests/test_select_filters.py`
Expected: all pass. `test_best_quality_wins_each_cluster` relies on the two groups being separated by colour hash *and* time; if clustering splits differently, print the labels before touching the algorithm.

- [ ] **Step 5: Commit**

```bash
git add src/mkgallery/select.py tests/test_select.py
git commit -m "Select images by clustering; chronological or varied order" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Bundle front-end libraries with licences

**Files:**
- Create: `src/mkgallery/vendor/jquery/{jquery.min.js,LICENSE.txt}`, `src/mkgallery/vendor/justifiedgallery/{jquery.justifiedGallery.min.js,justifiedGallery.min.css,LICENSE}`, `src/mkgallery/vendor/glightbox/{glightbox.min.js,glightbox.min.css,LICENSE.md}`, `src/mkgallery/vendor/VERSIONS.md`, `tests/test_vendor.py`

**Interfaces:**
- Produces: directory `src/mkgallery/vendor/` with exactly the three subdirectories above (`render.py` in Task 10 copies it to `<title>_assets/lib/`). Each subdirectory contains one file starting with `LICENSE`. File names are used verbatim by the page template: `lib/jquery/jquery.min.js`, `lib/justifiedgallery/jquery.justifiedGallery.min.js`, `lib/justifiedgallery/justifiedGallery.min.css`, `lib/glightbox/glightbox.min.js`, `lib/glightbox/glightbox.min.css`.

- [ ] **Step 1: Write the failing test**

`tests/test_vendor.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_vendor.py`
Expected: FAIL — vendor dir does not exist.

- [ ] **Step 3: Download the pinned files**

Needs network; if blocked by the sandbox, re-run with `allowed_domains`: `cdnjs.cloudflare.com`, `cdn.jsdelivr.net`, `raw.githubusercontent.com`, `data.jsdelivr.com`.

```bash
set -e
V=src/mkgallery/vendor
mkdir -p $V/jquery $V/justifiedgallery $V/glightbox

curl -fsSL https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js -o $V/jquery/jquery.min.js
curl -fsSL https://raw.githubusercontent.com/jquery/jquery/3.7.1/LICENSE.txt -o $V/jquery/LICENSE.txt

JG=https://cdn.jsdelivr.net/npm/justifiedGallery@3.8.1
curl -fsSL $JG/dist/js/jquery.justifiedGallery.min.js -o $V/justifiedgallery/jquery.justifiedGallery.min.js
curl -fsSL $JG/dist/css/justifiedGallery.min.css -o $V/justifiedgallery/justifiedGallery.min.css
curl -fsSL $JG/LICENSE -o $V/justifiedgallery/LICENSE

GL=https://cdn.jsdelivr.net/npm/glightbox@3.3.0
curl -fsSL $GL/dist/js/glightbox.min.js -o $V/glightbox/glightbox.min.js
curl -fsSL $GL/dist/css/glightbox.min.css -o $V/glightbox/glightbox.min.css
curl -fsSL $GL/LICENSE.md -o $V/glightbox/LICENSE.md
```
If a URL 404s (wrong version or licence filename), list the package files with `curl -fsSL "https://data.jsdelivr.com/v1/packages/npm/<pkg>@<version>?structure=flat"`, use the nearest published version/file, and record what you used in `VERSIONS.md`. Do not fetch licence text from anywhere except the library's own package or repository.

- [ ] **Step 4: Write `VERSIONS.md` and check banners**

`src/mkgallery/vendor/VERSIONS.md` — one section per library: name, pinned version actually downloaded, download URLs, licence (MIT), upstream homepage (`https://jquery.com`, `https://github.com/miromannino/Justified-Gallery`, `https://github.com/biati-digital/glightbox`). Add a line: "Files are unmodified except where noted." If `glightbox.min.js` or any other file has no copyright banner (test 2 fails), prepend a one-line comment `/*! GLightbox <version> | MIT License | https://github.com/biati-digital/glightbox */` (using the real copyright holder named in its LICENSE.md) and note the prepended line in `VERSIONS.md`.

- [ ] **Step 5: Run to verify pass**

Run: `hatch test tests/test_vendor.py`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/mkgallery/vendor tests/test_vendor.py
git commit -m "Bundle jQuery, Justified Gallery and GLightbox with licences" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Rendering (`render.py` + page template)

**Files:**
- Create: `src/mkgallery/render.py`, `src/mkgallery/templates/page.html.j2`, `tests/test_render.py`

**Interfaces:**
- Consumes: `Analyzed`; `load_full` from `imaging`; vendor dir from Task 9.
- Produces: `safe_stem(title: str) -> str`; `render(selected: Sequence[Analyzed], title: str, output_dir: Path) -> Path` (returns the HTML path; `output_dir` must exist); `LIBRARIES: dict[str, tuple[str, str]]` mapping vendor dir name → (display name, URL).

- [ ] **Step 1: Write failing tests**

`tests/test_render.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_render.py`
Expected: FAIL — `ModuleNotFoundError: mkgallery.render`.

- [ ] **Step 3: Implement `render.py`**

```python
"""Convert the selected images and write the gallery page."""

import hashlib
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, PackageLoader
from PIL import Image

from .imaging import load_full
from .model import Analyzed

THUMB_EDGE = 600
FULL_EDGE = 1600
VENDOR_DIR = Path(__file__).parent / "vendor"

LIBRARIES: dict[str, tuple[str, str]] = {
    "justifiedgallery": ("Justified Gallery", "https://github.com/miromannino/Justified-Gallery"),
    "glightbox": ("GLightbox", "https://github.com/biati-digital/glightbox"),
    "jquery": ("jQuery", "https://jquery.com"),
}

_UNSAFE_STEM = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class Photo:
    full: str
    thumb: str
    width: int
    height: int
    caption: str


def safe_stem(title: str) -> str:
    stem = _UNSAFE_STEM.sub("_", title).strip().lstrip(".")
    return stem or "gallery"


def _asset_name(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:6]
    return f"{_UNSAFE_NAME.sub('_', path.stem)}-{digest}.jpg"


def _caption(when: datetime) -> str:
    return f"{when.day} {when:%B %Y}"


def _summary(selected: Sequence[Analyzed]) -> str:
    count = len(selected)
    text = f"{count} photo{'' if count == 1 else 's'}"
    first = min(i.timestamp for i in selected)
    last = max(i.timestamp for i in selected)
    if (first.year, first.month) == (last.year, last.month):
        return f"{text} · {first:%B %Y}"
    if first.year == last.year:
        return f"{text} · {first:%B} – {last:%B %Y}"
    return f"{text} · {first:%b %Y} – {last:%b %Y}"


def _reset(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)


def render(selected: Sequence[Analyzed], title: str, output_dir: Path) -> Path:
    stem = safe_stem(title)
    assets = output_dir / f"{stem}_assets"
    _reset(assets / "thumbs")
    _reset(assets / "full")
    shutil.rmtree(assets / "lib", ignore_errors=True)
    shutil.copytree(VENDOR_DIR, assets / "lib", ignore=shutil.ignore_patterns("VERSIONS.md"))

    photos = []
    for item in selected:
        name = _asset_name(item.path)
        with load_full(item.path) as img:
            full = img.copy()
        full.thumbnail((FULL_EDGE, FULL_EDGE), Image.Resampling.LANCZOS)
        full.save(assets / "full" / name, "JPEG", quality=88, optimize=True, progressive=True)
        thumb = full.copy()
        thumb.thumbnail((THUMB_EDGE, THUMB_EDGE), Image.Resampling.LANCZOS)
        thumb.save(assets / "thumbs" / name, "JPEG", quality=82, optimize=True, progressive=True)
        photos.append(
            Photo(
                full=f"full/{name}",
                thumb=f"thumbs/{name}",
                width=thumb.width,
                height=thumb.height,
                caption=_caption(item.timestamp),
            )
        )

    env = Environment(loader=PackageLoader("mkgallery", "templates"), autoescape=True)
    html = env.get_template("page.html.j2").render(
        title=title,
        summary=_summary(selected),
        assets_url=quote(f"{stem}_assets"),
        photos=photos,
        libraries=list(LIBRARIES.values()),
    )
    page = output_dir / f"{stem}.html"
    page.write_text(html, encoding="utf-8")
    return page
```
(`VERSIONS.md` is excluded from the copy; the licence files and banners travel with the libraries. The test `test_footer_credits_every_bundled_library` compares `LIBRARIES` against vendor subdirectories, so `VERSIONS.md` being a file there is fine.)

- [ ] **Step 4: Write the template `src/mkgallery/templates/page.html.j2`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<!--
  Bundled libraries (all MIT licensed; licence texts are in {{ assets_url }}/lib/<name>/):
  jQuery, Justified Gallery (https://github.com/miromannino/Justified-Gallery),
  GLightbox (https://github.com/biati-digital/glightbox).
-->
<link rel="stylesheet" href="{{ assets_url }}/lib/justifiedgallery/justifiedGallery.min.css">
<link rel="stylesheet" href="{{ assets_url }}/lib/glightbox/glightbox.min.css">
<style>
  :root {
    --bg: #f7f5f1; --fg: #1c1b19; --muted: #8a857c; --rule: #ddd8ce; --accent: #b4562f;
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #131312; --fg: #ece9e2; --muted: #8d887e; --rule: #2c2b28; --accent: #d9825b; }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .page {
    max-width: 1760px; margin: 0 auto;
    padding: clamp(1.5rem, 5vw, 4.5rem) clamp(1rem, 3.5vw, 3rem) 2.5rem;
  }
  header { margin-bottom: clamp(1.5rem, 4vw, 3rem); }
  h1 {
    margin: 0; font-weight: 300; letter-spacing: -0.02em; line-height: 1.02;
    font-size: clamp(2.4rem, 7vw, 5.5rem); overflow-wrap: anywhere;
    font-family: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  }
  .summary {
    margin: 1rem 0 0; color: var(--muted); font-size: .8rem;
    letter-spacing: .14em; text-transform: uppercase;
  }
  .summary::before {
    content: ""; display: block; width: 3rem; height: 2px;
    background: var(--accent); margin-bottom: 1rem;
  }
  #gallery a { display: block; border-radius: 3px; background: var(--rule); }
  #gallery img { transition: transform .6s cubic-bezier(.2, .6, .2, 1); }
  #gallery a:hover img { transform: scale(1.035); }
  footer {
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--rule);
    color: var(--muted); font-size: .75rem;
  }
  footer a { color: inherit; text-decoration-color: var(--rule); text-underline-offset: 3px; }
  footer a:hover { color: var(--fg); }
  @media (prefers-reduced-motion: reduce) { #gallery img { transition: none; } }
</style>
</head>
<body>
<div class="page">
  <header>
    <h1>{{ title }}</h1>
    <p class="summary">{{ summary }}</p>
  </header>
  <main id="gallery">
{%- for photo in photos %}
    <a class="glightbox" href="{{ assets_url }}/{{ photo.full }}" data-description="{{ photo.caption }}"><img src="{{ assets_url }}/{{ photo.thumb }}" width="{{ photo.width }}" height="{{ photo.height }}" alt="Photo from {{ photo.caption }}"></a>
{%- endfor %}
  </main>
  <footer>
    Made with mkgallery · Built with
    {% for name, url in libraries -%}
      <a href="{{ url }}">{{ name }}</a>{{ ", " if not loop.last else "" }}
    {%- endfor %}
  </footer>
</div>
<script src="{{ assets_url }}/lib/jquery/jquery.min.js"></script>
<script src="{{ assets_url }}/lib/justifiedgallery/jquery.justifiedGallery.min.js"></script>
<script src="{{ assets_url }}/lib/glightbox/glightbox.min.js"></script>
<script>
  $("#gallery").justifiedGallery({
    rowHeight: 220, maxRowHeight: 360, margins: 6, border: 0, lastRow: "center"
  });
  GLightbox({ selector: ".glightbox", loop: true });
</script>
</body>
</html>
```
`test_single_photo_wording` asserts the summary text `1 photo ` or `1 photo<`; the summary line renders as `1 photo · March 2025`, which contains `1 photo `.

- [ ] **Step 5: Run to verify pass**

Run: `hatch test tests/test_render.py`
Expected: all pass. Then open a generated page once to look at it: render a few synthetic images with a throwaway script via `hatch run python -c …` into `$TMPDIR`, and open the HTML in a browser (or ask the user to). The design brief is "not a 90s homepage" — check spacing, title typography, dark mode, hover, phone width (narrow the window), and the lightbox. Adjust only the `<style>` block if something looks off; do not change structure the tests rely on.

- [ ] **Step 6: Commit**

```bash
git add src/mkgallery/render.py src/mkgallery/templates/page.html.j2 tests/test_render.py
git commit -m "Render gallery page with bundled Justified Gallery and GLightbox" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Build pipeline and CLI (`build.py`, `cli.py`)

**Files:**
- Create: `src/mkgallery/build.py`, `src/mkgallery/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `scan`, `Cache`, `analyze`, `select_images`, `render`.
- Produces:
  ```python
  class GalleryError(Exception): ...

  def build_gallery(src: Path, title: str, *, count: int = 48,
                    output_dir: Path = Path("."), cache_dir: Path | None = None,
                    jobs: int = 1, order: str = "chronological",
                    log: Callable[[str], None] = ..., 
                    progress: Callable[[int, int], None] | None = None) -> Path
  ```
  `cache_dir=None` means no persistent cache. Raises `GalleryError` when: output dir or cache dir equals/lies inside SRC; `exiftool` is not on PATH; no supported images; nothing readable; nothing passes the filters. Guards run before anything is written.
- CLI (`cli.py`, entry point `mkgallery`): `main` click command: `SRC` argument; `--title/-t` (required); `--count/-n` (default 48, ≥1); `--output-dir/-o` (default `.`); `--cache-dir` (default `platformdirs.user_cache_dir("mkgallery")`); `--no-cache`; `--jobs/-j` (default CPU count); `--order` (`chronological`|`varied`).

- [ ] **Step 1: Write failing tests**

`tests/test_cli.py`:
```python
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
    result = _run(src, "-t", "v", "-n", 4, "-o", tmp_path / "o", "--no-cache", "-j", 1, "--order", "varied")
    assert result.exit_code == 0, result.output


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
```

- [ ] **Step 2: Run to verify failure**

Run: `hatch test tests/test_cli.py`
Expected: FAIL — `ModuleNotFoundError: mkgallery.cli`.

- [ ] **Step 3: Implement `build.py`**

```python
"""Wire scan → analyse → select → render, with the safety guards."""

import shutil
from collections.abc import Callable
from pathlib import Path

from .analyze import analyze
from .cache import Cache
from .render import render
from .scan import scan
from .select import select_images


class GalleryError(Exception):
    """A problem the user can fix; reported without a traceback."""


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def build_gallery(
    src: Path,
    title: str,
    *,
    count: int = 48,
    output_dir: Path = Path("."),
    cache_dir: Path | None = None,
    jobs: int = 1,
    order: str = "chronological",
    log: Callable[[str], None] = lambda _message: None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    src = src.resolve()
    output_dir = output_dir.resolve()
    if _inside(output_dir, src):
        raise GalleryError(
            f"Output directory {output_dir} is inside the source directory {src}; "
            "nothing is ever written there."
        )
    if cache_dir is not None and _inside(cache_dir.resolve(), src):
        raise GalleryError(
            f"Cache directory {cache_dir} is inside the source directory {src}; "
            "nothing is ever written there."
        )
    if shutil.which("exiftool") is None:
        raise GalleryError("exiftool was not found on PATH; please install it.")

    paths = scan(src)
    if not paths:
        raise GalleryError(f"No supported images found below {src}")
    log(f"Found {len(paths)} images")

    with Cache(cache_dir) as cache:
        result = analyze(paths, cache, jobs, progress)
    for path, message in result.failures:
        log(f"Skipped {path}: {message}")
    if not result.items:
        raise GalleryError("None of the images could be read.")

    picks = select_images(result.items, count, order)
    if not picks:
        raise GalleryError("No image passed the quality filters.")
    log(f"Selected {len(picks)} of {len(result.items)} images")

    output_dir.mkdir(parents=True, exist_ok=True)
    return render(picks, title, output_dir)
```

- [ ] **Step 4: Implement `cli.py`**

```python
import os
import sys
from pathlib import Path

import click
import platformdirs

from .build import GalleryError, build_gallery


def _progress(done: int, total: int) -> None:
    if sys.stderr.isatty():
        click.echo(f"\rAnalysing {done}/{total}", nl=done == total, err=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("src", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--title", "-t", required=True, help="Page title; also names the output files.")
@click.option("--count", "-n", default=48, show_default=True, type=click.IntRange(min=1),
              help="Number of images to show.")
@click.option("--output-dir", "-o", default=Path("."), show_default=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Analysis cache (default: the user cache directory).")
@click.option("--no-cache", is_flag=True, help="Do not read or write the analysis cache.")
@click.option("--jobs", "-j", type=click.IntRange(min=1), default=None,
              help="Parallel workers (default: CPU count).")
@click.option("--order", type=click.Choice(["chronological", "varied"]), default="chronological",
              show_default=True,
              help="chronological: by capture time; varied: neighbours look as different as possible.")
def main(src, title, count, output_dir, cache_dir, no_cache, jobs, order):
    """Create TITLE.html and TITLE_assets/ summarising the photos below SRC."""
    if no_cache:
        cache = None
    else:
        cache = cache_dir or Path(platformdirs.user_cache_dir("mkgallery"))
    try:
        page = build_gallery(
            src,
            title,
            count=count,
            output_dir=output_dir,
            cache_dir=cache,
            jobs=jobs or os.cpu_count() or 1,
            order=order,
            log=lambda message: click.echo(message, err=True),
            progress=_progress,
        )
    except GalleryError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote {page}")
```

- [ ] **Step 5: Run to verify pass**

Run: `hatch test`
Expected: the whole suite passes. Then `hatch fmt` and `hatch run types:check`; fix everything they report (including in earlier files — don't leave findings behind because they came from an earlier task).

- [ ] **Step 6: Commit**

```bash
git add src/mkgallery/build.py src/mkgallery/cli.py tests/test_cli.py
git commit -m "Add build pipeline and mkgallery CLI" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
(Also `git add` any files `hatch fmt` / type fixes touched, by explicit path.)

---

### Task 12: README and real-collection check

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

Sections: what it does (one paragraph + the output layout); install (`hatch`/`pipx install .`, requires system `exiftool`); usage with the CLI options table; how selection works (filters → clustering → best per cluster → order; `varied` explained as "neighbours are as dissimilar as possible"); cache location and `--no-cache`; the guarantee that SRC is never written; **Credits** listing Justified Gallery, GLightbox, jQuery (MIT, links, licence files shipped in `<title>_assets/lib/<name>/`), and "selection approach ported from mkmapdiary".

- [ ] **Step 2: Manual check on real data (needs the user)**

Ask the user for a real folder with CR2 files (a small one first). Run
`hatch run mkgallery <folder> --title "Check" -o $TMPDIR/check --cache-dir $TMPDIR/cache`
Verify: CR2 files appear in the result; the second run reports no re-analysis (much faster); the original folder is unchanged (`find <folder> -newer <some earlier file>` / compare `ls -lR` before and after); the page looks right at desktop and phone width and in dark mode. Report the timing for N images. Tune nothing silently — report anything surprising (e.g. too many images dropped by the entropy/quality filters) to the user.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
