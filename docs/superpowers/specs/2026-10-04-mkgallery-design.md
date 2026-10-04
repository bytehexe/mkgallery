# mkgallery — design

## Goal

`mkgallery SRC --title "2025"` turns a large photo collection (≈5000 images/year,
mixed formats incl. CR2) into one static page: a title and a gallery of ~48
automatically chosen, visually varied, good-quality images.

Output: `<title>.html` + `<title>_assets/`. Nothing is ever written to SRC; no
config is read from SRC. Selection idea is ported from `../mkmapdiary`
(`lib/highlights.py`), without geo/map/pins/mkdocs/doit.

## Non-goals (YAGNI)

- Performance work for >5000 images (no time-slice pre-reduction etc.).
- Geo data, maps, per-day pages, pinned images, config files, timezone calibration.
- Video/audio.

## CLI

```
mkgallery SRC --title TITLE [--count 48] [--output-dir .] [--cache-dir DIR]
          [--no-cache] [-j N] [--order chronological|varied]
```

- `--order` defaults to `chronological` (the story of the year); `varied` uses the
  mkmapdiary annealing order (neighbours are visually dissimilar).

- Errors if `--output-dir` is inside SRC (or equals it).
- `TITLE` is used verbatim in the page (HTML-escaped) and, with path separators and
  control characters replaced, as the file/dir stem.
- Re-running overwrites `<title>.html` and replaces `<title>_assets/` contents.

## Pipeline

1. **Scan** (`scan.py`): recursive, read-only walk of SRC. Extensions: jpg, jpeg,
   png, webp, tif, tiff; RAW via rawpy: cr2, cr3, nef, arw, dng. Case-insensitive.
2. **Analyse** (`analyze.py`), process pool (`-j`, default CPU count):
   - Small decode only: JPEG via `Image.draft`; RAW via `rawpy.extract_thumb()`
     (falls back to a half-size `postprocess` if no preview). Result ≤1024px.
   - Metrics, ported from mkmapdiary: quality = 0.5·norm(Laplacian var) +
     0.5·norm(contrast) (normalised across the whole set); entropy; `colorhash`;
     `whash`.
   - Timestamp + orientation: one batched exiftool process over all files
     (`DateTimeOriginal`/`CreateDate`, `Orientation`); fallback file mtime.
     Timestamps are naive-local and only used for relative distances and per-day
     duplicate grouping.
3. **Cache** (`cache.py`): per-image raw metrics (not normalised scores) in a
   SQLite/msgpack file under `platformdirs.user_cache_dir("mkgallery")`, keyed by
   (absolute path, size, mtime_ns). Normalisation and selection are recomputed each
   run, so cache entries stay valid when the set changes.
4. **Select** (`select.py`, ported from `Highlights`, non-geo path):
   - Drop duplicates (per day: whash + time, complete-linkage threshold 10, keep best
     quality), bad quality (< mean − k·σ, k = 2, as in mkmapdiary's `defaults.yaml`), entropy ≤ 6.5.
   - Agglomerative (average linkage, precomputed) clustering into `count` clusters on
     colour-hash distance + time distance (each min-max normalised, summed); best
     quality per cluster. Hash distances vectorised with numpy.
   - If fewer than `count` candidates remain, take all.
   - Order: `chronological` (default) sorts the picks by timestamp. `varied` uses
     `dual_annealing` (seed 42) so neighbouring images are visually dissimilar (the tour minimises neighbour similarity), best image
     rotated to second position.
5. **Render** (`render.py`): for each selected image only, convert from the *original*
   (RAW developed here, not before) with orientation applied:
   `<title>_assets/thumbs/<name>.jpg` (≈600px long edge, q≈82) and
   `<title>_assets/full/<name>.jpg` (≈1600px long edge, q≈88). `<name>` =
   stem + 6-char hash of the source path (unique, stable). Then write the HTML.

## Page

Single HTML file (Jinja2 template `templates/page.html.j2`), inline CSS, no framework.

- Libraries bundled in `<title>_assets/lib/`: jQuery, Justified Gallery (JS+CSS),
  GLightbox (JS+CSS). Pinned versions, vendored in the package under
  `src/mkgallery/vendor/` with their original licence texts.
- Markup: `<h1>` title, `<main id="gallery">` with
  `<a class="glightbox" href="full/…"><img src="thumbs/…" alt="" loading="lazy" width height></a>`
  per image (width/height from the thumbnail, which Justified Gallery needs and which
  avoids layout shift). GLightbox caption: capture date.
- Justified Gallery handles mixed aspect ratios (rowHeight 220, maxRowHeight 360,
  margins 6, lastRow `center`).
- **Look**: not a "90s homepage". Generous whitespace; max-width container (~1700px)
  with fluid side padding; large, light-weight title in a system serif/sans stack
  (`ui-serif`/`Iowan Old Style`/Georgia fallbacks, tight tracking) with a thin accent
  rule or small photo-count line beneath; neutral off-white background and near-black
  text, with a `prefers-color-scheme: dark` variant (deep neutral background);
  thumbnails with a 3px radius, soft fade-in on load, subtle hover dim/scale;
  GLightbox themed to match; tiny muted footer. Responsive down to phone width.
- **Attribution**: licence texts in `<title>_assets/lib/licenses/` (jQuery, Justified
  Gallery, GLightbox; all MIT), original copyright banners left intact in the bundled
  files, an HTML comment listing them, and a muted footer line "Built with
  mkgallery · Justified Gallery · GLightbox · jQuery" linking to each project.
  A test asserts every bundled library has a licence file and footer link.

## Layout

```
src/mkgallery/{cli,scan,analyze,cache,select,render}.py
src/mkgallery/templates/page.html.j2
src/mkgallery/vendor/…            # bundled JS/CSS + licences
tests/
pyproject.toml                    # hatch (test, fmt, types), hatchling
```

Dependencies: click, Pillow, rawpy, imagehash, numpy, scipy, scikit-learn, pyexiftool,
platformdirs, Jinja2. Requires `exiftool` on PATH (clear error if missing).

## Testing

Generated synthetic images (varied colours, noise for quality, blur, exact duplicates,
EXIF dates written via Pillow). Cover: scan filtering; SRC untouched (hash tree before
and after); output-inside-SRC rejected; cache hit on second run (analysis not
re-invoked) and invalidation on mtime change; selection (duplicates/blur/low-entropy
excluded, count honoured, < count handled, deterministic order, both `--order` modes); render outputs and
HTML (escaped title, one entry per image, attribution present). RAW path tested with
a mocked `rawpy`; a real CR2 is exercised manually.

## Risks / open items

- Bundled library versions are fetched once at implementation time and committed.
- The quality threshold (k = 2) and the entropy cut-off (6.5) are taken from mkmapdiary and may need
  tuning on the real collection.
