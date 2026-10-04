# mkgallery

Summarise a large photo collection — say, a year — as **one static page**: a title and a
gallery of the most interesting pictures, picked automatically.

```
mkgallery ~/Photos/2025 --title "2025"
```

writes `2025.html` and a `2025_assets/` directory next to it. Nothing in the source
directory is touched, and no configuration is read from it.

## Requirements

- Python ≥ 3.10
- [`exiftool`](https://exiftool.org/) on `PATH`
- Formats: JPEG, PNG, WebP, TIFF and RAW (CR2, CR3, NEF, ARW, DNG)

## Install

```
pipx install .        # or: hatch run mkgallery …
```

## Usage

```
mkgallery SRC --title TITLE [options]
```

| Option | Default | |
|---|---|---|
| `-t, --title` | required | Page title; also names `TITLE.html` / `TITLE_assets/` |
| `-n, --count` | 48 | Number of images shown |
| `-o, --output-dir` | `.` | Where to write the output (must not be inside `SRC`) |
| `--order` | `chronological` | `chronological`, or `varied` (neighbours are as dissimilar as possible) |
| `--cache-dir` | user cache dir | Analysis cache (must not be inside `SRC`) |
| `--no-cache` | | Do not read or write the cache |
| `-j, --jobs` | CPU count | Parallel workers |

## How images are chosen

The selection is ported from [mkmapdiary](https://github.com/bytehexe/mkmapdiary)'s front page:

1. Each image is analysed from a small preview (for RAW files the embedded JPEG — nothing
   is developed at this stage): sharpness, contrast, entropy, perceptual and colour hashes.
   Results are cached by path, size and modification time.
2. Near-duplicates (per day), unusually soft images and flat, low-entropy images are dropped.
3. The rest is clustered by colour and capture time into `--count` groups; the sharpest,
   best-contrast image of each group is chosen. The picks therefore spread over the whole
   year and over different looks.
4. Only the chosen images are converted (RAW files are developed here) into a thumbnail and
   a 1600 px version.

## Credits and licences

mkgallery is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.txt).

The generated page bundles these libraries (all MIT licensed). Their licence texts and
copyright notices are shipped in `TITLE_assets/lib/<name>/` and in
`src/mkgallery/vendor/`:

- [jQuery](https://jquery.com)
- [Justified Gallery](https://github.com/miromannino/Justified-Gallery) — the row layout
- [GLightbox](https://github.com/biati-digital/glightbox) — the lightbox

The selection approach and metric code are adapted from mkmapdiary (same author).
