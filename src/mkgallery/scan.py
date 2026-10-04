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
