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
