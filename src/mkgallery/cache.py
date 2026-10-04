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
