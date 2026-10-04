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
