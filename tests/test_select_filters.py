from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from mkgallery.model import Analyzed, Metrics
from mkgallery.select import duplicates, eligible, low_quality, quality_scores


def bits(seed: int, n: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join("1" if b else "0" for b in rng.integers(0, 2, n))


def mk(name, when, *, lap=0.5, con=0.5, entropy=7.5, w=None, c=None) -> Analyzed:
    seed = sum(map(ord, name))
    return Analyzed(
        Path(name),
        when,
        Metrics(lap, con, entropy, w or bits(seed, 64), c or bits(seed + 1, 42)),
    )


T0 = datetime(2025, 6, 1, 12, 0, 0)


def test_quality_is_normalised_and_nan_free():
    items = [
        mk("a", T0, lap=0.0, con=0.0),
        mk("b", T0, lap=1.0, con=1.0),
        mk("c", T0, lap=0.5, con=0.5),
    ]
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
