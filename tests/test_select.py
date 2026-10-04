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


def test_varied_puts_dissimilar_images_next_to_each_other():
    reds = [mk(i, days=i, colour_seed=1, quality=0.3 + 0.05 * i) for i in range(4)]
    blues = [mk(10 + i, days=100 + i, colour_seed=2, quality=0.3 + 0.05 * i) for i in range(4)]
    group = {p.path: p.path.name < "img010" for p in reds + blues}

    def same_neighbours(picks):
        return sum(group[a.path] == group[b.path] for a, b in zip(picks, picks[1:], strict=False))

    chrono = select_images(reds + blues, 8, order="chronological")
    varied = select_images(reds + blues, 8, order="varied")
    assert same_neighbours(chrono) == 6
    assert same_neighbours(varied) <= 1
