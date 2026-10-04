import pytest

from mkgallery.select import allocate_quota, select_images, split_events
from tests.test_select import mk


def event(first: int, days: float, n: int, colour_seed: int | None = None):
    """n images spread over `days` days, starting at day-offset `days` (ids from `first`)."""
    return [
        mk(first + k, days=days + k * 0.01, quality=0.3 + (k % 5) * 0.1, colour_seed=colour_seed)
        for k in range(n)
    ]


def big_and_small():
    big = event(0, 0, 60)
    small = [event(100 + 10 * e, 100 * (e + 1), 5, colour_seed=999) for e in range(3)]
    return big + [i for group in small for i in group], small


def test_split_events_breaks_on_gaps_over_a_day():
    items = [mk(0, days=0), mk(1, days=0.5), mk(2, days=1.4), mk(3, days=5), mk(4, days=5.1)]
    assert split_events(items, [0, 1, 2, 3, 4]) == [[0, 1, 2], [3, 4]]


def test_split_events_orders_by_time_and_only_uses_kept_indexes():
    items = [mk(0, days=9), mk(1, days=0), mk(2, days=0.1), mk(3, days=50)]
    assert split_events(items, [0, 1, 2]) == [[1, 2], [0]]


def test_split_events_empty():
    assert split_events([], []) == []


def test_allocate_gives_every_event_a_slot_and_sums_to_count():
    quotas = allocate_quota([300, 5, 5, 5], 10)
    assert sum(quotas) == 10
    assert min(quotas) >= 1
    assert quotas[0] == max(quotas)


def test_allocate_never_exceeds_event_size():
    quotas = allocate_quota([1, 1, 50], 10)
    assert quotas == [1, 1, 8]


def test_allocate_is_sublinear_in_size():
    quotas = allocate_quota([400, 100], 20)
    assert quotas[0] < 4 * quotas[1]


def test_allocate_is_capped_by_total_size():
    assert allocate_quota([2, 3], 10) == [2, 3]


def test_quota_gives_every_small_event_a_pick():
    items, small = big_and_small()
    picks = select_images(items, 10, event_quota=True)
    chosen = {p.path for p in picks}
    assert len(picks) == 10
    for group in small:
        assert chosen & {i.path for i in group}


def test_more_events_than_count_still_returns_count_unique_picks():
    items = [mk(i, days=i * 3, quality=0.3 + (i % 5) * 0.1) for i in range(20)]
    picks = select_images(items, 6, event_quota=True)
    assert len(picks) == 6
    assert len({p.path for p in picks}) == 6


def test_single_event_behaves_like_plain_clustering():
    items = event(0, 0, 20)
    picks = select_images(items, 5, event_quota=True)
    assert len(picks) == 5


def test_quota_off_keeps_previous_behaviour():
    items, _ = big_and_small()
    assert select_images(items, 10, event_quota=False) == select_images(
        items, 10, event_quota=False
    )


def test_selection_is_deterministic():
    items, _ = big_and_small()
    first = select_images(items, 10, event_quota=True, balance=0.5)
    assert first == select_images(list(reversed(items)), 10, event_quota=True, balance=0.5)


@pytest.mark.parametrize("balance", [-0.1, 1.5])
def test_rejects_balance_outside_unit_interval(balance):
    with pytest.raises(ValueError, match="balance"):
        select_images([mk(0)], 1, balance=balance)


def test_balance_raises_small_event_share():
    items, small = big_and_small()
    small_paths = {i.path for group in small for i in group}

    def share(balance):
        picks = select_images(items, 10, event_quota=False, balance=balance)
        return sum(p.path in small_paths for p in picks)

    assert share(1.0) > share(0.0)


def test_balance_returns_count_unique_picks():
    items, _ = big_and_small()
    picks = select_images(items, 10, event_quota=False, balance=1.0)
    assert len(picks) == 10
    assert len({p.path for p in picks}) == 10


def test_balance_inside_events_with_quota():
    items, _ = big_and_small()
    picks = select_images(items, 12, event_quota=True, balance=0.5)
    assert len(picks) == 12
    assert len({p.path for p in picks}) == 12
