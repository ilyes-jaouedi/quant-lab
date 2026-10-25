import pandas as pd
import pandas.testing as pdt

from fin.data import load_bars, load_market_data


def test_first_load_fetches_and_caches(provider, cache):
    bars = load_bars("X", "2020-01-01", "2020-06-30", provider=provider, cache=cache)
    assert bars.index[0] == pd.Timestamp("2020-01-01")
    assert bars.index[-1] == pd.Timestamp("2020-06-30")
    assert len(provider.calls) == 1

    again = load_bars("X", "2020-01-01", "2020-06-30", provider=provider, cache=cache)
    assert len(provider.calls) == 1, "second load must come from cache"
    pdt.assert_frame_equal(bars, again, check_freq=False)


def test_subrange_is_served_from_cache(provider, cache):
    load_bars("X", "2020-01-01", "2020-06-30", provider=provider, cache=cache)
    sub = load_bars("X", "2020-03-01", "2020-03-31", provider=provider, cache=cache)
    assert len(provider.calls) == 1
    assert sub.index.min() >= pd.Timestamp("2020-03-01")
    assert sub.index.max() <= pd.Timestamp("2020-03-31")


def test_extending_end_fetches_only_the_tail(provider, cache):
    load_bars("X", "2020-01-01", "2020-06-30", provider=provider, cache=cache)
    bars = load_bars("X", "2020-01-01", "2020-12-31", provider=provider, cache=cache)

    assert len(provider.calls) == 2
    _, tail_start, _ = provider.calls[1]
    assert tail_start > pd.Timestamp("2020-06-01"), "should not re-download the whole series"
    expected = provider.history.loc["2020-01-01":"2020-12-31"]
    pdt.assert_frame_equal(bars, expected, check_freq=False)


def test_extending_start_refetches(provider, cache):
    load_bars("X", "2020-06-01", "2020-12-31", provider=provider, cache=cache)
    bars = load_bars("X", "2020-01-01", "2020-12-31", provider=provider, cache=cache)
    assert len(provider.calls) == 2
    assert bars.index[0] == pd.Timestamp("2020-01-01")


def test_start_before_inception_is_not_refetched(provider, cache):
    # The fake series starts in 2020; asking from 2000 must not trigger a download every time.
    load_bars("X", "2000-01-01", "2020-12-31", provider=provider, cache=cache)
    load_bars("X", "2000-01-01", "2020-12-31", provider=provider, cache=cache)
    assert len(provider.calls) == 1


def test_revised_history_triggers_full_refetch(provider, cache):
    load_bars("X", "2020-01-01", "2020-06-30", provider=provider, cache=cache)
    # A dividend re-adjusts all past adjusted prices.
    provider.history["adj_close"] *= 0.98
    bars = load_bars("X", "2020-01-01", "2020-12-31", provider=provider, cache=cache)

    assert len(provider.calls) == 3  # initial, tail (detects revision), full refetch
    expected = provider.history.loc["2020-01-01":"2020-12-31"]
    pdt.assert_frame_equal(bars, expected, check_freq=False)


def test_refresh_ignores_cache(provider, cache):
    load_bars("X", "2020-01-01", "2020-06-30", provider=provider, cache=cache)
    load_bars("X", "2020-01-01", "2020-06-30", provider=provider, cache=cache, refresh=True)
    assert len(provider.calls) == 2


def test_load_market_data_aligns_tickers(provider, cache):
    md = load_market_data(["A", "B"], "2020-01-01", "2020-03-31", provider=provider, cache=cache)
    assert md.tickers == ["A", "B"]
    assert md.close.shape[1] == 2
