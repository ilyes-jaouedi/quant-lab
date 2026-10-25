import numpy as np
import pandas as pd
import pytest

from fin.data import BarCache, DataProvider, normalize_bars


class FakeProvider(DataProvider):
    """Serves slices of a synthetic series and records every request."""

    name = "fake"

    def __init__(self, start="2020-01-01", end="2020-12-31", seed=0):
        index = pd.bdate_range(start, end, name="date")
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
        self.history = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close,
                "volume": 1e6,
            },
            index=index,
        )
        self.calls: list[tuple] = []

    def fetch(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        return normalize_bars(self.history.loc[start:end])


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def cache(tmp_path):
    return BarCache(tmp_path)
