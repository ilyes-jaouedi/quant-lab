import numpy as np
import pandas as pd
import pytest

from fin.data import MarketData


def _bars(dates, prices):
    idx = pd.DatetimeIndex(dates, name="date")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"open": p, "high": p, "low": p, "close": p, "adj_close": p, "volume": 1.0}, index=idx
    )


@pytest.fixture
def md():
    # B does not trade on 2020-01-02 (different exchange calendar).
    a = _bars(["2020-01-01", "2020-01-02", "2020-01-03"], [100, 110, 99])
    b = _bars(["2020-01-01", "2020-01-03"], [50, 55])
    return MarketData.from_bars({"A": a, "B": b})


def test_alignment_keeps_gaps_as_nan(md):
    assert len(md.index) == 3
    assert np.isnan(md.close.loc["2020-01-02", "B"])


def test_until_has_no_future_data(md):
    view = md.until("2020-01-02")
    assert view.index.max() == pd.Timestamp("2020-01-02")
    for field in ("open", "close", "adj_close", "volume"):
        assert getattr(view, field).index.max() <= pd.Timestamp("2020-01-02")


def test_returns_use_each_tickers_own_trading_days(md):
    r = md.returns(log=False)
    assert r.loc["2020-01-02", "A"] == pytest.approx(0.10)
    assert r.loc["2020-01-03", "A"] == pytest.approx(-0.10)
    # B's return over the gap is attributed to the day it traded again, not lost.
    assert r.loc["2020-01-03", "B"] == pytest.approx(0.10)
    assert np.isnan(r.loc["2020-01-02", "B"])


def test_log_returns(md):
    r = md.returns(log=True)
    assert r.loc["2020-01-02", "A"] == pytest.approx(np.log(1.1))
