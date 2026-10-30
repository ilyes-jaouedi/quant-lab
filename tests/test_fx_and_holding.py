import numpy as np
import pandas as pd
import pytest
from test_engine import FREE, Const, make_md

from fin.backtest import BacktestConfig, run_backtest
from fin.data import BarCache, DataProvider, load_market_data, normalize_bars
from fin.strategy import create

DATES = pd.bdate_range("2020-01-01", periods=5, name="date")


def _bars(prices):
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"open": p, "high": p, "low": p, "close": p, "adj_close": p, "volume": 1.0}, index=DATES
    )


class MultiProvider(DataProvider):
    """Serves fixed series per ticker, with quote currencies."""

    name = "multi"

    def __init__(self, series: dict[str, list], currencies: dict[str, str]):
        self.series = {t: _bars(p) for t, p in series.items()}
        self.currencies = currencies

    def fetch(self, ticker, start, end):
        return normalize_bars(self.series[ticker].loc[start:end])

    def currency(self, ticker):
        return self.currencies.get(ticker)


@pytest.fixture
def provider():
    return MultiProvider(
        {
            "US": [100, 100, 100, 100, 100],  # flat in USD
            "LDN": [500, 500, 500, 500, 500],  # pence
            "EUROPE": [10, 11, 12, 13, 14],
            "EURUSD=X": [1.0, 1.0, 1.25, 1.25, 1.0],  # USD per EUR
            "EURGBP=X": [0.5, 0.5, 0.5, 0.5, 0.5],
        },
        {"US": "USD", "LDN": "GBp", "EUROPE": "EUR"},
    )


def test_usd_asset_converted_to_eur(provider, tmp_path):
    md = load_market_data(["US"], provider=provider, cache=BarCache(tmp_path), currency="EUR")
    # A flat USD price loses value in EUR when the euro strengthens (1.0 -> 1.25 USD per EUR).
    np.testing.assert_allclose(md.close["US"].to_numpy(), [100, 100, 80, 80, 100])


def test_pence_quotes_are_scaled_to_pounds_then_converted(provider, tmp_path):
    md = load_market_data(["LDN"], provider=provider, cache=BarCache(tmp_path), currency="EUR")
    # 500 GBp = 5 GBP = 5 / 0.5 EUR
    np.testing.assert_allclose(md.close["LDN"].to_numpy(), 10.0)


def test_same_currency_is_untouched_and_no_currency_keeps_quotes(provider, tmp_path):
    cache = BarCache(tmp_path)
    md = load_market_data(["EUROPE", "US"], provider=provider, cache=cache, currency="EUR")
    np.testing.assert_allclose(md.close["EUROPE"].to_numpy(), [10, 11, 12, 13, 14])
    raw = load_market_data(["US"], provider=provider, cache=cache)
    np.testing.assert_allclose(raw.close["US"].to_numpy(), 100.0)


def test_currency_is_recorded_in_cache(provider, tmp_path):
    cache = BarCache(tmp_path)
    load_market_data(["US"], provider=provider, cache=cache, currency="EUR")
    assert cache.read("multi", "US").meta.currency == "USD"


# --- buy-and-hold vs constant mix ------------------------------------------------------------


def _two_assets():
    a = [100, 100, 150, 200, 200, 200, 200, 200]
    b = [100] * 8
    return make_md({"A": a, "B": b}, start="2020-01-27")  # spans a month end


def test_buy_and_hold_lets_weights_drift_and_never_sells():
    res = run_backtest(
        create("buy_and_hold", weights={"A": 0.5, "B": 0.5}),
        _two_assets(),
        BacktestConfig(costs=FREE),
    )
    assert len(res.trades) == 2  # the initial purchase only
    assert (res.trades["notional"] > 0).all()
    np.testing.assert_allclose(res.weights.iloc[-1].to_numpy(), [2 / 3, 1 / 3])


def test_constant_mix_rebalances_back_to_targets():
    strategy = create("constant_mix", weights={"A": 0.5, "B": 0.5})
    strategy.rebalance = "D"
    res = run_backtest(strategy, _two_assets(), BacktestConfig(costs=FREE))
    assert (res.trades["notional"] < 0).any()
    np.testing.assert_allclose(res.weights.iloc[-1].to_numpy(), [0.5, 0.5])


def test_buy_only_invests_new_cash_in_the_underweight_asset():
    md = _two_assets()
    cfg = BacktestConfig(initial_capital=1000, contribution=100, costs=FREE)
    res = run_backtest(create("buy_and_hold", weights={"A": 0.5, "B": 0.5}), md, cfg)
    feb_buys = res.trades[res.trades["date"] > pd.Timestamp("2020-02-01")]
    assert set(feb_buys["ticker"]) == {"B"}  # A ran up, so new money goes to B
    assert (res.trades["notional"] > 0).all()


def test_rebalance_never_decides_only_at_start():
    class Counter(Const):
        name = "counter"
        calls = 0

        def target_weights(self, data):
            type(self).calls += 1
            return super().target_weights(data)

    strategy = Counter({"A": 1.0})
    strategy.rebalance = "never"
    run_backtest(strategy, make_md({"A": [100] * 60}), BacktestConfig(costs=FREE))
    assert Counter.calls == 1
