import numpy as np
import pandas as pd
import pytest

from fin.backtest import BacktestConfig, Costs, run_backtest
from fin.data import MarketData
from fin.strategy import Strategy

FREE = Costs(bps=0, fixed=0)


def make_md(prices: dict[str, list], opens: dict[str, list] | None = None, start="2020-01-01"):
    n = len(next(iter(prices.values())))
    idx = pd.bdate_range(start, periods=n, name="date")
    bars = {}
    for t, p in prices.items():
        p = np.asarray(p, dtype=float)
        o = p if opens is None else np.asarray(opens[t], dtype=float)
        bars[t] = pd.DataFrame(
            {"open": o, "high": p, "low": p, "close": p, "adj_close": p, "volume": 1.0}, index=idx
        )
    return MarketData.from_bars(bars)


class Const(Strategy):
    name = "const"

    def __init__(self, weights, rebalance="M"):
        self.weights = weights
        self.universe = list(weights)
        self.rebalance = rebalance

    def target_weights(self, data):
        return pd.Series(self.weights, dtype=float)


class Spy(Const):
    """Records the last date it was allowed to see on each call."""

    name = "spy"

    def __init__(self, weights):
        super().__init__(weights, rebalance="D")
        self.seen = []

    def target_weights(self, data):
        self.seen.append(data.index.max())
        return super().target_weights(data)


def test_buy_and_hold_tracks_price():
    p = [100, 101, 103, 99, 105, 110]
    res = run_backtest(Const({"A": 1.0}), make_md({"A": p}), BacktestConfig(costs=FREE))
    # Decided at day 0's close, bought at day 1's open (= close here).
    assert res.nav.iloc[-1] == pytest.approx(10_000 * 110 / 101)
    assert res.weights["A"].iloc[-1] == pytest.approx(1.0)


def test_execution_happens_at_next_open_not_decision_close():
    close = [100, 100, 100, 100]
    opens = [100, 125, 100, 100]
    res = run_backtest(
        Const({"A": 1.0}), make_md({"A": close}, {"A": opens}), BacktestConfig(costs=FREE)
    )
    assert res.trades["price"].iloc[0] == 125
    assert res.nav.iloc[1] == pytest.approx(10_000 * 100 / 125)


def test_next_close_execution():
    close = [100, 120, 120]
    cfg = BacktestConfig(costs=FREE, execution="next_close")
    res = run_backtest(Const({"A": 1.0}), make_md({"A": close}), cfg)
    assert res.trades["price"].iloc[0] == 120


def test_strategy_only_sees_data_up_to_decision_date():
    md = make_md({"A": np.linspace(100, 120, 30)})
    spy = Spy({"A": 1.0})
    res = run_backtest(spy, md, BacktestConfig(costs=FREE))
    assert spy.seen == list(res.nav.index)


def test_proportional_costs():
    res = run_backtest(
        Const({"A": 1.0}), make_md({"A": [100] * 5}), BacktestConfig(costs=Costs(bps=10))
    )
    # Buy notional n with n * 1.001 = 10_000.
    assert res.nav.iloc[-1] == pytest.approx(10_000 / 1.001)
    assert res.trades["cost"].sum() == pytest.approx(10_000 / 1.001 * 0.001)


def test_fixed_fee_per_order():
    res = run_backtest(
        Const({"A": 0.5, "B": 0.5}),
        make_md({"A": [100] * 5, "B": [50] * 5}),
        BacktestConfig(costs=Costs(bps=0, fixed=1.0)),
    )
    assert len(res.trades) == 2
    assert res.nav.iloc[-1] == pytest.approx(10_000 - 2.0)


def test_contributions_with_flat_prices_keep_value_equal_to_money_in():
    md = make_md({"A": [100] * 70})  # ~3.5 months of business days
    cfg = BacktestConfig(initial_capital=0, contribution=100, costs=FREE)
    res = run_backtest(Const({"A": 1.0}), md, cfg)
    assert res.flows.sum() == 400  # Jan, Feb, Mar, Apr
    assert res.nav.iloc[-1] == pytest.approx(400)
    assert np.allclose(res.returns, 0)


def test_time_weighted_return_ignores_contributions():
    p = 100 * 1.001 ** np.arange(120)
    opens = np.r_[p[0], p[:-1]]  # open = previous close: no overnight gap
    md = make_md({"A": p}, {"A": opens})
    cfg = BacktestConfig(initial_capital=0, contribution=100, costs=FREE)
    res = run_backtest(Const({"A": 1.0}), md, cfg)
    # Fully invested from day 1, so every daily TWR equals the asset's return.
    np.testing.assert_allclose(res.returns.iloc[2:], 0.001, rtol=1e-9)
    assert res.nav.iloc[-1] > res.flows.sum()


def test_cash_earns_interest():
    cfg = BacktestConfig(costs=FREE, cash_rate=0.0252)  # 1 bp per day
    res = run_backtest(Const({"A": 0.0}), make_md({"A": [100] * 11}), cfg)
    assert res.nav.iloc[-1] == pytest.approx(10_000 * 1.0001**10)


def test_rebalancing_restores_weights():
    a = [100, 200, 200, 200, 200]
    b = [100, 100, 100, 100, 100]
    res = run_backtest(
        Const({"A": 0.5, "B": 0.5}, rebalance="D"),
        make_md({"A": a, "B": b}),
        BacktestConfig(costs=FREE),
    )
    np.testing.assert_allclose(res.weights.iloc[-1].to_numpy(), [0.5, 0.5])


def test_none_keeps_portfolio_in_cash():
    class Never(Const):
        name = "never"

        def target_weights(self, data):
            return None

    res = run_backtest(
        Never({"A": 1.0}), make_md({"A": [100, 200, 300]}), BacktestConfig(costs=FREE)
    )
    assert res.nav.iloc[-1] == 10_000
    assert res.trades.empty


@pytest.mark.parametrize(
    "weights, match",
    [({"A": 1.5}, "leverage"), ({"A": -0.5}, "shorting"), ({"Z": 1.0}, "outside the universe")],
)
def test_invalid_weights_are_rejected(weights, match):
    class Bad(Const):
        name = "bad"

        def target_weights(self, data):
            return pd.Series(weights)

    with pytest.raises(ValueError, match=match):
        run_backtest(Bad({"A": 1.0}), make_md({"A": [100] * 3}))


def test_start_end_window_uses_earlier_history_for_decisions():
    md = make_md({"A": np.linspace(100, 200, 60)})
    spy = Spy({"A": 1.0})
    start = md.index[30]
    res = run_backtest(spy, md, BacktestConfig(start=start, costs=FREE))
    assert res.nav.index[0] == start
    assert len(md.until(spy.seen[0]).index) == 31  # full history up to the first decision
