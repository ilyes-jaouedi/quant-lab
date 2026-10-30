"""Daily, long-only portfolio simulator.

Timeline of one trading day ``d``:

1. cash earns one day of interest (``cash_rate``);
2. orders decided on the previous decision date execute, at ``d``'s open (default) or close;
3. the portfolio is valued at ``d``'s close: this gives the day's time-weighted return;
4. external cash flows arrive (initial capital, periodic contributions);
5. on decision dates, the strategy sees ``data.until(d)`` and returns target weights.

A decision made with data up to ``d`` can therefore never trade at a price from ``d``.
Prices are dividend/split adjusted, so holding an asset earns its total return.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from fin.data import MarketData
from fin.strategy import Strategy


@dataclass(frozen=True)
class Costs:
    """Proportional costs (commission + spread + slippage) plus a fixed fee per order."""

    bps: float = 10.0
    fixed: float = 0.0

    def of(self, notional: float) -> float:
        return 0.0 if notional == 0 else abs(notional) * self.bps / 1e4 + self.fixed


@dataclass
class BacktestConfig:
    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    initial_capital: float = 10_000.0
    contribution: float = 0.0
    contribution_freq: str = "M"  # contributions arrive on the first trading day of each period
    costs: Costs = field(default_factory=Costs)
    cash_rate: float | pd.Series = 0.0  # annualized, decimal; a Series is forward-filled
    execution: Literal["next_open", "next_close"] = "next_open"
    min_trade: float = 0.0  # skip orders below this notional (a no-trade band)


@dataclass
class BacktestResult:
    label: str
    strategy: Strategy
    config: BacktestConfig
    nav: pd.Series  # portfolio value at close, after the day's cash flows
    returns: pd.Series  # time-weighted daily returns (unaffected by cash flows)
    flows: pd.Series  # external cash flows in
    weights: pd.DataFrame  # end-of-day holdings as fractions of nav
    trades: pd.DataFrame
    cash_rate: pd.Series  # daily rate actually applied, for excess-return metrics
    vs_benchmark: dict | None = None  # significance tests, filled in by the experiment runner

    def metrics(self) -> dict:
        from fin.backtest.metrics import result_metrics

        return result_metrics(self)


def period_dates(index: pd.DatetimeIndex, freq: str, which: Literal["first", "last"]):
    """First or last trading day in ``index`` of each calendar period (none for "never")."""
    if freq == "D":
        return index
    if freq == "never":
        return pd.DatetimeIndex([])
    groups = index.to_series().groupby(index.to_period(freq))
    return pd.DatetimeIndex(groups.min() if which == "first" else groups.max())


def run_backtest(
    strategy: Strategy,
    data: MarketData,
    config: BacktestConfig | None = None,
    label: str | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    tickers = list(strategy.universe)
    missing = set(tickers) - set(data.tickers)
    if missing:
        raise ValueError(f"no data for {sorted(missing)}")

    adj = data.adj_close[tickers]
    close = adj.ffill()
    open_ = data.open[tickers] * (adj / data.close[tickers])
    open_ = open_.where(open_ > 0).fillna(adj)

    index = close.loc[config.start : config.end].dropna(how="all").index
    if len(index) < 2:
        raise ValueError("fewer than 2 trading days in the backtest window")

    contrib_dates = (
        set(period_dates(index, config.contribution_freq, "first"))
        if config.contribution
        else set()
    )
    decision_dates = set(period_dates(index, strategy.rebalance, "last")) | {index[0]}
    decision_dates |= contrib_dates  # invest new money right away

    rate = _daily_rate(config.cash_rate, index)
    close_np = close.loc[index].to_numpy(dtype=float)
    open_np = open_.loc[index].to_numpy(dtype=float)
    exec_np = open_np if config.execution == "next_open" else close_np

    n, k = len(index), len(tickers)
    shares = np.zeros(k)
    cash = 0.0
    pending: np.ndarray | None = None
    nav_out, ret_out, flow_out = np.zeros(n), np.zeros(n), np.zeros(n)
    w_out = np.zeros((n, k))
    trades: list[dict] = []
    prev_nav = 0.0

    for i, d in enumerate(index):
        if i > 0:
            cash *= 1 + rate[i - 1]
        if pending is not None:
            cash = _rebalance(
                d, pending, shares, cash, exec_np[i], config, tickers, trades, strategy.sells
            )
            pending = None

        held = np.where(shares != 0, shares * close_np[i], 0.0)
        nav = cash + held.sum()
        ret_out[i] = nav / prev_nav - 1 if prev_nav > 0 else 0.0

        flow = (config.initial_capital if i == 0 else 0.0) + (
            config.contribution if d in contrib_dates else 0.0
        )
        cash += flow
        nav += flow

        if d in decision_dates:
            target = strategy.target_weights(data.until(d))
            if target is not None:
                pending = _validate(target, tickers)

        nav_out[i], flow_out[i] = nav, flow
        w_out[i] = held / nav if nav > 0 else 0.0
        prev_nav = nav

    trade_cols = ["date", "ticker", "shares", "price", "notional", "cost"]
    return BacktestResult(
        label=label or strategy.name,
        strategy=strategy,
        config=config,
        nav=pd.Series(nav_out, index=index, name="nav"),
        returns=pd.Series(ret_out, index=index, name="returns"),
        flows=pd.Series(flow_out, index=index, name="flows"),
        weights=pd.DataFrame(w_out, index=index, columns=tickers),
        trades=pd.DataFrame(trades, columns=trade_cols),
        cash_rate=pd.Series(rate, index=index, name="cash_rate"),
    )


def _daily_rate(cash_rate, index: pd.DatetimeIndex) -> np.ndarray:
    if isinstance(cash_rate, pd.Series):
        annual = cash_rate.reindex(cash_rate.index.union(index)).ffill().reindex(index).fillna(0)
        return annual.to_numpy(dtype=float) / 252
    return np.full(len(index), float(cash_rate) / 252)


def _validate(target: pd.Series, tickers: list[str]) -> np.ndarray:
    extra = set(target.index) - set(tickers)
    if extra:
        raise ValueError(f"weights for tickers outside the universe: {sorted(extra)}")
    w = target.reindex(tickers).fillna(0.0).astype(float)
    if (w < -1e-12).any():
        raise ValueError(f"negative weights (shorting) are not supported: {w.to_dict()}")
    if w.sum() > 1 + 1e-9:
        raise ValueError(f"weights sum to {w.sum():.4f} > 1 (leverage is not supported)")
    return w.clip(lower=0).to_numpy()


def _rebalance(d, w, shares, cash, px, config, tickers, trades, sells=True) -> float:
    """Trade toward weights ``w`` at prices ``px``. Mutates ``shares``; returns new cash.

    With ``sells=False`` only the buys are kept: new cash goes to the underweight assets.
    """
    tradable = ~np.isnan(px)
    value = np.where(tradable & (shares != 0), shares * np.nan_to_num(px), 0.0)
    nav = cash + value.sum()
    delta = np.where(tradable, w * nav - value, 0.0)
    if not sells:
        delta = np.maximum(delta, 0.0)
    delta[np.abs(delta) < max(config.min_trade, 1e-9 * max(nav, 1.0))] = 0.0
    costs = config.costs

    def record(j, notional, cost):
        trades.append(
            {
                "date": d,
                "ticker": tickers[j],
                "shares": notional / px[j],
                "price": px[j],
                "notional": notional,
                "cost": cost,
            }
        )

    for j in np.flatnonzero(delta < 0):
        notional = delta[j]
        cost = costs.of(notional)
        shares[j] = 0.0 if w[j] == 0 else shares[j] + notional / px[j]
        cash += -notional - cost
        record(j, notional, cost)

    buys = np.flatnonzero(delta > 0)
    if buys.size:
        # Scale buys so that notional + costs fit in the available cash.
        budget = cash - costs.fixed * buys.size
        need = delta[buys].sum() * (1 + costs.bps / 1e4)
        scale = min(1.0, max(budget, 0.0) / need)
        for j in buys:
            notional = delta[j] * scale
            if notional <= 0:
                continue
            cost = costs.of(notional)
            shares[j] += notional / px[j]
            cash -= notional + cost
            record(j, notional, cost)
    return 0.0 if abs(cash) < 1e-9 else cash
