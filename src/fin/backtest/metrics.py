"""Performance metrics.

Return-based metrics use time-weighted returns, so they measure the strategy, not the timing
of contributions. The money-weighted IRR answers the investor's question instead: "what annual
rate did my actual euros earn?"
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

PERIODS = 252


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def return_metrics(returns: pd.Series, rf_daily: pd.Series | float = 0.0) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {}
    years = len(r) / PERIODS
    excess = r - (
        rf_daily.reindex(r.index).fillna(0) if isinstance(rf_daily, pd.Series) else rf_daily
    )
    vol = r.std() * np.sqrt(PERIODS)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * np.sqrt(PERIODS)
    cagr = (1 + r).prod() ** (1 / years) - 1
    mdd = max_drawdown(r)
    ann_excess = excess.mean() * PERIODS
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": ann_excess / vol if vol > 0 else np.nan,
        "sortino": ann_excess / downside if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "years": years,
    }


def xirr(dates: pd.DatetimeIndex, cash_flows: np.ndarray) -> float:
    """Annual rate r with sum(cf_i / (1+r)^t_i) = 0, t in years.

    Investor's view: money put in is negative, money taken out (or final value) positive.
    """
    t = np.asarray((dates - dates[0]).days, dtype=float) / 365.25
    cf = np.asarray(cash_flows, dtype=float)
    if not (cf < 0).any() or not (cf > 0).any():
        return np.nan

    def npv(rate):
        return np.sum(cf / (1 + rate) ** t)

    try:
        return brentq(npv, -0.9999, 100.0)
    except ValueError:
        return np.nan


def result_metrics(result) -> dict:
    """All metrics for a :class:`~fin.backtest.engine.BacktestResult`."""
    active = result.nav > 0
    if not active.any():
        return {}
    first = active.idxmax()
    r = result.returns.loc[first:].iloc[1:]  # first day only receives the initial flow
    out = return_metrics(r, result.cash_rate)

    flows = result.flows.loc[first:]
    nav = result.nav.loc[first:]
    investor_cf = -flows.to_numpy().copy()
    investor_cf[-1] += nav.iloc[-1]
    trades = result.trades
    traded = trades["notional"].abs().sum() if len(trades) else 0.0

    out.update(
        {
            "final_value": float(nav.iloc[-1]),
            "contributed": float(flows.sum()),
            "irr": xirr(nav.index, investor_cf),
            "avg_exposure": float(result.weights.loc[first:].sum(axis=1).mean()),
            "turnover": traded / 2 / nav.mean() / out.get("years", np.nan),
            "costs": float(trades["cost"].sum()) if len(trades) else 0.0,
            "n_trades": len(trades),
        }
    )
    return out
