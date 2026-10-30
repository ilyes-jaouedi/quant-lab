"""Judge each strategy of an experiment against its benchmark, with significance tests."""

import numpy as np

from fin.backtest.engine import BacktestResult
from fin.stats import compare_returns, deflated_sharpe_ratio, probabilistic_sharpe_ratio


def active_returns(result: BacktestResult):
    """Daily returns from the first day the account holds money (excluding that funding day)."""
    active = result.nav > 0
    if not active.any():
        return result.returns.iloc[0:0]
    return result.returns.loc[active.idxmax() :].iloc[1:]


def versus_benchmark(
    results: dict[str, BacktestResult], benchmark: str, n_boot: int = 2000
) -> dict[str, dict]:
    """Per strategy: paired bootstrap tests against ``benchmark``, PSR and deflated Sharpe.

    The deflated Sharpe ratio counts every strategy of the experiment as a trial. That is a
    lower bound: ideas tried and discarded elsewhere count too.
    """
    bench = results[benchmark]
    sharpes = [r.metrics().get("sharpe", np.nan) for r in results.values()]
    sharpe_std = float(np.nanstd(sharpes, ddof=1)) if len(results) > 1 else 0.0
    n_trials = len(results)

    out = {}
    for label, res in results.items():
        r = active_returns(res)
        excess = r - res.cash_rate.reindex(r.index).fillna(0.0)
        entry = {
            "psr": probabilistic_sharpe_ratio(excess),
            "dsr": deflated_sharpe_ratio(excess, n_trials, sharpe_std),
            "n_trials": n_trials,
        }
        if label != benchmark:
            entry |= compare_returns(r, active_returns(bench), res.cash_rate, n_boot=n_boot)
        out[label] = entry
    return out
