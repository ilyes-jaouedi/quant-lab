"""Walk-forward evaluation: choose parameters on the past, score them on the future.

For each test window, every parameter set in ``grid`` is backtested on the preceding training
window; the best one (by ``metric``) is then run on the test window. Stitching the test windows
gives an out-of-sample return series: the honest estimate of what parameter tuning would have
delivered. If the chosen parameters jump around between windows, the "edge" is probably noise.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from fin.backtest.engine import BacktestConfig, run_backtest
from fin.backtest.metrics import return_metrics
from fin.data import MarketData


@dataclass
class WalkForwardResult:
    returns: pd.Series  # stitched out-of-sample daily returns
    choices: pd.DataFrame  # one row per test window: dates, chosen params, train score
    cash_rate: pd.Series

    def metrics(self) -> dict:
        return return_metrics(self.returns, self.cash_rate)


def walk_forward(
    strategy_cls,
    grid: list[dict],
    data: MarketData,
    config: BacktestConfig,
    train_years: int = 5,
    test_years: int = 1,
    metric: str = "sharpe",
    anchored: bool = False,
) -> WalkForwardResult:
    """``anchored=True`` trains on all history up to each test window instead of a rolling one."""
    base = replace(config, contribution=0.0, initial_capital=config.initial_capital or 10_000.0)
    universe = strategy_cls(**grid[0]).universe
    calendar = data.adj_close[universe].loc[config.start : config.end].dropna(how="all").index
    first, last = calendar[0], calendar[-1]

    pieces, rates, rows = [], [], []
    test_start = first + pd.DateOffset(years=train_years)
    while test_start < last:
        test_end = min(test_start + pd.DateOffset(years=test_years), last + pd.Timedelta(days=1))
        train_start = first if anchored else test_start - pd.DateOffset(years=train_years)
        train_cfg = replace(base, start=train_start, end=test_start - pd.Timedelta(days=1))

        scores = []
        for params in grid:
            res = run_backtest(strategy_cls(**params), data, train_cfg)
            scores.append(res.metrics().get(metric, np.nan))
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
        best = int(np.argmax(scores))

        test_cfg = replace(base, start=test_start, end=test_end - pd.Timedelta(days=1))
        oos = run_backtest(strategy_cls(**grid[best]), data, test_cfg)
        pieces.append(oos.returns.iloc[1:])  # day 1 only funds the account
        rates.append(oos.cash_rate.iloc[1:])
        rows.append(
            {
                "test_start": test_start,
                "test_end": test_end - pd.Timedelta(days=1),
                **grid[best],
                f"train_{metric}": scores[best],
            }
        )
        test_start = test_end

    if not pieces:
        raise ValueError("not enough data for a single train/test split")
    return WalkForwardResult(pd.concat(pieces), pd.DataFrame(rows), pd.concat(rates))
