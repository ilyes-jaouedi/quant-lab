"""L1 trend filtering for regime detection: invested while the current trend segment slopes up.

On each decision date, fit a piecewise-linear trend to the last ``lookback`` log prices and
look at the slope of its final segment. ``lam_rel`` in (0, 1) sets the sparsity relative to
lam_max: small values give many short segments (reactive, noisy), values near 1 approach a
single straight line.
"""

import numpy as np
import pandas as pd

from fin.signals import l1_trend_filter, l1_trend_lam_max
from fin.strategy import Strategy


class L1Trend(Strategy):
    name = "l1_trend"

    def __init__(
        self, ticker: str, lam_rel: float = 0.05, lookback: int = 504, min_history: int = 252
    ):
        self.ticker = ticker
        self.lam_rel = lam_rel
        self.lookback = lookback
        self.min_history = min_history
        self.universe = [ticker]

    def target_weights(self, data):
        price = data.adj_close[self.ticker].dropna().iloc[-self.lookback :]
        if len(price) < self.min_history:
            return None
        y = np.log(price.to_numpy())
        trend = l1_trend_filter(y, self.lam_rel * l1_trend_lam_max(y))
        slope = trend[-1] - trend[-2]
        return pd.Series({self.ticker: 1.0 if slope > 0 else 0.0})
