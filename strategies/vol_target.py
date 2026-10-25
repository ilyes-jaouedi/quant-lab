"""Volatility targeting (Moreira & Muir, 2017): hold less when forecast volatility is high.

weight = min(max_weight, target_vol / forecast_vol), the rest in cash. Motivated by
research/001: volatility is predictable, returns are not. With max_weight = 1 (no leverage)
this can only de-risk; it's a bet that high-volatility periods have worse risk/reward.
"""

import pandas as pd

from fin.signals import annualize, ewma_vol
from fin.strategy import Strategy


class VolTarget(Strategy):
    name = "vol_target"

    def __init__(
        self,
        ticker: str,
        target_vol: float = 0.15,
        lam: float = 0.94,
        max_weight: float = 1.0,
        min_history: int = 63,
    ):
        self.ticker = ticker
        self.target_vol = target_vol
        self.lam = lam
        self.max_weight = max_weight
        self.min_history = min_history
        self.universe = [ticker]

    def target_weights(self, data):
        price = data.adj_close[self.ticker].dropna().iloc[-1000:]
        r = price.pct_change().dropna()
        if len(r) < self.min_history:
            return None
        vol = annualize(ewma_vol(r, self.lam).iloc[-1])
        return pd.Series({self.ticker: min(self.max_weight, self.target_vol / vol)})
