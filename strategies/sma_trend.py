"""Classic trend filter (Faber, 2007): invested when price is above its N-month average."""

import pandas as pd

from fin.strategy import Strategy


class SmaTrend(Strategy):
    name = "sma_trend"

    def __init__(self, ticker: str, months: int = 10):
        self.ticker = ticker
        self.months = months
        self.universe = [ticker]

    def target_weights(self, data):
        price = data.adj_close[self.ticker].dropna()
        n = 21 * self.months
        if len(price) < n:
            return None
        invested = price.iloc[-1] > price.iloc[-n:].mean()
        return pd.Series({self.ticker: 1.0 if invested else 0.0})
