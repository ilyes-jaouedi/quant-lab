"""Multi-asset trend following, long or cash (time-series momentum, Moskowitz-Ooi-Pedersen 2012).

Each asset is held while its trend signal is positive and replaced by cash otherwise. The
signal is the sign of a linear filter of past returns (see fin.signals.trend):

- "momentum": the lookback return (box kernel);
- "sma": price above its simple moving average (triangular kernel);
- "ema": price above its exponential moving average of the same lag (exponential kernel).

Weights before the signal: "equal" (1/N) or "inverse_vol" (risk parity: proportional to
1 / recent volatility, so each asset contributes similar risk). ``always_invested=True``
switches the signal off, giving the matching benchmark with the same weighting.
"""

import numpy as np
import pandas as pd

from fin.strategy import Strategy


class TrendFollowing(Strategy):
    name = "trend_following"

    def __init__(
        self,
        tickers: list[str],
        lookback_months: int = 12,
        signal: str = "momentum",
        weighting: str = "equal",
        vol_lookback: int = 63,
        always_invested: bool = False,
    ):
        if signal not in {"momentum", "sma", "ema"}:
            raise ValueError(f"unknown signal {signal!r}")
        if weighting not in {"equal", "inverse_vol"}:
            raise ValueError(f"unknown weighting {weighting!r}")
        self.tickers = list(tickers)
        self.lookback_months = lookback_months
        self.signal = signal
        self.weighting = weighting
        self.vol_lookback = vol_lookback
        self.always_invested = always_invested
        self.universe = list(tickers)

    def target_weights(self, data):
        n = 21 * self.lookback_months
        px = data.adj_close[self.tickers].iloc[-(n + 1) * 3 :].ffill()
        window = px.iloc[-(n + 1) :]
        if len(window) < n + 1 or window.isna().any().any():
            return None  # wait until every asset has a full lookback

        log_px = np.log(window)
        if self.always_invested:
            on = pd.Series(True, index=self.tickers)
        elif self.signal == "momentum":
            on = log_px.iloc[-1] > log_px.iloc[0]
        elif self.signal == "sma":
            on = log_px.iloc[-1] > log_px.iloc[-n:].mean()
        else:
            on = log_px.iloc[-1] > log_px.ewm(span=n, adjust=False).mean().iloc[-1]

        if self.weighting == "equal":
            base = pd.Series(1.0 / len(self.tickers), index=self.tickers)
        else:
            vol = np.log(px.iloc[-(self.vol_lookback + 1) :]).diff().std()
            inv = 1.0 / vol
            base = inv / inv.sum()
        return base.where(on, 0.0)
