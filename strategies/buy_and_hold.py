"""Buy once and hold: the benchmark every strategy has to beat.

Weights are only targets for the initial purchase (and for new cash, which goes to whatever is
underweight). Nothing is ever sold, so weights drift with prices, as in a real portfolio you
leave alone. For a portfolio rebalanced back to fixed weights, see ``constant_mix``.
"""

import pandas as pd

from fin.strategy import Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"
    rebalance = "never"
    sells = False

    def __init__(self, weights: dict[str, float]):
        self.weights = dict(weights)
        self.universe = list(weights)

    def target_weights(self, data):
        return pd.Series(self.weights, dtype=float)
