"""Hold fixed weights. With monthly contributions this is plain DCA: the benchmark to beat."""

import pandas as pd

from fin.strategy import Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def __init__(self, weights: dict[str, float]):
        self.weights = dict(weights)
        self.universe = list(weights)

    def target_weights(self, data):
        return pd.Series(self.weights, dtype=float)
