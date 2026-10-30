"""Fixed weights, rebalanced every period (monthly by default): sells winners, buys losers.

The fair benchmark for multi-asset strategies that trade the same assets.
"""

import pandas as pd

from fin.strategy import Strategy


class ConstantMix(Strategy):
    name = "constant_mix"

    def __init__(self, weights: dict[str, float]):
        self.weights = dict(weights)
        self.universe = list(weights)

    def target_weights(self, data):
        return pd.Series(self.weights, dtype=float)
