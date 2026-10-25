"""The Strategy interface: the only thing that changes between ideas."""

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from fin.data import MarketData


class Strategy(ABC):
    """Answers one question: given the data up to today, what portfolio weights do I want?

    Subclasses set ``name`` (registry key) and, in ``__init__``, ``universe`` (tickers the
    strategy may hold). Weights are fractions of portfolio value; they must be >= 0 and sum to
    at most 1, the remainder being cash. The engine calls :meth:`target_weights` on each
    rebalance date with a point-in-time view of the data, and executes the trades afterwards.
    """

    name: ClassVar[str]
    universe: list[str]
    rebalance: str = "M"  # "D", "W", "M", "Q" or "Y": decide on the last trading day of each period

    @abstractmethod
    def target_weights(self, data: MarketData) -> pd.Series | None:
        """Target weights indexed by ticker, or ``None`` to keep the current portfolio."""

    def params(self) -> dict:
        """Constructor parameters, recorded with each run for reproducibility."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_") and k != "universe"}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"
