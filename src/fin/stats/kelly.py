"""Growth-optimal betting: the Kelly criterion, volatility drag and the rebalancing bonus.

For a fraction ``f`` of wealth in an asset with arithmetic mean ``mu`` and volatility
``sigma`` (annualized), the rest earning ``rf``, the long-run growth rate of wealth is

    g(f) = rf + f (mu - rf) - f^2 sigma^2 / 2          (continuous-time / small-step limit)

The ``-sigma^2 / 2`` term is the volatility drag. ``g`` is a parabola maximized at the Kelly
fraction ``f* = (mu - rf) / sigma^2``; it gives ``g(f*) = rf + SR^2 / 2`` with ``SR`` the
Sharpe ratio, and falls back to ``rf`` at ``2 f*``. Half Kelly keeps 3/4 of the excess growth.
"""

import numpy as np
import pandas as pd


def growth_rate(f, mu: float, sigma: float, rf: float = 0.0):
    """Long-run log growth rate of a constant fraction ``f`` (scalar or array)."""
    f = np.asarray(f, dtype=float)
    return rf + f * (mu - rf) - 0.5 * f**2 * sigma**2


def kelly_fraction(mu: float, sigma: float, rf: float = 0.0) -> float:
    """Growth-optimal fraction ``(mu - rf) / sigma^2`` (above 1 means leverage)."""
    return (mu - rf) / sigma**2


def kelly_fraction_se(sigma: float, years: float) -> float:
    """Standard error of an estimated Kelly fraction, from the noise in the mean alone.

    ``f_hat = m_hat / sigma^2`` with ``se(m_hat) = sigma / sqrt(years)``, so the error is
    ``1 / (sigma sqrt(years))``: it depends only on how long and how volatile the sample is.
    """
    return 1.0 / (sigma * np.sqrt(years))


def kelly_shrinkage(f_hat: float, se: float) -> float:
    """Fraction of an estimated Kelly bet to take when ``f_hat`` is noisy (plug-in Wiener gain).

    Betting ``c f_hat`` with ``f_hat = f* + noise`` maximizes expected growth at
    ``c = f*^2 / (f*^2 + se^2)``, the gain ``SNR / (1 + SNR)`` of a Wiener filter.
    ``f*`` is unknown, so ``f_hat`` stands in for it, which makes this an optimistic upper bound.
    """
    return f_hat**2 / (f_hat**2 + se**2)


def bet_growth(f, p: float, b: float = 1.0):
    """Expected log growth per round of betting ``f`` at odds ``b:1``, win probability ``p``."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        return p * np.log1p(f * b) + (1 - p) * np.log1p(-f)


def kelly_bet(p: float, b: float = 1.0) -> float:
    """Kelly fraction for a binary bet: ``p - (1 - p) / b`` (zero when the bet has no edge)."""
    return max(p - (1 - p) / b, 0.0)


def leveraged_returns(r: pd.Series, leverage: float, rf=0.0, spread: float = 0.0) -> pd.Series:
    """Returns of holding ``leverage`` times ``r``, rebalanced every period.

    ``rf`` (per period, scalar or Series) is earned on the cash left when ``leverage < 1`` and
    paid, plus ``spread`` per period, on the amount borrowed when ``leverage > 1``.
    """
    cash = 1.0 - leverage
    rate = rf if cash >= 0 else rf + spread
    return leverage * r + cash * rate


def log_growth(r, periods_per_year: int = 252) -> float:
    """Realized annualized log growth rate: mean of ``log(1 + r)`` times periods per year.

    Returns ``-inf`` when any period loses everything.
    """
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if (r <= -1).any():
        return -np.inf
    return float(np.mean(np.log1p(r)) * periods_per_year)


def rebalancing_bonus(weights, cov) -> float:
    """Extra growth of a constant mix over the weighted growth of its assets.

    ``(sum_i w_i sigma_i^2 - w' Sigma w) / 2``: half the gap between the average variance and
    the portfolio variance. It is zero for one asset or perfectly correlated assets and grows
    with diversification. Same units as ``cov`` (annualized covariance gives annual growth).
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    return 0.5 * float(w @ np.diag(cov) - w @ cov @ w)
