"""Volatility forecasters.

Convention for every function here: the value stored at date ``t`` uses returns up to and
including ``t`` and is the forecast for day ``t+1``. Units are daily (not annualized); use
:func:`annualize` for display.

EWMA and GARCH(1,1) are the same first-order IIR filter on squared innovations:

    s[t+1] = omega + alpha * u[t] + beta * s[t],   u[t] = (r[t] - mu)^2

EWMA (RiskMetrics) is the special case omega=0, alpha=1-lambda, beta=lambda.
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

TRADING_DAYS = 252


def annualize(daily_vol, periods: int = TRADING_DAYS):
    return daily_vol * np.sqrt(periods)


def rolling_vol(r: pd.Series, window: int = 21) -> pd.Series:
    """Sample standard deviation over the last ``window`` returns."""
    return r.rolling(window, min_periods=window).std()


def ewma_vol(r: pd.Series, lam: float = 0.94, warmup: int = 21) -> pd.Series:
    """RiskMetrics exponentially weighted volatility (zero mean, decay ``lam``)."""
    x = r.dropna()
    s0 = float(np.mean(x.iloc[:warmup] ** 2))
    var = garch_filter(x.to_numpy() ** 2, omega=0.0, alpha=1 - lam, beta=lam, s0=s0)
    out = pd.Series(np.sqrt(var), index=x.index)
    out.iloc[: warmup - 1] = np.nan
    return out.reindex(r.index)


def garch_filter(u: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
    """Run s[t+1] = omega + alpha*u[t] + beta*s[t] and return s[1:], i.e. one-step forecasts.

    Element ``t`` of the output is the variance forecast for ``t+1`` given ``u[:t+1]``.
    """
    v = omega + alpha * np.asarray(u, dtype=float)
    out, _ = lfilter([1.0], [1.0, -beta], v, zi=[beta * s0])
    return out


def garch_vol(
    r: pd.Series,
    min_obs: int = 1000,
    refit_every: int = 252,
    dist: str = "normal",
) -> pd.Series:
    """Out-of-sample GARCH(1,1) volatility forecasts with periodic re-estimation.

    Parameters are fit on an expanding window ending at each refit date and then held fixed
    for the next ``refit_every`` days, so every forecast only uses information available at
    the time. The first ``min_obs`` days have no forecast.
    """
    from arch import arch_model

    x = r.dropna()
    scaled = x.to_numpy() * 100.0  # arch optimizes better on percent returns
    n = len(scaled)
    out = np.full(n, np.nan)

    for k in range(min_obs, n + 1, refit_every):
        train = scaled[:k]
        fit = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist=dist).fit(disp="off")
        mu, omega, alpha, beta = (fit.params[p] for p in ("mu", "omega", "alpha[1]", "beta[1]"))
        # These parameters produce the forecasts made on days k-1 .. k-2+refit_every
        # (s[t] forecasts t+1 from data up to t), including the final day's.
        stop = min(k - 1 + refit_every, n)
        u = (scaled[:stop] - mu) ** 2
        s = garch_filter(u, omega, alpha, beta, s0=float(np.var(train)))
        out[k - 1 : stop] = s[k - 1 : stop]

    return pd.Series(np.sqrt(out) / 100.0, index=x.index).reindex(r.index)
