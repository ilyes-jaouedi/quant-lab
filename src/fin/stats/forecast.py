"""Forecast evaluation for returns and volatility.

All functions take a forecast stored at ``t`` (made with data up to ``t``) and the realized
target already aligned to ``t`` (i.e. shifted back from ``t+1``). Use :func:`next_period` to
build that target, so alignment mistakes can't silently create look-ahead.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def next_period(target: pd.Series, horizon: int = 1) -> pd.Series:
    """Target realized over (t, t+horizon], stored at t: the mean of the next ``horizon`` values.

    E.g. ``next_period(r**2, 5)`` is the average squared return over the following 5 days.
    """
    return target.rolling(horizon).mean().shift(-horizon)


def _align(*series: pd.Series) -> list[pd.Series]:
    df = pd.concat(series, axis=1).dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]


def oos_r2(y: pd.Series, forecast: pd.Series, benchmark: pd.Series) -> float:
    """Campbell-Thompson out-of-sample R^2: 1 - MSE(forecast) / MSE(benchmark).

    Positive means the forecast beats the benchmark (typically the historical mean or zero).
    """
    y, f, b = _align(y, forecast, benchmark)
    return 1 - np.sum((y - f) ** 2) / np.sum((y - b) ** 2)


def mincer_zarnowitz(y: pd.Series, forecast: pd.Series) -> dict:
    """Regress y = a + b * forecast. Unbiased forecasts have a=0, b=1; R^2 measures information."""
    y, f = _align(y, forecast)
    fit = sm.OLS(y.to_numpy(), sm.add_constant(f.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    return {"a": fit.params[0], "b": fit.params[1], "r2": fit.rsquared, "n": int(fit.nobs)}


def qlike(realized_var: pd.Series, forecast_var: pd.Series) -> float:
    """Mean QLIKE loss, log(f) + y/f. Lower is better.

    Robust to noise in the realized-variance proxy (Patton, 2011), unlike MSE, which is
    dominated by a few crash days.
    """
    y, f = _align(realized_var, forecast_var)
    return float(np.mean(np.log(f) + y / f))
