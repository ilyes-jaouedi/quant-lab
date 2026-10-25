"""Serial dependence tests: autocorrelation, Ljung-Box, Lo-MacKinlay variance ratio."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def acf(x: pd.Series, nlags: int = 40) -> pd.Series:
    """Sample autocorrelation at lags 1..nlags (NaNs dropped).

    Under the i.i.d. null each value is approximately N(0, 1/n): see :func:`acf_band`.
    """
    from statsmodels.tsa.stattools import acf as _acf

    values = _acf(x.dropna().to_numpy(), nlags=nlags, fft=True)
    return pd.Series(values[1:], index=pd.RangeIndex(1, nlags + 1, name="lag"), name="acf")


def acf_band(n: int, level: float = 0.95) -> float:
    """Half-width of the i.i.d. confidence band for a sample ACF of length ``n``."""
    return stats.norm.ppf(0.5 + level / 2) / np.sqrt(n)


def ljung_box(x: pd.Series, lags: Iterable[int] = (5, 10, 20)) -> pd.DataFrame:
    """Ljung-Box Q test of no autocorrelation up to each lag."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    out = acorr_ljungbox(x.dropna().to_numpy(), lags=list(lags))
    return out.rename(columns={"lb_stat": "Q", "lb_pvalue": "p_value"}).rename_axis("lag")


def variance_ratio(x: pd.Series, q: int) -> dict:
    """Lo-MacKinlay (1988) variance ratio test with overlapping q-period sums.

    VR(q) = Var(q-period return) / (q * Var(1-period return)). For a random walk VR = 1;
    VR > 1 means positive autocorrelation (trending), VR < 1 negative (mean reverting).
    Approximately, VR(q) - 1 = 2 * sum_{j<q} (1 - j/q) * rho_j.

    Returns the statistic plus the homoskedastic ``z`` and the heteroskedasticity-robust
    ``z_robust`` (the one to trust for returns, whose variance clusters).
    """
    r = x.dropna().to_numpy(dtype=float)
    n = len(r)
    if q < 2 or n <= 2 * q:
        raise ValueError(f"need q >= 2 and more than 2q observations (q={q}, n={n})")
    e = r - r.mean()

    var_1 = np.sum(e**2) / (n - 1)
    sums_q = np.convolve(e, np.ones(q), mode="valid")  # overlapping q-period sums
    m = q * (n - q + 1) * (1 - q / n)
    var_q = np.sum(sums_q**2) / m
    vr = var_q / var_1

    phi = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
    e2 = e**2
    denom = np.sum(e2) ** 2
    theta = 0.0
    for j in range(1, q):
        delta_j = np.sum(e2[j:] * e2[:-j]) / denom
        theta += (2 * (q - j) / q) ** 2 * delta_j

    z = (vr - 1) / np.sqrt(phi)
    z_robust = (vr - 1) / np.sqrt(theta)
    return {
        "q": q,
        "vr": vr,
        "z": z,
        "z_robust": z_robust,
        "p_value": 2 * stats.norm.sf(abs(z_robust)),
        "n": n,
    }


def variance_ratio_table(x: pd.Series, qs: Iterable[int] = (2, 5, 10, 20, 60)) -> pd.DataFrame:
    return pd.DataFrame([variance_ratio(x, q) for q in qs]).set_index("q")
