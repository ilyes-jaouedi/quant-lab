"""Is a performance difference real, or luck?

Daily returns are not i.i.d. (volatility clusters, see research/001), so the classic t-test
overstates confidence. We resample with the stationary bootstrap (Politis & Romano, 1994):
blocks of random, geometrically distributed length, which keeps short-range dependence.
Comparisons with a benchmark are paired: both series are resampled with the same indices, so
their correlation (usually very high) is preserved.

The Probabilistic and Deflated Sharpe Ratios (Bailey & Lopez de Prado, 2012, 2014) account for
fat tails, skew, sample length and, for the deflated one, the number of strategies tried:
testing many ideas and keeping the best is like lowering a detection threshold, so the best
Sharpe ratio is inflated even when every idea is worthless.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

PERIODS = 252
EULER_GAMMA = 0.5772156649015329


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot, n) resampling indices: blocks start at random points, mean length ``mean_block``.

    Within a block indices advance by one (wrapping around); each step starts a new block with
    probability 1 / mean_block.
    """
    t = np.arange(n)
    restart = rng.random((n_boot, n)) < 1.0 / mean_block
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    last = np.maximum.accumulate(np.where(restart, t, 0), axis=1)
    rows = np.arange(n_boot)[:, None]
    return (starts[rows, last] + (t - last)) % n


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    n_boot: int = 2000,
    mean_block: float = 21.0,
    seed: int = 0,
    chunk: int = 250,
) -> np.ndarray:
    """Bootstrap distribution of statistic(a) - statistic(b), with shared indices.

    ``statistic`` maps an (k, n) array of resampled series to k values (row-wise).
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for lo in range(0, n_boot, chunk):
        k = min(chunk, n_boot - lo)
        idx = stationary_bootstrap_indices(len(a), k, mean_block, rng)
        out[lo : lo + k] = statistic(a[idx]) - statistic(b[idx])
    return out


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, x.mean(axis=-1) / sd * np.sqrt(PERIODS), 0.0)


def _mean_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1) * PERIODS


def compare_returns(
    r: pd.Series,
    r_bench: pd.Series,
    rf_daily: pd.Series | float = 0.0,
    n_boot: int = 2000,
    mean_block: float = 21.0,
    seed: int = 0,
) -> dict:
    """Paired bootstrap tests of a strategy against a benchmark on their common dates.

    - ``return_diff``: difference in annualized mean log return (~ difference in growth rate);
    - ``sharpe_diff``: difference in annualized Sharpe ratio (excess over ``rf_daily``).

    p-values are one-sided, for H0 "the strategy is not better", from the bootstrap
    distribution re-centred on zero. CIs are 95% percentile intervals.
    """
    df = pd.concat([r, r_bench], axis=1, keys=["a", "b"]).dropna()
    rf = rf_daily.reindex(df.index).fillna(0.0) if isinstance(rf_daily, pd.Series) else rf_daily
    log_a, log_b = np.log1p(df["a"].to_numpy()), np.log1p(df["b"].to_numpy())
    ex_a, ex_b = (df["a"] - rf).to_numpy(), (df["b"] - rf).to_numpy()

    out = {"n": len(df)}
    for name, (a, b, stat) in {
        "return_diff": (log_a, log_b, _mean_rows),
        "sharpe_diff": (ex_a, ex_b, _sharpe_rows),
    }.items():
        observed = float(stat(a[None])[0] - stat(b[None])[0])
        boot = paired_bootstrap(a, b, stat, n_boot=n_boot, mean_block=mean_block, seed=seed)
        out[name] = observed
        out[f"{name}_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out[f"{name}_p"] = float(np.mean(boot - observed >= observed))
    return out


def probabilistic_sharpe_ratio(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark) given the sample, accounting for skew and fat tails.

    ``sr_benchmark`` is annualized. Bailey & Lopez de Prado (2012).
    """
    x = pd.Series(r).dropna().to_numpy()
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return np.nan
    sr = x.mean() / x.std(ddof=1)
    sr0 = sr_benchmark / np.sqrt(PERIODS)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected best Sharpe among ``n_trials`` worthless strategies (true Sharpe 0).

    ``sharpe_std`` is the standard deviation of the trials' Sharpe ratios (same units as the
    result). Bailey & Lopez de Prado (2014), via extreme value theory.
    """
    if n_trials < 2:
        return 0.0
    return sharpe_std * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )


def deflated_sharpe_ratio(r: pd.Series, n_trials: int, trial_sharpe_std: float) -> float:
    """P(true Sharpe > 0) after correcting for having picked the best of ``n_trials``.

    ``trial_sharpe_std``: std of the annualized Sharpe ratios of all strategies tried.
    """
    return probabilistic_sharpe_ratio(r, expected_max_sharpe(n_trials, trial_sharpe_std))
