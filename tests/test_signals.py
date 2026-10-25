import numpy as np
import pandas as pd
import pytest

from fin.signals import ewma_vol, garch_filter, garch_vol, rolling_vol


def _garch_returns(n=3000, omega=0.02, alpha=0.08, beta=0.9, seed=0):
    """Simulated GARCH(1,1) percent-scale returns, returned as decimals."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(s) * rng.standard_normal()
        s = omega + alpha * r[t] ** 2 + beta * s
    return pd.Series(r / 100, index=pd.bdate_range("2000-01-03", periods=n))


def test_garch_filter_matches_explicit_recursion():
    rng = np.random.default_rng(1)
    u = rng.random(50)
    omega, alpha, beta, s0 = 0.1, 0.05, 0.9, 1.0
    expected, s = [], s0
    for ut in u:
        s = omega + alpha * ut + beta * s
        expected.append(s)
    np.testing.assert_allclose(garch_filter(u, omega, alpha, beta, s0), expected)


def test_ewma_matches_pandas_after_warmup():
    r = _garch_returns(500)
    ours = ewma_vol(r, lam=0.94) ** 2
    ref = (r**2).ewm(alpha=0.06, adjust=False).mean()
    # Different initialization; the effect decays as 0.94^t.
    np.testing.assert_allclose(ours.iloc[200:], ref.iloc[200:], rtol=1e-4)


@pytest.mark.parametrize(
    "forecaster",
    [
        lambda r: rolling_vol(r, 21),
        lambda r: ewma_vol(r),
        lambda r: garch_vol(r, min_obs=500, refit_every=250),
    ],
    ids=["rolling", "ewma", "garch"],
)
def test_forecasts_do_not_use_future_data(forecaster):
    r = _garch_returns(1200)
    cut = 900
    altered = r.copy()
    altered.iloc[cut + 1 :] *= 5  # wreck everything after the cut
    a, b = forecaster(r), forecaster(altered)
    pd.testing.assert_series_equal(a.iloc[: cut + 1], b.iloc[: cut + 1])


def test_garch_recovers_simulated_persistence():
    r = _garch_returns(4000)
    vol = garch_vol(r, min_obs=1000, refit_every=500)
    assert vol.iloc[:999].isna().all()
    assert vol.iloc[999:].notna().all()
    # Forecasts should track the true conditional vol: correlated with next-day |r|.
    realized = r.abs().shift(-1)
    assert vol.corr(realized) > 0.2
