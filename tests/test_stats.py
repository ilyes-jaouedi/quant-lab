import numpy as np
import pandas as pd
import pytest

from fin.stats import acf, acf_band, next_period, oos_r2, qlike, variance_ratio


def _ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(x)


def test_acf_of_ar1():
    a = acf(_ar1(20000, 0.5), nlags=3)
    np.testing.assert_allclose(a.values, [0.5, 0.25, 0.125], atol=0.03)
    assert list(a.index) == [1, 2, 3]


def test_acf_band():
    assert acf_band(10000) == pytest.approx(1.96 / 100, rel=1e-3)


def test_variance_ratio_random_walk_not_rejected():
    rng = np.random.default_rng(0)
    res = variance_ratio(pd.Series(rng.standard_normal(5000)), q=5)
    assert res["vr"] == pytest.approx(1.0, abs=0.06)
    assert abs(res["z_robust"]) < 3


def test_variance_ratio_detects_autocorrelation():
    # VR(2) = 1 + rho_1
    pos = variance_ratio(_ar1(20000, 0.2), q=2)
    assert pos["vr"] == pytest.approx(1.2, abs=0.03)
    assert pos["z_robust"] > 5
    neg = variance_ratio(_ar1(20000, -0.2), q=2)
    assert neg["vr"] == pytest.approx(0.8, abs=0.03)
    assert neg["z_robust"] < -5


def test_next_period_alignment():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(next_period(s, 1).values[:4], [2, 3, 4, 5])
    np.testing.assert_allclose(next_period(s, 2).values[:3], [2.5, 3.5, 4.5])
    assert np.isnan(next_period(s, 2).values[3:]).all()


def test_oos_r2():
    y = pd.Series([1.0, -1.0, 1.0, -1.0])
    zero = pd.Series(0.0, index=y.index)
    assert oos_r2(y, y, zero) == 1.0
    assert oos_r2(y, zero, zero) == 0.0
    assert oos_r2(y, -y, zero) < 0


def test_qlike_prefers_true_variance():
    rng = np.random.default_rng(0)
    true_var = pd.Series(rng.uniform(0.5, 2.0, 5000))
    realized = true_var * rng.standard_normal(5000) ** 2
    assert qlike(realized, true_var) < qlike(
        realized, pd.Series(true_var.mean(), index=true_var.index)
    )
