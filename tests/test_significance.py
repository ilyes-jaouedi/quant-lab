import numpy as np
import pandas as pd
import pytest

from fin.stats import (
    compare_returns,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    stationary_bootstrap_indices,
)


def test_bootstrap_indices_shape_range_and_block_length():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(1000, 50, mean_block=20, rng=rng)
    assert idx.shape == (50, 1000)
    assert idx.min() >= 0 and idx.max() < 1000
    # Consecutive steps continue the block (+1 mod n) about 1 - 1/20 of the time.
    cont = np.mean((idx[:, 1:] - idx[:, :-1]) % 1000 == 1)
    assert cont == pytest.approx(0.95, abs=0.01)


def test_bootstrap_preserves_autocorrelation_better_than_iid():
    rng = np.random.default_rng(1)
    e = rng.standard_normal(5000)
    x = np.convolve(e, np.ones(10) / 10, mode="same")  # strongly autocorrelated
    idx = stationary_bootstrap_indices(len(x), 20, mean_block=50, rng=rng)
    rho = np.mean([np.corrcoef(row[1:], row[:-1])[0, 1] for row in x[idx]])
    assert rho > 0.8  # i.i.d. resampling would give ~0


def _series(mu, sigma, n=2520, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=pd.bdate_range("2010-01-01", periods=n))


def test_identical_series_are_not_significant():
    r = _series(0.0004, 0.01)
    res = compare_returns(r, r, n_boot=500)
    assert res["return_diff"] == 0 and res["sharpe_diff"] == 0
    assert res["return_diff_p"] == 1.0


def test_clearly_better_strategy_is_significant():
    bench = _series(0.0002, 0.01, seed=1)
    better = bench + 0.0004  # +10%/year, same risk
    res = compare_returns(better, bench, n_boot=500)
    assert res["return_diff"] == pytest.approx(0.1, rel=0.05)
    assert res["return_diff_p"] < 0.01
    assert res["return_diff_ci"][0] > 0


def test_noise_difference_is_not_significant():
    bench = _series(0.0003, 0.01, seed=2)
    noisy = bench + _series(0.0, 0.003, seed=3)  # same mean, extra noise
    res = compare_returns(noisy, bench, n_boot=1000)
    assert res["return_diff_p"] > 0.05
    assert res["return_diff_ci"][0] < 0 < res["return_diff_ci"][1]


def test_probabilistic_sharpe_ratio():
    assert probabilistic_sharpe_ratio(_series(0.0, 0.01, seed=4)) == pytest.approx(0.5, abs=0.25)
    assert probabilistic_sharpe_ratio(_series(0.001, 0.01, seed=5)) > 0.99
    # The same returns judged against a higher hurdle are less convincing.
    r = _series(0.0004, 0.01, seed=6)
    assert probabilistic_sharpe_ratio(r, 1.0) < probabilistic_sharpe_ratio(r, 0.0)


def test_expected_max_sharpe_grows_with_trials():
    values = [expected_max_sharpe(n, 0.5) for n in (1, 2, 10, 100, 1000)]
    assert values[0] == 0
    assert all(b > a for a, b in zip(values, values[1:], strict=False))


def test_deflated_sharpe_is_stricter_than_psr():
    r = _series(0.0005, 0.01, seed=7)
    assert deflated_sharpe_ratio(r, 50, 0.5) < probabilistic_sharpe_ratio(r)
