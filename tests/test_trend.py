import numpy as np
import pandas as pd
import pytest

from fin.signals.trend import (
    apply_kernel,
    crossover_kernel,
    ema_kernel,
    frequency_response,
    lag,
    momentum_kernel,
    noise_gain,
    price_minus_smoother,
    sma_kernel,
)


@pytest.fixture
def log_price():
    rng = np.random.default_rng(0)
    return np.cumsum(rng.normal(0.0003, 0.01, 2000))


def test_momentum_kernel_equals_n_day_log_return(log_price):
    n = 50
    filtered = apply_kernel(np.diff(log_price), momentum_kernel(n))
    # filtered[t] is built from returns up to r[t] = x[t+1] - x[t], so it equals x[t+1] - x[t+1-n].
    expected = log_price[n:] - log_price[:-n]
    np.testing.assert_allclose(filtered[n - 1 :], expected)
    assert np.isnan(filtered[: n - 1]).all()


def test_price_minus_sma_is_triangular_kernel_on_returns(log_price):
    n = 30
    w = price_minus_smoother(sma_kernel(n))
    np.testing.assert_allclose(w, (n - 1 - np.arange(n - 1)) / n)
    # Identity check against the direct computation x_t - SMA_n(x)_t.
    x = pd.Series(log_price)
    direct = (x - x.rolling(n).mean()).to_numpy()[1:]
    filtered = apply_kernel(np.diff(log_price), w)
    np.testing.assert_allclose(filtered[n:], direct[n:], atol=1e-12)


def test_price_minus_ema_matches_pandas(log_price):
    span = 40
    h = ema_kernel(span, length=400)
    w = price_minus_smoother(h)
    x = pd.Series(log_price)
    direct = (x - x.ewm(span=span, adjust=False).mean()).to_numpy()
    filtered = apply_kernel(np.diff(log_price), w)
    # The truncated kernel and pandas' recursion agree once the start-up transient has decayed.
    np.testing.assert_allclose(filtered[600:], direct[601:], atol=1e-6)


def test_equal_lag_convention_for_sma_and_ema():
    assert lag(sma_kernel(21)) == pytest.approx(10.0)
    assert lag(ema_kernel(21)) == pytest.approx(10.0, abs=1e-3)


def test_sma_is_smoother_than_ema_at_equal_lag():
    # For a given delay the box average has the least noise gain of all kernels (1/n).
    assert noise_gain(sma_kernel(21)) == pytest.approx(1 / 21)
    assert noise_gain(sma_kernel(21)) < noise_gain(ema_kernel(21))


def test_crossover_kernel_matches_difference_of_smoothers(log_price):
    fast, slow = sma_kernel(10), sma_kernel(40)
    w = crossover_kernel(fast, slow)
    x = pd.Series(log_price)
    direct = (x.rolling(10).mean() - x.rolling(40).mean()).to_numpy()[1:]
    filtered = apply_kernel(np.diff(log_price), w)
    np.testing.assert_allclose(filtered[60:], direct[60:], atol=1e-12)


def test_frequency_response_is_low_pass_with_unit_dc_gain():
    freq, h = frequency_response(sma_kernel(20))
    assert abs(h[0]) == pytest.approx(1.0)
    assert freq[-1] < 0.5 + 1e-9
    # First null of a length-20 box filter at 1/20 cycles per day.
    i = np.argmin(np.abs(freq - 1 / 20))
    assert abs(h[i]) < 1e-3
