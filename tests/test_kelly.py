import numpy as np
import pandas as pd
import pytest

from fin.stats import (
    bet_growth,
    growth_rate,
    kelly_bet,
    kelly_fraction,
    kelly_fraction_se,
    kelly_shrinkage,
    leveraged_returns,
    log_growth,
    rebalancing_bonus,
)

MU, SIGMA, RF = 0.08, 0.18, 0.02


def test_kelly_fraction_maximizes_growth():
    f = np.linspace(-1, 5, 60001)
    best = f[np.argmax(growth_rate(f, MU, SIGMA, RF))]
    assert best == pytest.approx(kelly_fraction(MU, SIGMA, RF), abs=1e-4)


def test_growth_parabola_landmarks():
    k = kelly_fraction(MU, SIGMA, RF)
    sharpe = (MU - RF) / SIGMA
    assert growth_rate(k, MU, SIGMA, RF) == pytest.approx(RF + sharpe**2 / 2)
    assert growth_rate(2 * k, MU, SIGMA, RF) == pytest.approx(RF)  # overbetting x2 = cash
    excess = growth_rate(k, MU, SIGMA, RF) - RF
    assert growth_rate(k / 2, MU, SIGMA, RF) - RF == pytest.approx(0.75 * excess)


def test_kelly_bet_on_a_biased_coin():
    assert kelly_bet(0.6) == pytest.approx(0.2)
    assert kelly_bet(0.5) == 0.0
    f = np.linspace(0, 0.99, 9901)
    assert f[np.argmax(bet_growth(f, 0.55, 2.0))] == pytest.approx(kelly_bet(0.55, 2.0), abs=1e-4)
    assert bet_growth(1.0, 0.6) == -np.inf  # all-in: one loss is ruin


def test_leveraged_returns_borrow_and_lend():
    r = pd.Series([0.01, -0.02, 0.03])
    pd.testing.assert_series_equal(leveraged_returns(r, 1.0, rf=0.001), r)
    np.testing.assert_allclose(leveraged_returns(r, 2.0, rf=0.001, spread=0.0005), 2 * r - 0.0015)
    np.testing.assert_allclose(leveraged_returns(r, 0.5, rf=0.001, spread=0.0005), 0.5 * r + 0.0005)


def test_log_growth():
    assert log_growth(np.full(252, 0.001)) == pytest.approx(252 * np.log(1.001))
    assert log_growth([0.1, -1.0, 0.1]) == -np.inf


def test_simulated_growth_follows_the_parabola():
    rng = np.random.default_rng(0)
    n = 252 * 2000
    r = pd.Series(rng.normal(MU / 252, SIGMA / np.sqrt(252), n))
    for lev in [0.5, 1.0, 2.0, 3.0]:
        realized = log_growth(leveraged_returns(r, lev, rf=RF / 252))
        # The sample mean is noisy (se of mu over 2000 years ~ 0.4%), so compare with the
        # parabola evaluated at the sample moments.
        mu_hat, sig_hat = r.mean() * 252, r.std() * np.sqrt(252)
        assert realized == pytest.approx(growth_rate(lev, mu_hat, sig_hat, RF), abs=2e-3)


def test_kelly_se_and_shrinkage():
    assert kelly_fraction_se(0.2, 25) == pytest.approx(1.0)
    assert kelly_shrinkage(2.0, 0.0) == 1.0
    assert kelly_shrinkage(1.0, 1.0) == pytest.approx(0.5)


def test_rebalancing_bonus_formula_and_simulation():
    sigma = 0.3
    cov = np.diag([sigma**2, sigma**2])
    w = np.array([0.5, 0.5])
    assert rebalancing_bonus(w, cov) == pytest.approx(sigma**2 / 4)
    assert rebalancing_bonus([1.0], [[0.04]]) == 0.0
    assert rebalancing_bonus(w, np.full((2, 2), 0.04)) == pytest.approx(0.0)

    # Two independent assets with zero growth each: the daily rebalanced mix still grows.
    rng = np.random.default_rng(1)
    mu = sigma**2 / 2  # arithmetic mean that makes each asset's log growth zero
    r = rng.normal(mu / 252, sigma / np.sqrt(252), (252 * 2000, 2))
    assets = [log_growth(r[:, i]) for i in range(2)]
    mix = log_growth(r @ w)
    assert mix - np.dot(w, assets) == pytest.approx(sigma**2 / 4, abs=2e-3)
