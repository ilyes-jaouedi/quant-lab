import numpy as np
import pandas as pd
import pytest
from test_engine import FREE, Const, make_md

from fin.backtest import (
    BacktestConfig,
    max_drawdown,
    return_metrics,
    run_backtest,
    walk_forward,
    xirr,
)
from fin.signals import kinks, l1_trend_filter, l1_trend_lam_max
from fin.strategy import create, discover
from fin.tracking import list_runs, load_run, save_run

# --- metrics ---------------------------------------------------------------------------------


def test_max_drawdown():
    r = pd.Series([0.1, -0.5, 0.2])
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_xirr_one_year_ten_percent():
    dates = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])
    assert xirr(dates, np.array([-100.0, 110.0])) == pytest.approx(0.10, abs=2e-3)


def test_cagr_of_constant_growth():
    r = pd.Series([0.0003] * 2520)
    assert return_metrics(r)["cagr"] == pytest.approx(1.0003**252 - 1)


def test_result_metrics_dca_irr_matches_flat_price():
    md = make_md({"A": [100] * 200})
    res = run_backtest(
        Const({"A": 1.0}), md, BacktestConfig(initial_capital=0, contribution=100, costs=FREE)
    )
    m = res.metrics()
    assert m["final_value"] == pytest.approx(m["contributed"])
    assert m["irr"] == pytest.approx(0.0, abs=1e-6)


# --- L1 trend filter -------------------------------------------------------------------------


def test_l1_trend_recovers_piecewise_linear_signal():
    t = np.arange(300, dtype=float)
    truth = np.where(
        t < 100, 0.01 * t, np.where(t < 200, 1 - 0.005 * (t - 100), 0.5 + 0.02 * (t - 200))
    )
    y = truth + np.random.default_rng(0).normal(0, 0.02, len(t))
    x = l1_trend_filter(y, 0.02 * l1_trend_lam_max(y))
    assert np.sqrt(np.mean((x - truth) ** 2)) < 0.02
    # Kinks cluster around the true change points at 100 and 200.
    k = kinks(x, tol=1e-3)
    assert any(abs(k - 100) < 10) and any(abs(k - 200) < 10)


@pytest.mark.parametrize("lam_rel", [0.01, 0.1, 0.5])
def test_l1_trend_matches_generic_convex_solver(lam_rel):
    cp = pytest.importorskip("cvxpy")
    from fin.signals import second_difference

    y = np.log(100) + np.random.default_rng(2).normal(0.0003, 0.01, 400).cumsum()
    lam = lam_rel * l1_trend_lam_max(y)
    x = cp.Variable(len(y))
    objective = 0.5 * cp.sum_squares(y - x) + lam * cp.norm1(second_difference(len(y)) @ x)
    cp.Problem(cp.Minimize(objective)).solve(
        solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12
    )
    np.testing.assert_allclose(l1_trend_filter(y, lam), x.value, atol=1e-5)


def test_l1_trend_above_lam_max_is_a_straight_line():
    y = np.random.default_rng(1).normal(size=80).cumsum()
    x = l1_trend_filter(y, 1.01 * l1_trend_lam_max(y))
    line = np.polyval(np.polyfit(np.arange(80), y, 1), np.arange(80))
    np.testing.assert_allclose(x, line, atol=1e-3 * np.ptp(y))


# --- walk-forward ----------------------------------------------------------------------------


class OnOff(Const):
    name = "on_off"

    def __init__(self, on: bool):
        super().__init__({"A": 1.0 if on else 0.0})
        self.on = on


def test_walk_forward_picks_params_on_past_and_stitches_test_windows():
    rng = np.random.default_rng(0)
    p = 100 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.005, 252 * 6)))  # rising market
    md = make_md({"A": p}, start="2010-01-01")
    wf = walk_forward(
        OnOff,
        [{"on": False}, {"on": True}],
        md,
        BacktestConfig(costs=FREE, cash_rate=0.01),
        train_years=2,
        test_years=1,
    )
    assert len(wf.choices) >= 3
    assert wf.choices["on"].all()
    assert wf.returns.index.is_monotonic_increasing
    assert wf.returns.index[0] >= md.index[0] + pd.DateOffset(years=2)
    assert wf.metrics()["cagr"] > 0


# --- registry and run store ------------------------------------------------------------------


def test_registry_discovers_project_strategies():
    names = set(discover())
    assert {"buy_and_hold", "vol_target", "sma_trend", "l1_trend"} <= names
    s = create("buy_and_hold", weights={"X": 1.0})
    assert s.universe == ["X"] and s.params() == {"weights": {"X": 1.0}}


def test_run_store_roundtrip(tmp_path):
    md = make_md({"A": [100, 101, 102, 103, 104]})
    res = run_backtest(Const({"A": 1.0}), md, BacktestConfig(costs=FREE), label="my test")
    run_id = save_run(res, spec={"note": "x"}, root=tmp_path)
    runs = list_runs(tmp_path)
    assert runs["id"].tolist() == [run_id]
    loaded = load_run(run_id, root=tmp_path)
    pd.testing.assert_series_equal(loaded["series"]["nav"], res.nav, check_freq=False)
    assert loaded["meta"]["label"] == "my test"
    assert loaded["meta"]["metrics"]["final_value"] == pytest.approx(res.nav.iloc[-1])
