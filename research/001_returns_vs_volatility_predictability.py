# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 001 — Are returns predictable? Is volatility?
#
# **Hypotheses**
#
# - **H1** Daily returns are (almost) serially uncorrelated: past returns barely predict
#   future returns.
# - **H2** Absolute returns are strongly and persistently autocorrelated: volatility *is*
#   predictable.
# - **H3** The small return autocorrelation that once existed has faded over time (markets
#   adapt once a pattern is known and traded).
#
# **Data**: S&P 500 index (`^GSPC`, 1962 → today) for the long history, and the iShares MSCI
# World ETF (`IWDA.AS`, 2009 → today) as the asset actually held. Log returns from adjusted
# prices.
#
# **Rules**: every forecast is out-of-sample: the value at date *t* only uses data up to *t*
# and is scored against what happens after *t*.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fin.data import load_market_data
from fin.plotting import SERIES, use_style
from fin.signals import annualize, ewma_vol, garch_vol, rolling_vol
from fin.stats import (
    acf,
    acf_band,
    ljung_box,
    mincer_zarnowitz,
    next_period,
    oos_r2,
    qlike,
    variance_ratio_table,
)

use_style()
pd.set_option("display.float_format", "{:.4f}".format)

md = load_market_data(["^GSPC", "IWDA.AS"], start="1962-01-01")
r = md.returns()
assets = {"S&P 500": r["^GSPC"].dropna(), "MSCI World (IWDA)": r["IWDA.AS"].dropna()}
spx = assets["S&P 500"]
{name: f"{s.index[0].date()} → {s.index[-1].date()} ({len(s)} days)" for name, s in assets.items()}

# %% [markdown]
# ## 1. Stylized facts
#
# Before testing anything: what do the returns look like? Two features matter for everything
# below: **heavy tails** (excess kurtosis far above the Gaussian 0) and **volatility
# clustering** (turbulent days come in bursts).

# %%
pd.DataFrame(
    {
        name: {
            "ann. mean": s.mean() * 252,
            "ann. vol": annualize(s.std()),
            "skew": s.skew(),
            "excess kurtosis": s.kurt(),
            "worst day": s.min(),
            "best day": s.max(),
        }
        for name, s in assets.items()
    }
)

# %%
fig, ax = plt.subplots()
ax.plot(spx.index, spx * 100, lw=0.4, color=SERIES[0])
ax.set_title("S&P 500 daily log returns (%): calm and turbulent periods cluster")
ax.set_ylabel("daily return, %")
plt.show()

# %% [markdown]
# ## 2. Autocorrelation: returns vs absolute returns (H1, H2)
#
# If returns were i.i.d., the sample ACF at each lag would be ~N(0, 1/n): inside the
# dashed band 95% of the time. We compare the ACF of $r_t$ with the ACF of $|r_t|$.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (label, x, nlags) in zip(
    axes, [("returns  r", spx, 40), ("absolute returns  |r|", spx.abs(), 250)], strict=True
):
    a = acf(x, nlags)
    band = acf_band(len(x))
    ax.vlines(a.index, 0, a.values, color=SERIES[0], lw=1)
    ax.axhspan(-band, band, color=SERIES[0], alpha=0.12, lw=0)
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_title(f"S&P 500: ACF of {label}")
    ax.set_xlabel("lag (trading days)")
axes[0].set_ylabel("autocorrelation")
plt.show()

# %%
pd.concat(
    {
        f"{name} {kind}": ljung_box(x, lags=[5, 10, 20])
        for name, s in assets.items()
        for kind, x in [("r", s), ("|r|", s.abs())]
    },
    axis=1,
)

# %% [markdown]
# ## 3. Variance ratio test (H1, H3)
#
# Lo & MacKinlay (1988): for a random walk, the variance of q-day returns is q times the
# variance of 1-day returns, so $VR(q) = 1$. $VR > 1$ means trending (positive
# autocorrelation), $VR < 1$ mean reversion. We use the heteroskedasticity-robust z, since
# section 2 shows the variance is anything but constant.

# %%
pd.concat(
    {name: variance_ratio_table(s, qs=[2, 5, 10, 20, 60]) for name, s in assets.items()},
    axis=1,
).loc[:, (slice(None), ["vr", "z_robust", "p_value"])]

# %% [markdown]
# Is the dependence stable over time? Same test, decade by decade (H3).

# %%
by_decade = {}
for decade, s in spx.groupby((spx.index.year // 10) * 10):
    t = variance_ratio_table(s, qs=[2, 5])
    by_decade[f"{decade}s"] = {
        "lag-1 autocorr": s.autocorr(1),
        "VR(2)": t.loc[2, "vr"],
        "z(2)": t.loc[2, "z_robust"],
        "VR(5)": t.loc[5, "vr"],
        "z(5)": t.loc[5, "z_robust"],
        "days": len(s),
    }
pd.DataFrame(by_decade).T

# %%
window = 5 * 252
rolling_rho = spx.rolling(window).corr(spx.shift(1))
band = acf_band(window)
fig, ax = plt.subplots()
ax.plot(rolling_rho.index, rolling_rho, color=SERIES[0])
ax.axhspan(-band, band, color=SERIES[0], alpha=0.12, lw=0)
ax.axhline(0, color="#52514e", lw=0.6)
ax.set_title("S&P 500: lag-1 autocorrelation of daily returns, rolling 5-year window")
ax.set_ylabel("autocorrelation")
plt.show()

# %% [markdown]
# ## 4. Volatility forecasting, out-of-sample (H2)
#
# Four forecasters of tomorrow's variance, all using only data up to today:
#
# | model | idea |
# |---|---|
# | **historical** | expanding-window average of $r^2$: "volatility is constant" (the baseline) |
# | **rolling 21d** | sample variance of the last month |
# | **EWMA** | RiskMetrics, λ = 0.94: a one-pole IIR low-pass on $r^2$ |
# | **GARCH(1,1)** | same IIR filter plus a mean-reversion term, parameters re-fit yearly |
#
# Targets: next-day $r^2$ (very noisy proxy) and the mean of $r^2$ over the next 21 days.
# Scores: Mincer-Zarnowitz $R^2$ (higher is better) and QLIKE loss (lower is better).


# %%
def vol_forecasts(s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "historical": s.pow(2).expanding(252).mean(),
            "rolling 21d": rolling_vol(s, 21) ** 2,
            "EWMA": ewma_vol(s) ** 2,
            "GARCH(1,1)": garch_vol(s, min_obs=1000, refit_every=252) ** 2,
        }
    ).dropna()


def score_vol(s: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for horizon in (1, 21):
        target = next_period(s**2, horizon)
        for name, f in forecasts.items():
            rows[(f"{horizon}d", name)] = {
                "MZ R2": mincer_zarnowitz(target, f)["r2"],
                "MZ slope b": mincer_zarnowitz(target, f)["b"],
                "QLIKE": qlike(target, f),
            }
    return pd.DataFrame(rows).T


vol_f = {name: vol_forecasts(s) for name, s in assets.items()}
pd.concat({name: score_vol(assets[name], f) for name, f in vol_f.items()}, axis=1)

# %%
f = vol_f["S&P 500"].loc["2019-07":"2021-06"]
realized = np.sqrt(next_period(spx**2, 21)).loc[f.index]
fig, ax = plt.subplots()
ax.plot(realized.index, annualize(realized) * 100, color=SERIES[2], label="realized, next 21 days")
ax.plot(f.index, annualize(np.sqrt(f["GARCH(1,1)"])) * 100, color=SERIES[0], label="GARCH(1,1)")
ax.plot(f.index, annualize(np.sqrt(f["historical"])) * 100, color=SERIES[1], label="historical")
ax.set_title("S&P 500 volatility forecasts around the COVID crash (annualized %)")
ax.set_ylabel("volatility, % per year")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# ## 5. Return forecasting, out-of-sample (H1)
#
# The fair comparison: the same simple model (AR(1), coefficients re-estimated every day on
# an expanding window) used to predict **tomorrow's return** vs **tomorrow's absolute
# return**. Score: Campbell-Thompson out-of-sample $R^2$ against the historical mean
# (positive = beats "just use the average").


# %%
def expanding_ar1_forecast(y: pd.Series, min_obs: int = 1000) -> pd.Series:
    """Forecast of y[t+1] made at t: a_t + b_t * y[t], OLS on pairs (y[s-1], y[s]) with s <= t."""
    x, z = y.shift(1), y
    valid = x.notna() & z.notna()
    x, z = x.where(valid), z.where(valid)
    n = valid.cumsum()
    mx, mz = x.expanding().mean(), z.expanding().mean()
    cov = (x * z).expanding().mean() - mx * mz
    var = (x * x).expanding().mean() - mx**2
    b = cov / var
    a = mz - b * mx
    return (a + b * y).where(n >= min_obs)


def score_returns(s: pd.Series) -> dict:
    out = {}
    for label, y in [("r", s), ("|r|", s.abs())]:
        target = next_period(y, 1)
        hist_mean = y.expanding(1000).mean()
        out[f"OOS R2 for {label}"] = oos_r2(target, expanding_ar1_forecast(y), hist_mean)
    return out


pd.DataFrame({name: score_returns(s) for name, s in assets.items()})

# %% [markdown]
# ## 6. Conclusions
#
# Numbers quoted from the run on data to 2026-10-23.
#
# **H1 (returns ≈ unpredictable): supported.**
# - The return ACF stays within or near the i.i.d. band. Ljung-Box does reject for the S&P 500
#   at 10–20 lags, but it assumes i.i.d. data and over-rejects under volatility clustering.
#   The heteroskedasticity-robust variance ratio test rejects at **no** horizon (all
#   |z| < 1.5, both assets).
# - Out-of-sample, an AR(1) forecast of tomorrow's return is **worse** than the historical
#   mean (OOS R² ≈ −0.2%). Statistically visible ≠ exploitable.
#
# **H2 (volatility is predictable): strongly supported.**
# - ACF of |r| ≈ 0.3 at lag 1 and still ≈ 0.05 after 250 days: long memory, a slow
#   (roughly hyperbolic) decay rather than GARCH's exponential one.
# - The same AR(1) that fails on returns gets OOS R² ≈ +7% on |r|.
# - GARCH(1,1) explains ~21% (S&P) / ~26% (MSCI World) of next-month realized variance; the
#   constant-volatility baseline explains ~0%. Ranking by QLIKE: GARCH < EWMA < rolling 21d <
#   historical, on both assets and horizons.
# - Around COVID, GARCH lags the shock (it's a causal filter, so there's phase delay) but adapts within
#   days, while "historical" never moves.
# - MZ slopes < 1, especially at 21 days: the forecasts over-react. Partly because a
#   1-step forecast is used for a 21-day horizon without GARCH's mean reversion.
#
# **H3 (dependence fades over time): supported, with a caveat.**
# - Lag-1 autocorrelation: +0.18 / +0.25 in the 1960s/70s, ~0 in the 1980s/90s, negative
#   since the 2000s (VR(2) significant at both ends).
# - Caveat: much of the early positive autocorrelation is a known measurement artifact of
#   indices (non-synchronous trading: stale prices of illiquid constituents), not a tradable
#   effect, since the index itself couldn't be traded back then.
# - The rolling chart jumps when one crash day (Oct 1987, Mar 2020) enters or leaves the
#   window: with excess kurtosis ≈ 25, Pearson autocorrelation is fragile.
#
# **So what**
# - The predictable part of the market is its **risk**, not its direction. The natural first
#   strategy to test is **volatility targeting**: scale exposure by target_vol / forecast_vol
#   (Moreira & Muir, 2017, *Volatility-Managed Portfolios*) against buy-and-hold and DCA.
# - Nothing here argues for changing a long-term DCA.
#
# **Follow-ups**
# - HAR-RV model (Corsi, 2009) to capture the long memory GARCH(1,1) misses.
# - Proper multi-step GARCH forecasts for the 21-day horizon.
# - Robust autocorrelation (rank-based, or after volatility normalization r/σ̂) for H3.
# - L1 trend filtering for regime detection (planned for Phase 2).
