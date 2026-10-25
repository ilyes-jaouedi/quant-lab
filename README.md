# fin

Personal quant research infrastructure. The library in `src/fin/` is stable and tested;
strategies are small plugins; research is free-form.

```
src/fin/      infrastructure library (data, signals, stats, backtest, execution, api)
strategies/   one file per strategy (the only thing that changes between ideas)
research/     notebooks, numbered: 001_topic.ipynb — messy is fine
configs/      experiment configs
tests/        tests for the infrastructure
data/         local market-data cache (git-ignored)
runs/         saved experiment results (git-ignored)
```

Code that gets reused in `research/` graduates into `src/fin/` with a test.

## Library modules

| Module | Contents |
|---|---|
| `fin.data` | providers, Parquet cache, `MarketData` |
| `fin.signals` | volatility forecasters: rolling, EWMA, out-of-sample GARCH(1,1) |
| `fin.stats` | ACF, Ljung-Box, Lo-MacKinlay variance ratio, OOS R², Mincer-Zarnowitz, QLIKE |
| `fin.plotting` | shared matplotlib style (`use_style()`, fixed `SERIES` colors) |

Convention for every signal/forecast: the value at date `t` uses data up to `t` only and is a
forecast for after `t`. `fin.stats.next_period` builds matching targets.

## Research notebooks

Notebooks are paired with a `.py` file via jupytext: edit either one in Jupyter and both stay
in sync, and the `.py` keeps git diffs readable.

```sh
uv run jupyter lab                    # open research/*.ipynb
uv run jupytext --sync research/*.py  # after editing the .py outside Jupyter
```

| Notebook | Finding |
|---|---|
| `001_returns_vs_volatility_predictability` | returns ≈ unpredictable (OOS R² < 0); volatility predictable (GARCH explains ~20–25% of next-month variance) |

## Setup

```sh
uv sync                     # create .venv and install
uv run pytest               # run tests
uv run fin fetch IWDA.AS    # download + cache daily bars
uv run fin cache            # list cached series
```

## Data layer

```python
from fin.data import load_market_data

md = load_market_data(["IWDA.AS", "^GSPC"], start="2010-01-01")
md.close  # wide DataFrame: dates x tickers
md.returns()  # log returns from adjusted close
md.until("2015-12-31")  # point-in-time view: nothing after that date
```

Bars are cached as Parquet under `data/<provider>/`. Later calls only download the missing
tail; if the provider has revised past adjusted prices (e.g. a dividend), the series is
re-downloaded in full. Set `FIN_DATA_DIR` to put the cache elsewhere.
