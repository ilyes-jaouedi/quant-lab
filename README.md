# fin

Personal quant research infrastructure. The library in `src/fin/` is stable and tested;
strategies are small plugins; research is free-form.

```
src/fin/      infrastructure library (data, signals, stats, backtest, execution, api)
strategies/   one file per strategy (the only thing that changes between ideas)
research/     notebooks, numbered: 001_topic.ipynb — messy is fine
configs/      experiment configs
ui/           web dashboard (React + shadcn/ui), talks to the API in fin.api
tests/        tests for the infrastructure
data/         local market-data cache (git-ignored)
runs/         saved experiment results (git-ignored)
```

Code that gets reused in `research/` graduates into `src/fin/` with a test.

## Library modules

| Module | Contents |
|---|---|
| `fin.data` | providers, Parquet cache, `MarketData`, currency conversion (`currency="EUR"`) |
| `fin.signals` | volatility forecasters (rolling, EWMA, out-of-sample GARCH(1,1)); L1 trend filter (banded primal-dual interior point); trend rules as kernels on returns, with lag, noise gain and frequency response (`fin.signals.trend`) |
| `fin.stats` | ACF, Ljung-Box, Lo-MacKinlay variance ratio, OOS R², Mincer-Zarnowitz, QLIKE; significance: stationary bootstrap, paired tests vs a benchmark, probabilistic and deflated Sharpe ratios; growth-optimal betting: Kelly fraction and its standard error, growth rate with volatility drag, leveraged returns, rebalancing bonus (`fin.stats.kelly`) |
| `fin.plotting` | shared matplotlib style (`use_style()`, fixed `SERIES` colors) |
| `fin.strategy` | `Strategy` base class; discovers strategies from `strategies/*.py` |
| `fin.backtest` | daily long-only engine (costs, contributions, cash interest), metrics, walk-forward, YAML runner |
| `fin.tracking` | run store: every backtest saved under `runs/` with config, metrics and git commit |
| `fin.api` | FastAPI: experiments (list, run), batches of runs, trades, cached market data |

Convention for every signal/forecast: the value at date `t` uses data up to `t` only and is a
forecast for after `t`. `fin.stats.next_period` builds matching targets.

## Strategies and backtests

A strategy is one small class in `strategies/`:

```python
class SmaTrend(Strategy):
    name = "sma_trend"          # used in configs
    rebalance = "M"             # decide on the last trading day of each month

    def __init__(self, ticker: str, months: int = 10):
        self.ticker, self.months, self.universe = ticker, months, [ticker]

    def target_weights(self, data):        # data = MarketData up to today only
        price = data.adj_close[self.ticker].dropna()
        if len(price) < 21 * self.months:
            return None                    # not enough history: keep the portfolio
        up = price.iloc[-1] > price.iloc[-21 * self.months:].mean()
        return pd.Series({self.ticker: 1.0 if up else 0.0})   # rest stays in cash
```

Experiments are YAML files in `configs/`: one period, cash-flow schedule, cost model and cash
rate shared by every strategy listed, so the results are comparable.

```sh
uv run fin backtest configs/world_cto.yaml   # run, print comparison, save runs
uv run fin runs                              # list saved runs
```

Every experiment tests each strategy against its benchmark (the first strategy, or
`benchmark:`) with a paired stationary bootstrap, and reports the probabilistic and deflated
Sharpe ratios: `return_diff_p` and `sharpe_diff_p` near 0 mean the edge is unlikely to be luck.

`buy_and_hold` never sells (weights drift, new cash buys what is underweight); `constant_mix`
rebalances to fixed weights. Configs convert all prices to `currency:` (default EUR) with
daily FX rates.

Engine rules: decisions use data up to the close of day `t` and execute at the next open;
prices are total-return adjusted; long-only, no leverage (weights >= 0, sum <= 1); returns are
time-weighted, and `irr` gives the money-weighted return of the actual cash flows.

## Dashboard

```sh
cd ui && npm install && npm run build && cd ..   # once, and after UI changes
uv run fin serve                                 # API + dashboard on http://localhost:8000
```

- **Experiments**: run any config in `configs/` and browse past results.
- **Experiment page**: one sentence per strategy against the benchmark (the first strategy in the
  config, or `benchmark:`), growth and drawdown charts with 1Y-All windows, the full metrics
  table, and each strategy's exposure and orders.
- **Market data**: download Yahoo tickers into the cache and plot them.

UI development with hot reload: `uv run fin serve` in one terminal, `cd ui && npm run dev` in
another (Vite proxies `/api` to port 8000). Stack: Vite, React, TypeScript, Tailwind, shadcn/ui
(Chalk Slate theme from 21st.dev, Instrument Sans), TradingView lightweight-charts.

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
| `002_luck_and_multiple_testing` | the best of 500 random strategies "passes" a naive test (p = 0.04), exactly as extreme value theory predicts; SMA lookback ranks in-sample don't predict out-of-sample (ρ = −0.08) |
| `003_trend_as_filter_design` | trend rules are kernels on past returns; the SMA vs momentum gap was a lag mismatch; 11/12 trend variants beat on Sharpe and halve drawdowns, none significant alone; payoff convex (t = 3.5) |
| `004_kelly_and_volatility_drag` | growth follows g(f) = r + f(μ−r) − f²σ²/2 on the S&P 500 (Kelly f* ≈ 2.4, −96% drawdown); f* is too noisy to estimate (se ≈ 1.8 over 10 years): estimated Kelly loses to holding the index, −11%/yr vs +6%/yr since 2000; rebalancing adds 1.4%/yr over the average asset |

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
