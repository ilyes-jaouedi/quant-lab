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
