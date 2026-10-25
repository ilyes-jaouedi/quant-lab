"""Run experiments described by YAML configs (see ``configs/``).

Every strategy in a config shares the same period, cash flows, costs and cash rate, so their
results are directly comparable.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from fin.backtest.engine import BacktestConfig, BacktestResult, Costs, run_backtest
from fin.data import load_bars, load_market_data
from fin.strategy import create


def load_config(path: Path | str) -> dict:
    """Parse an experiment YAML; ``name`` defaults to the file name."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg.setdefault("name", path.stem)
    return cfg


def resolve_cash_rate(spec) -> float | pd.Series:
    """A number (annual, decimal) or a ticker quoting an annual yield in percent, e.g. ^IRX."""
    if spec is None:
        return 0.0
    if isinstance(spec, (int, float)):
        return float(spec)
    return load_bars(str(spec))["close"] / 100.0


def backtest_config(cfg: dict) -> BacktestConfig:
    contribution = cfg.get("contribution") or {}
    return BacktestConfig(
        start=cfg.get("start"),
        end=cfg.get("end"),
        initial_capital=float(cfg.get("initial_capital", 10_000.0)),
        contribution=float(contribution.get("amount", 0.0)),
        contribution_freq=contribution.get("frequency", "M"),
        costs=Costs(**(cfg.get("costs") or {})),
        cash_rate=resolve_cash_rate(cfg.get("cash_rate")),
        execution=cfg.get("execution", "next_open"),
        min_trade=float(cfg.get("min_trade", 0.0)),
    )


def run_config(
    cfg: dict, save: bool = True, root: Path | None = None
) -> list[tuple[BacktestResult, str | None]]:
    """Backtest every strategy in ``cfg``; returns (result, run id or None) pairs.

    The first strategy is the benchmark unless the config names one with ``benchmark:``.
    """
    from fin.tracking import save_run

    strategies = []
    for spec in cfg["strategies"]:
        strategy = create(spec["strategy"], **(spec.get("params") or {}))
        if "rebalance" in spec:
            strategy.rebalance = spec["rebalance"]
        strategies.append((spec.get("label", spec["strategy"]), strategy, spec))

    tickers = sorted({t for _, s, _ in strategies for t in s.universe})
    data = load_market_data(tickers)  # full history: strategies may need warm-up data
    bt_cfg = backtest_config(cfg)

    # Runs from one invocation form a batch: same data, period and settings, so comparable.
    batch = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.get('name', 'experiment')}"
    benchmark = cfg.get("benchmark", strategies[0][0])
    out = []
    for order, (label, strategy, spec) in enumerate(strategies):
        result = run_backtest(strategy, data, bt_cfg, label=label)
        run_id = None
        if save:
            run_id = save_run(
                result,
                spec={"config": cfg, "strategy": spec},
                root=root,
                extra={
                    "batch": batch,
                    "experiment": cfg.get("name"),
                    "order": order,
                    "benchmark": benchmark,
                },
            )
        out.append((result, run_id))
    return out
