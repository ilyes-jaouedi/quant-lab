"""Command line entry point: ``uv run fin --help``."""

from typing import Annotated

import typer

from fin.config import DATA_DIR
from fin.data import BarCache, DataError, load_bars

app = typer.Typer(no_args_is_help=True, help="Personal quant research toolkit.")


@app.command()
def fetch(
    tickers: Annotated[list[str], typer.Argument(help="Yahoo tickers, e.g. IWDA.AS ^GSPC")],
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD; default: full history")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD; default: today")] = None,
    provider: str = "yahoo",
    refresh: Annotated[bool, typer.Option(help="Ignore the cache and re-download")] = False,
):
    """Download daily bars into the local cache."""
    failed = False
    for ticker in tickers:
        try:
            bars = load_bars(ticker, start, end, provider=provider, refresh=refresh)
        except DataError as e:
            typer.secho(f"{ticker:<12} FAILED: {e}", fg="red")
            failed = True
            continue
        first, last = bars.index[0].date(), bars.index[-1].date()
        typer.echo(f"{ticker:<12} {len(bars):>6} bars  {first} -> {last}")
    if failed:
        raise typer.Exit(1)


METRIC_FORMATS = {
    "cagr": "{:.2%}",
    "vol": "{:.2%}",
    "sharpe": "{:.2f}",
    "sortino": "{:.2f}",
    "max_drawdown": "{:.1%}",
    "calmar": "{:.2f}",
    "irr": "{:.2%}",
    "final_value": "{:,.0f}",
    "contributed": "{:,.0f}",
    "avg_exposure": "{:.0%}",
    "turnover": "{:.2f}",
    "costs": "{:,.0f}",
    "n_trades": "{:.0f}",
}


def _metrics_table(rows: dict[str, dict]) -> str:
    import pandas as pd

    table = {
        label: {
            metric: fmt.format(m[metric]) if pd.notna(m.get(metric, None)) else "-"
            for metric, fmt in METRIC_FORMATS.items()
        }
        for label, m in rows.items()
    }
    return pd.DataFrame(table).to_string()


@app.command()
def backtest(
    config: Annotated[str, typer.Argument(help="Experiment YAML, e.g. configs/world_cto.yaml")],
    save: Annotated[bool, typer.Option(help="Save runs under runs/")] = True,
):
    """Backtest every strategy in a config and print a comparison."""
    from fin.backtest.runner import load_config, run_config

    results = run_config(load_config(config), save=save)
    first = results[0][0].nav.index
    typer.echo(f"{first[0].date()} -> {first[-1].date()}\n")
    typer.echo(_metrics_table({r.label: r.metrics() for r, _ in results}))
    if save:
        typer.echo("\nsaved: " + ", ".join(run_id for _, run_id in results))


@app.command()
def runs(limit: int = 20):
    """List saved backtest runs, newest first."""
    from fin.tracking import list_runs

    df = list_runs()
    if df.empty:
        typer.echo("no runs yet")
        return
    cols = [c for c in ["id", "cagr", "vol", "sharpe", "max_drawdown", "irr"] if c in df]
    typer.echo(df[cols].head(limit).to_string(index=False, float_format="{:.4f}".format))


@app.command()
def cache():
    """List cached series."""
    entries = BarCache(DATA_DIR).entries()
    if not entries:
        typer.echo(f"cache is empty ({DATA_DIR})")
        return
    for e in entries:
        first, last = e.bars.index[0].date(), e.bars.index[-1].date()
        typer.echo(f"{e.provider:<8} {e.ticker:<12} {len(e.bars):>6} bars  {first} -> {last}")


if __name__ == "__main__":
    app()
