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
