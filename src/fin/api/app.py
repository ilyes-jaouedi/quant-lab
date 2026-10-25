"""HTTP API over the library: experiments, saved runs and cached market data.

``fin serve`` runs it; the dashboard in ``ui/`` is its client. When ``ui/dist`` exists (after
``npm run build``), the API also serves the dashboard itself on ``/``.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fin.backtest.runner import load_config, run_config
from fin.config import DATA_DIR, PROJECT_ROOT, RUNS_DIR
from fin.data import BarCache, DataError, load_bars
from fin.tracking import clean_json, load_run


def _dates(index: pd.DatetimeIndex) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in index]


def _values(s: pd.Series, digits: int = 6) -> list:
    return clean_json(np.round(s.to_numpy(dtype=float), digits).tolist())


def _leading_comment(path: Path) -> str:
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("# ").rstrip())
    return " ".join(lines)


def create_app(
    runs_dir: Path | None = None,
    configs_dir: Path | None = None,
    data_dir: Path | None = None,
    ui_dir: Path | None = None,
) -> FastAPI:
    runs_dir = Path(runs_dir or RUNS_DIR)
    configs_dir = Path(configs_dir or PROJECT_ROOT / "configs")
    data_dir = Path(data_dir or DATA_DIR)
    ui_dir = Path(ui_dir or PROJECT_ROOT / "ui" / "dist")
    app = FastAPI(title="fin", version="0.1.0")

    def metas() -> list[dict]:
        out = []
        for p in runs_dir.glob("*/meta.json"):
            meta = json.loads(p.read_text())
            meta.setdefault("batch", meta["id"])
            meta.setdefault("experiment", meta["label"])
            meta.setdefault("order", 0)
            out.append(meta)
        return out

    def summarize(batch: str, runs: list[dict]) -> dict:
        runs = sorted(runs, key=lambda m: m["order"])
        first = runs[0]
        return {
            "batch": batch,
            "experiment": first["experiment"],
            "created": min(m["created"] for m in runs),
            "start": first["config"]["start"],
            "end": first["config"]["end"],
            "benchmark": first.get("benchmark", first["label"]),
            "config": first["config"],
            "runs": [
                {
                    "id": m["id"],
                    "label": m["label"],
                    "strategy": m["strategy"]["name"],
                    "params": m["strategy"]["params"],
                    "metrics": m["metrics"],
                }
                for m in runs
            ],
        }

    @app.get("/api/experiments")
    def experiments():
        out = []
        for path in sorted(configs_dir.glob("*.yaml")):
            cfg = load_config(path)
            out.append(
                {
                    "name": cfg["name"],
                    "description": _leading_comment(path),
                    "start": str(cfg.get("start") or ""),
                    "strategies": [s.get("label", s["strategy"]) for s in cfg["strategies"]],
                }
            )
        return out

    @app.post("/api/experiments/{name}/run")
    def run_experiment(name: str):
        path = configs_dir / f"{name}.yaml"
        if not path.exists():
            raise HTTPException(404, f"No experiment config named {name!r} in configs/")
        try:
            results = run_config(load_config(path), save=True, root=runs_dir)
        except (DataError, ValueError) as e:
            raise HTTPException(422, str(e)) from e
        run_id = results[0][1]
        return {"batch": json.loads((runs_dir / run_id / "meta.json").read_text())["batch"]}

    @app.get("/api/batches")
    def batches():
        groups = defaultdict(list)
        for meta in metas():
            groups[meta["batch"]].append(meta)
        out = [summarize(b, runs) for b, runs in groups.items()]
        return clean_json(sorted(out, key=lambda b: b["created"], reverse=True))

    @app.get("/api/batches/{batch}")
    def batch_detail(batch: str):
        runs = [m for m in metas() if m["batch"] == batch]
        if not runs:
            raise HTTPException(404, f"No batch {batch!r}")
        out = summarize(batch, runs)
        for run in out["runs"]:
            loaded = load_run(run["id"], root=runs_dir)
            s = loaded["series"]
            active = s.index[s["nav"] > 0]
            s = s.loc[active[0] :] if len(active) else s
            growth = (1 + s["returns"].iloc[1:]).cumprod()
            growth = pd.concat([pd.Series([1.0], index=s.index[:1]), growth])
            exposure = loaded["weights"].loc[s.index].sum(axis=1)
            run["series"] = {
                "dates": _dates(s.index),
                "value": _values(s["nav"], 2),
                "growth": _values(growth),
                "drawdown": _values(growth / growth.cummax() - 1),
                "exposure": _values(exposure, 4),
            }
        return clean_json(out)

    @app.get("/api/runs/{run_id}/trades")
    def trades(run_id: str):
        try:
            t = load_run(run_id, root=runs_dir)["trades"]
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        t = t.assign(date=t["date"].dt.strftime("%Y-%m-%d")) if len(t) else t
        return clean_json(t.to_dict(orient="records"))

    @app.get("/api/data")
    def cached_data():
        return [
            {
                "ticker": e.ticker,
                "provider": e.provider,
                "first": e.bars.index[0].strftime("%Y-%m-%d"),
                "last": e.bars.index[-1].strftime("%Y-%m-%d"),
                "days": len(e.bars),
            }
            for e in BarCache(data_dir).entries()
        ]

    @app.get("/api/data/{ticker}")
    def price_series(ticker: str, provider: str = "yahoo"):
        entry = BarCache(data_dir).read(provider, ticker)
        if entry is None:
            raise HTTPException(404, f"{ticker} is not downloaded yet")
        bars = entry.bars
        return {
            "ticker": ticker,
            "dates": _dates(bars.index),
            "close": _values(bars["close"], 4),
            "adj_close": _values(bars["adj_close"], 4),
        }

    @app.post("/api/data/{ticker}/fetch")
    def fetch_ticker(ticker: str):
        try:
            bars = load_bars(ticker, cache=BarCache(data_dir))
        except DataError as e:
            raise HTTPException(404, f"Yahoo has no data for {ticker}. Check the ticker.") from e
        return {"ticker": ticker, "days": len(bars)}

    if ui_dir.exists():
        app.mount("/assets", StaticFiles(directory=ui_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            file = ui_dir / path
            return FileResponse(file if path and file.is_file() else ui_dir / "index.html")

    return app
