"""Experiment store: every backtest is saved with its config, metrics and code version.

Layout: ``runs/<YYYYmmdd-HHMMSS>_<label>/`` with ``meta.json``, ``series.parquet``
(nav, returns, flows), ``weights.parquet`` and ``trades.parquet``.
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fin.config import PROJECT_ROOT, RUNS_DIR


def git_info() -> dict:
    def git(*args):
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, pd.Series):
        return f"<series {x.name}>"
    return str(x)


def clean_json(x):
    """Replace NaN/inf by None, recursively, so the value is strict JSON."""
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def save_run(
    result, spec: dict | None = None, root: Path | None = None, extra: dict | None = None
) -> str:
    """Persist a BacktestResult; returns the run id. ``extra`` is merged into meta.json."""
    root = Path(root or RUNS_DIR)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", result.label).strip("-")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{slug}"
    run_dir = root / run_id
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    pd.concat([result.nav, result.returns, result.flows], axis=1).to_parquet(
        run_dir / "series.parquet"
    )
    result.weights.to_parquet(run_dir / "weights.parquet")
    result.trades.to_parquet(run_dir / "trades.parquet")

    cfg = result.config
    meta = {
        "id": run_dir.name,
        "label": result.label,
        "created": datetime.now().isoformat(timespec="seconds"),
        "strategy": {"name": result.strategy.name, "params": result.strategy.params()},
        "rebalance": result.strategy.rebalance,
        "config": {
            "start": result.nav.index[0].date().isoformat(),
            "end": result.nav.index[-1].date().isoformat(),
            "initial_capital": cfg.initial_capital,
            "contribution": cfg.contribution,
            "contribution_freq": cfg.contribution_freq,
            "costs": {"bps": cfg.costs.bps, "fixed": cfg.costs.fixed},
            "cash_rate": cfg.cash_rate,
            "execution": cfg.execution,
            "min_trade": cfg.min_trade,
        },
        "spec": spec or {},
        "metrics": result.metrics(),
        "git": git_info(),
    } | (extra or {})
    meta = clean_json(meta)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_jsonable))
    return run_dir.name


def list_runs(root: Path | None = None) -> pd.DataFrame:
    """One row per saved run, newest first: id, label, created and the main metrics."""
    root = Path(root or RUNS_DIR)
    rows = []
    for meta_path in root.glob("*/meta.json"):
        meta = json.loads(meta_path.read_text())
        rows.append(
            {"id": meta["id"], "label": meta["label"], "created": meta["created"]}
            | meta.get("metrics", {})
        )
    if not rows:
        return pd.DataFrame(columns=["id", "label", "created"])
    return pd.DataFrame(rows).sort_values("id", ascending=False, ignore_index=True)


def load_run(run_id: str, root: Path | None = None) -> dict:
    run_dir = Path(root or RUNS_DIR) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"no run {run_id!r} in {run_dir.parent}")
    return {
        "meta": json.loads((run_dir / "meta.json").read_text()),
        "series": pd.read_parquet(run_dir / "series.parquet"),
        "weights": pd.read_parquet(run_dir / "weights.parquet"),
        "trades": pd.read_parquet(run_dir / "trades.parquet"),
    }
