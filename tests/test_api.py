import pytest
from fastapi.testclient import TestClient
from test_engine import FREE, Const, make_md

from fin.api import create_app
from fin.backtest import BacktestConfig, run_backtest
from fin.tracking import save_run


@pytest.fixture
def client(tmp_path):
    runs, configs, data = tmp_path / "runs", tmp_path / "configs", tmp_path / "data"
    configs.mkdir()
    (configs / "demo.yaml").write_text(
        "# A demo experiment.\nstrategies:\n  - {label: hold, strategy: buy_and_hold, "
        "params: {weights: {A: 1.0}}}\n"
    )
    md = make_md({"A": [100, 101, 99, 103, 104]})
    for order, (label, w) in enumerate([("hold", 1.0), ("half", 0.5)]):
        res = run_backtest(Const({"A": w}), md, BacktestConfig(costs=FREE), label=label)
        save_run(
            res,
            root=runs,
            extra={"batch": "b1", "experiment": "demo", "order": order, "benchmark": "hold"},
        )
    return TestClient(create_app(runs_dir=runs, configs_dir=configs, data_dir=data))


def test_experiments_lists_configs(client):
    exps = client.get("/api/experiments").json()
    assert exps == [
        {"name": "demo", "description": "A demo experiment.", "start": "", "strategies": ["hold"]}
    ]


def test_batches_group_runs_in_config_order(client):
    batches = client.get("/api/batches").json()
    assert len(batches) == 1
    assert batches[0]["benchmark"] == "hold"
    assert [r["label"] for r in batches[0]["runs"]] == ["hold", "half"]


def test_batch_detail_has_aligned_series(client):
    detail = client.get("/api/batches/b1").json()
    s = detail["runs"][1]["series"]
    n = len(s["dates"])
    assert all(len(s[k]) == n for k in ("value", "growth", "drawdown", "exposure"))
    assert s["growth"][0] == 1.0
    assert max(s["drawdown"]) <= 0


def test_trades_and_missing_resources(client):
    run_id = client.get("/api/batches").json()[0]["runs"][0]["id"]
    trades = client.get(f"/api/runs/{run_id}/trades").json()
    assert trades[0]["ticker"] == "A"
    assert client.get("/api/batches/nope").status_code == 404
    assert client.get("/api/data/NOPE").status_code == 404
    assert client.post("/api/experiments/nope/run").status_code == 404
