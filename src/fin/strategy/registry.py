"""Discover strategies from plain .py files in the project's ``strategies/`` folder."""

import importlib.util
import inspect
from pathlib import Path

from fin.config import PROJECT_ROOT
from fin.strategy.base import Strategy

STRATEGIES_DIR = PROJECT_ROOT / "strategies"


def discover(directory: Path | None = None) -> dict[str, type[Strategy]]:
    """Import every ``*.py`` in ``directory`` and collect the Strategy subclasses it defines."""
    directory = Path(directory or STRATEGIES_DIR)
    found: dict[str, type[Strategy]] = {}
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"fin_strategies.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, Strategy)
                and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__
            ):
                if obj.name in found:
                    raise ValueError(f"duplicate strategy name {obj.name!r} in {path}")
                found[obj.name] = obj
    return found


def create(name: str, directory: Path | None = None, **params) -> Strategy:
    strategies = discover(directory)
    if name not in strategies:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(strategies)}")
    return strategies[name](**params)
