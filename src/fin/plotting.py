"""Shared matplotlib style so every notebook and report looks the same.

Categorical colors are assigned in this fixed order and follow the entity, never its rank.
"""

import matplotlib as mpl
from cycler import cycler

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (10, 4),
            "figure.dpi": 110,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.prop_cycle": cycler(color=SERIES),
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 9.5,
        }
    )
