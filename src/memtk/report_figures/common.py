"""Shared style and saving for the report figures (`scripts/r_report_figures.py`).

Every report figure is sized for one A4 text column (16 cm wide), with fonts of 7-8 pt at print size,
no suptitle (the caption is written in the report) and saved twice: PNG at 300 dpi (Word, Google Docs)
and PDF (vector, LaTeX). Colours are fixed across figures: S red, T blue, held-out grey; E1 blue, E2
orange.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from matplotlib import rc_context
from matplotlib.figure import Figure

from memtk import config

WIDTH_IN = 16 / 2.54          # one A4 text column
DPI = 300
STYLE = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
         "lines.linewidth": 1.2}
COLORS = {"S": "tab:red", "T": "tab:blue", "H": "0.45", "H-eval": "0.45", "E1": "tab:blue", "E2": "tab:orange",
          "either": "k"}


@contextmanager
def style():
    """The report style for every figure drawn inside the block."""
    with rc_context(STYLE):  # pyright: ignore[reportArgumentType]  # matplotlib stub types rc keys as a literal union
        yield


def figure(height_cm: float, width_in: float = WIDTH_IN) -> Figure:
    """An empty figure of the column width and the given height."""
    return Figure(figsize=(width_in, height_cm / 2.54))


def save_both(fig: Figure, out: Path, name: str) -> list[Path]:
    """Lay out and save the figure as `<name>.png` (300 dpi) and `<name>.pdf` in `out`, cropped to its
    content (a small pad), so no empty margin follows it on the page."""
    fig.set_layout_engine("constrained")
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.03)
    return paths


def index_row(name: str, section: str, content: str, sources: Sequence[str | Path]) -> dict:
    """One row of a stage's `figures.csv`: file stem, report section, what it shows, source files
    (repository-relative)."""
    return {"figure": name, "section": section, "content": content,
            "sources": "; ".join(config.repo_relative(s) if isinstance(s, Path) else s for s in sources)}


def write_index(rows: list[dict], out: Path, filename: str = "figures.csv") -> Path:
    """The stage's index (`figures.csv`, or `tables.csv` for the tables; one row per item)."""
    path = out / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
