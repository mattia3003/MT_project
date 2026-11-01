"""Shared figure helpers.

Figures are built with matplotlib's Figure object directly (no pyplot), so importing memtk never
changes the plotting backend of a notebook or script.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def tile_figure(n_rows: int, n_cols: int, tile: float = 2.2) -> tuple[Figure, np.ndarray]:
    """A figure with an (n_rows, n_cols) grid of image tiles; axes always come back 2D."""
    fig = Figure(figsize=(tile * n_cols, (tile + 0.25) * n_rows))
    return fig, fig.subplots(n_rows, n_cols, squeeze=False)


def show_slice(ax, image: np.ndarray, outlines: Sequence[tuple[np.ndarray, str]] = (), title: str = "",
               vmax: float = 1.0) -> None:
    """Grey image in a fixed display range [0, vmax], with coloured outlines of boolean masks."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    for mask, color in outlines:
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=0.7)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def hide_unused(axes: np.ndarray, n_used: int) -> None:
    """Turn off the axes after the first n_used tiles of a grid."""
    for ax in axes.ravel()[n_used:]:
        ax.axis("off")


def save(fig: Figure, path: Path, title: str = "", dpi: int = 90) -> Path:
    """Add an optional title, lay out and save; creates the folder."""
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path
