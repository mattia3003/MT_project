"""Shared figure helpers.

Figures are built with matplotlib's Figure object directly (no pyplot), so importing memtk never
changes the plotting backend of a notebook or script.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from memtk.lesions import LesionLabels, label_components

TITLE_FONTSIZE = 10

# Outline colours used in every E0 figure
BRAIN_COLOR, LESION_COLOR, BOX_COLOR, HIGHLIGHT_COLOR, SPECK_COLOR = "cyan", "red", "yellow", "lime", "orange"
# One colour per lesion number (L1, L2, ...), avoiding the brain, box and speck colours
LESION_COLORS = ["red", "lime", "magenta", "royalblue", "white", "mediumpurple", "lightpink", "olive"]
# One colour per data set in charts
SET_COLORS = {"T": "tab:blue", "H-cal": "tab:green", "H-eval": "tab:olive", "H": "tab:green",
              "S main": "tab:red", "S all": "tab:gray"}


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


def draw_box(ax, box: tuple | None) -> None:
    """An inclusive box (r0, r1, c0, c1), e.g. the brain box, as a dashed rectangle; nothing for None."""
    if box is not None:
        r0, r1, c0, c1 = box
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1, fill=False,
                               edgecolor=BOX_COLOR, linestyle="--", linewidth=0.6))


def hide_unused(axes: np.ndarray, n_used: int) -> None:
    """Turn off the axes after the first n_used tiles of a grid."""
    for ax in axes.ravel()[n_used:]:
        ax.axis("off")


def save(fig: Figure, path: Path, title: str = "", dpi: int = 90) -> Path:
    """Add an optional title, lay out and save; creates the folder.

    The title is wrapped to the figure width, and the constrained layout reserves room for it, so
    it neither covers the tile titles of the first row nor runs off the image.
    """
    if title:
        chars_per_line = max(30, int(fig.get_figwidth() * 72 / (0.6 * TITLE_FONTSIZE)))
        fig.suptitle(textwrap.fill(title, chars_per_line), fontsize=TITLE_FONTSIZE)
    fig.set_layout_engine("constrained")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


# --- labelled slices (lesion colours, strays, specks) -----------------------------------------

def draw_lesions(ax, image: np.ndarray, res: LesionLabels, brain: np.ndarray, box: tuple | None, title: str,
                 fontsize: float = 5) -> None:
    """One slice after attach-or-drop: each lesion filled in its own colour with its number
    (L1 = largest) next to it, attached strays in a lighter shade of their lesion's colour,
    dropped specks marked x, brain outline and box."""
    show_slice(ax, image, [(brain, BRAIN_COLOR)], title)
    overlay = np.zeros((*image.shape, 4))
    for n in range(1, res.n_lesions + 1):
        lesion = res.labels == n
        color = to_rgba(LESION_COLORS[(n - 1) % len(LESION_COLORS)])
        overlay[lesion & ~res.attached] = (*color[:3], 0.65)
        overlay[lesion & res.attached] = (*color[:3], 0.3)
        ax.contour(lesion.astype(float), levels=[0.5], colors=[color], linewidths=0.6)
        rows, cols = np.nonzero(lesion)
        # lesion-coloured number with a black outline, readable on the lesion and the image alike
        ax.text(cols.max() + 1.5, rows.min() - 1, f"L{n}", color=color, fontsize=fontsize, fontweight="bold",
                ha="left", va="bottom", path_effects=[patheffects.withStroke(linewidth=fontsize / 3, foreground="black")])
    ax.imshow(overlay, interpolation="nearest")
    specks, n_specks = label_components(res.dropped)
    if n_specks:
        centres = np.array([np.argwhere(specks == i).mean(axis=0) for i in range(1, n_specks + 1)])
        ax.scatter(centres[:, 1], centres[:, 0], marker="x", s=fontsize * 2, c=SPECK_COLOR, linewidths=0.6)
    draw_box(ax, box)


def lesion_sheet(items: list[tuple[str, np.ndarray, LesionLabels, np.ndarray, tuple | None]], path: Path,
                 title: str, n_cols: int = 8) -> Path:
    """Tiles of (title, image, attach-or-drop result, brain, box), drawn by `draw_lesions`."""
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = tile_figure(n_rows, n_cols, tile=2.2)
    for ax, (tile_title, image, res, brain, box) in zip(axes.ravel(), items):
        draw_lesions(ax, image, res, brain, box, tile_title)
    hide_unused(axes, len(items))
    return save(fig, path, title)
