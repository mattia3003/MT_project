"""Shared figure helpers.

Figures are built with matplotlib's Figure object directly (no pyplot), so importing memtk never
changes the plotting backend of a notebook or script.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from memtk.lesions import LesionLabels, label_components

TITLE_FONTSIZE = 10
SAVE_DPI = 90          # default resolution of `save`; `resolution` changes it for a block
SAVE_FOR_PRINT = False # `resolution(..., for_print=True)`: crop to content and add a PDF
SAVE_MAX_WIDTH_IN = None  # `resolution(..., max_width_in=w)`: wider figures are shrunk to w (text keeps its size)

# Outline colours used in every E0 figure
BRAIN_COLOR, LESION_COLOR, BOX_COLOR, HIGHLIGHT_COLOR, SPECK_COLOR = "cyan", "red", "yellow", "lime", "orange"
# One colour per lesion number (L1, L2, ...), avoiding the brain, box and speck colours
LESION_COLORS = ["red", "lime", "magenta", "royalblue", "white", "mediumpurple", "lightpink", "olive"]
# One colour per data set in charts
SET_COLORS = {"T": "tab:blue", "H-cal": "tab:green", "H-eval": "tab:olive", "H": "tab:green",
              "S main": "tab:red", "S all": "tab:gray"}
# Zoomed tiles (`zoom_window`): px around the lesions, smallest window
ZOOM_MARGIN, ZOOM_MIN_CROP = 6, 24


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


def save(fig: Figure, path: Path, title: str = "", dpi: int | None = None) -> Path:
    """Add an optional title, lay out and save; creates the folder. The resolution is `dpi`, else
    the current `SAVE_DPI` (90; see `resolution`).

    The title is wrapped to the figure width, and the constrained layout reserves room for it, so
    it neither covers the tile titles of the first row nor runs off the image.
    """
    if SAVE_MAX_WIDTH_IN and fig.get_figwidth() > SAVE_MAX_WIDTH_IN:
        w, h = fig.get_size_inches()
        fig.set_size_inches(SAVE_MAX_WIDTH_IN, h * SAVE_MAX_WIDTH_IN / w)
    if title:
        chars_per_line = max(30, int(fig.get_figwidth() * 72 / (0.6 * TITLE_FONTSIZE)))
        fig.suptitle(textwrap.fill(title, chars_per_line), fontsize=TITLE_FONTSIZE)
    fig.set_layout_engine("constrained")
    path.parent.mkdir(parents=True, exist_ok=True)
    tight = {"bbox_inches": "tight", "pad_inches": 0.03} if SAVE_FOR_PRINT else {}
    fig.savefig(path, dpi=dpi or SAVE_DPI, **tight)
    if SAVE_FOR_PRINT:
        fig.savefig(path.with_suffix(".pdf"), **tight)
    return path


@contextmanager
def resolution(dpi: int, for_print: bool = False, max_width_in: float | None = None):
    """Every `save` inside the block writes at `dpi`; with `for_print` it also crops the figure to its
    content and writes a PDF next to the PNG; with `max_width_in` a wider figure is shrunk to that width
    (aspect kept) while its text keeps its point size, so it stays legible on the printed page (added
    2026-11-16/17 for the report's appendix, which redraws some experiment figures; the stages keep
    90 dpi, uncropped, PNG only, full size)."""
    global SAVE_DPI, SAVE_FOR_PRINT, SAVE_MAX_WIDTH_IN
    before = SAVE_DPI, SAVE_FOR_PRINT, SAVE_MAX_WIDTH_IN
    SAVE_DPI, SAVE_FOR_PRINT, SAVE_MAX_WIDTH_IN = dpi, for_print, max_width_in
    try:
        yield
    finally:
        SAVE_DPI, SAVE_FOR_PRINT, SAVE_MAX_WIDTH_IN = before


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


# --- zoomed label maps (moved from `memtk.e4.plant` / `e4.validate` on 2026-11-15; E4 and E5) ---

def zoom_window(masks: list[np.ndarray]) -> tuple[slice, slice]:
    """A square window around the pixels of all masks, ZOOM_MARGIN px wide on each side and at
    least ZOOM_MIN_CROP px, kept inside the image."""
    rows, cols = np.nonzero(np.any(masks, axis=0))
    h, w = masks[0].shape
    side = min(max(int(rows.max() - rows.min()), int(cols.max() - cols.min())) + 1 + 2 * ZOOM_MARGIN, h, w)
    side = max(side, min(ZOOM_MIN_CROP, h, w))
    r0 = int(np.clip(round((rows.min() + rows.max()) / 2 - side / 2), 0, h - side))
    c0 = int(np.clip(round((cols.min() + cols.max()) / 2 - side / 2), 0, w - side))
    return slice(r0, r0 + side), slice(c0, c0 + side)


def fill(ax, regions: list[tuple[np.ndarray, str]], alpha: float = 0.55) -> None:
    """Fill boolean regions in their colours, half transparent, over what is drawn."""
    overlay = np.zeros((*regions[0][0].shape, 4))
    for region, color in regions:
        overlay[region] = (*to_rgba(color)[:3], alpha)
    ax.imshow(overlay, interpolation="nearest")


def lesion_colors(labels: np.ndarray) -> dict[int, str]:
    """One colour per lesion number of a label map (LESION_COLORS, repeating)."""
    return {n: LESION_COLORS[(n - 1) % len(LESION_COLORS)] for n in range(1, int(labels.max()) + 1)}


def draw_labels(ax, image: np.ndarray, labels: np.ndarray, colors: dict[int, str], title: str,
                filled: bool = True) -> None:
    """A slice with each lesion of `labels` outlined in its colour and numbered; filled as well unless
    `filled` is False (outlines only leave the image under the lesion visible; E7 reports)."""
    show_slice(ax, image, [], title)
    regions = []
    for n, color in colors.items():
        lesion = labels == n
        if not lesion.any():
            continue
        rgba = to_rgba(color)
        regions.append((lesion, color))
        ax.contour(lesion.astype(float), levels=[0.5], colors=[rgba], linewidths=0.6 if filled else 1.0)
        rows, cols = np.nonzero(lesion)
        ax.text(cols.max() + 1, rows.min() - 0.5, f"L{n}", color=rgba, fontsize=5, fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="black")])
    if regions and filled:
        fill(ax, regions)
