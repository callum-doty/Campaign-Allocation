"""Palette and matplotlib settings shared by every figure. Party colors carry
party meaning only; other comparisons use orange / aqua / violet against gray."""

INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DEM, REP, NEUTRAL = "#2a78d6", "#e34948", "#b4b2a9"
ORANGE, AQUA, VIOLET = "#eb6834", "#1baf7a", "#6250d6"

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": SECONDARY, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.titlelocation": "left", "axes.titlepad": 12,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
}
