import matplotlib.pyplot as plt
import pandas as pd

from .style import AQUA, AXIS, INK, RC, SECONDARY

TIER_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]


def plot_calibration(backtests, path):
    """Predicted vs actual Democratic win rate by tier, pooled over the
    forward backtests."""
    plt.rcParams.update(RC)
    pooled = pd.concat([b["calibration"].assign(won=lambda t: t.actual * t.n, pred=lambda t: t.predicted * t.n)
                        for b in backtests]).groupby(level=0).sum()
    pooled = pooled.reindex(TIER_ORDER)
    pooled["predicted"], pooled["actual"] = pooled.pred / pooled.n, pooled.won / pooled.n
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(pooled))[::-1]
    for yi, (tier, row) in zip(y, pooled.iterrows()):
        ax.plot([row.predicted * 100, row.actual * 100], [yi, yi], color=AXIS, lw=2, zorder=1)
        ax.scatter(row.predicted * 100, yi, s=70, color=AQUA, zorder=2, edgecolor="white", linewidth=1.5)
        ax.scatter(row.actual * 100, yi, s=70, color=INK, marker="D", zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{row.predicted:.0%} vs {row.actual:.0%}  (n={int(row.n)})", (102, yi), va="center",
                    color=SECONDARY, fontsize=9, annotation_clip=False)
    ax.set_yticks(list(y), pooled.index)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Democratic win rate (%)")
    ax.scatter([], [], s=50, color=AQUA, label="Predicted")
    ax.scatter([], [], s=50, color=INK, marker="D", label="Actual")
    ax.legend(loc="lower right")
    years = ", ".join(str(b["cycle"]) for b in backtests)
    ax.set_title(f"Forecast vs result by tier, each year predicted only from earlier years ({years})")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
