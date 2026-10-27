import matplotlib.pyplot as plt

from ..compare_maps import compare_maps
from .style import AXIS, NEUTRAL, RC, SECONDARY, VIOLET


def plot_battleground(fc, path, low=0.2, high=0.8):
    """Every race the forecast puts between `low` and `high`, ranked, with
    the districts whose lines changed for this cycle highlighted."""
    plt.rcParams.update(RC)
    changed = set(compare_maps(new_lines=fc["cycle"]).query("changed").district_id)
    close = fc["races"][(fc["races"].p > low) & (fc["races"].p < high)].sort_values("p")
    fig, ax = plt.subplots(figsize=(8, 0.2 * len(close) + 1.4))
    ax.axvline(50, color=AXIS, lw=1.5)
    for i, r in enumerate(close.itertuples()):
        redrawn = r.district_id in changed
        ax.scatter(r.p * 100, i, s=36, color=VIOLET if redrawn else NEUTRAL, edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(f"{r.p:.0%}  {r.status}", (r.p * 100, i), xytext=(7, 0), textcoords="offset points",
                    va="center", color=SECONDARY, fontsize=7.5)
    ax.set_yticks(range(len(close)), close.district_id, fontsize=8)
    ax.set_xlim(low * 100 - 2, high * 100 + 12)
    ax.set_xlabel("Forecast chance of a Democratic win (%)")
    n_changed = sum(d in changed for d in close.district_id)
    ax.scatter([], [], s=36, color=VIOLET, label=f"Lines changed for {fc['cycle']} ({n_changed})")
    ax.scatter([], [], s=36, color=NEUTRAL, label=f"Same lines ({len(close) - n_changed})")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"{fc['cycle']} battleground: {len(close)} races between {low:.0%} and {high:.0%}")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
