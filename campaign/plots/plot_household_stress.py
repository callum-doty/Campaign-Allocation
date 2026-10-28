import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import AXIS, DEM, RC, REP, SECONDARY, SURFACE


def plot_household_stress(table, path):
    """How the president's party ran against its final polls, against each
    household-stress composite, one point per election. The two largest
    shocks in each panel are labeled."""
    plt.rcParams.update(RC)
    gap = table.s * table.miss
    colors = table.president.map({"D": DEM, "R": REP})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, col, title, xlabel in ((axes[0], "basket", "Cost basket", "Price change, June over June (%)"),
                                   (axes[1], "stress", "Credit and labor stress", "Change, June over June (points)")):
        ax.axhline(0, color=AXIS, lw=1)
        ax.scatter(table[col], gap, s=70, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
        for year in table[col].nlargest(2).index:
            ax.annotate(str(year), (table[col][year], gap[year]), xytext=(6, 4), textcoords="offset points",
                        color=SECONDARY, fontsize=9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("President's party: House vote minus\nits final polls (points)")
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, mec=SURFACE, mew=2) for c in (DEM, REP)]
    axes[1].legend(handles, ["Democratic president", "Republican president"], loc="lower right")
    fig.suptitle("Household stress doesn't predict where the president's party lands against its polls",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(w_pad=2)
    fig.savefig(path)
    plt.close(fig)
