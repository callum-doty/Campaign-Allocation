import matplotlib.pyplot as plt

from .style import AQUA, AXIS, NEUTRAL, ORANGE, RC, SURFACE


def plot_poll_lean(table, path):
    """How far the generic ballot overstated Democrats each year, on Sept 1
    and in the final average, against the national House vote. The gray line
    joining each pair is the fall drift."""
    plt.rcParams.update(RC)
    sept, final = table.gb_sept - table.house_vote, table.gb - table.house_vote
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axhline(0, color=AXIS, lw=1)
    ax.vlines(table.index, sept, final, color=NEUTRAL, lw=1.2, zorder=2)
    for values, color, name in ((sept, ORANGE, "Sept 1 polls"), (final, AQUA, "Final polls")):
        ax.plot(table.index, values, ls="", marker="o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3,
                label=f"{name} (average {values.mean():+.1f})")
    ax.set_xticks(table.index)
    ax.set_ylabel("Polls minus House vote (points,\n+ = polls more Democratic)")
    ax.set_title(f"The polls overstated Democrats in {int((sept > 0).sum())} of {len(table)} Septembers", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.3)
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)
