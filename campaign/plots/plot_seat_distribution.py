import matplotlib.pyplot as plt

from .style import AQUA, GRID, MUTED, NEUTRAL, RC, SECONDARY


def plot_seat_distribution(fc, path):
    """Distribution of the Democratic seat total among modeled races, with
    the majority line and the 80% range."""
    plt.rcParams.update(RC)
    s = fc["seats"]
    pmf = s["pmf"][(s["pmf"].index >= s["p50"] - 45) & (s["pmf"].index <= s["p50"] + 45)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvspan(s["p10"] - 0.5, s["p90"] + 0.5, color=GRID, alpha=0.6, lw=0)
    ax.bar(pmf.index, pmf.values * 100, width=0.9,
           color=[AQUA if seats >= s["need"] else NEUTRAL for seats in pmf.index])
    ax.axvline(s["need"] - 0.5, color=MUTED, lw=1, ls="--")
    top = pmf.max() * 100
    ax.annotate(f"majority: {s['need']}+ of these {len(fc['races'])} races", (s["need"] - 0.5, top * 1.02),
                xytext=(-4, 0), textcoords="offset points", ha="right", color=SECONDARY, fontsize=9)
    ax.annotate(f"80% range: {s['p10']}-{s['p90']}", (s["p90"] + 0.5, top * 0.9), xytext=(4, 0),
                textcoords="offset points", color=SECONDARY, fontsize=9)
    ax.set_ylim(0, top * 1.12)
    ax.set_xlabel("Democratic seats among modeled races")
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"{fc['cycle']}: {s['p_majority']:.0%} chance of a Democratic majority, "
                 f"with polling error shared across races")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)
