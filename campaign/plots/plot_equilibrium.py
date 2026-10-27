import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .style import DEM, RC, REP, SECONDARY

BANDS = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
LABELS = ["<5%", "5-20%", "20-40%", "40-60%", "60-80%", "80-95%", ">95%"]


def plot_equilibrium(backtests, games, path):
    """Where each committee's money goes, actual vs. the two-player
    equilibrium, by each race's forecast chance of a Democratic win: DCCC's
    share of its budget above the line, NRCC's below, pooled over cycles."""
    plt.rcParams.update(RC)
    pooled = pd.concat([pd.DataFrame({"p": b["races"].p.to_numpy(), "dccc": b["races"].dccc.to_numpy(),
                                      "nrcc": b["races"].nrcc.to_numpy(), "eq_d": games[b["cycle"]]["x_d"],
                                      "eq_r": games[b["cycle"]]["x_r"]}) for b in backtests])
    band = pd.cut(pooled.p, BANDS, labels=LABELS, include_lowest=True)
    shares = pooled.groupby(band, observed=False)[["dccc", "nrcc", "eq_d", "eq_r"]].sum()
    shares = shares / shares.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    x = np.arange(len(LABELS))
    top = shares.max().max() * 1.18
    for ax, (title, d_col, r_col) in zip(axes, [("What the committees actually did", "dccc", "nrcc"),
                                                 ("Equilibrium: each side's best reply to the other", "eq_d", "eq_r")]):
        ax.bar(x, shares[d_col], color=DEM, width=0.72, label="DCCC")
        ax.bar(x, -shares[r_col], color=REP, width=0.72, label="NRCC")
        for xi, up, down in zip(x, shares[d_col], shares[r_col]):
            if up >= 2:
                ax.text(xi, up + top * 0.02, f"{up:.0f}%", ha="center", va="bottom", fontsize=8, color=SECONDARY)
            if down >= 2:
                ax.text(xi, -down - top * 0.02, f"{down:.0f}%", ha="center", va="top", fontsize=8, color=SECONDARY)
        ax.axhline(0, color=SECONDARY, lw=0.8)
        ax.set_xticks(x, LABELS, fontsize=8)
        ax.set_ylim(-top, top)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}%"))
    axes[0].set_ylabel("Share of each committee's budget\n(DCCC above the line, NRCC below)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.supxlabel("Forecast chance of a Democratic win", fontsize=10, color=SECONDARY)
    years = f"{backtests[0]['cycle']}-{backtests[-1]['cycle']}"
    fig.suptitle(f"At equilibrium both parties pile into the same close races; in reality both spread money much wider ({years})",
                 x=0.01, ha="left", fontsize=11.5)
    fig.savefig(path)
    plt.close(fig)
