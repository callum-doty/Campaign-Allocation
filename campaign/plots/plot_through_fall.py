import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..dated_forecast import CHECKPOINTS
from .style import NEUTRAL, RC, SECONDARY, SURFACE, VIOLET


def plot_through_fall(backtests, path):
    """The forecast at each checkpoint of the fall: how uncertain the national
    environment still is (the spread of the move to the final polls), and how
    accurate the race forecasts are (log loss over the backtest years). Two
    panels, since the two measures don't share a scale."""
    plt.rcParams.update(RC)
    labels = [f"{pd.Timestamp(f'2000-{d}'):%b} {int(d[3:])}" for d in CHECKPOINTS]
    drift = backtests[-1]["when_spent"]["drift"]
    spread = [drift[d]["sd"] for d in CHECKPOINTS]
    ll = [np.mean([b["scores"][d]["log_loss"] for b in backtests]) for d in CHECKPOINTS]
    ll_final = np.mean([b["scores"]["with"]["log_loss"] for b in backtests])
    x = np.arange(len(CHECKPOINTS))
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, values, fmt in ((left, spread, "{:.1f}"), (right, ll, "{:.3f}")):
        ax.plot(x, values, color=VIOLET, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
        for i in (0, len(x) - 1):
            ax.annotate(fmt.format(values[i]), (x[i], values[i]), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=SECONDARY, fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, max(values) * 1.35)
        ax.grid(axis="x", visible=False)
    left.set_title("National uncertainty still to resolve")
    left.set_ylabel("Spread of the move to the final polls (points)")
    right.axhline(ll_final, color=NEUTRAL, lw=1.2, zorder=2)
    right.annotate(f"final polls, in hindsight: {ll_final:.3f}", (x[-1], ll_final), xytext=(0, -14),
                   textcoords="offset points", ha="right", color=SECONDARY, fontsize=9)
    right.set_title("Race forecasts: no sharper by late October")
    right.set_ylabel(f"Log loss, {backtests[0]['cycle']}-{backtests[-1]['cycle']} (lower is better)")
    fig.tight_layout(w_pad=3)
    fig.savefig(path)
    plt.close(fig)
