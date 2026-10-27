import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import AQUA, NEUTRAL, RC, SECONDARY

BANDS = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
LABELS = ["<5%", "5-20%", "20-40%", "40-60%", "60-80%", "80-95%", ">95%"]


def plot_allocation(backtests, path):
    """Where DCCC's money went vs. where the model would put the same
    budget, by each race's forecast chance of a Democratic win (pooled)."""
    plt.rcParams.update(RC)
    races = pd.concat([b["races"] for b in backtests])
    band = pd.cut(races.p, BANDS, labels=LABELS, include_lowest=True)
    shares = pd.DataFrame({"DCCC": races.dccc.groupby(band, observed=False).sum(),
                           "Model": races.model_dccc.groupby(band, observed=False).sum()})
    shares = shares / shares.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(LABELS))
    for offset, col, color in ((-0.19, "DCCC", NEUTRAL), (0.19, "Model", AQUA)):
        bars = ax.bar(x + offset, shares[col], width=0.36, color=color, label=col)
        for b, v in zip(bars, shares[col]):
            if v >= 1:
                ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", color=SECONDARY, fontsize=8)
    ax.set_xticks(x, LABELS)
    ax.set_xlabel("Forecast chance of a Democratic win")
    ax.set_ylabel("Share of DCCC's budget (%)")
    ax.legend(["DCCC actual", "Model, same budget"], loc="upper left")
    years = f"{backtests[0]['cycle']}-{backtests[-1]['cycle']}"
    ax.set_title(f"Where DCCC's own money went, {years} pooled")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)
