import matplotlib.pyplot as plt
import numpy as np

from .style import AQUA, NEUTRAL, ORANGE, RC, SECONDARY, SURFACE


def plot_safe_seat_share(backtests, path):
    """Share of DCCC's money in races the forecast called 80%+ for Democrats,
    by year and pooled, three ways: with the final polls, with the Sept 1
    forecast for every dollar, and with each dollar judged by the forecast
    when it was spent."""
    plt.rcParams.update(RC)

    def shares(b):
        r, t = b["races"], b["when_spent"]["tranches"]
        return np.array([r.dccc[r.p > 0.8].sum(), r.dccc[r.p_sept > 0.8].sum(),
                         t.amount[t.p > 0.8].sum() / t.amount.sum() * r.dccc.sum()]), r.dccc.sum()
    dollars, budgets = zip(*(shares(b) for b in backtests))
    table = [d / total * 100 for d, total in zip(dollars, budgets)] + [sum(dollars) / sum(budgets) * 100]
    groups = [str(b["cycle"]) for b in backtests] + ["Pooled"]
    series = (("Final polls (hindsight)", NEUTRAL), ("Sept 1 forecast", ORANGE), ("Forecast when each dollar was spent", AQUA))
    x = np.arange(len(groups)) * 1.15
    width = 0.3
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, (name, color) in enumerate(series):
        values = [row[k] for row in table]
        bars = ax.bar(x + (k - 1) * width, values, width=width, color=color, label=name,
                      edgecolor=SURFACE, linewidth=2)
        pooled = bars[-1]
        ax.annotate(f"{values[-1]:.0f}%", (pooled.get_x() + pooled.get_width() / 2, values[-1]), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=SECONDARY, fontsize=9)
    ax.axvline(x[-1] - 0.575, color=NEUTRAL, lw=0.8)
    ax.set_xticks(x, groups)
    ax.set_ylabel("DCCC's budget in races called 80%+ (%)")
    ax.set_ylim(0, max(max(row) for row in table) * 1.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.3)
    ax.set_title("DCCC's money in 'safe' Democratic seats, by when the forecast is judged", pad=30)
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)
