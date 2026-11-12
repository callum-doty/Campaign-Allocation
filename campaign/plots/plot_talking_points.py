import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import AXIS, DEM, GRID, INK, MUTED, NEUTRAL, RC, REP, SECONDARY, SURFACE

TERMS = (("2017-01-20", "Trump"), ("2021-01-20", "Biden"), ("2025-01-20", "Trump"))
ELECTIONS = ("2016-11-08", "2018-11-06", "2020-11-03", "2022-11-08", "2024-11-05", "2026-11-03")


def _months(idx):
    return pd.PeriodIndex(idx, freq="M").to_timestamp() + pd.Timedelta(days=14)


def _shade(ax):
    for (start, _), end in zip(TERMS, [t for t, _ in TERMS[1:]] + ["2026-12-31"]):
        if TERMS.index((start, _)) % 2 == 0:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=GRID, alpha=0.35, lw=0, zorder=0)
    for e in ELECTIONS:
        ax.axvline(pd.Timestamp(e), color=AXIS, lw=0.8, ls=(0, (2, 3)), zorder=1)


def _side_panel(ax, frame, measure, title, ylabel, label_sides=False):
    for side, color, name in (("L", DEM, "Left-leaning outlets"), ("R", REP, "Right-leaning outlets")):
        s = (1 - frame[f"{measure}_{side}"]) * 100  # repeat share, percent
        s.index = pd.PeriodIndex(s.index, freq="M")
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))  # gaps break the line
        x = _months(s.index)
        ax.scatter(x, s.values, s=9, color=color, alpha=0.25, lw=0, zorder=2)
        smooth = s.rolling(6, min_periods=3, center=True).mean().where(s.rolling(3, min_periods=1, center=True).count() > 0)
        ax.plot(x, smooth.values, color=color, lw=2, zorder=3, label=name)
        if label_sides:
            ax.text(x[-1] + pd.Timedelta(days=40), smooth.dropna().values[-1], name.split()[0],
                    color=SECONDARY, fontsize=9, va="center")
    _shade(ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


HEAD = "headline corrected"
THEMES = "themes corrected"


def plot_talking_points(results, path):
    """How much of each side's month repeats its usual material, 2016-2026, over the generic
    ballot. Repeat share = 1 - turnover: the share of a month's terms that were among the side's
    most common in months m-5..m-3. Outlets weighted equally, column titles removed. Dots are
    months, lines are 6-month centered means; gray bands are Trump's terms, dashed lines are
    Election Days."""
    plt.rcParams.update(RC)
    head = pd.DataFrame(results["monthly"][HEAD]).set_index("month")
    them = pd.DataFrame(results["monthly"][THEMES]).set_index("month")
    gb = pd.Series(results["ballot_monthly"]).sort_index()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.6), sharex=True,
                             gridspec_kw={"height_ratios": (1, 1, 0.75), "hspace": 0.42})
    _side_panel(axes[0], head, "turnover", "Headline wording: share of a month's word pairs among the side's 100 usual ones",
                "Repeated (%)", label_sides=True)
    _side_panel(axes[1], them, "turnover", "Topics: share of a month's GDELT theme tags among the side's 25 usual ones",
                "Repeated (%)", label_sides=True)
    axes[0].legend(loc="upper left", ncols=2, fontsize=9)

    ax = axes[2]
    x = _months(gb.index)
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(x, gb.values, color=NEUTRAL, lw=1.2, zorder=2)
    ax.plot(x, gb.rolling(3, min_periods=2, center=True).mean().values, color=INK, lw=2, zorder=3)
    _shade(ax)
    ax.set_title("Generic ballot, Democratic margin (polls ending each month)")
    ax.set_ylabel("Points")
    ax.grid(axis="x", visible=False)
    for (start, name), end in zip(TERMS, [t for t, _ in TERMS[1:]] + ["2026-11-30"]):
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=MUTED, fontsize=9)
    ax.set_xlim(pd.Timestamp("2016-01-01"), pd.Timestamp("2027-03-01"))
    fig.savefig(path)
    plt.close(fig)
