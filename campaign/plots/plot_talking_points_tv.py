import matplotlib.pyplot as plt
import pandas as pd

from .plot_talking_points import TERMS, _months, _shade
from .style import AXIS, DEM, INK, MUTED, NEUTRAL, RC, REP, SECONDARY

RUN = "pairs"
CHANNELS = (("R", REP, "Fox News"), ("L", DEM, "MSNBC"), ("C", NEUTRAL, "CNN"))


def _series(frame, col):
    s = (1 - frame[col]) * 100  # repeat share, percent
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))  # gaps break the line


def _smooth(s):
    return s.rolling(6, min_periods=3, center=True).mean().where(s.rolling(3, min_periods=1, center=True).count() > 0)


def plot_talking_points_tv(results, path):
    """How much of each cable channel's month repeats its usual word pairs, 2016-2026, and the
    Fox-minus-MSNBC gap. Repeat share = the share of a month's word-pair uses that were among the
    channel's 100 most used pairs in months m-5..m-3. Dots are months, lines are 6-month centered
    means; gray bands are Trump's terms, dashed lines are Election Days."""
    plt.rcParams.update(RC)
    f = pd.DataFrame(results["monthly"][RUN]).set_index("month")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": (1.25, 1), "hspace": 0.4})
    ax = axes[0]
    for side, color, name in CHANNELS:
        s = _series(f, f"turnover_{side}")
        x = _months(s.index)
        ax.scatter(x, s.values, s=9, color=color, alpha=0.25, lw=0, zorder=2)
        sm = _smooth(s)
        ax.plot(x, sm.values, color=color, lw=2 if side != "C" else 1.5, zorder=3, label=name)
        ax.text(x[-1] + pd.Timedelta(days=40), sm.dropna().values[-1], name, color=SECONDARY, fontsize=9, va="center")
    _shade(ax)
    ax.set_title("Share of a month's word pairs among the channel's 100 usual ones")
    ax.set_ylabel("Repeated (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=3, fontsize=9)
    for (start, name), end in zip(TERMS, [t for t, _ in TERMS[1:]] + ["2026-11-30"]):
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=MUTED, fontsize=9)

    ax = axes[1]
    gap = _series(f, "turnover_R") - _series(f, "turnover_L")
    x = _months(gap.index)
    ax.axhline(0, color=AXIS, lw=1)
    ax.scatter(x, gap.values, s=9, color=INK, alpha=0.25, lw=0, zorder=2)
    ax.plot(x, _smooth(gap).values, color=INK, lw=2, zorder=3)
    _shade(ax)
    ax.set_title("Fox News minus MSNBC (above zero: Fox repeats more)")
    ax.set_ylabel("Points")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(pd.Timestamp("2016-01-01"), pd.Timestamp("2027-03-01"))
    fig.savefig(path)
    plt.close(fig)
