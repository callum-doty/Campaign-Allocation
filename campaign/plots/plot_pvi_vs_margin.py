import matplotlib.pyplot as plt
import numpy as np

from .style import DEM, MUTED, NEUTRAL, RC, REP, SECONDARY

STATUS = [("D incumbent", DEM, 1, 0), ("Open", NEUTRAL, 0, 1), ("R incumbent", REP, 0, 0)]


def plot_pvi_vs_margin(races, model, path):
    """Actual margins against PVI for one cycle's contested races, with the
    fitted baseline for each incumbency status (at the median floor share)."""
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(8, 5))
    contested = races[races.margin.abs() < 80]
    c = model["coef"]
    xs = np.linspace(-30, 30, 2)
    gb = contested.gb.iloc[0]
    for status, color, inc, opn in STATUS:
        pts = contested[contested.status == status]
        ax.scatter(pts.pvi, pts.margin, s=14, color=color, alpha=0.45, linewidths=0)
        fs = contested.floor_share.median()
        ys = c["intercept"] + c["pvi"] * xs + c["d_incumbent"] * inc + c["open_seat"] * opn + c["floor_share"] * fs + gb
        ax.plot(xs, ys, color=color if color != NEUTRAL else MUTED, lw=2, solid_capstyle="round")
        ax.annotate(status, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=SECONDARY, fontsize=9)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-32, 38)
    ax.set_ylim(-80, 80)
    ax.set_xlabel("District lean, PVI (points, + = Democratic)")
    ax.set_ylabel("Democratic margin (points)")
    ax.set_title(f"{int(contested.cycle.iloc[0])}: each PVI point is worth {c['pvi']:.1f} margin points; "
                 f"incumbency shifts the line")
    fig.savefig(path)
    plt.close(fig)
