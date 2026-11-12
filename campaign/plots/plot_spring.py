import matplotlib.pyplot as plt
import numpy as np

from .style import AQUA, AXIS, DEM, INK, MUTED, NEUTRAL, ORANGE, RC, REP, SECONDARY, SURFACE, VIOLET

MOMENTUM = (0.62, -0.45)   # an illustrative pendulum: the fitted spring's p1 plus a p2 that overshoots


def response(p1, p2=0.0, start=10.0, n=6):
    """What is left of a swing of `start` points after each election."""
    path = [start, p1 * start]
    while len(path) < n:
        path.append(p1 * path[-1] + p2 * path[-2])
    return np.array(path)


def plot_spring(results, path):
    """The spring in three panels: the House vote around its center, how
    much of each swing the next election keeps, and how a swing fades
    against a random walk and a pendulum with momentum. Uses the spring
    with forces and the center moved in 1994; the fixed-center spring is
    the other line in the last panel."""
    plt.rcParams.update(RC)
    series = {int(year): v for year, v in results["series"].items()}
    years = np.array(sorted(series))
    y = np.array([series[year] for year in years])
    c, p1, b_s, b_sm, shift = results["realigned"]["models"]["spring+forces"]["coef"]
    p1_fixed = results["main"]["models"]["spring+forces"]["coef"][1]
    center = lambda year: (c + shift * (year >= 1994)) / (1 - p1)
    preds = results["readings"]["predictions"]
    gb_2026 = results["readings"]["gb_2026"]
    in_2028 = sorted(preds[run]["spring+forces"]["2028 if 2026 = gb"] for run in preds)

    fig = plt.figure(figsize=(10, 8.6))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.05), hspace=0.55, wspace=0.28)

    # 1. The series around its center
    ax = fig.add_subplot(grid[0, :])
    ax.axhline(0, color=AXIS, lw=1)
    ax.text(2029.2, -0.6, "50/50", color=MUTED, fontsize=9, va="top")
    for a, b in ((1945, 1993), (1993, 2029)):
        ax.plot([a, b], [center(a + 1)] * 2, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.plot(years, y, color=NEUTRAL, lw=1.2, zorder=3)
    ax.scatter(years, y, s=42, c=np.where(y >= 0, DEM, REP), edgecolors=SURFACE, linewidths=1.5, zorder=4)
    ax.plot([2024, 2026], [y[-1], gb_2026], color=NEUTRAL, lw=1.2, ls=":", zorder=3)
    ax.scatter([2026], [gb_2026], s=42, facecolors=SURFACE, edgecolors=DEM, linewidths=1.8, zorder=4)
    ax.annotate(f"2026 final polls, D+{gb_2026:.1f}\n(result not in the data yet)", (2026, gb_2026), (2015.2, 15.2),
                fontsize=9, color=SECONDARY, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.vlines(2028, *in_2028, color=ORANGE, lw=4, zorder=4, capstyle="round")
    ax.text(2028.6, np.mean(in_2028), f"2028 by the\nspring: D+{in_2028[0]:.1f}\nto D+{in_2028[1]:.1f}",
            fontsize=9, color=SECONDARY, va="center")
    ax.set_xlim(1944, 2034)
    ax.set_ylim(-12, 19)
    ax.set_xticks(range(1946, 2030, 8))
    ax.set_ylabel("House vote, Democratic margin\n(points)")
    ax.set_title("The House vote swings around a center, and the center moved in 1994", pad=26)
    ax.text(0, 1.03, f"Dashed: the spring's center, D+{center(1950):.1f} before 1994 and D+{center(2000):.1f} since",
            transform=ax.transAxes, color=SECONDARY, fontsize=9.5)
    ax.grid(axis="x", visible=False)

    # 2. The spring law: how much of last election's swing is left
    ax = fig.add_subplot(grid[1, 0])
    rows = years >= 1950
    lagged = np.array([series[year - 2] for year in years[rows]])
    forces = np.array([b_s * s + b_sm * s * (year % 4 == 2) for year, s in
                       zip(years[rows], [1.0 if results["president"][str(year)] == "D" else -1.0 for year in years[rows]])])
    x = lagged - center(years[rows])
    kept = y[rows] - forces - center(years[rows])
    span = np.array([-14, 14])
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.plot(span, span, color=NEUTRAL, lw=2, ls=(0, (4, 3)), label="random walk: keeps all of it")
    ax.plot(span, p1 * span, color=ORANGE, lw=2, solid_capstyle="round", label=f"spring: keeps {p1:.0%}")
    ax.scatter(x, kept, s=36, color=INK, alpha=0.75, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    kicks = np.abs(kept - p1 * x)
    for i in np.argsort(kicks)[-4:]:
        ax.annotate(str(years[rows][i]), (x[i], kept[i]), (5, 3), textcoords="offset points", fontsize=8.5, color=SECONDARY)
    ax.legend(loc="upper left", fontsize=8.5, handlelength=2.4)
    ax.set_xlim(*span)
    ax.set_ylim(*span)
    ax.set_xlabel("Last election, distance from center (points)")
    ax.set_ylabel("This election, distance from center\n(president's party and midterms removed)")
    ax.set_title("Each election keeps about a third of the last swing", fontsize=11, pad=24)
    ax.text(0, 1.03, f"Center moved in 1994. With a fixed center it keeps {p1_fixed:.0%}",
            transform=ax.transAxes, color=SECONDARY, fontsize=9)

    # 3. A swing fading out
    ax = fig.add_subplot(grid[1, 1])
    steps = np.arange(6)
    ax.axhline(0, color=AXIS, lw=1)
    lines = ((response(1.0), NEUTRAL, "random walk", (0, (4, 3))),
             (response(p1), ORANGE, f"spring, center moved ({p1:.2f})", "-"),
             (response(p1_fixed), AQUA, f"spring, center fixed ({p1_fixed:.2f})", "-"),
             (response(*MOMENTUM), VIOLET, "pendulum with momentum\n(illustration, not fitted)", (0, (1.5, 2))))
    for values, color, label, style in lines:
        ax.plot(steps, values, color=color, lw=2, ls=style, solid_capstyle="round", label=label)
        ax.scatter(steps, values, s=28, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.legend(loc="upper center", ncol=2, fontsize=8, handlelength=2.4, columnspacing=1.2)
    ax.annotate("overshoots\nthe center", (3, response(*MOMENTUM)[3]), (3.6, -5.6), fontsize=8.5, color=SECONDARY,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xticks(steps, ["swing", "+1", "+2", "+3", "+4", "+5"])
    ax.set_xlabel("Elections after a 10-point swing")
    ax.set_ylabel("Points of the swing left")
    ax.set_ylim(-7.5, 16.5)
    ax.set_yticks([-5, 0, 5, 10])
    ax.set_title("A swing fades without overshooting", fontsize=11, pad=24)
    ax.text(0, 1.03, "The fitted pendulum tracks the spring", transform=ax.transAxes, color=SECONDARY, fontsize=9)
    ax.grid(axis="x", visible=False)

    fig.savefig(path)
    plt.close(fig)
