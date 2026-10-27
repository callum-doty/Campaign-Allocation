import matplotlib.pyplot as plt

from .style import AQUA, GRID, INK, MUTED, RC, SECONDARY

SWING = 3.0
MAJORITY = 218


def plot_seats_by_gb(fc, path):
    """Expected Democratic seats against the national environment, with the
    current polling average and a +/-3 point polling-miss band."""
    plt.rcParams.update(RC)
    gb, seats = fc["gb"], fc["seats_by_gb"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(gb - SWING, gb + SWING, color=GRID, alpha=0.6, lw=0)
    ax.plot(list(seats), list(seats.values()), color=AQUA, lw=2.5)
    ax.axhline(MAJORITY, color=MUTED, lw=1, ls="--")
    ax.annotate("majority (218)", (min(seats), MAJORITY), xytext=(2, 4), textcoords="offset points",
                color=SECONDARY, fontsize=9)
    for g in (gb - SWING, gb, gb + SWING):
        s = seats[round(g, 1)]
        ax.scatter(g, s, s=60 if g == gb else 30, color=INK if g == gb else AQUA, zorder=3,
                   edgecolor="white", linewidth=1.5)
        label = f"D+{g:.1f}: {s:.0f}" + (" (now)" if g == gb else "")
        ax.annotate(label, (g, s), xytext=(6, -12), textcoords="offset points", color=SECONDARY, fontsize=9)
    ax.set_xlabel("Generic ballot, Democrats minus Republicans (points)")
    ax.set_ylabel(f"Expected Democratic seats\n({len(fc['races'])} modeled races)")
    ax.set_title(f"{fc['cycle']}: the national environment moves more seats than the new maps")
    fig.savefig(path)
    plt.close(fig)
