import matplotlib.pyplot as plt

from ..compare_maps import compare_maps
from .style import DEM, NEUTRAL, RC, REP, SECONDARY


def plot_map_changes(path, old_lines=2024, new_lines=2026):
    """Net change in districts Harris carried in 2024, per redrawn state:
    the same votes counted on the old and new maps."""
    plt.rcParams.update(RC)
    m = compare_maps(old_lines, new_lines)
    states = sorted(m[m.changed].district_id.str[:2].unique())
    rows = []
    for s in states:
        g = m[m.district_id.str.startswith(s + "-")]
        before, after = int((g.old_share > 0.5).sum()), int((g.new_share > 0.5).sum())
        rows.append((s, before, after, len(g)))
    rows.sort(key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    for i, (s, before, after, n) in enumerate(rows):
        delta = after - before
        ax.barh(i, delta if delta else 0.06, color=DEM if delta > 0 else REP if delta < 0 else NEUTRAL, height=0.6)
        ax.annotate(f"{before} to {after} of {n}", (max(delta, 0), i), xytext=(6, 0), textcoords="offset points",
                    va="center", color=SECONDARY, fontsize=9)
    net = sum(r[2] - r[1] for r in rows)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.axvline(0, color=SECONDARY, lw=0.8)
    ax.set_xlabel("Change in districts Harris won in 2024")
    ax.set_title(f"New maps for {new_lines}: net {net:+d} Harris-won districts across {len(rows)} redrawn states")
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
