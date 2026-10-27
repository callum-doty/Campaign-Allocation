import matplotlib.pyplot as plt
import numpy as np

from ..spending_effect import HALF_SATURATION, SCALE_PTS, spending_effect
from .style import AQUA, NEUTRAL, RC, SECONDARY


def plot_spending_effect(path, scales=(0.5, 1.5, 4.5)):
    """The assumed margin shift from party money, default and alternatives."""
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    dollars = np.linspace(0, 15e6, 200)
    for scale in scales:
        default = scale == SCALE_PTS
        shift = spending_effect(dollars, scale, HALF_SATURATION)
        ax.plot(dollars / 1e6, shift, color=AQUA if default else NEUTRAL, lw=2.5 if default else 1.5)
        ax.annotate(f"{scale:g}-pt scale" + (" (default)" if default else ""), (15, shift[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=SECONDARY, fontsize=9)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, None)
    ax.set_xlabel("Party money in the race ($M)")
    ax.set_ylabel("Margin shift (points)")
    ax.set_title("Assumed effect of party money: diminishing returns in dollars, "
                 f"half-saturation at ${HALF_SATURATION / 1e6:.1f}M")
    fig.savefig(path)
    plt.close(fig)
