import matplotlib.pyplot as plt
import pandas as pd

from ..channel_mix import BAND_LABELS
from .style import AQUA, NEUTRAL, ORANGE, RC, SECONDARY, VIOLET, INK

SHOWN = [("TV", ORANGE), ("Digital", AQUA), ("Mail", VIOLET)]


def plot_channel_mix(mixes, path):
    """Channel shares by TV-market cost, one panel per spender group,
    pooled over the cycles in `mixes` (channel_mix() results)."""
    plt.rcParams.update(RC)
    mix = pd.concat(mixes)
    mix["channel"] = mix.channel.where(mix.channel.isin([c for c, _ in SHOWN]), "Everything else")
    groups = list(dict.fromkeys(mix.group))
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 1.5 * len(groups) + 1.2), sharex=True)
    for ax, group in zip(axes, groups):
        g = mix[mix.group == group].groupby(["band", "channel"], observed=False).amount.sum()
        share = (g / g.groupby(level="band", observed=False).transform("sum")).unstack().reindex(BAND_LABELS[::-1]) * 100
        left = pd.Series(0.0, index=share.index)
        for channel, color in SHOWN + [("Everything else", NEUTRAL)]:
            vals = share.get(channel, pd.Series(0.0, index=share.index)).fillna(0)
            ax.barh(range(len(share)), vals, left=left, color=color, height=0.7, edgecolor="white", linewidth=1.5,
                    label=channel if group == groups[0] else None)
            for y, (v, l) in enumerate(zip(vals, left)):
                if v >= 5:
                    ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                            color=INK if color in (AQUA, NEUTRAL) else "white")
            left += vals
        ax.set_yticks(range(len(share)), share.index, fontsize=9)
        ax.set_title(group, fontsize=10, color=SECONDARY, pad=4)
        ax.set_xlim(0, 100)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("Share of spending (%)")
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    cycles = "-".join(str(c) for c in sorted(mix.cycle.unique()))
    fig.suptitle(f"In the most expensive TV markets, nobody shifts money to mail ({cycles})",
                 x=0.02, ha="left", y=1.05, fontsize=12)
    fig.savefig(path)
    plt.close(fig)
