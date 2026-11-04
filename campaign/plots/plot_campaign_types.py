import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from ..mine_spending import NO_STRUCTURE
from .style import AQUA, AXIS, INK, MUTED, NEUTRAL, ORANGE, RC, SECONDARY, SURFACE, VIOLET

COLORS = {"Early TV": ORANGE, "Incumbent maintenance": VIOLET, "Broad build-up": AQUA}
BENCHMARK_COLORS = (MUTED, NEUTRAL)
CHOSEN = 3
MINUS = "\u2212"


def scores_by_k(ax, actual, benchmarks, column, floor, ceiling, legend_at):
    """One measure against k: the real campaigns in ink, benchmarks in gray, and k = CHOSEN marked."""
    for (label, frame), color in zip(benchmarks.items(), BENCHMARK_COLORS):
        ax.plot(frame.index, frame[column], color=color, linewidth=2, solid_capstyle="round", label=label)
    ax.plot(actual.index, actual[column], color=INK, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round", label="Real campaigns")
    ax.annotate(f"k = {CHOSEN}, used in the report: {actual[column][CHOSEN]:.2f}", (CHOSEN, actual[column][CHOSEN]), xytext=(9, 7), textcoords="offset points", fontsize=8, color=SECONDARY,
                path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=legend_at, fontsize=8)
    ax.set_xticks(actual.index)
    ax.set_xlim(actual.index[0] - 0.2, actual.index[-1] + 0.2)
    ax.set_ylim(floor, ceiling)
    ax.set_xlabel("Clusters (k)")
    ax.grid(axis="x", visible=False)


def silhouette_widths(ax, scores, labels, order, names):
    """Each campaign's silhouette, sorted within its cluster: long bars sit clearly inside a cluster, short ones near a border."""
    gap = 0.04 * len(scores)
    bottom = 0
    for c in order[::-1]:
        s = np.sort(scores[labels == c])
        ax.fill_betweenx(bottom + np.arange(len(s)), 0, s, color=COLORS[names[c]], linewidth=0)
        ax.text(s.max() + 0.015, bottom + len(s) / 2, f"{names[c]}\nmean {s.mean():.2f}", fontsize=8, color=SECONDARY, va="center")
        bottom += len(s) + gap
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.axvline(NO_STRUCTURE, color=SECONDARY, linewidth=1)
    ax.text(NO_STRUCTURE + 0.008, bottom * 1.005, f"Rule of thumb: {NO_STRUCTURE}", fontsize=8, color=SECONDARY, va="bottom")
    ax.set_yticks([])
    ax.set_ylim(0, bottom * 1.07)
    ax.set_xlim(-0.08, 0.62)
    ax.set_xlabel("Silhouette of each campaign")
    ax.grid(False)
    ax.spines["left"].set_visible(False)


def cluster_map(ax, x, labels, order, names):
    """The campaigns on their first two principal components."""
    pca = PCA(2).fit(x)
    xy = pca.transform(x)
    shuffle = np.random.default_rng(0).permutation(len(xy))
    ax.scatter(*xy[shuffle].T, s=7, c=[COLORS[names[c]] for c in labels[shuffle]], alpha=0.45, linewidth=0, rasterized=True)
    for c in order:
        center = np.median(xy[labels == c], axis=0)
        ax.text(*center, names[c], fontsize=9, color=INK, ha="center", va="center",
                path_effects=[patheffects.withStroke(linewidth=3, foreground=SURFACE)])
    share = pca.explained_variance_ratio_
    ax.set_xlabel(f"First component ({share[0]:.0%} of the variation)")
    ax.set_ylabel(f"Second component ({share[1]:.0%})")
    ax.grid(False)


def profiles(ax, profile, order, names):
    """Mean share of spending by category and cluster, the categories that differ most on top."""
    spread = (profile.max() - profile.min()).sort_values()
    for y, column in enumerate(spread.index):
        ax.hlines(y, profile[column].min(), profile[column].max(), color=AXIS, linewidth=1, zorder=1)
        for c in order:
            ax.scatter(profile.loc[c, column], y, s=52, color=COLORS[names[c]], edgecolor=SURFACE, linewidth=1.2, zorder=2)
        if y >= len(spread) - 3:
            ax.text(profile[column].max() + 1.2, y, f"{profile[column].max():.0f}%", fontsize=8, color=SECONDARY, va="center")
    ax.set_yticks(range(len(spread)), spread.index, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1.5, 37)
    ax.set_xlabel("Average share of spending by June 30 (%)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


def forecast_change(ax, forecast):
    """Change in log loss from adding the clusters, by year, with the miss removed. Above zero is worse."""
    change = forecast.within
    ax.bar(range(len(change)), change, width=0.42, color=NEUTRAL)
    for i, v in enumerate(change):
        ax.text(i, v + (0.00012 if v > 0 else -0.00012), f"{v:+.4f}".replace("-", MINUS), ha="center", va="bottom" if v > 0 else "top", fontsize=8, color=SECONDARY)
    ax.axhline(0, color=SECONDARY, linewidth=1)
    ax.axhline(change.mean(), color=INK, linewidth=1)
    ax.text(-0.38, change.mean() + 0.00008, f"Mean {change.mean():+.5f}", ha="left", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(change)), change.index)
    ax.set_ylim(-0.0024, 0.0042)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.3f}".replace("-", MINUS) if v else "0")
    ax.set_ylabel("Change in log loss (above zero is worse)")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(False)


def plot_campaign_types(x, clustered, profile, names, order, ks, benchmarks, forecast, path):
    """Section 3 of SPENDING.md on one page. `x`: centered log-ratios; `clustered`: cluster and ici for each campaign; `profile`: mean shares by cluster; `names`: cluster to name;
    `order`: clusters as shown; `ks`: `choose_k`; `benchmarks`: label to `benchmark_k`; `forecast`: `cluster_signal`."""
    plt.rcParams.update(RC)
    labels = clustered.cluster.to_numpy()
    silhouette = silhouette_samples(x, labels)
    fig = plt.figure(figsize=(14.5, 9.2))
    grid = GridSpec(2, 3, figure=fig, top=0.84, bottom=0.1, left=0.05, right=0.97, hspace=0.36, wspace=0.28)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(3)]

    scores_by_k(axes[0], ks, benchmarks, "silhouette", 0, 0.3, (0.62, 0.45))
    axes[0].axhline(NO_STRUCTURE, color=SECONDARY, linewidth=1, zorder=0)
    axes[0].text(ks.index[-1] + 0.2, NO_STRUCTURE + 0.005, "At or below 0.25: no substantial structure, by rule of thumb", ha="right", va="bottom", fontsize=8, color=SECONDARY)
    axes[0].set_ylabel("Mean silhouette")
    axes[0].set_title("Separation is weak at every k, and k = 2 scores highest", fontsize=11)
    scores_by_k(axes[1], ks, benchmarks, "stability", 0.4, 1.03, "lower left")
    axes[1].set_ylabel("Stability on resamples (adjusted Rand)")
    axes[1].set_title("Shuffled data is about as stable, but only up to k = 4", fontsize=11)
    silhouette_widths(axes[2], silhouette, labels, order, names)
    axes[2].set_title(f"{(silhouette > NO_STRUCTURE).mean():.0%} of campaigns clear 0.25, and the highest is {silhouette.max():.2f}", fontsize=11)
    cluster_map(axes[3], x, labels, order, names)
    axes[3].set_title("The groups are neighboring slices of one cloud", fontsize=11)
    profiles(axes[4], profile.loc[order] * 100, order, names)
    axes[4].set_title("Broadcast marks Early TV; fundraising marks maintenance", fontsize=11)
    forecast_change(axes[5], forecast)
    axes[5].set_title(f"The types make the forecast worse in {(forecast.within > 0).sum()} of {len(forecast)} years", fontsize=11)

    handles = []
    for c in order:
        group = clustered[clustered.cluster == c]
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=8, color=COLORS[names[c]],
                              label=f"{names[c]}: {len(group):,} campaigns, {(group.ici == 'I').mean():.0%} incumbents"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.043, 0.925), ncol=3, fontsize=10, handletextpad=0.3, columnspacing=1.8)
    fig.suptitle("Campaign spending types exist but are weak, and they don't help the forecast", x=0.05, ha="left", y=0.965, fontsize=15)
    fig.text(0.05, 0.035, f"{len(clustered):,} House campaigns with at least $100K spent by June 30, 2012-2024. K-means on centered log-ratios of spending shares, with total spending left out. Log loss is against the forecast's baseline plus surplus,\nwith clusters fit only on earlier years and the year's national miss removed. Benchmarks run the same steps on data with no kinds of campaign in it, averaged over three draws. The map shows two components of many. The 0.25 rule of thumb is from Kaufman and Rousseeuw (1990).",
             fontsize=7.5, color=MUTED, va="top")
    fig.savefig(path)
    plt.close(fig)
