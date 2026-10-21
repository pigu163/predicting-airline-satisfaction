"""Notebook display helpers; every figure is saved as PDF only."""
from pathlib import Path
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

from plot.airline_eda import BLUE, plot_rate_pages


def _save(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)  # The notebook explicitly displays each figure exactly once.
    return fig


def plot_strength(result, output):
    t = result['summary'].query("kind != 'id'").sort_values('cramers_v')
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), layout='constrained', sharey=True)
    labels = [textwrap.fill(x, 26) for x in t.feature]
    for ax, column, title in zip(axes, ['cramers_v', 'entropy_reduction'],
                                 ["Cramer's V", 'Empirical I(Y;G) / H(Y)']):
        ax.barh(np.arange(len(t)), t[column], color=BLUE)
        ax.set(xlim=(0, 1), title=title, xlabel='Association (not validation performance)')
        for i, v in enumerate(t[column]):
            ax.text(v+.012, i, f'{v:.3f}', va='center', fontsize=7)
    axes[0].set_yticks(np.arange(len(t)), labels)
    fig.suptitle('All 21 business features | continuous variables grouped into quantile bins', fontsize=10)
    return _save(fig, output, '01_feature_strength')


def plot_numeric(result, output):
    t = result['summary'].query("kind in ['continuous', 'rating']").sort_values('cramers_v', ascending=False)
    columns = ['point_biserial', 'spearman', 'kendall_tau_b', 'somers_d_auc']
    values = t[columns].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7.5), layout='constrained')
    im = ax.imshow(values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(4), ['Point-biserial r', 'Spearman rho', 'Kendall tau-b', 'Somers D = 2*AUC-1'], rotation=18, ha='right')
    ax.set_yticks(range(len(t)), t.feature)
    for (i, j), v in np.ndenumerate(values):
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7, color='white' if abs(v) > .65 else 'black')
    ax.set_title('Raw-value direction / ordering | nonmissing rows only\nNo category codes for text features; no ID in business ranking')
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label='Signed coefficient; scales differ across columns', shrink=.7)
    return _save(fig, output, '02_numeric_direction')


def plot_sensitivity(result, output):
    t = result['sensitivity']
    features = t.loc[t.feature.ne('id'), 'feature'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    for feature in features:
        s = t[t.feature.eq(feature)]
        for ax, metric in zip(axes, ['cramers_v', 'entropy_reduction']):
            ax.plot(s.requested_bins, s[metric], 'o-', ms=4, lw=1, label=feature)
    for ax, title in zip(axes, ["Cramer's V", 'I(Y;G) / H(Y)']):
        ax.set(xlabel='Requested quantile bins (ties collapsed; delay zero separate)',
               ylabel=title, ylim=(0, 1), xticks=[10, 20, 40])
    axes[1].legend(loc='upper left', fontsize=6)
    return _save(fig, output, '03_bin_sensitivity')


def plot_strata(result, output):
    t = result['strata'].copy()
    t['panel'] = t.stratifier + ': ' + t.stratum
    features = result['summary'].query("kind != 'id'").sort_values('cramers_v', ascending=False).feature
    matrix = t.pivot(index='feature', columns='panel', values='cramers_v').reindex(features)
    fig, ax = plt.subplots(figsize=(9.5, 8), layout='constrained')
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#EEEEEE')
    im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_xticks(range(len(matrix.columns)), [textwrap.fill(c, 22) for c in matrix.columns], rotation=30, ha='right')
    for (i, j), v in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, 'N/A' if np.isnan(v) else f'{v:.3f}', ha='center', va='center', fontsize=7,
                color='white' if v > .65 else 'black')
    ax.set_title("Within-group Cramer's V | globally fixed feature bins\nN/A: feature is the stratifier; compare with group sizes and baseline rates")
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label="Within-stratum V", shrink=.65)
    return _save(fig, output, '04_stratified_strength')


def plot_joint(result, output):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    # Color encodes satisfaction rate, not association strength; annotate cell counts.
    for ax, pair in zip(axes.flat, result['joint'].itertuples()):
        t = result['joint_rates'].query('feature_a == @pair.feature_a and feature_b == @pair.feature_b')
        rates = t.pivot(index='level_a', columns='level_b', values='rate').sort_index().sort_index(axis=1)
        counts = t.pivot(index='level_a', columns='level_b', values='n').reindex(index=rates.index, columns=rates.columns)
        im = ax.imshow(rates.to_numpy(), vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_yticks(range(len(rates)), rates.index)
        ax.set_xticks(range(len(rates.columns)), [textwrap.fill(str(v), 12) for v in rates.columns], rotation=30, ha='right')
        ax.set(xlabel=pair.feature_b, ylabel=pair.feature_a)
        for (i, j), v in np.ndenumerate(rates.to_numpy()):
            if not np.isnan(v):
                ax.text(j, i, f'{v:.0%}\nn={int(counts.iloc[i,j]):,}', ha='center', va='center', fontsize=6,
                        color='white' if v > .65 else 'black')
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, label='Empirical satisfaction rate', shrink=.65)
    fig.suptitle('Selected combinations | all observed cells, no fitted interaction model', fontsize=10)
    return _save(fig, output, '05_joint_rates')


def plot_redundancy(result, output):
    matrix = result['feature_matrix']
    linkage = result['linkage']
    order = hierarchy.leaves_list(linkage)
    m = matrix.iloc[order, order]
    fig = plt.figure(figsize=(13, 8.5), layout='constrained')
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.7])
    left, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    hierarchy.dendrogram(linkage, orientation='left', labels=matrix.index.tolist(), leaf_font_size=6, ax=left,
                         color_threshold=0, above_threshold_color=BLUE)
    left.set(xlabel='Average-linkage dissimilarity (1 - V)', title='Feature grouping; no target used')
    im = right.imshow(m, vmin=0, vmax=1, cmap='Blues')
    right.set_xticks(range(len(m)), m.columns, rotation=90, fontsize=6)
    right.set_yticks(range(len(m)), m.index, fontsize=6)
    right.tick_params(length=0)
    right.set_title("Pairwise feature-feature Cramer's V\nContinuous features binned; association is not conditional redundancy")
    fig.colorbar(im, ax=right, label='Feature-feature V', shrink=.6)
    return _save(fig, output, '06_feature_hierarchy')


def plot_profiles(result, baseline, output):
    """Reuse existing rate panels, including counts and Wilson intervals, for ALL features."""
    figures = []
    for kind, prefix in [('continuous', '07_continuous'), ('rating', '08_rating'),
                         ('categorical', '09_categorical'), ('id', '10_id_diagnostic')]:
        features = result['catalog'].loc[result['catalog'].kind.eq(kind), 'feature'].tolist()
        pages = plot_rate_pages(result['rates'], features, baseline, prefix, output=output)
        for fig in pages:
            plt.close(fig)
        figures += pages
    return figures
