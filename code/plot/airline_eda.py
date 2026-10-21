# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) Counts/category rates -> BarCategorical: incompatible benchmark aggregates -> param inherit
# (b) Conditional-rate curves -> LineTrend: incompatible cumulative time series -> param inherit
# (c) Association overview -> cross-type inherit (BarCategorical) -> param inherit
# (d) Correlation -> CorrelationMatrix: ggpairs differs from compact heatmap -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.
# Inherited: equal facets, probability range 0..1, clean spines, no legend frame.
# Baseline overrides asset typography/colors/dimensions to the journal specification.
# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill export settings, adapted for PDF-only output.
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Export a vector PDF only."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)

from pathlib import Path
import textwrap
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, MaxNLocator

TARGET = 'satisfaction'
CONTINUOUS = ['Age', 'Flight Distance', 'Departure Delay in Minutes',
              'Arrival Delay in Minutes']
CATEGORICAL_FEATURES = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
BLUE = CATEGORICAL[0]


def feature_groups(train):
    """Dataset-specific roles; id is a diagnostic, never a business predictor."""
    ratings = [c for c in train.columns if c not in
               [TARGET, 'id', *CONTINUOUS, *CATEGORICAL_FEATURES]]
    return {'continuous': CONTINUOUS, 'rating': ratings,
            'categorical': CATEGORICAL_FEATURES}


def validate_data(train, test):
    required = {TARGET, 'id', *CONTINUOUS, *CATEGORICAL_FEATURES}
    if not required.issubset(train):
        raise ValueError(f'Missing columns: {required - set(train)}')
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {False, True}:
        raise ValueError('Expected a nonmissing binary satisfaction label (0/1 or bool).')
    if set(test.columns) != set(train.columns) - {TARGET}:
        raise ValueError('Train/test predictor columns differ.')
    for frame in (train, test):
        if not frame['id'].is_unique:
            raise ValueError('IDs must be unique within each dataset.')
        numeric = frame.select_dtypes(include='number')
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError('Infinite numeric values require explicit handling.')


def data_profile(train, test):
    """All columns from both datasets; no row filtering or imputation."""
    rows = []
    for name, frame in [('train', train), ('test', test)]:
        for c in frame:
            rows.append(dict(dataset=name, feature=c, dtype=str(frame[c].dtype),
                             rows=len(frame), missing=int(frame[c].isna().sum()),
                             unique=int(frame[c].nunique()),
                             missing_fraction=float(frame[c].isna().mean())))
    return pd.DataFrame(rows)


def grouped_rates(frame, feature, kind, bins=20):
    """Full-data conditional means and pointwise 95% Wilson intervals.

    Continuous: quantile bins (duplicate edges collapsed), represented at median x.
    Delays: zero is separate, positive values quantile-binned. Missing is a group.
    Ordinal/categorical: every original level. No rows are dropped.
    """
    if not 2 <= bins <= 100:
        raise ValueError('bins must be between 2 and 100')
    x = frame[feature]
    labels = pd.Series('<missing>', index=frame.index, dtype='object')
    if kind in ('continuous', 'id'):
        valid = x.notna()
        if 'Delay' in feature:
            labels.loc[valid & x.eq(0)] = 'zero'
            valid &= x.ne(0)
        if valid.any():
            if x[valid].nunique() <= 1:
                labels.loc[valid] = 'constant'
            else:
                if kind == 'id':
                    binned = pd.cut(x[valid], bins=bins, duplicates='drop')
                else:
                    binned = pd.qcut(x[valid], q=bins, duplicates='drop')
                labels.loc[valid] = binned.astype(str)
    else:
        labels.loc[x.notna()] = x[x.notna()].astype(str)
    work = pd.DataFrame({'group': labels, 'x': x, 'y': frame[TARGET].astype(int)})
    result = work.groupby('group', sort=False, observed=True).agg(
        n=('y', 'size'), positive=('y', 'sum'), rate=('y', 'mean'))
    if kind != 'categorical':
        result['x'] = work.groupby('group', sort=False, observed=True)['x'].median()
        result['x_min'] = work.groupby('group', sort=False, observed=True)['x'].min()
        result['x_max'] = work.groupby('group', sort=False, observed=True)['x'].max()
        result = result.sort_values('x', na_position='last')
    else:
        result = result.sort_index()
        result['x'] = np.arange(len(result))
    n, p, z = result['n'], result['rate'], 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    result['low'], result['high'] = np.maximum(0, center-half), np.minimum(1, center+half)
    result = result.reset_index()
    result.insert(0, 'feature', feature)
    result.insert(1, 'kind', kind)
    assert result.n.sum() == len(frame), (feature, 'grouped row count changed')
    assert result.positive.sum() == frame[TARGET].sum()
    assert ((result.low >= 0) & (result.high <= 1+1e-12)).all()
    return result


def association_summary(train, tables):
    """Descriptive eta-squared of grouped rates, not validated feature importance."""
    overall = train[TARGET].mean()
    rows = []
    for feature, table in tables.items():
        weights = table.n / table.n.sum()
        eta2 = float((weights*(table.rate-overall)**2).sum() / (overall*(1-overall)))
        supported = table.loc[table.n >= max(100, int(np.ceil(len(train)*.001)))]
        gap = supported.rate.max()-supported.rate.min() if len(supported) > 1 else np.nan
        numeric = pd.api.types.is_numeric_dtype(train[feature])
        pearson = train[feature].corr(train[TARGET].astype(int)) if numeric else np.nan
        spearman = train[feature].corr(train[TARGET].astype(int), method='spearman') if numeric else np.nan
        rows.append(dict(feature=feature, grouped_eta_squared=eta2,
                         supported_rate_gap=gap, pearson=pearson, spearman=spearman,
                         groups=len(table), minimum_group_n=int(table.n.min())))
    return pd.DataFrame(rows).sort_values('grouped_eta_squared', ascending=False).reset_index(drop=True)


def _save(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_cns_figure(fig, output / name)
    return fig


def plot_overview(train, test, profile, output='image'):
    fig, axes = plt.subplots(1, 2, figsize=(183/25.4, 85/25.4), layout='constrained')
    counts = train[TARGET].value_counts().reindex([False, True])
    bars = axes[0].bar(['Unsatisfied (0)', 'Satisfied (1)'], counts,
                       color=[GREY, BLUE], width=.6)
    for bar, value in zip(bars, counts):
        axes[0].text(bar.get_x()+bar.get_width()/2, value, f'{value:,}\n{value/len(train):.2%}',
                     ha='center', va='bottom', fontsize=7)
    axes[0].set(ylim=(0, counts.max()*1.3), ylabel='Training rows', title='(a) Binary target balance')
    missing = profile.loc[profile.missing.gt(0)].copy()
    if len(missing):
        labels = [f'{r.dataset}:\n'+textwrap.fill(r.feature, 22) for r in missing.itertuples()]
        axes[1].bar(labels, missing.missing_fraction, color=BLUE, width=.5)
        for i, r in enumerate(missing.itertuples()):
            axes[1].text(i, r.missing_fraction, f'n={r.missing:,}', ha='center', va='bottom', fontsize=7)
        axes[1].set_ylim(0, missing.missing_fraction.max()*1.4)
    else:
        axes[1].text(.5, .5, 'No missing values', transform=axes[1].transAxes, ha='center')
    axes[1].set(title='(b) Missingness (nonzero columns)', ylabel='Fraction of rows')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=2))
    return _save(fig, output, '01_target_and_missingness')


def plot_rate_pages(tables, features, baseline, prefix, output='image', per_page=6):
    """Equal-size feature facets, each with a matched sample-count strip."""
    figures = []
    for start in range(0, len(features), per_page):
        selected = features[start:start+per_page]
        ncols = min(2, len(selected))
        rows = (len(selected)+ncols-1)//ncols
        width = 89 if ncols == 1 else 183
        fig = plt.figure(figsize=(width/25.4, (rows*67+18)/25.4), layout='constrained')
        grid = fig.add_gridspec(rows, ncols)
        caption = ('Satisfaction: mean and 95% Wilson CI\nDashed: overall rate; lower strip: counts'
                   if ncols == 1 else
                   'Training satisfaction rate | points: mean; bars: 95% Wilson CI\n'
                   'Dashed: overall rate; lower strips: all row counts')
        fig.suptitle(caption, fontsize=8)
        for j, feature in enumerate(selected):
            table = tables[feature]
            nested = grid[j//ncols, j%ncols].subgridspec(2, 1, height_ratios=[3, 1], hspace=.05)
            ax = fig.add_subplot(nested[0])
            count_ax = fig.add_subplot(nested[1], sharex=ax)
            kind = table.kind.iloc[0]
            widths = .65
            if kind in ('continuous', 'id'):
                xpos = table.x.to_numpy().copy()
                missing = table.group.eq('<missing>').to_numpy()
                finite = xpos[~missing]
                span = max(float(np.ptp(finite)), 1.)
                xpos[missing] = finite.max() + span*.15
                if 'Delay' in feature:
                    xpos[missing] = max(finite.max()*2, 1)
                    ax.set_xscale('symlog', linthresh=1)
                    ticks = [v for v in [0, 1, 10, 100] if v <= finite.max()]
                    tick_labels = [str(v) for v in ticks]
                    xlabel = 'Minutes (symlog; linear from 0 to 1)'
                else:
                    ticks = np.linspace(finite.min(), finite.max(), 4).tolist()
                    tick_labels = [f'{v:,.0f}' for v in ticks]
                    xlabel = 'Bin median (original feature scale)'
                if missing.any():
                    ticks += xpos[missing].tolist()
                    tick_labels += ['NA']*int(missing.sum())
                count_ax.set_xticks(ticks, tick_labels)
                count_ax.set_xlabel(xlabel, fontsize=7)
                # Narrow count bars preserve median positions without suggesting density.
                delta = np.diff(np.unique(xpos))
                widths = .65*delta.min() if len(delta) else .65
            else:
                xpos = np.arange(len(table))
                labels = ['NA' if v == '<missing>' else textwrap.fill(v, 17) for v in table.group]
                count_ax.set_xticks(xpos, labels)
                count_ax.set_xlabel('Observed level', fontsize=7)
            if kind != 'categorical':
                # Do not imply continuity to missingness or between zero and positive delay bins.
                connect = ~table.group.isin(['<missing>', 'zero'])
                ax.plot(xpos[connect], table.loc[connect, 'rate'], color=BLUE, lw=1)
            ax.errorbar(xpos, table.rate,
                        yerr=np.vstack([table.rate-table.low, table.high-table.rate]),
                        fmt='o', color=BLUE, ms=3, lw=.8, capsize=2)
            ax.axhline(baseline, color=GREY, ls='--', lw=.6)
            ax.set(ylim=(-.03, 1.03), ylabel='Satisfied',
                   title=f'({chr(97+j)}) '+textwrap.fill(feature, 34))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.tick_params(axis='x', labelbottom=False)
            count_ax.bar(xpos, table.n, color=GREY, width=widths)
            count_ax.set_ylabel('n', rotation=0, labelpad=9)
            count_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            count_ax.yaxis.get_offset_text().set_fontsize(6)
            count_ax.yaxis.set_major_locator(MaxNLocator(2))
            count_ax.tick_params(axis='x', labelsize=6)
        figures.append(_save(fig, output, f'{prefix}_{start//per_page+1:02d}'))
    return figures


def plot_associations(summary, output='image'):
    fig, ax = plt.subplots(figsize=(183/25.4, 158/25.4), layout='constrained')
    ordered = summary.iloc[::-1]
    ax.barh(ordered.feature, ordered.grouped_eta_squared, color=BLUE, height=.65)
    ax.set(xlabel='Grouped between-rate variance / binary target variance',
           title='Descriptive univariate association (all 21 business features)\n'
                 'Quantile bins for continuous features; original levels otherwise')
    return _save(fig, output, '05_feature_associations')


def plot_correlations(train, output='image'):
    columns = [c for c in train.select_dtypes(include='number').columns if c != 'id']
    columns += [TARGET]
    corr = train[columns].astype(float).corr(method='spearman')
    fig, ax = plt.subplots(figsize=(183/25.4, 183/25.4), layout='constrained')
    cmap = LinearSegmentedColormap.from_list('academic_diverging', DIVERGING)
    shown = np.ma.masked_where(np.triu(np.ones(corr.shape), 1).astype(bool), corr.to_numpy())
    im = ax.pcolormesh(shown, cmap=cmap, vmin=-1, vmax=1, shading='flat')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    names = [textwrap.fill(c, 24) for c in corr.columns]
    ax.set_xticks(np.arange(len(corr))+.5, names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(corr))+.5, names, fontsize=6)
    ax.tick_params(length=0)
    ax.set_title('Spearman correlations | pairwise complete observations\n'
                 'ID excluded; label encoded as 0/1; missing counts in source tables')
    colorbar = fig.colorbar(im, ax=ax, shrink=.65, label='Spearman rho', fraction=.04)
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor('face')  # Avoid PDF viewer seams between vector color strips.
    for side in ax.spines.values():
        side.set_visible(False)
    fig = _save(fig, output, '06_spearman_correlations')
    return fig, corr


def run_eda(train, test, output='image'):
    """Rebuild all PDFs and traceable source tables. No model fit."""
    validate_data(train, test)
    output = Path(output)
    source = output / 'source_data'
    source.mkdir(parents=True, exist_ok=True)
    metadata = dict(python=sys.version, executable=sys.executable, pandas=pd.__version__,
                    numpy=np.__version__, matplotlib=mpl.__version__,
                    training_rows=len(train), test_rows=len(test), quantile_bins=20,
                    sampling=False, imputation=False, model_fit=False)
    (source/'runtime.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    profile = data_profile(train, test)
    groups = feature_groups(train)
    tables = {c: grouped_rates(train, c, kind) for kind, cols in groups.items() for c in cols}
    summary = association_summary(train, tables)
    profile.to_csv(source/'data_profile.csv', index=False)
    pd.concat(tables.values(), ignore_index=True).to_csv(source/'feature_rates.csv', index=False)
    summary.to_csv(source/'feature_associations.csv', index=False)
    figures = [plot_overview(train, test, profile, output)]
    for kind, prefix in [('continuous', '02_continuous_rates'), ('rating', '03_rating_rates'),
                         ('categorical', '04_category_rates')]:
        figures += plot_rate_pages(tables, groups[kind], train[TARGET].mean(), prefix, output)
    figures.append(plot_associations(summary, output))
    corr_fig, corr = plot_correlations(train, output)
    figures.append(corr_fig)
    corr.to_csv(source/'spearman_correlations.csv')
    numeric = train[corr.columns].notna().astype('int64')
    numeric.T.dot(numeric).to_csv(source/'correlation_pair_counts.csv')
    # ID is inspected for ordering artifacts, never included in business rankings.
    id_table = grouped_rates(train, 'id', 'id')
    id_table.to_csv(source/'id_rates.csv', index=False)
    figures += plot_rate_pages({'id': id_table}, ['id'], train[TARGET].mean(), '07_id_diagnostic', output)
    return {'profile': profile, 'rates': tables, 'associations': summary,
            'correlations': corr, 'figures': figures}
