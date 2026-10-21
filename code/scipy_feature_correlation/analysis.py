"""Full-training-set association diagnostics; no predictive model or target encoding.

Discrete estimates use complete contingency counts, including a missing-value group.
Numeric rank/linear estimates use pairwise nonmissing rows. See the notebook for
formulas, assumptions, and why these are not validation performance or causality.
"""
from itertools import combinations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from plot.airline_eda import CONTINUOUS, CATEGORICAL_FEATURES, feature_groups

TARGET = 'satisfaction'


def catalog(train):
    """Assign semantic roles rather than treating every integer as continuous."""
    required = {TARGET, 'id', *CONTINUOUS, *CATEGORICAL_FEATURES}
    if not required.issubset(train):
        raise ValueError(f'Missing columns: {required - set(train)}')
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {False, True}:
        raise ValueError('Expected both nonmissing binary labels, 0/1 or bool.')
    if not train.id.is_unique:
        raise ValueError('Duplicate IDs.')
    if np.isinf(train.select_dtypes('number').to_numpy()).any():
        raise ValueError('Infinite numeric input.')
    roles = {c: kind for kind, columns in feature_groups(train).items() for c in columns}
    roles['id'] = 'id'
    for c, kind in roles.items():
        if kind != 'categorical' and not pd.api.types.is_numeric_dtype(train[c]):
            raise ValueError(f'{c}: expected numeric data')
        if kind == 'rating' and not train[c].dropna().isin(range(6)).all():
            raise ValueError(f'{c}: expected integer service ratings 0..5')
    return pd.DataFrame([
        dict(feature=c, kind=roles[c], dtype=str(train[c].dtype), n=len(train),
             missing=int(train[c].isna().sum()), unique=int(train[c].nunique()))
        for c in train.columns if c != TARGET
    ])


def group_values(x, kind, bins=20):
    """Label-free grouping: quantiles; delay zero separate; ID equal-width bins.

    Fit once on all train X for descriptive analysis; reuse the same groups within
    strata. Never use these full-data results as validation target encodings.
    """
    if not 2 <= bins <= 100:
        raise ValueError('bins must be between 2 and 100')
    groups = pd.Series('<missing>', index=x.index, dtype='object')
    valid = x.notna()
    if kind in ('continuous', 'id'):
        if 'Delay' in str(x.name):
            zero = valid & x.eq(0)
            groups.loc[zero] = 'zero'
            valid &= ~zero
        if x[valid].nunique() <= 1:
            groups.loc[valid] = 'constant'
        else:
            cut = pd.cut if kind == 'id' else pd.qcut
            groups.loc[valid] = cut(x[valid], bins, duplicates='drop').astype(str)
    else:
        if x[valid].astype(str).eq('<missing>').any():
            raise ValueError('Reserved missing-value label in input')
        groups.loc[valid] = x[valid].astype(str)
    return groups


def contingency(a, b):
    """Fast exact counts; factorize removes empty categories, preserves all rows."""
    a, av = pd.factorize(a, sort=False)
    b, bv = pd.factorize(b, sort=False)
    if len(a) != len(b) or len(a) == 0 or (a < 0).any() or (b < 0).any():
        raise ValueError('Expected equal nonempty arrays without missing group codes')
    return np.bincount(a * len(bv) + b, minlength=len(av)*len(bv)).reshape(len(av), len(bv))


def table_metrics(table):
    """Uncorrected Pearson chi-square, V and empirical MI in bits.

    H(Y|G) = sum_g P(g) H(Y|g); MI = H(Y)-H(Y|G).
    entropy_reduction = MI/H(Y), asymmetric and NOT classification accuracy.
    """
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or (t < 0).any() or not np.isfinite(t).all() or t.sum() == 0:
        raise ValueError('Expected a finite, nonnegative, nonempty contingency table')
    if not np.equal(t, np.floor(t)).all():
        raise ValueError('Expected integer observation counts')
    t = t.astype(np.int64)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    n = t.sum()
    hy = stats.entropy(t.sum(axis=0), base=2)
    conditional = np.dot(t.sum(axis=1)/n, stats.entropy(t, base=2, axis=1))
    mi = max(0., float(hy-conditional))
    result = dict(n=int(n), groups=len(t), label_entropy_bits=float(hy),
                  conditional_entropy_bits=float(conditional), mi_bits=mi,
                  entropy_reduction=mi/hy if hy > 0 else np.nan)
    if min(t.shape) < 2:
        # Constant X has no association; a constant Y has undefined normalized strength.
        result.update(cramers_v=0. if hy > 0 else np.nan, chi2=0., chi2_p=1.,
                      dof=0, min_expected=np.nan, expected_below5=np.nan)
    else:
        chi, p, dof, expected = stats.chi2_contingency(t, correction=False)
        result.update(cramers_v=float(stats.contingency.association(t, correction=False)),
                      chi2=float(chi), chi2_p=float(p), dof=int(dof),
                      min_expected=float(expected.min()), expected_below5=float((expected < 5).mean()))
    return result


def numeric_metrics(x, y):
    """Pairwise complete observations, keeping score ordering and all tied pairs.

    U/(n1*n0) is the AUC of the raw numeric value, with half credit for ties.
    D(score|label)=2*AUC-1; label MUST be first in scipy.stats.somersd.
    Somers is computed on the compact label-by-value table to avoid n-by-n work.
    """
    valid = x.notna() & y.notna()
    x, y = x[valid].to_numpy(), y[valid].astype(int).to_numpy()
    out = dict(n_numeric=len(x), point_biserial=np.nan, spearman=np.nan,
               kendall_tau_b=np.nan, raw_auc=np.nan, somers_d_auc=np.nan,
               somers_d_scipy=np.nan)
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return out
    out['point_biserial'] = float(stats.pointbiserialr(y, x).statistic)
    out['spearman'] = float(stats.spearmanr(x, y).statistic)
    out['kendall_tau_b'] = float(stats.kendalltau(x, y, variant='b', method='asymptotic').statistic)
    positive, negative = x[y == 1], x[y == 0]
    u = stats.mannwhitneyu(positive, negative, method='asymptotic').statistic
    out['raw_auc'] = float(u / (len(positive)*len(negative)))
    out['somers_d_auc'] = 2*out['raw_auc']-1
    # Only low-cardinality values: SciPy's table algorithm is costly for near-unique ID.
    if len(np.unique(x)) <= 100:
        t = pd.crosstab(y, x).reindex(index=[0, 1], fill_value=0).to_numpy()
        out['somers_d_scipy'] = float(stats.somersd(t).statistic)
        if not np.isclose(out['somers_d_scipy'], out['somers_d_auc'], atol=1e-10):
            raise AssertionError('Somers orientation/AUC mismatch')
    return out


def rate_table(train, feature, kind, groups):
    """All groups, including missing; Wilson intervals are pointwise, not simultaneous."""
    work = pd.DataFrame({'group': groups, 'y': train[TARGET].astype(int), 'x': train[feature]})
    t = work.groupby('group', sort=False, observed=True).agg(n=('y', 'size'), positive=('y', 'sum'))
    t['rate'] = t.positive/t.n
    if kind == 'categorical':
        t = t.sort_index()
        t['x'] = np.arange(len(t))
        t['x_min'] = t['x_max'] = np.nan
    else:
        t['x'] = work.groupby('group', observed=True).x.median()
        t['x_min'] = work.groupby('group', observed=True).x.min()
        t['x_max'] = work.groupby('group', observed=True).x.max()
        t = t.sort_values('x', na_position='last')
    z = stats.norm.ppf(.975)
    center = (t.rate+z*z/(2*t.n))/(1+z*z/t.n)
    half = z*np.sqrt(t.rate*(1-t.rate)/t.n+z*z/(4*t.n*t.n))/(1+z*z/t.n)
    t['low'], t['high'] = np.maximum(0., center-half), np.minimum(1., center+half)
    t = t.reset_index()
    t.insert(0, 'feature', feature)
    t.insert(1, 'kind', kind)
    return t


def stratified_analysis(groups, y, features, strata=('Class', 'Type of Travel')):
    """Within-stratum strengths + I(Y;X|Z), conditioning on ONE named Z at a time.

    Each stratum uses the globally defined X bins. Weighted within-stratum MI,
    not an average of V, gives the conditional information. No adjustment for
    every other feature, interaction test, or causal effect is claimed.
    """
    rows, conditional = [], []
    for z in strata:
        h_y_z = table_metrics(contingency(groups[z], y))['conditional_entropy_bits']
        for x in features:
            if x == z:
                continue
            cmi = 0.
            for level in pd.unique(groups[z]):
                mask = groups[z].eq(level)
                m = table_metrics(contingency(groups.loc[mask, x], y[mask]))
                cmi += mask.mean()*m['mi_bits']
                rows.append(dict(feature=x, stratifier=z, stratum=level,
                                 satisfaction_rate=float(y[mask].mean()), **m))
            conditional.append(dict(feature=x, stratifier=z, conditional_mi_bits=cmi,
                                    conditional_entropy_reduction=cmi/h_y_z if h_y_z > 0 else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(conditional)


def joint_analysis(groups, y, pairs):
    """Joint discrete MI and conditional gains from the same full sample/bins."""
    rows, rates = [], []
    for a, b in pairs:
        # Integer pair IDs avoid ambiguity from concatenating category strings.
        ac = pd.factorize(groups[a])[0]
        bc = pd.factorize(groups[b])[0]
        joint = ac*(bc.max()+1)+bc
        ma = table_metrics(contingency(groups[a], y))
        mb = table_metrics(contingency(groups[b], y))
        mj = table_metrics(contingency(joint, y))
        rows.append(dict(feature_a=a, feature_b=b, mi_a_bits=ma['mi_bits'],
                         mi_b_bits=mb['mi_bits'], joint_mi_bits=mj['mi_bits'],
                         gain_a_given_b_bits=max(0., mj['mi_bits']-mb['mi_bits']),
                         gain_b_given_a_bits=max(0., mj['mi_bits']-ma['mi_bits']),
                         joint_v=mj['cramers_v'], joint_groups=mj['groups']))
        t = pd.DataFrame({'level_a': groups[a], 'level_b': groups[b], 'y': y})
        t = t.groupby(['level_a', 'level_b'], observed=True).y.agg(n='size', positive='sum').reset_index()
        t['rate'] = t.positive/t.n
        t.insert(0, 'feature_a', a)
        t.insert(1, 'feature_b', b)
        rates.append(t)
    return pd.DataFrame(rows), pd.concat(rates, ignore_index=True)


def redundancy_analysis(groups, features):
    """Feature-feature V, with average linkage on dissimilarity 1-V (not Ward)."""
    matrix = pd.DataFrame(np.eye(len(features)), index=features, columns=features)
    rows = []
    for a, b in combinations(features, 2):
        v = table_metrics(contingency(groups[a], groups[b]))['cramers_v']
        matrix.loc[a, b] = matrix.loc[b, a] = v
        rows.append(dict(feature_a=a, feature_b=b, cramers_v=v))
    distance = np.clip(1-matrix.to_numpy(), 0., 1.)
    np.fill_diagonal(distance, 0.)
    linkage = hierarchy.linkage(squareform(distance, checks=True), method='average', optimal_ordering=True)
    return matrix, pd.DataFrame(rows).sort_values('cramers_v', ascending=False), linkage


def run_analysis(train, output='image/scipy_feature_correlation', bins=20):
    """Compute all summaries from train only; write reproducible local source tables."""
    metadata = catalog(train)
    y = train[TARGET].astype(int)
    groups = pd.DataFrame(index=train.index)
    rows, rates, sensitivity = [], {}, []
    for record in metadata.to_dict('records'):
        feature, kind = record['feature'], record['kind']
        groups[feature] = group_values(train[feature], kind, bins)
        table = rate_table(train, feature, kind, groups[feature])
        rates[feature] = table
        supported = table[table.n >= max(100, int(np.ceil(.001*len(train))))]
        row = dict(**record, **{k: v for k, v in table_metrics(contingency(groups[feature], y)).items() if k != 'n'},
                   min_group_n=int(table.n.min()),
                   supported_rate_min=supported.rate.min(), supported_rate_max=supported.rate.max())
        row['supported_rate_gap'] = row['supported_rate_max']-row['supported_rate_min']
        if kind != 'categorical':
            row.update(numeric_metrics(train[feature], y))
        rows.append(row)
        if kind in ('continuous', 'id'):
            for q in (10, 20, 40):
                metrics = table_metrics(contingency(group_values(train[feature], kind, q), y))
                sensitivity.append(dict(feature=feature, requested_bins=q, **metrics))
    summary = pd.DataFrame(rows)
    business = summary.kind.ne('id')
    summary['chi2_q_by'] = np.nan
    # One declared family: the business-feature chi-square tests; arbitrary dependence allowed.
    summary.loc[business, 'chi2_q_by'] = stats.false_discovery_control(summary.loc[business, 'chi2_p'], method='by')
    summary['rank_v'] = summary.loc[business, 'cramers_v'].rank(ascending=False, method='min')
    summary = summary.sort_values(['kind', 'cramers_v'], ascending=[True, False]).reset_index(drop=True)
    features = summary.loc[summary.kind.ne('id')].sort_values('cramers_v', ascending=False).feature.tolist()
    strata, conditional = stratified_analysis(groups, y, features)
    pairs = [('Online boarding', 'Inflight wifi service'), ('Online boarding', 'Class'),
             ('Inflight wifi service', 'Class'), ('Class', 'Type of Travel')]
    joint, joint_rates = joint_analysis(groups, y, pairs)
    matrix, redundancy, linkage = redundancy_analysis(groups, features)
    result = dict(catalog=metadata, summary=summary, rates=rates,
                  sensitivity=pd.DataFrame(sensitivity), strata=strata, conditional=conditional,
                  joint=joint, joint_rates=joint_rates, redundancy=redundancy,
                  feature_matrix=matrix, linkage=linkage)
    output = Path(output)
    source = output/'source_data'
    source.mkdir(parents=True, exist_ok=True)
    for name in ['catalog', 'summary', 'sensitivity', 'strata', 'conditional', 'joint', 'joint_rates', 'redundancy']:
        result[name].to_csv(source/f'{name}.csv', index=False)
    matrix.to_csv(source/'feature_matrix.csv')
    pd.concat(rates.values(), ignore_index=True).to_csv(source/'group_rates.csv', index=False)
    pd.DataFrame(linkage, columns=['left', 'right', 'distance', 'count']).to_csv(source/'linkage.csv', index=False)
    runtime = dict(python=sys.version, executable=sys.executable, scipy=scipy.__version__,
                   pandas=pd.__version__, numpy=np.__version__, rows=len(train), positives=int(y.sum()),
                   bins=bins, sensitivity_bins=[10, 20, 40], model_fit=False, sampling=False,
                   multiple_testing='BY; business-feature chi-square family only',
                   numeric_missing='pairwise exclusion', grouped_missing='explicit group')
    (source/'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2))
    return result


def feature_guide(result):
    """One readable record per business feature; recommendations are hypotheses."""
    descriptions = {
        'Online boarding': '优先：比较等级类别编码与数值编码；检查 0 和 3→4 跳变。',
        'Inflight wifi service': '优先：保留 0 特殊组；不要先加单调约束；检查与舱位交互。',
        'Inflight entertainment': '比较非线性/类别编码；检查与座椅、清洁的冗余。',
        'Seat comfort': '比较非线性/类别编码；检查与娱乐、清洁的冗余。',
        'Class': '类别编码；检查分层基线与评分交互。',
        'Type of Travel': '类别编码；检查与舱位及客户类型的组合。',
        'Customer Type': '类别编码；检查客户类型是否改变服务评分的关系。',
        'Gender': '边际关联小，保留作基线对照；不凭单变量结果直接删列。',
        'Age': '检查非单调：树模型或样条；不只拟合一个线性系数。',
        'Flight Distance': '检查阈值及舱位混杂；比较非线性表达。',
        'Departure Delay in Minutes': '零值标记＋原始值/log1p；保留长尾，不随意截断。',
        'Arrival Delay in Minutes': '保留缺失标记、零值和长尾；填补必须在训练折内拟合。',
        'Gate location': '线性/排序关联小也可能有组间差异；检查 U 形与类别编码。',
        'Departure/Arrival time convenient': '边际关联较小；先保留，验证是否有条件或交互增益。',
    }
    t = result['summary'].query("kind != 'id'").sort_values('cramers_v', ascending=False).copy()
    t['建模检查建议'] = t.feature.map(descriptions).fillna('比较等级类别编码与数值编码；检查分组曲线及冗余。')
    return t[['feature', 'kind', 'cramers_v', 'entropy_reduction', 'raw_auc', '建模检查建议']]
