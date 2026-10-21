"""Run from the project root: python code/scipy_feature_correlation/check_analysis.py."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy_feature_correlation.analysis import (
    contingency, group_values, table_metrics, numeric_metrics, rate_table,
    stratified_analysis, joint_analysis,
)


def check():
    # Independence, perfect prediction and constant/empty margins.
    independent = table_metrics([[10, 10], [10, 10]])
    perfect = table_metrics([[20, 0], [0, 20]])
    assert independent['cramers_v'] == independent['mi_bits'] == 0
    assert np.isclose(perfect['cramers_v'], 1) and np.isclose(perfect['entropy_reduction'], 1)
    assert table_metrics([[10, 10], [0, 0]])['cramers_v'] == 0
    assert np.isnan(table_metrics([[10, 0], [20, 0]])['cramers_v'])

    # U shape: zero signed monotonic trend can coexist with perfect association.
    x = pd.Series([0, 1, 2]*10, name='rating')
    y = pd.Series([1, 0, 1]*10)
    assert np.isclose(numeric_metrics(x, y)['spearman'], 0)
    assert np.isclose(table_metrics(contingency(x, y))['cramers_v'], 1)
    assert np.isclose(numeric_metrics(x, y)['raw_auc'], .5)
    forward = numeric_metrics(pd.Series([0, 1, 2, 2, 3, 4]), pd.Series([0, 0, 0, 1, 1, 1]))
    assert np.isclose(forward['somers_d_scipy'], 2*forward['raw_auc']-1)
    assert np.isclose(forward['raw_auc'], 8.5/9)

    # Wilson and missing counts; numeric metrics exclude just the missing row.
    f = pd.DataFrame({'x': [0, 0, 1, 2, np.nan, 100], 'satisfaction': [0, 1, 0, 1, 1, 0]})
    f.x.name = 'Arrival Delay in Minutes'
    g = group_values(f.x, 'continuous', 3)
    assert g.eq('zero').sum() == 2 and g.eq('<missing>').sum() == 1
    t = rate_table(f, 'x', 'continuous', g)
    assert t.n.sum() == 6 and t.positive.sum() == 3
    assert numeric_metrics(f.x, f.satisfaction)['n_numeric'] == 5
    for row in t.itertuples():
        ci = stats.binomtest(row.positive, row.n).proportion_ci(method='wilson')
        assert np.allclose([row.low, row.high], [ci.low, ci.high])

    # XOR: marginal signals absent, joint/conditional information exactly one bit.
    g = pd.DataFrame({'a': [0, 0, 1, 1]*10, 'b': [0, 1, 0, 1]*10})
    y = pd.Series([0, 1, 1, 0]*10)
    joint, _ = joint_analysis(g, y, [('a', 'b')])
    assert np.isclose(joint.joint_mi_bits.iloc[0], 1)
    assert np.isclose(joint.mi_a_bits.iloc[0], 0)
    strata, conditional = stratified_analysis(g, y, ['a'], strata=['b'])
    assert strata.n.sum() == len(g)
    assert np.isclose(conditional.conditional_mi_bits.iloc[0], 1)
    # I(Y;A,B)-I(Y;B) == weighted within-B I(Y;A).
    assert np.isclose(joint.gain_a_given_b_bits.iloc[0], conditional.conditional_mi_bits.iloc[0])
    print('PASS: independence/perfect/U-shape, ties and AUC orientation, missing/Wilson, XOR/conditional MI.')


if __name__ == '__main__':
    check()
