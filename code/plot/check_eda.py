"""Small runnable checks of aggregation and data integrity (no ML fit)."""
import numpy as np
import pandas as pd
from airline_eda import grouped_rates, feature_groups, validate_data


def check():
    # Missing, tied continuous values, all-zero/all-one outcomes and special zero.
    toy = pd.DataFrame({'satisfaction': [0, 1, 1, 0, 1, 0],
                        'x': [0, 0, 1, 1, np.nan, 4],
                        'Arrival Delay in Minutes': [0, 0, 1, 2, np.nan, 4]})
    for kind in ['continuous', 'rating', 'categorical', 'id']:
        result = grouped_rates(toy, 'x', kind, bins=3)
        assert result.n.sum() == 6 and result.positive.sum() == 3
        assert result.loc[result.group.eq('<missing>'), 'n'].iloc[0] == 1
        assert np.all(result.low <= result.rate) and np.all(result.high >= result.rate)
    delay = grouped_rates(toy, 'Arrival Delay in Minutes', 'continuous', bins=3)
    assert delay.loc[delay.group.eq('zero'), 'n'].iloc[0] == 2
    assert delay.loc[delay.group.eq('<missing>'), 'n'].iloc[0] == 1
    for outcome in [0, 1]:
        constant = toy.assign(x=1, satisfaction=outcome)
        assert len(grouped_rates(constant, 'x', 'continuous')) == 1
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    train, test = pd.read_csv(root/'train.csv'), pd.read_csv(root/'test.csv')
    validate_data(train, test)
    groups = feature_groups(train)
    assert sum(map(len, groups.values())) == 21
    source = pd.read_csv(root/'image/source_data/feature_rates.csv')
    assert set(source.feature) == set(train.columns)-{'id', 'satisfaction'}
    for _, t in source.groupby('feature'):
        assert t.n.sum() == len(train) and t.positive.sum() == train.satisfaction.sum()
    print('PASS: edge cases, all 21 feature groups, full row/label conservation.')


if __name__ == '__main__':
    check()
