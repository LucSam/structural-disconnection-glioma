#!/usr/bin/env python3
"""Describe shared samples and classifications without adding significance tests."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from _common import ROOT, OUTCOMES


def main():
    display = pd.read_csv(ROOT / 'data/private/display_participants.csv')
    aat = display[display.battery.eq('AAT')].set_index('subject_id')
    dem = display[display.battery.eq('DemTect')].set_index('subject_id')
    paired = aat[['burden', 'tertile', 'preserved']].join(
        dem[['burden', 'tertile', 'preserved']], how='inner', lsuffix='_aat', rsuffix='_dem')
    outcomes = pd.read_csv(ROOT / 'data/private/outcomes.csv').set_index('subject_id')
    raw_aat = outcomes.loc[aat.index, list(OUTCOMES['AAT'].values())].to_numpy()
    raw_dem = outcomes.loc[dem.index, list(OUTCOMES['DemTect'].values())].to_numpy()
    assert np.isfinite(raw_aat).all() and np.isfinite(raw_dem).all()
    assert np.array_equal((raw_aat >= [63,63,64,64]).all(axis=1), aat.preserved)
    assert np.array_equal(outcomes.loc[dem.index,'DemTect_global'].ge(13), dem.preserved)
    high = paired[paired.tertile_aat.eq(3) & paired.tertile_dem.eq(3)]
    rows = []
    for sample, data in [('Both complete batteries', paired), ('Both highest thirds', high)]:
        for a in [True, False]:
            for d in [True, False]:
                rows.append({'sample': sample, 'AAT': 'No deficit' if a else 'Deficit',
                             'DemTect': 'No deficit' if d else 'Deficit',
                             'n': int((data.preserved_aat.eq(a) & data.preserved_dem.eq(d)).sum())})
    pd.DataFrame(rows).to_csv(ROOT / 'results/battery_status_overlap.csv', index=False)
    regional = pd.read_csv(ROOT / 'data/private/regional_chaco.csv').set_index('subject_id')
    atlas = pd.read_csv(ROOT / 'data/resources/network_definitions.csv')
    left = ['chaco_roi_' + str(i) for i in atlas.loc[atlas.roi_name.str.endswith('(L)'), 'roi_index']]
    burden_left, burden_all = regional[left].mean(axis=1), regional.mean(axis=1)
    np.testing.assert_allclose(paired.burden_aat, burden_left.loc[paired.index], atol=1e-12)
    np.testing.assert_allclose(paired.burden_dem, burden_all.loc[paired.index], atol=1e-12)
    summary = {
        'n_both_complete': len(paired),
        'n_same_status': int(paired.preserved_aat.eq(paired.preserved_dem).sum()),
        'n_both_no_deficit': int((paired.preserved_aat & paired.preserved_dem).sum()),
        'n_both_deficit': int((~paired.preserved_aat & ~paired.preserved_dem).sum()),
        'n_high_aat': int(aat.tertile.eq(3).sum()), 'n_high_demtect': int(dem.tertile.eq(3).sum()),
        'n_both_high': len(high),
        'n_high_same_status': int(high.preserved_aat.eq(high.preserved_dem).sum()),
        'burden_spearman_rho_paired': float(spearmanr(paired.burden_aat, paired.burden_dem).statistic),
        'burden_spearman_rho_all_163': float(spearmanr(burden_left, burden_all).statistic),
        'correlation_sample': '104 patients with both complete batteries; mean left-labelled versus whole-brain regional ChaCo',
        'interpretation': 'Descriptive overlap; no p values, inferential family or test of map similarity',
    }
    (ROOT / 'metadata/battery_overlap.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
