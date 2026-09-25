#!/usr/bin/env python3
"""Descriptive anatomy of high-burden groups and QoL display smoothers.

No subgroup hypothesis tests, matching, new cutoffs or smaller correction
families are introduced. Regional means use the same high-burden definition
and complete cases as the central models.
"""
import json

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from _common import ROOT, anatomical_group


def main():
    patients = pd.read_csv(ROOT / 'data/private/display_participants.csv')
    regional = pd.read_csv(ROOT / 'data/private/regional_chaco.csv').set_index('subject_id')
    atlas = pd.read_csv(ROOT / 'data/resources/network_definitions.csv').sort_values('roi_index')
    atlas['anatomical_group'] = atlas.roi_name.map(anatomical_group)
    rows, summaries, grouped = [], [], []
    for battery in ['AAT', 'DemTect']:
        high = patients[patients.battery.eq(battery) & patients.tertile.eq(3)]
        for preserved, group in [(True, 'No deficit'), (False, 'Deficit')]:
            frame = high[high.preserved.eq(preserved)]
            values = regional.loc[frame.subject_id, [f'chaco_roi_{i}' for i in atlas.roi_index]]
            means = values.mean(axis=0).to_numpy()
            assert len(values) == len(frame) and np.isfinite(means).all()
            summaries.append({'battery': battery, 'group': group, 'preserved': preserved,
                              'n': len(frame), 'mean_burden': frame.burden.mean(),
                              'sd_burden': frame.burden.std(), 'min_burden': frame.burden.min(), 'max_burden': frame.burden.max()})
            for r, mean in zip(atlas.itertuples(), means, strict=True):
                rows.append({'battery': battery, 'group': group, 'n': len(frame),
                             'roi_index': r.roi_index, 'roi_name': r.roi_name,
                             'anatomical_group': r.anatomical_group, 'mean_chaco': mean})
            for scope in ['all parcels', 'left-labelled parcels']:
                selected = atlas if scope == 'all parcels' else atlas[atlas.roi_name.str.endswith('(L)')]
                for name, group_atlas in selected.groupby('anatomical_group'):
                    cols = [f'chaco_roi_{i}' for i in group_atlas.roi_index]
                    per_patient = values[cols].mean(axis=1)
                    grouped.append({'battery': battery, 'group': group, 'scope': scope,
                                    'anatomical_group': name, 'n': len(frame), 'n_parcels': len(cols),
                                    'mean_chaco': per_patient.mean(), 'sd_chaco': per_patient.std()})
    pd.DataFrame(rows).to_csv(ROOT / 'results/high_burden_parcel_means.csv', index=False)
    pd.DataFrame(summaries).to_csv(ROOT / 'results/high_burden_group_summary.csv', index=False)
    pd.DataFrame(grouped).to_csv(ROOT / 'results/high_burden_anatomical_means.csv', index=False)
    outcomes = pd.read_csv(ROOT / 'data/private/outcomes.csv')
    smoothers = []
    q = 'EORTC_QLQ_BN20_communication_deficit'
    for battery, score in [('AAT', 'AAT_mean_T_score'), ('DemTect', 'DemTect_global')]:
        frame = outcomes[[score, q]].dropna()
        fit = lowess(frame[q], frame[score], frac=2/3, it=3, return_sorted=True)
        assert np.isfinite(fit).all()
        # Repeated test values share a fitted ordinate; retain one plotting
        # coordinate per distinct score, rather than patient-level rows.
        display = pd.DataFrame({'score': fit[:, 0], 'fitted_communication': fit[:, 1]}).groupby('score', as_index=False).mean()
        display['battery'] = battery
        smoothers.append(display)
    pd.concat(smoothers, ignore_index=True).to_csv(ROOT / 'results/qol_display_smoothers.csv', index=False)
    (ROOT / 'metadata/group_anatomy_definitions.json').write_text(json.dumps({
        'groups': 'Highest third of existing battery-specific burden; existing deficit/no-deficit definitions',
        'map': 'All 191 parcel means in each group, filled atlas projections with a common colour scale',
        'regional_summaries': 'Both all-parcel and left-labelled summaries exported; scope specified whenever values are reported',
        'inference': 'Unadjusted subgroup descriptions, no matching or new significance test',
        'LOWESS': {'frac': 2/3, 'iterations': 3, 'input': 'Original, unjittered paired values',
                   'purpose': 'Visual guide; Spearman correlations and original full-family q values remain the inferential results'}
    }, indent=2) + '\n')
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
