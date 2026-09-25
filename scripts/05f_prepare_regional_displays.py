#!/usr/bin/env python3
"""Regional effect-size maps and descriptive high-burden differences.

These exports contain coefficients, not a new set of hypothesis tests.
Every atlas parcel is retained; no significance-based parcel selection is used.
"""
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from _common import ROOT, OUTCOMES, sha256


def adjusted_rho(frame, outcome, columns):
    covariates = ['age_at_inclusion', 'grade', 'tumor_volume_ml', 'hemisphere_R']
    d = frame.dropna(subset=[outcome, *covariates, *columns])
    c = np.column_stack([np.ones(len(d)), rankdata(d[covariates], axis=0)])
    y = rankdata(d[outcome].to_numpy(dtype=float))
    x = rankdata(d[columns].to_numpy(dtype=float), axis=0)
    yr = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    xr = x - c @ np.linalg.lstsq(c, x, rcond=None)[0]
    denominator = np.sqrt(np.sum(yr*yr) * np.sum(xr*xr, axis=0))
    return len(d), np.divide(-(yr @ xr), denominator, out=np.full(len(columns), np.nan), where=denominator>1e-10)


def main():
    original = ROOT.parent / 'manuscript-data'
    atlas_source = original / 'data/resources/nemo_fs191_parcellation.nii.gz'
    atlas_target = ROOT / 'data/resources/nemo_fs191_parcellation.nii.gz'
    if not atlas_target.exists():
        shutil.copy2(atlas_source, atlas_target)
    d = pd.read_csv(ROOT / 'data/private/cohort.csv').merge(
        pd.read_csv(ROOT / 'data/private/outcomes.csv'), on='subject_id', validate='one_to_one').merge(
        pd.read_csv(ROOT / 'data/private/regional_chaco.csv'), on='subject_id', validate='one_to_one')
    d['hemisphere_R'] = d.hemisphere.eq('R').astype(float)
    labels = pd.read_csv(ROOT / 'data/resources/network_definitions.csv').sort_values('roi_index')
    columns = [f'chaco_roi_{i}' for i in labels.roi_index]
    outcomes = [('AAT', 'Mean AAT T-score', 'AAT_mean_T_score', 'global'),
                ('DemTect', 'DemTect global', 'DemTect_global', 'global')]
    outcomes += [(b, name, column, 'subtest') for b, tests in OUTCOMES.items() for name,column in tests.items()]
    rows = []
    for battery, name, column, scope in outcomes:
        n, rho = adjusted_rho(d, column, columns)
        for roi,value in zip(labels.itertuples(),rho,strict=True):
            rows.append({'battery':battery, 'outcome':name, 'scope':scope, 'n':n,
                         'roi_index':roi.roi_index,'roi_name':roi.roi_name,'deficit_rho':value})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / 'results/regional_display_correlations.csv', index=False)
    means = pd.read_csv(ROOT / 'results/high_burden_parcel_means.csv')
    differences = means.pivot(index=['battery','roi_index','roi_name'],columns='group',values='mean_chaco').reset_index()
    differences['difference_percentage_points'] = 100*(differences['Deficit']-differences['No deficit'])
    differences.to_csv(ROOT / 'results/high_burden_regional_differences.csv', index=False)
    # Independent check against the saved, hemisphere-adjusted DemTect map.
    reference_source = original / 'results/analysis/regional_chaco/regional_chaco_global_score_partial_sensitivity.csv'
    reference_target = ROOT / 'data/reference/regional_global_sensitivity.csv'
    if not reference_target.exists():
        shutil.copy2(reference_source, reference_target)
    reference = pd.read_csv(reference_target)
    ref = reference[reference.sensitivity.eq('hemisphere_adjusted_all_subjects') & reference.outcome.eq('DemTect_global')].sort_values('roi_index')
    current = result[result.outcome.eq('DemTect global')].sort_values('roi_index')
    error = float(np.max(np.abs(current.deficit_rho.to_numpy()-ref.deficit_rho_partial.to_numpy())))
    assert error < 1e-10, error
    metadata = {'role':'Descriptive effect-size maps; no p values, new correction family or selected parcels',
                'covariates':['age','WHO grade','tumour volume','hemisphere'],
                'coefficient':'Negative correlation between rank residuals; positive indicates lower performance with greater disconnection',
                'mean_burden_adjusted':False,'aat_global_measure':'Recorded mean T-score, not raw total',
                'n_maps':11,'n_parcels_per_map':191,'max_error_saved_DemTect_map':error,
                'difference':'Deficit group minus no-deficit group, percentage points; unadjusted, not matched',
                'input_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [atlas_target,reference_target]}}
    (ROOT / 'metadata/regional_displays.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(result.groupby(['battery','outcome'],sort=False).agg(n=('n','first'),minimum=('deficit_rho','min'),maximum=('deficit_rho','max')).to_string())


if __name__ == '__main__':
    main()
