#!/usr/bin/env python3
"""Validate saved TFNBS results and compare hemisphere and lobe models."""
import json
import numpy as np
import pandas as pd
from _common import ROOT

GROUPS=['Subcortical/limbic','Cerebellum','Frontal/insula','Perisylvian/central sulci','Temporal','Parietal','Occipital/visual','Cingulate/medial']


def number(value):
    return int(str(value).replace(',',''))


def summarise_revision(original):
    hemisphere = pd.read_csv(ROOT/'results/tfnbs_hemisphere/edge_statistics.csv.gz')
    counts = pd.read_csv(ROOT/'results/tfnbs_hemisphere/counts.csv')
    lobe = original[original.model_family.eq('lobe')]
    comparison, anatomy = [], []
    for row in counts.itertuples():
        h = hemisphere[hemisphere.subtest_label.eq(row.subtest)]
        l = lobe[lobe.subtest_label.eq(row.subtest)]
        assert int(h.significant_fwe_edges.sum()) == row.edge_FWE
        assert int(h.significant_battery_bonferroni.sum()) == row.across_subtests
        keys = lambda frame: set(zip(frame.roi_i, frame.roi_j))
        shared = keys(h[h.significant_battery_bonferroni]) & keys(l[l.significant_battery_bonferroni])
        comparison.append({'battery': row.battery, 'subtest': row.subtest, 'n': row.n,
                           'hemisphere_edge_FWE': row.edge_FWE, 'hemisphere_across_subtests': row.across_subtests,
                           'lobe_edge_FWE': int(l.significant_fwe_edges.sum()),
                           'lobe_across_subtests': int(l.significant_battery_bonferroni.sum()),
                           'across_subtest_edges_shared_by_models': len(shared)})
    for family, frame in [('hemisphere', hemisphere), ('lobe', lobe)]:
        frame = frame[frame.significant_battery_bonferroni].copy()
        frame['class_pair'] = [' - '.join(sorted([a,b],key=GROUPS.index))
                               for a,b in zip(frame.roi_i_anatomical_group,frame.roi_j_anatomical_group)]
        side = lambda value: 'L' if value.endswith('(L)') else 'R' if value.endswith('(R)') else 'V'
        frame['hemisphere_pair'] = ['–'.join(sorted([side(a),side(b)])) for a,b in zip(frame.roi_i_name,frame.roi_j_name)]
        frame['cerebellar_endpoint'] = frame.roi_i_anatomical_group.eq('Cerebellum') | frame.roi_j_anatomical_group.eq('Cerebellum')
        anatomy.append(frame)
    detailed = pd.concat(anatomy,ignore_index=True)
    detailed.to_csv(ROOT/'results/revision_retained_connections.csv',index=False)
    detailed.groupby(['model_family','battery','subtest_label','class_pair','hemisphere_pair','cerebellar_endpoint']).size().rename('n_edges').reset_index().to_csv(ROOT/'results/revision_connection_anatomy.csv',index=False)
    pd.DataFrame(comparison).to_csv(ROOT/'results/connection_model_comparison.csv',index=False)
    print(pd.DataFrame(comparison).to_string(index=False))


def main():
    edges=pd.read_csv(ROOT/'data/reference/tfnbs_edges.csv')
    summary=pd.read_csv(ROOT/'data/reference/tfnbs_summary.csv')
    checks=[]
    for row in summary.to_dict('records'):
        for family,prefix in [('clinical','Clinical'),('lobe','Location-adjusted')]:
            subset=edges[edges.battery.eq(row['Battery']) & edges.subtest_label.eq(row['Subtest']) & edges.model_family.eq(family)]
            for flag,suffix in [('significant_fwe_edges','edge FWE'),('significant_battery_bonferroni','across subtests')]:
                count=int(subset[flag].sum())
                expected=number(row[f'{prefix}: {suffix}'])
                if count!=expected:raise ValueError(f'TFNBS count mismatch: {row["Subtest"]}, {family}, {suffix}')
                checks.append({'battery':row['Battery'],'subtest':row['Subtest'],'model':family,'correction':suffix,'n_edges':count})
    pd.DataFrame(checks).to_csv(ROOT/'results/tfnbs_count_verification.csv',index=False)
    primary=edges[edges.model_family.eq('lobe') & edges.significant_battery_bonferroni].copy()
    primary.to_csv(ROOT/'results/location_adjusted_edges.csv',index=False)
    group_counts=primary.groupby(['battery','subtest_label','class_pair']).size().rename('n_edges').reset_index()
    group_counts.to_csv(ROOT/'results/location_adjusted_anatomical_pairs.csv',index=False)
    table=summary.drop(columns='Leading anatomical pairs (clinical model)').copy()
    for col in table.columns[2:]:table[col]=table[col].map(number)
    table.to_csv(ROOT/'results/connection_counts.csv',index=False)
    (ROOT/'metadata/tfnbs_validation.json').write_text(json.dumps({'all_36_counts_match':True,'new_permutations_run':False,
        'role':'Previously validated clinical-only reference and lobe-adjusted sensitivity; new hemisphere-adjusted results are recorded separately',
        'source':'Previously validated TFNBS output; unchanged data, design, contrast and correction'},indent=2)+'\n')
    print('TFNBS: all 36 counts verified; location-adjusted across-subtest edges:',len(primary))
    summarise_revision(edges)


if __name__=='__main__':main()
