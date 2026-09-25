#!/usr/bin/env python3
"""Structural coverage, group descriptions and the disconnection–QoL link.

Anatomical groups use fs191 parcel names only. New displays have no hypothesis
tests. Previously reported regional QoL tests retain their original families.
"""
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

from _common import ROOT, QOL, anatomical_group, safe_spearman, bh_fdr, sha256

GROUPS = ['Frontal/insula', 'Perisylvian/central sulci', 'Temporal', 'Parietal',
          'Occipital/visual', 'Cingulate/medial', 'Subcortical/limbic', 'Cerebellum']
FEATURES = {'left_mean': 'Mean left-labelled ChaCo', 'whole_mean': 'Mean whole-brain ChaCo',
            'AAT_edges': 'ChaCoConn: AAT-associated edges',
            'DemTect_edges': 'ChaCoConn: DemTect-associated edges'}


def main():
    private = ROOT / 'data/private'
    regional = pd.read_csv(private / 'regional_chaco.csv').set_index('subject_id')
    labels = pd.read_csv(ROOT / 'data/resources/network_definitions.csv')[['roi_index','roi_name']].sort_values('roi_index')
    labels['anatomical_group'] = labels.roi_name.map(anatomical_group)
    assert set(labels.anatomical_group) == set(GROUPS)
    labels['side'] = labels.roi_name.str.extract(r'\(([LRV])\)$')
    labels.to_csv(ROOT / 'results/structural_parcel_groups.csv', index=False)
    cols = [f'chaco_roi_{i}' for i in labels.roi_index]
    rv = regional[cols].to_numpy()
    coverage = labels.copy()
    coverage['n'] = len(regional)
    coverage['mean_chaco'] = rv.mean(axis=0)
    coverage['n_nonzero'] = (rv > 0).sum(axis=0)
    coverage.to_csv(ROOT / 'results/cohort_regional_coverage.csv', index=False)

    source_path = private / 'tfnbs_hemisphere/chacoconn_edges.npz'
    source = np.load(source_path)
    ids = source['subject_ids'].astype(str)
    edges = source['values'].astype(float)
    i,j = source['roi_i'],source['roi_j']
    assert edges.shape == (163,18145) and len(set(zip(i,j))) == 18145 and np.all(i<j)
    index = {sid:k for k,sid in enumerate(ids)}
    meta = labels.set_index('roi_index')
    gi = meta.loc[i,'anatomical_group'].map({g:k for k,g in enumerate(GROUPS)}).to_numpy()
    gj = meta.loc[j,'anatomical_group'].map({g:k for k,g in enumerate(GROUPS)}).to_numpy()
    lo,hi = np.minimum(gi,gj),np.maximum(gi,gj)
    blocks = {(a,b): np.flatnonzero((lo==a)&(hi==b)) for a in range(8) for b in range(a,8)}
    assert sum(map(len,blocks.values())) == 18145
    grouped_values = {key:edges[:,idx].mean(axis=1) for key,idx in blocks.items()}
    edge_coverage = pd.DataFrame({'roi_i':i,'roi_j':j,'mean_chacoconn':edges.mean(axis=0),
                                 'n_nonzero':(edges>0).sum(axis=0),'n':len(ids)})
    edge_coverage.to_csv(ROOT / 'results/cohort_edge_coverage.csv',index=False)
    grouped_coverage=[]
    for (a,b),value in grouped_values.items():
        grouped_coverage.append({'group_i':GROUPS[a],'group_j':GROUPS[b],
                                 'n_edge_pairs':len(blocks[(a,b)]),'n':len(ids),
                                 'mean_chacoconn':value.mean()})
    pd.DataFrame(grouped_coverage).to_csv(ROOT / 'results/cohort_edge_group_coverage.csv',index=False)

    patients = pd.read_csv(private / 'display_participants.csv')
    group_rows, all_edges = [],[]
    for battery in ['AAT','DemTect']:
        high = patients[patients.battery.eq(battery)&patients.tertile.eq(3)]
        for preserved,name in [(True,'No deficit'),(False,'Deficit')]:
            cases = high[high.preserved.eq(preserved)]
            selection = np.array([index[sid] for sid in cases.subject_id])
            means = edges[selection].mean(axis=0)
            all_edges.append(pd.DataFrame({'battery':battery,'group':name,'n':len(cases),
                                          'roi_i':i,'roi_j':j,'mean_chacoconn':means}))
            for (a,b),value in grouped_values.items():
                group_rows.append({'battery':battery,'group':name,'n':len(cases),
                    'group_i':GROUPS[a],'group_j':GROUPS[b],'n_edge_pairs':len(blocks[(a,b)]),
                    'mean_chacoconn':value[selection].mean()})
    pd.concat(all_edges,ignore_index=True).to_csv(ROOT / 'results/high_burden_edge_means.csv.gz',index=False,
                                                compression={'method':'gzip','mtime':0})
    groups = pd.DataFrame(group_rows)
    groups.to_csv(ROOT / 'results/high_burden_edge_group_means.csv',index=False)
    diff=groups.pivot(index=['battery','group_i','group_j','n_edge_pairs'],columns='group',values='mean_chacoconn').reset_index()
    diff['difference_percentage_points']=100*(diff['Deficit']-diff['No deficit'])
    diff.to_csv(ROOT / 'results/high_burden_edge_group_differences.csv',index=False)

    features=pd.DataFrame(index=regional.index)
    left=labels.loc[labels.side.eq('L'),'roi_index']
    features['left_mean']=regional[[f'chaco_roi_{x}' for x in left]].mean(axis=1)
    features['whole_mean']=regional[cols].mean(axis=1)
    features['top_decile']=np.sort(rv,axis=1)[:,-int(np.ceil(len(cols)*.1)):].mean(axis=1)
    stats=pd.read_csv(ROOT / 'results/tfnbs_hemisphere/edge_statistics.csv.gz')
    edge_index={pair:k for k,pair in enumerate(zip(i,j))}
    set_rows=[];set_counts={}
    for battery in ['AAT','DemTect']:
        selected=stats[stats.battery.eq(battery)&stats.significant_battery_bonferroni]
        pairs=selected.groupby(['roi_i','roi_j']).size().rename('n_subtests').reset_index()
        positions=[edge_index[(r.roi_i,r.roi_j)] for r in pairs.itertuples()]
        assert len(positions)>0
        features[f'{battery}_edges']=pd.Series(edges[:,positions].mean(axis=1),index=ids).reindex(features.index)
        pairs['battery']=battery;set_rows.append(pairs);set_counts[battery]=len(pairs)
    pd.concat(set_rows,ignore_index=True).to_csv(ROOT / 'results/qol_connection_sets.csv',index=False)
    outcomes=pd.read_csv(private / 'outcomes.csv').set_index('subject_id')
    data=outcomes.join(features,validate='one_to_one')
    assert data[list(features.columns)].notna().all().all()
    data.to_csv(private / 'network_qol_participants.csv')
    correlations=[]
    for key,label in FEATURES.items():
        for col,(name,_,focus) in QOL.items():
            if not focus:continue
            d=data[[key,col]].dropna()
            rho=float(spearmanr(d[key],d[col]).statistic)
            correlations.append({'feature':key,'label':label,'qol_measure':col,'qol_label':name,
                'n':len(d),'rho':rho,'selected_using_performance':key.endswith('_edges'),
                'role':'Descriptive coefficient; no p value or new significance family'})
    correlations=pd.DataFrame(correlations)
    correlations.to_csv(ROOT / 'results/network_qol_correlations.csv',index=False)

    # Reproduce and preserve the historical regional and joint structural families.
    refdir=ROOT / 'data/reference'
    copied=[]
    for filename in ['qol_structural_disconnection_correlations.csv','qol_structural_joint_51_sensitivity.csv']:
        target=refdir / filename
        if not target.exists():shutil.copy2(ROOT.parent/'manuscript-data/results/analysis/qol'/filename,target)
        copied.append(target)
    reference=pd.read_csv(copied[0]);joint=pd.read_csv(copied[1])
    assert len(joint)==51 and np.allclose(joint.q_fdr_joint_51,bh_fdr(joint.p))
    rows=[]
    for key,label in [('whole_mean','Mean regional NeMo ChaCo'),('top_decile','Top-decile regional NeMo ChaCo')]:
        for col,(name,_,focus) in QOL.items():
            if not focus:continue
            n,rho,p=safe_spearman(data[key],data[col])
            old=reference[reference.structural_feature.eq(label)&reference.qol_link.eq(name)].iloc[0]
            old_joint=joint[joint.structural_feature.eq(label)&joint.qol_link.eq(name)].iloc[0]
            np.testing.assert_allclose([n,rho,p],[old.n,old.rho_feature_vs_better_qol,old.p],atol=1e-12)
            rows.append({'feature':key,'label':label,'qol_measure':col,'qol_label':name,
                         'n':n,'rho':rho,'p':p,'q_six':old.q_fdr_6_fixed_regional_qol_tests,
                         'q_joint_51':old_joint.q_fdr_joint_51})
    prior=pd.DataFrame(rows)
    np.testing.assert_allclose(prior.q_six,bh_fdr(prior.p))
    prior.to_csv(ROOT / 'results/prior_regional_qol_families.csv',index=False)
    smoothers=[]
    communication='EORTC_QLQ_BN20_communication_deficit'
    for key in ['left_mean','whole_mean']:
        d=data[[key,communication]].dropna()
        fit=lowess(d[communication],100*d[key],frac=2/3,it=3,return_sorted=True)
        smoother=pd.DataFrame({'x':fit[:,0],'fitted_communication':fit[:,1]}).groupby('x',as_index=False).mean()
        smoother['feature']=key;smoothers.append(smoother)
    pd.concat(smoothers,ignore_index=True).to_csv(ROOT / 'results/network_qol_smoothers.csv',index=False)

    # Audit the three right-sided regions mentioned in the global DemTect map.
    cohort=pd.read_csv(private / 'cohort.csv').set_index('subject_id')
    audit=data[['DemTect_global']].join(cohort[['hemisphere']]).join(regional[cols])
    audit=audit.dropna(subset=['DemTect_global','hemisphere'])
    audit_rows=[]
    for side in ['L','R']:
        for deficit in [False,True]:
            d=audit[audit.hemisphere.eq(side)&audit.DemTect_global.lt(13).eq(deficit)]
            for roi in labels.itertuples():
                x=d[f'chaco_roi_{roi.roi_index}']
                audit_rows.append({'lesion_side':side,'deficit':deficit,'n':len(d),
                                  'roi_index':roi.roi_index,'roi_name':roi.roi_name,
                                  'mean_chaco':x.mean(),'n_nonzero':x.gt(0).sum()})
    audit_table=pd.DataFrame(audit_rows)
    audit_table.to_csv(ROOT / 'results/demtect_regional_support.csv',index=False)
    metadata={'role':'Descriptive structural context and QoL bridge; no new hypothesis tests',
        'anatomical_grouping':'fs191 names only; no functional-system labels',
        'edge_group_mean':'Equal-weight mean of all unordered atlas-pair ChaCoConn estimates within each anatomical pair; zeros included',
        'coverage':'Unselected means across all 163 patients; all 191 regions and 18,145 atlas pairs',
        'high_burden_groups':'Existing thirds and test boundaries; unadjusted and unmatched',
        'edge_sets':set_counts,'selection':'Union within each battery of main-model edges surviving edge-FWE and across-subtest correction; duplicates counted once',
        'qol_sample':'All available QoL pairs; missing neuropsychological assessments do not exclude a structural-QoL pair',
        'inference':'All 12 structural display coefficients descriptive; original 6- and 51-test corrections retained separately',
        'prespecification':'Fixed regional formulas do not establish prior specification of the analysis family',
        'mediation':'Not tested; no comparison of correlation strengths or causal interpretation',
        'input_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [source_path,*copied,
            private/'regional_chaco.csv',private/'outcomes.csv',private/'cohort.csv',
            private/'display_participants.csv',ROOT/'data/resources/network_definitions.csv',
            ROOT/'results/tfnbs_hemisphere/edge_statistics.csv.gz']}}
    (ROOT / 'metadata/network_context.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Unique selected edge sets:',set_counts)
    print(correlations[['feature','qol_label','n','rho']].to_string(index=False))
    selected_names=['S_subparietal (R)','G_cingul-Post-dorsal (R)','G_parietal_sup (R)']
    print(audit_table[audit_table.lesion_side.eq('L')&audit_table.roi_name.isin(selected_names)].to_string(index=False))
    print('Largest absolute anatomical-pair differences (percentage points):')
    for battery in ['AAT','DemTect']:
        d=diff[diff.battery.eq(battery)]
        print(d.loc[d.difference_percentage_points.abs().nlargest(5).index,['battery','group_i','group_j','difference_percentage_points']].to_string(index=False))


if __name__=='__main__':
    main()
