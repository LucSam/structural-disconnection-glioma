#!/usr/bin/env python3
"""Clinical distributions, high-burden preserved performance and the full QoL family."""
import json
import numpy as np
import pandas as pd
from _common import ROOT, OUTCOMES, NEURO, QOL, load_inputs, safe_spearman, bh_fdr

THRESHOLDS=np.array([63.,63.,64.,64.])


def main():
    blocks,_,_=load_inputs(ROOT)
    outcomes=pd.read_csv(ROOT/'data/private/outcomes.csv').set_index('subject_id')
    rows, tertiles, qol_preserved, definitions=[],[],[],{}
    for battery,b in blocks.items():
        local=outcomes.loc[b['subject_ids']].copy()
        burden=b['cov'][:,-1]
        cut1,cut2=np.quantile(burden,[1/3,2/3],method='linear')
        # A value equal to a boundary stays in the lower third; ties are never split.
        third=np.searchsorted([cut1,cut2],burden,side='left')+1
        score=local['AAT_mean_T_score' if battery=='AAT' else 'DemTect_global'].to_numpy()
        if not np.isfinite(score).all():
            raise ValueError('A descriptive global score is missing in a model-complete case')
        global_low=score<(63.5 if battery=='AAT' else 13.)
        preserved=(b['raw']>=THRESHOLDS).all(axis=1) if battery=='AAT' else ~global_low
        high=third==3
        frame=pd.DataFrame({'subject_id':b['subject_ids'],'battery':battery,'burden':burden,'score':score,
                            'tertile':third,'global_below_cutoff':global_low,'preserved':preserved,
                            'high_burden_preserved':high & preserved})
        if battery=='AAT':
            frame['n_low_subtests']=(b['raw']<THRESHOLDS).sum(axis=1)
            frame['score_category']=np.where(preserved,'No AAT subtest deficit',np.where(global_low,'Mean T-score below 63.5','Mean at least 63.5; at least one low subtest'))
        else:
            frame['n_low_subtests']=np.nan
            frame['score_category']=np.where(preserved,'DemTect at least 13','DemTect below 13')
        for col in QOL:frame[col]=local[col].to_numpy()
        rows.append(frame)
        definitions[battery]={'n':len(frame),'burden':b['burden'],'lower_tertile_boundary':float(cut1),'upper_tertile_boundary':float(cut2),
                              'high_rule':'strictly above empirical 2/3 quantile; boundary ties remain in lower group',
                              'n_preserved':int(preserved.sum()),'n_high_burden':int(high.sum()),'n_high_burden_preserved':int((high&preserved).sum()),
                              'n_global_below':int(global_low.sum())}
        for t in (1,2,3):
            sub=frame[frame.tertile.eq(t)]
            tertiles.append({'battery':battery,'tertile':t,'n':len(sub),'burden_min':sub.burden.min(),'burden_max':sub.burden.max(),
                             'global_below_cutoff':int(sub.global_below_cutoff.sum()),'any_subtest_below_cutoff':int((~sub.preserved).sum()) if battery=='AAT' else np.nan,
                             'preserved':int(sub.preserved.sum()),'preserved_percent':100*sub.preserved.mean()})
        for col,(label,_,focus) in QOL.items():
            if not focus:continue
            values=frame.loc[frame.high_burden_preserved,col].dropna()
            qol_preserved.append({'battery':battery,'qol_measure':col,'qol_label':label,'n_preserved_high_total':int((high&preserved).sum()),
                                  'n_qol_available':len(values),'min':values.min(),'max':values.max(),'median':values.median()})
    participants=pd.concat(rows,ignore_index=True)
    participants.to_csv(ROOT/'data/private/analysis_participants.csv',index=False)
    pd.DataFrame(tertiles).to_csv(ROOT/'results/burden_tertiles.csv',index=False)
    pd.DataFrame(qol_preserved).to_csv(ROOT/'results/high_burden_preserved_qol.csv',index=False)
    (ROOT/'metadata/clinical_definitions.json').write_text(json.dumps(definitions,indent=2)+'\n')
    corr=[]
    for ncol,nlabel in NEURO.items():
        for qcol,(qlabel,higher,focus) in QOL.items():
            n,r,p=safe_spearman(outcomes[ncol],outcomes[qcol])
            corr.append({'neuropsych_measure':ncol,'neuropsych_label':nlabel,'qol_measure':qcol,'qol_label':qlabel,
                         'qol_higher_better':higher,'qol_focus':focus,'n':n,'spearman_rho':r,'p_value':p})
    corr=pd.DataFrame(corr)
    corr['q_fdr_all_neuropsych_qol_tests']=bh_fdr(corr.p_value)
    corr.to_csv(ROOT/'results/qol_correlations.csv',index=False)
    original=pd.read_csv(ROOT/'data/reference/qol_neuropsych_correlations.csv')
    keys=['neuropsych_measure','qol_measure']
    joined=corr.merge(original,on=keys,suffixes=('_v2','_original'),validate='one_to_one')
    for col in ['n','spearman_rho','p_value','q_fdr_all_neuropsych_qol_tests']:
        np.testing.assert_allclose(joined[col+'_v2'],joined[col+'_original'],atol=1e-12,equal_nan=True)
    print(pd.DataFrame(tertiles).to_string(index=False))
    print(pd.DataFrame(qol_preserved).to_string(index=False))
    print('QoL: identical to original full family;',corr.p_value.notna().sum(),'estimable tests')


if __name__=='__main__':
    main()
