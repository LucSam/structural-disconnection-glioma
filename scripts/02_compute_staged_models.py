#!/usr/bin/env python3
"""Compare clinical covariates, additional burden, and additional spatial pattern."""
import json
import numpy as np
import pandas as pd
from _common import ROOT, load_inputs, permutation_test, reproduce_existing, bh_fdr, sha256

N_PERM = 19999
SEED = 20260923


def main():
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    plan = {
        'status': 'Exploratory revision; model scope fixed before this run',
        'outcomes': 'Four AAT subtests or five DemTect subtests, inverse-normal ranks within complete cases',
        'M0': 'Intercept, age, WHO grade, log1p tumour volume, hemisphere',
        'M1': 'M0 plus mean regional ChaCo (left hemisphere AAT; whole brain DemTect)',
        'M2': 'M1 plus eight regional ChaCo PCs',
        'sensitivity': 'Replace PCs with eight AAT or seven nonredundant DemTect anatomical means',
        'tests': 'Burden: M1 versus M0; pattern: M2 versus M1; anatomical pattern versus M1, each battery',
        'correction': 'Benjamini-Hochberg FDR jointly across all six model comparisons',
        'permutations': N_PERM, 'seed_AAT': SEED, 'seed_DemTect': SEED+1,
        'method': 'Pillai trace; Freedman-Lane residual-vector permutations under the comparison-specific reduced model',
        'description': 'No prediction, no causal or resilience test, no significance-based model selection',
    }
    (ROOT/'metadata/staged_model_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    blocks, inputs, diagnostics = load_inputs(ROOT)
    reproduced = reproduce_existing(blocks, pd.read_csv(inputs['files']['existing_joint']))
    reproduced.to_csv(out/'original_joint_reproduction.csv', index=False)
    rows, directions = [], []
    for battery,b in blocks.items():
        for question, representation, x, cov in [
            ('Extent beyond clinical factors', 'Mean regional ChaCo', b['cov'][:,[-1]], b['cov'][:,:-1]),
            ('Pattern beyond extent', 'Eight PCs', b['pc8'], b['cov']),
            ('Pattern sensitivity', 'Anatomical means', b['anatomical'], b['cov']),
        ]:
            r = permutation_test(b['rank'],x,cov,N_PERM,np.random.default_rng(SEED+int(battery=='DemTect')))
            rows.append({'battery':battery,'question':question,'representation':representation,'burden_covariate':b['burden'],**r})
            if question.startswith('Extent'):
                design=np.column_stack([np.ones(len(x)),cov,x])
                coef=np.linalg.lstsq(design,b['rank'],rcond=None)[0][-1]
                iqr=float(np.subtract(*np.percentile(x,[75,25])))
                for label,value in zip(b['labels'],coef,strict=True):
                    directions.append({'battery':battery,'subtest':label,'burden_iqr':iqr,'rank_normal_score_change_per_burden_iqr':float(value*iqr)})
            print(f'Computed {battery}: {question}',flush=True)
    results=pd.DataFrame(rows)
    results['q_fdr_six_models']=bh_fdr(results['p_permutation'])
    results.to_csv(out/'staged_models.csv',index=False)
    pd.DataFrame(directions).to_csv(out/'extent_coefficient_directions.csv',index=False)
    (ROOT/'metadata/model_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(results[['battery','question','n','pillai','p_permutation','q_fdr_six_models']].to_string(index=False))
    assert all(sha256(ROOT/p)==h for p,h in inputs['hashes'].items()), 'Input changed'


if __name__=='__main__':
    main()
