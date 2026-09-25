#!/usr/bin/env python3
"""Derive display quantities from the same complete cases and fitted joint models."""
import json
import numpy as np
import pandas as pd
from _common import ROOT, load_inputs, orthogonal_design


def main():
    blocks, _, _ = load_inputs(ROOT)
    participants = pd.read_csv(ROOT / 'data/private/analysis_participants.csv')
    effects, margins = [], []
    models = pd.read_csv(ROOT / 'results/staged_models.csv')
    for battery, b in blocks.items():
        qc, qx = orthogonal_design(b['pc8'], b['cov'])
        residual = b['rank'] - qc @ (qc.T @ b['rank'])
        fitted = qx @ (qx.T @ residual)
        numerator = np.sum(fitted**2, axis=0)
        denominator = np.sum(residual**2, axis=0)
        values = numerator / denominator
        joint = models[models.battery.eq(battery) & models.question.eq('Pattern beyond extent')].iloc[0]
        np.testing.assert_allclose(numerator.sum()/denominator.sum(), joint.partial_r2_trace_in_sample, atol=1e-12)
        for label, value in zip(b['labels'], values, strict=True):
            effects.append({'battery': battery, 'subtest': label, 'n': len(b['raw']),
                            'partial_r2': float(value), 'scale': 'inverse-normal ranks',
                            'model': 'Same joint PC8 model, conditional on clinical factors and mean regional ChaCo',
                            'interpretation': 'Descriptive in-sample fit; no additional significance test'})
        if battery == 'AAT':
            margin = np.min(b['raw'] - np.array([63., 63., 64., 64.]), axis=1)
        else:
            margin = np.full(len(b['raw']), np.nan)
        margins.extend({'subject_id': sid, 'battery': battery, 'minimum_aat_margin': value}
                       for sid, value in zip(b['subject_ids'], margin, strict=True))
    # A separate private display table leaves the original clinical output unchanged.
    participants = participants.merge(pd.DataFrame(margins), on=['subject_id','battery'], validate='one_to_one')
    aat = participants[participants.battery.eq('AAT')]
    assert np.array_equal(aat.minimum_aat_margin.ge(0).to_numpy(), aat.preserved.to_numpy())
    participants.to_csv(ROOT / 'data/private/display_participants.csv', index=False)
    pd.DataFrame(effects).to_csv(ROOT / 'results/subtest_pattern_partial_r2.csv', index=False)
    (ROOT / 'metadata/display_definitions.json').write_text(json.dumps({
        'AAT_y': 'Minimum of Token Test minus 63, Repetition minus 63, Naming minus 64 and Comprehension minus 64',
        'DemTect_y': 'DemTect global score; threshold 13',
        'status_colours': {'no_deficit':'#2A6F97','AAT_subtest_only':'#BD8A35','global_below':'#8B1E3F'},
        'highlight': 'Black outline: highest burden third and no deficit',
        'partial_r2': 'Per-outcome partial R-squared from the exact joint model samples and outcome transformation; no new inferential test',
    }, indent=2) + '\n')
    print(pd.DataFrame(effects)[['battery','subtest','n','partial_r2']].to_string(index=False))


if __name__ == '__main__':
    main()
